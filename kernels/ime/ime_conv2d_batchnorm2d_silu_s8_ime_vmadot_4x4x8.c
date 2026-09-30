/* source: curated */
/* algorithm: ime_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* origin: kernels/ime/ime_conv2d_s8_ime_vmadot_4x4x8.c (the im2col->vmadot MAC
 *   core) with the BN and SiLU stages of
 *   kernels/rvv/rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c
 *   folded into its store path.
 *
 *   WHY THIS FILE EXISTS. The graph fuses Conv->BN->SiLU into one op named
 *   `conv2d_batchnorm2d_silu_s8`, and curated kernels are looked up by EXACT op
 *   name. The IME library had only `conv2d_s8`, so 57 of the deployed
 *   yolov8_nano_64x96's 90 dispatches -- every backbone convolution, and the
 *   most expensive ones -- could not reach the matrix engine at all, whatever
 *   the shape said.
 *
 *   BIT-EXACTNESS. Three stages, each taken unchanged from the kernel that
 *   already owns it:
 *     conv  -- int32 vmadot MAC + the scalar Q0.31 requantize tail of the IME
 *              conv kernel (bias in the accumulator domain, (acc*mult +
 *              (1<<30))>>31, rounding shift, +conv_output_offset, clamp).
 *     BN    -- mb_icbs_bn_stage(): float fv = conv_int8 * bn_scale_in;
 *              y = bn_s*fv + bn_b; roundf(y / bn_scale_out); clamp.
 *     SiLU  -- the 256-entry table of the RVV kernel: x/(1+exp(-x)) on the BN
 *              stage's int8 output, roundf(/silu_scale_out), clamp.
 *   The MAC is integer, so it is identical to RVV's by construction; the two
 *   float stages are evaluated by the SAME expressions in the same order, so
 *   the results are identical rather than merely close. Nothing below touches
 *   them -- every change in this file moves BYTES, not arithmetic.
 *
 *   THE EPILOGUE IS TABULATED ONLY ABOVE A BREAK-EVEN. Both float stages
 *   consume an int8 and produce an int8, so together they are a 256-entry
 *   function of (conv_int8) per output channel -- memoization, not
 *   approximation. But building that table costs 256 evaluations per channel
 *   while a panel only produces M = OH*OW outputs per channel: tabulating
 *   unconditionally cost 16k evaluations per dispatch against 6k outputs on
 *   this net's 8x12 layers, measured 163 ms against 47 ms for the whole model.
 *   So the table is built only when M >= MB_ICBS_LUT_BREAKEVEN -- the same
 *   trade, and the same threshold, as the RVV kernel's MB_CBS_LUT_BREAKEVEN.
 *
 *   ---------------------------------------------------------------------
 *   WHAT THE FIRST VERSION SPENT ITS TIME ON, AND WHAT THAT COST (measured
 *   per shape with scripts/ime_fused_conv_bench.py against the deployed RVV
 *   fused kernel, board, cluster 0, 57 real dispatches). It won 9 of 57 and
 *   the whole-net best-of was 41.03 ms against 43.57 ms all-RVV. The losses
 *   were not the MAC -- they were the two packing loops:
 *
 *   1. B WAS RE-PACKED FOR EVERY 4-CHANNEL PANEL. The old nest was
 *      `for n0 { pack B panel n0; for mt { MAC } }`, and a panel's pack walks
 *      the weight tensor at stride OC picking 4 bytes out of every OC-byte row.
 *      With OC=256 that is one cache line fetched per byte kept, repeated
 *      OC/4 = 64 times -- 64 passes over a 288 KB weight tensor, ~18 MB of line
 *      traffic, to feed a MAC that only has M=6 rows to amortize it over. This
 *      is why the small-spatial layers lost worst: l7.conv (IC=128 4x6 OC=256
 *      3x3, M=6) measured 1.96 ms against RVV's 0.97 ms.
 *      NOW: B is packed ONCE per dispatch, in one linear pass over the weight
 *      in its native k-major order (weight[k*OC + oc] is contiguous in oc), so
 *      the tensor is read once and written once.
 *
 *   2. THE A GATHER DERIVED ITS TAPS WITH TWO INTEGER DIVISIONS PER BYTE.
 *      `ic = k / (KH*KW); kh = (k - ic*KH*KW) / KW;` ran inside the innermost
 *      loop over packed bytes, so a 3x3 layer paid ~2 hardware divides for each
 *      of its tens of thousands of im2col bytes: l0.conv (1536x27 im2col, 49 KB
 *      packed) measured 2.51 ms against RVV's 0.95 ms, ~80 cycles per packed
 *      byte. NOW the pack loop is nested over (ic, kh, kw) so k is *counted*,
 *      never divided, and each (tap, output row) pair is copied with a vector
 *      strided load / strided store instead of a scalar byte loop.
 *
 *   THE LAYOUT CHANGE THAT MAKES (2) POSSIBLE, and it is the only structural
 *   change to the MAC. vmadot wants its operands as 32-byte 4x8 tiles [i][q].
 *   The old packing order was [m-tile][k-slab][i][q], in which the bytes of one
 *   fixed tap k across consecutive output pixels sit at stride 8 *inside* a
 *   tile and then jump by (k_slabs*32 - 24) at the tile boundary -- not a
 *   constant stride, so the copy could not be vectorised. Storing instead as
 *   [k-slab][m-tile][i][q] makes that stride a uniform 8 bytes for every
 *   pixel, because crossing a tile boundary advances by 32-24 = 8 as well. A
 *   fixed tap's source bytes are a stride-SW run along an input row, so each
 *   (tap, row) pair becomes exactly one vlse8 -> vsse8. B is packed
 *   [k-slab][panel][j][q] for the same reason: at fixed k, consecutive output
 *   channels are 8 bytes apart, so one weight row is one vle8 -> vsse8.
 *   The MAC then walks the slab axis with a stride (m_tiles*32 for A,
 *   panels*32 for B) instead of a fixed 32 -- one `add` in place of one `addi`.
 *   The tiles the engine sees are byte-for-byte the ones it saw before.
 *
 *   AND A THIRD PACKING FIX, worth less than the other two: the general path
 *   packs from a ZERO-PADDED COPY of the input, so ih = oh*SH + kh and
 *   iw = ow*SW + kw are in bounds for every tap and every tap of a k-slab
 *   spans the same run of output pixels -- which is what lets the slab go
 *   through one segment store as well. Whole-net all-IME 32.19 -> 31.80 ms.
 *
 *   WHAT IT MEASURES NOW, on the deployed yolov8_nano_64x96, 20 iterations,
 *   median per dispatch, every run `MODELBLASTER_VERIFY === max_abs_err=0`
 *   (results/codesign_feedback/ros_traced/yolo_standalone/ime_fused_v5_1core.txt
 *   and ime_shard4_v5_4harts.txt):
 *
 *                       fused-conv dispatches   all-IME    per-dispatch best-of
 *     1 hart   before        11 / 57            90.00 ms      45.40 ms
 *     1 hart   after         50 / 57            32.58 ms      32.19 ms   (RVV 47.15)
 *     4 harts  before         0 / 57            52.35 ms      24.25 ms
 *     4 harts  after         45 / 57            16.77 ms      15.71 ms   (RVV 24.26)
 *
 *   Forcing the engine on every fused dispatch is now FASTER than all-RVV
 *   (32.58 vs 47.15 ms at one hart), which it was not before -- so the engine
 *   is no longer only a per-dispatch alternative, though the best-of is still
 *   better than either pure arm and the 7 remaining losses are real.
 *
 *   WHAT IS LEFT, and why it is structural. The losses are the M=6 (2x3
 *   feature map) 3x3 convolutions -- detect.cv{2,3}_2_*, l7, l8.m0, l21.m0,
 *   l19 -- at 0.81-0.99x. M=6 fills two 4-row tiles with two rows wasted AND
 *   amortizes B's transpose (a full pass over the weight tensor, which RVV
 *   never has to make) over only those two tiles. That is the same small-M
 *   limit the matmul kernel's header records (M=7 -> 0.25x); it is now 0.81x
 *   rather than 0.30x, but it does not go away by packing faster.
 *
 *   ONE THING THAT DID NOT WORK, recorded so it is not retried: the general
 *   path's copies are 2-3 bytes long on a 2x3 map, so a scalar loop for n <= 4
 *   inside mb_icbs_copy_strided looks free. It cost 31 of the 49 wins --
 *   all-IME 32.38 -> 48.59 ms -- and the regression was NOT on the short
 *   copies: detect.cv2_1_0 (runs of 5-6, never taking the scalar branch) went
 *   0.661 -> 0.934 ms. The branch makes the helper too big to inline, so every
 *   call becomes a real call with its own vsetvl. Keep the helper branchless.
 *
 *   CLUSTER 0 ONLY. smt.vmadot is illegal on harts 4-7 (they SIGILL), exactly
 *   as for the conv and matmul kernels this is built from.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <riscv_vector.h>

/* Below this many output pixels per channel, building a 256-entry epilogue table costs more
 * than evaluating the two float stages per element (the RVV kernel's MB_CBS_LUT_BREAKEVEN). */
#define MB_ICBS_LUT_BREAKEVEN 256

/* Stack budget for the packed-once B. The deployed net's largest is l7.conv
 * (K=1152, OC=256 -> 288 KB); above the budget B falls back to being packed one
 * panel at a time into the same buffer, which is what the first version always
 * did. */
#define MB_ICBS_B_PACK_BUDGET (512u * 1024u)

/* Stack budget for the zero-padded input copy the general (non-1x1) path packs
 * from; above it the bounds-checked per-tap path runs instead. */
#define MB_ICBS_PAD_BUDGET (256u * 1024u)

/* Scalar Q0.31 requantize, identical to the IME/RVV conv kernels. */
static inline int32_t mb_icbs_q31_requant(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return (scaled + round) >> shift;
    }
    return scaled << (-shift);
}

/* BN stage, character for character the RVV fused kernel's mb_cbs_bn_stage. */
static inline int8_t mb_icbs_bn_stage(int8_t conv_int8,
                                      float bn_s, float bn_b,
                                      float bn_scale_in, float bn_scale_out,
                                      int bn_activation_min,
                                      int bn_activation_max)
{
    float fv = (float)conv_int8 * bn_scale_in;
    float y = bn_s * fv + bn_b;
    int32_t v = (int32_t)roundf(y / bn_scale_out);
    if (v < bn_activation_min) v = bn_activation_min;
    if (v > bn_activation_max) v = bn_activation_max;
    return (int8_t)v;
}

/* The one primitive both packers need: copy `n` int8 from a strided source to a
 * strided destination. Integer e8 only -- no float op is issued under an
 * integer SEW, and the vmadot asm below sets its own vtype on entry. */
static inline void mb_icbs_copy_strided(const int8_t *src, ptrdiff_t src_stride,
                                        int8_t *dst, ptrdiff_t dst_stride, int n)
{
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)n);
        vint8m1_t v = (src_stride == 1)
            ? __riscv_vle8_v_i8m1(src, vl)
            : __riscv_vlse8_v_i8m1(src, src_stride, vl);
        __riscv_vsse8_v_i8m1(dst, dst_stride, v, vl);
        src += (ptrdiff_t)vl * src_stride;
        dst += (ptrdiff_t)vl * dst_stride;
        n -= (int)vl;
    }
}

/* A WHOLE K-SLAB AT ONCE. The destination of a packed slab is the eight taps of
 * one k-slab interleaved 8-ways over the rows -- dst[i*8 + q] -- which is
 * exactly what an 8-field segment store writes. Feeding it eight source runs
 * turns eight stride-8 scatters (each 32 elements spread over 256 bytes, so
 * four lines touched per vector store) into one contiguous 256-byte store.
 * A NULL pointer is a tap past K, whose field must read as zero.
 *
 * MEASURED: with stride-8 scatters instead, the small-spatial layers were still
 * losing to RVV because B's transpose is amortized over only M/4 = 2 m-tiles --
 * detect.cv2_2_0 (K=2304, OC=64, M=6) ran 0.538 ms against RVV's 0.402 ms. */
static inline void mb_icbs_interleave8(
    const int8_t *p0, const int8_t *p1, const int8_t *p2, const int8_t *p3,
    const int8_t *p4, const int8_t *p5, const int8_t *p6, const int8_t *p7,
    ptrdiff_t src_stride, int8_t *dst, int n)
{
#define MB_ICBS_LANE(p) ((p) == NULL ? __riscv_vmv_v_x_i8m1(0, vl)                 \
                         : ((src_stride == 1) ? __riscv_vle8_v_i8m1((p), vl)       \
                            : __riscv_vlse8_v_i8m1((p), src_stride, vl)))
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)n);
        vint8m1x8_t t = __riscv_vundefined_i8m1x8();
        t = __riscv_vset_v_i8m1_i8m1x8(t, 0, MB_ICBS_LANE(p0));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 1, MB_ICBS_LANE(p1));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 2, MB_ICBS_LANE(p2));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 3, MB_ICBS_LANE(p3));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 4, MB_ICBS_LANE(p4));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 5, MB_ICBS_LANE(p5));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 6, MB_ICBS_LANE(p6));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 7, MB_ICBS_LANE(p7));
        __riscv_vsseg8e8_v_i8m1x8(dst, t, vl);
        ptrdiff_t adv = (ptrdiff_t)vl * src_stride;
        if (p0) { p0 += adv; }
        if (p1) { p1 += adv; }
        if (p2) { p2 += adv; }
        if (p3) { p3 += adv; }
        if (p4) { p4 += adv; }
        if (p5) { p5 += adv; }
        if (p6) { p6 += adv; }
        if (p7) { p7 += adv; }
        dst += (ptrdiff_t)vl * 8;
        n -= (int)vl;
    }
#undef MB_ICBS_LANE
}

void kernel_conv2d_batchnorm2d_silu_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max,
    float silu_scale_in, float silu_scale_out,
    int silu_activation_min, int silu_activation_max)
{
    if (OC <= 0 || IC <= 0) return;
    /* Symmetric int8 only -- an asymmetric conv stays on RVV (the picker's job),
     * because vmadot's raw product is the MAC only when both offsets are zero. */
    if (input_offset != 0 || filter_offset != 0) return;

    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW;                 /* im2col rows, per batch element   */
    int K = IC * KH * KW;            /* im2col inner (filter tap) length */
    if (M <= 0 || K <= 0) return;

    /* SiLU stage table: indexed by the BN stage's int8 output + 128. Built once
     * per dispatch; 256 expf() calls total, as in the RVV kernel. */
    int8_t silu_lut[256];
    for (int v = 0; v < 256; v++) {
        int8_t bn_int8 = (int8_t)(v - 128);
        float fbv = (float)bn_int8 * silu_scale_in;
        float sy = fbv / (1.0f + expf(-fbv));
        int32_t q = (int32_t)roundf(sy / silu_scale_out);
        if (q < silu_activation_min) q = silu_activation_min;
        if (q > silu_activation_max) q = silu_activation_max;
        silu_lut[v] = (int8_t)q;
    }

    int m_tiles = (M + 3) / 4;
    int k_slabs = (K + 7) / 8;
    int n_panels = (OC + 3) / 4;
    size_t bytes_per_m_tile = (size_t)k_slabs * 32u;

    size_t max_a_pack_bytes = 256u * 1024u;
    int block_m_tiles = (int)(max_a_pack_bytes / bytes_per_m_tile);
    if (block_m_tiles < 1) block_m_tiles = 1;
    if (block_m_tiles > m_tiles) block_m_tiles = m_tiles;

    /* B packed once for the whole dispatch when it fits the stack budget; the
     * fallback keeps exactly one panel resident, which is the [k-slab][1][j][q]
     * special case of the same layout. */
    size_t b_all_bytes = (size_t)n_panels * (size_t)k_slabs * 32u;
    const int pack_b_once = (b_all_bytes <= MB_ICBS_B_PACK_BUDGET);
    const int b_panels_resident = pack_b_once ? n_panels : 1;
    const size_t b_slab_stride = (size_t)b_panels_resident * 32u;

    int8_t packed_a[(size_t)block_m_tiles * bytes_per_m_tile];
    int8_t packed_b[(size_t)k_slabs * b_slab_stride];
    int32_t tile_output[16];
    int8_t epi_lut[4][256];          /* BN+SiLU for the 4 channels of one panel */

    /* ---- the padded-input form of the general path ----
     * With a zero-padded copy of the input, ih = oh*SH + kh and iw = ow*SW + kw
     * are in bounds for EVERY (oh, ow, kh, kw) the loop can produce, so the tap
     * loop has no bounds test and -- the point -- every tap of a k-slab covers
     * the same run of output pixels, which is the condition for packing the
     * whole slab with one segment store instead of eight strided ones. The
     * 2x3 layers were issuing ~3000 two-byte strided copies per dispatch.
     * The copy costs IC*(IH+2PH)*(IW+2PW) bytes, three orders below the im2col
     * it feeds; above the budget the bounds-checked path below still runs. */
    const int IHp = IH + 2 * PH, IWp = IW + 2 * PW;
    const size_t pad_bytes = (size_t)IC * IHp * IWp;
    const int use_pad = (PH != 0 || PW != 0) && (pad_bytes <= MB_ICBS_PAD_BUDGET);
    const int direct_pad = (PH == 0 && PW == 0);      /* no copy needed at all */
    const int slabwise_a = (use_pad || direct_pad);
    int8_t pad_buf[slabwise_a && !direct_pad ? pad_bytes : 1];
    /* tap_off[k] is the offset of tap k's (oh=0, ow=0) byte in that padded
     * tensor; built by counting (ic, kh, kw), so no index is ever divided. */
    int32_t tap_off[slabwise_a ? (size_t)K : 1];
    if (slabwise_a) {
        int t = 0;
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++)
                    tap_off[t++] = ((int32_t)ic * IHp + kh) * IWp + kw;
    }

    /* ---- pack B, once, in one linear pass over the weight ----
     * weight is IHWOC: weight[k*OC + oc] is contiguous in oc, and the
     * destination for a fixed k is stride-8 in oc, so one weight row is one
     * vle8 -> vsse8. The tail slots (k >= K, oc >= OC) must read as zero. */
    if (pack_b_once) {
        if (OC & 3)                      /* only the tail panel's unused j slots */
            memset(packed_b, 0, (size_t)k_slabs * b_slab_stride);
        for (int ks = 0; ks < k_slabs; ks++) {
            int k0 = ks * 8;
            const int8_t *w0 = weight + (size_t)(k0 + 0) * OC;
            ptrdiff_t st = OC;
            /* dst offset for (oc, q) is (oc/4)*32 + (oc%4)*8 + q == oc*8 + q */
            mb_icbs_interleave8(
                (k0 + 0 < K) ? w0 : NULL, (k0 + 1 < K) ? w0 + st : NULL,
                (k0 + 2 < K) ? w0 + 2*st : NULL, (k0 + 3 < K) ? w0 + 3*st : NULL,
                (k0 + 4 < K) ? w0 + 4*st : NULL, (k0 + 5 < K) ? w0 + 5*st : NULL,
                (k0 + 6 < K) ? w0 + 6*st : NULL, (k0 + 7 < K) ? w0 + 7*st : NULL,
                1, packed_b + (size_t)ks * b_slab_stride, OC);
        }
    }

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;

        for (int mt_outer = 0; mt_outer < m_tiles; mt_outer += block_m_tiles) {
            int current_m_tiles = m_tiles - mt_outer;
            if (current_m_tiles > block_m_tiles) current_m_tiles = block_m_tiles;
            const size_t a_slab_stride = (size_t)current_m_tiles * 32u;
            int m_lo = mt_outer * 4;
            int m_hi = m_lo + current_m_tiles * 4;
            if (m_hi > M) m_hi = M;

            /* ---- pack A ----
             * Everything not written below has to read as zero: the rows past M
             * in the tail tile, the taps past K in the tail slab, and every
             * out-of-bounds (padded) input position. The general path clears the
             * block first, which is one linear memset and makes each copy
             * unconditional; the 1x1 path writes every tap of every slab (a tap
             * past K is stored as a zero field) so it only has to clear when a
             * tail tile has rows past M. */
            const int one_by_one = (KH == 1 && KW == 1 && SH == 1 && SW == 1
                                    && PH == 0 && PW == 0);
            if (one_by_one) {
                /* A[m][k] = in_n[k*M + m]: the im2col matrix is the input
                 * transposed, so a tap is one contiguous run of M pixels, and a
                 * whole k-slab is one segment store. Only the rows past M in the
                 * tail tile still need clearing. */
                int rows = m_hi - m_lo;
                if (rows & 3)
                    memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    const int8_t *c0 = in_n + (size_t)k0 * M + m_lo;
                    ptrdiff_t st = M;
                    mb_icbs_interleave8(
                        (k0 + 0 < K) ? c0 : NULL, (k0 + 1 < K) ? c0 + st : NULL,
                        (k0 + 2 < K) ? c0 + 2*st : NULL, (k0 + 3 < K) ? c0 + 3*st : NULL,
                        (k0 + 4 < K) ? c0 + 4*st : NULL, (k0 + 5 < K) ? c0 + 5*st : NULL,
                        (k0 + 6 < K) ? c0 + 6*st : NULL, (k0 + 7 < K) ? c0 + 7*st : NULL,
                        1, packed_a + (size_t)ks * a_slab_stride, rows);
                }
            } else if (slabwise_a) {
                int rows = m_hi - m_lo;
                if (rows & 3)
                    memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                const int8_t *base_in = in_n;
                if (!direct_pad) {
                    /* build the zero-padded copy once per (n, block) */
                    memset(pad_buf, 0, pad_bytes);
                    for (int ic = 0; ic < IC; ic++)
                        for (int ih = 0; ih < IH; ih++)
                            memcpy(pad_buf + ((size_t)ic * IHp + ih + PH) * IWp + PW,
                                   in_n + ((size_t)ic * IH + ih) * IW, (size_t)IW);
                    base_in = pad_buf;
                }
                int oh_lo = m_lo / OW;
                int oh_hi = (m_hi - 1) / OW;
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    int8_t *slab = packed_a + (size_t)ks * a_slab_stride;
                    for (int oh = oh_lo; oh <= oh_hi; oh++) {
                        int base = oh * OW;
                        int ow0 = 0, ow1 = OW - 1;
                        if (base + ow0 < m_lo) ow0 = m_lo - base;
                        if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                        if (ow0 > ow1) continue;
                        ptrdiff_t o = (ptrdiff_t)oh * SH * IWp + (ptrdiff_t)ow0 * SW;
                        const int8_t *q0 = base_in + o;
                        mb_icbs_interleave8(
                            (k0+0 < K) ? q0 + tap_off[k0+0] : NULL,
                            (k0+1 < K) ? q0 + tap_off[k0+1] : NULL,
                            (k0+2 < K) ? q0 + tap_off[k0+2] : NULL,
                            (k0+3 < K) ? q0 + tap_off[k0+3] : NULL,
                            (k0+4 < K) ? q0 + tap_off[k0+4] : NULL,
                            (k0+5 < K) ? q0 + tap_off[k0+5] : NULL,
                            (k0+6 < K) ? q0 + tap_off[k0+6] : NULL,
                            (k0+7 < K) ? q0 + tap_off[k0+7] : NULL,
                            SW, slab + (size_t)(base + ow0 - m_lo) * 8u,
                            ow1 - ow0 + 1);
                    }
                }
            } else {
                memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                /* Fallback for an input too large to pad on the stack: the taps
                 * are still counted, never divided, but each (tap, row) pair is
                 * its own bounds-checked strided copy. */
                int oh_lo = m_lo / OW;
                int oh_hi = (m_hi - 1) / OW;
                int k = 0;
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++, k++) {
                            int8_t *slab = packed_a
                                + (size_t)(k >> 3) * a_slab_stride + (k & 7);
                            /* ow range for which iw = ow*SW - PW + kw is in [0,IW) */
                            int ow_first = (PW - kw + SW - 1) / SW;   /* SW > 0 */
                            if (ow_first < 0) ow_first = 0;
                            int ow_last_num = IW - 1 - kw + PW;
                            if (ow_last_num < 0) continue;      /* tap never in bounds */
                            int ow_last = ow_last_num / SW;
                            if (ow_last > OW - 1) ow_last = OW - 1;
                            if (ow_first > ow_last) continue;
                            for (int oh = oh_lo; oh <= oh_hi; oh++) {
                                int ih = oh * SH - PH + kh;
                                if (ih < 0 || ih >= IH) continue;
                                int base = oh * OW;
                                int ow0 = ow_first, ow1 = ow_last;
                                if (base + ow0 < m_lo) ow0 = m_lo - base;
                                if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                                if (ow0 > ow1) continue;
                                const int8_t *src = in_n
                                    + ((size_t)ic * IH + ih) * IW
                                    + (size_t)(ow0 * SW - PW + kw);
                                int8_t *dst = slab + (size_t)(base + ow0 - m_lo) * 8u;
                                mb_icbs_copy_strided(src, SW, dst, 8, ow1 - ow0 + 1);
                            }
                        }
                    }
                }
            }

            /* ---- for each 4-col panel of B (= 4 output channels) ---- */
            for (int n0 = 0; n0 < OC; n0 += 4) {
                int panel = n0 >> 2;
                const int8_t *b_panel = packed_b
                    + (pack_b_once ? (size_t)panel * 32u : 0u);
                if (!pack_b_once) {
                    memset(packed_b, 0, (size_t)k_slabs * 32u);
                    for (int k = 0; k < K; k++) {
                        int8_t *dst = packed_b + (size_t)(k >> 3) * 32u + (k & 7);
                        int cols = OC - n0; if (cols > 4) cols = 4;
                        mb_icbs_copy_strided(weight + (size_t)k * OC + n0, 1,
                                             dst, 8, cols);
                    }
                }

                /* The two float stages for these four channels. Tabulating them over the 256
                 * possible conv outputs is memoization of the same expressions -- but the build
                 * costs 256 evaluations per channel, and a panel only produces M outputs per
                 * channel. Below the break-even the per-element path is cheaper, which is the
                 * same trade (and the same threshold) the RVV kernel makes with
                 * MB_CBS_LUT_BREAKEVEN; getting it wrong the other way cost 16k evaluations per
                 * dispatch against 6k outputs on this net's 8x12 layers. */
                const int use_lut = (M >= MB_ICBS_LUT_BREAKEVEN);
                if (use_lut) {
                    for (int j = 0; j < 4; j++) {
                        int oc = n0 + j;
                        if (oc >= OC) break;
                        float bn_s = bn_scale[oc], bn_b = bn_bias[oc];
                        for (int v = 0; v < 256; v++) {
                            int8_t bn_int8 = mb_icbs_bn_stage(
                                (int8_t)(v - 128), bn_s, bn_b,
                                bn_scale_in, bn_scale_out,
                                bn_activation_min, bn_activation_max);
                            epi_lut[j][v] = silu_lut[(int)bn_int8 + 128];
                        }
                    }
                }

                for (int mt = 0; mt < current_m_tiles; mt++) {
                    int m0 = (mt_outer + mt) * 4;
                    const int8_t *a_tile = packed_a + (size_t)mt * 32u;
                    const int8_t *b_tile = b_panel;
                    int slabs = k_slabs;
                    size_t n32 = 32, n8 = 8;
                    size_t astep = a_slab_stride, bstep = b_slab_stride;

                    __asm__ volatile(
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        "vmv.v.i v8, 0\n\t"
                        "vmv.v.i v9, 0\n\t"
                        "1:\n\t"
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        "vle8.v v0, (%[pa])\n\t"
                        "vle8.v v4, (%[pb])\n\t"
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
                        "add %[pa], %[pa], %[as]\n\t"
                        "add %[pb], %[pb], %[bs]\n\t"
                        "addi %[ks], %[ks], -1\n\t"
                        "bnez %[ks], 1b\n\t"
                        "vsetvli t0, %[n8], e32, m1, ta, ma\n\t"
                        "vse32.v v8, (%[o0])\n\t"
                        "vse32.v v9, (%[o1])\n\t"
                        : [pa] "+r"(a_tile), [pb] "+r"(b_tile), [ks] "+r"(slabs)
                        : [o0] "r"(tile_output), [o1] "r"(tile_output + 8),
                          [n32] "r"(n32), [n8] "r"(n8),
                          [as] "r"(astep), [bs] "r"(bstep)
                        : "t0", "memory", "v0", "v4", "v8", "v9");

                    /* ---- conv Q0.31 tail, then BN and SiLU from the table ----
                     * The NCHW output index is (n*OC+oc)*OH*OW + oh*OW + ow and
                     * OH*OW is M while oh*OW + ow is the im2col row, so the
                     * store needs no oh/ow at all. */
                    for (int i = 0; i < 4 && m0 + i < M; i++) {
                        int row = m0 + i;
                        for (int j = 0; j < 4 && n0 + j < OC; j++) {
                            int oc = n0 + j;
                            int32_t acc = tile_output[i * 4 + j];
                            if (bias) acc += bias[oc];
                            int32_t s = mb_icbs_q31_requant(acc,
                                            conv_output_multiplier,
                                            conv_output_shift);
                            s += conv_output_offset;
                            if (s < conv_activation_min) s = conv_activation_min;
                            if (s > conv_activation_max) s = conv_activation_max;
                            int8_t conv_i8 = (int8_t)s;
                            int8_t epi;
                            if (use_lut) {
                                epi = epi_lut[j][(int)conv_i8 + 128];
                            } else {
                                int8_t bn_int8 = mb_icbs_bn_stage(
                                    conv_i8, bn_scale[oc], bn_bias[oc],
                                    bn_scale_in, bn_scale_out,
                                    bn_activation_min, bn_activation_max);
                                epi = silu_lut[(int)bn_int8 + 128];
                            }
                            output[((size_t)n * OC + oc) * M + row] = epi;
                        }
                    }
                }
            }
        }
    }
}
