/* source: curated */
/* algorithm: ime_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* K3 X100 conv2d_batchnorm2d_silu_s8 (ime_x100, K3-only backend; Director 2026-10-29),
 * "xf1". Starts from the team's kernels/ime/ime_conv2d_batchnorm2d_silu_s8_ime_vmadot_4x4x8.c
 * (same packing, byte for byte; helpers renamed mb_xf1_) and changes two things:
 *  1. MAC: 2 m-tiles x 4 panels (8 rows x 16 channels) per pass, 8 accumulators,
 *     instead of one 4x4 tile at a time (the X100 linear's 2x4 recipe);
 *  2. tail: bias add + Q0.31 requantize + clamp in vector registers, then BN+SiLU
 *     from a per-channel 256-entry table filled lazily on first use (the A100 fused
 *     kernel's memo) instead of per-element float math or a table built for every value.
 * Bit-exact (same integer MAC; vsmul with vxrm=rnu == (x*mult + 2^30) >> 31; BN and SiLU
 * by the kernel's own expressions). Measured 2026-10-29 on X100 cpu 0, all YOLOv8-nano
 * fused shapes bit-exact: 1.60x the K1 kernel at 160x160, 1.39x at 64x96
 * (experiments/xpurt-runs/x100-fused-opt/). The original header follows.
 */
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
 *     BN    -- mb_xf1_bn_stage(): float fv = conv_int8 * bn_scale_in;
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
 *   So the table is built only when M >= MB_XF1_LUT_BREAKEVEN -- the same
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
 *   inside mb_xf1_copy_strided looks free. It cost 31 of the 49 wins --
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
#include <stdlib.h>
#include <riscv_vector.h>

/* Below this many output pixels per channel, building a 256-entry epilogue table costs more
 * than evaluating the two float stages per element (the RVV kernel's MB_CBS_LUT_BREAKEVEN). */
#define MB_XF1_LUT_BREAKEVEN 256

/* Stack budget for the packed-once B. The deployed net's largest is l7.conv
 * (K=1152, OC=256 -> 288 KB); above the budget B falls back to being packed one
 * panel at a time into the same buffer, which is what the first version always
 * did. */
#define MB_XF1_B_PACK_BUDGET (512u * 1024u)

/* Stack budget for the zero-padded input copy the general (non-1x1) path packs
 * from; above it the bounds-checked per-tap path runs instead. */
#define MB_XF1_PAD_BUDGET (256u * 1024u)

/* Scalar Q0.31 requantize, identical to the IME/RVV conv kernels. */
static inline int32_t mb_xf1_q31_requant(int32_t x, int32_t mult, int32_t shift) {
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
static inline int8_t mb_xf1_bn_stage(int8_t conv_int8,
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
static inline void mb_xf1_copy_strided(const int8_t *src, ptrdiff_t src_stride,
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
static inline void mb_xf1_interleave8(
    const int8_t *p0, const int8_t *p1, const int8_t *p2, const int8_t *p3,
    const int8_t *p4, const int8_t *p5, const int8_t *p6, const int8_t *p7,
    ptrdiff_t src_stride, int8_t *dst, int n)
{
#define MB_XF1_LANE(p) ((p) == NULL ? __riscv_vmv_v_x_i8m1(0, vl)                 \
                         : ((src_stride == 1) ? __riscv_vle8_v_i8m1((p), vl)       \
                            : __riscv_vlse8_v_i8m1((p), src_stride, vl)))
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)n);
        vint8m1x8_t t = __riscv_vundefined_i8m1x8();
        t = __riscv_vset_v_i8m1_i8m1x8(t, 0, MB_XF1_LANE(p0));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 1, MB_XF1_LANE(p1));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 2, MB_XF1_LANE(p2));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 3, MB_XF1_LANE(p3));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 4, MB_XF1_LANE(p4));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 5, MB_XF1_LANE(p5));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 6, MB_XF1_LANE(p6));
        t = __riscv_vset_v_i8m1_i8m1x8(t, 7, MB_XF1_LANE(p7));
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
#undef MB_XF1_LANE
}


/* ---- xf1 additions ---- */
#define XF1_EMPTY ((int16_t)0x8080)      /* memset(0x80) value; never an int8 */

/* BN+SiLU of one conv output for channel oc, computed once per (oc, value). */
/* The memo can be shared by threads running the same layer (xf1_tables), so entries
 * are read and written as relaxed atomics (plain lh / sh); every writer stores the
 * same value. */
static inline __attribute__((always_inline)) int8_t xf1_map(int16_t *memo, const int8_t *silu_lut,
        const float *bn_scale, const float *bn_bias, float bn_scale_in, float bn_scale_out,
        int bn_min, int bn_max, int oc, int8_t c)
{
    int16_t *m = &memo[(size_t)oc * 256 + (uint8_t)(c + 128)];
    int16_t v = __atomic_load_n(m, __ATOMIC_RELAXED);
    if (v == XF1_EMPTY) {
        int8_t b = mb_xf1_bn_stage(c, bn_scale[oc], bn_bias[oc], bn_scale_in, bn_scale_out, bn_min, bn_max);
        v = silu_lut[(int)b + 128];
        __atomic_store_n(m, v, __ATOMIC_RELAXED);
    }
    return (int8_t)v;
}

/* BN+SiLU tables kept across calls (TASK-013, as in the A100 fused kernel): the
 * per-call memo cost a memset of OC * 512 bytes plus one BN stage (a roundf) per
 * entry used, and the SiLU table 256 expf, on every call. A slot holds one layer's
 * tables, keyed by every parameter they depend on: OC, the scales and limits, and a
 * copy of the per-channel bn_scale / bn_bias values, compared on every call, so a
 * reused buffer with new values never sees old entries. Slots are claimed once
 * (0 free -> 1 claiming -> 2 ready) and never freed; a call that finds none (all
 * taken, another thread claiming this one, no memory) uses per-call tables. */
#define XF1_SLOTS 256
typedef struct {
    int state, OC;
    float bsi, bso, ssi, sso;
    int bmin, bmax, smin, smax;
    float *bs, *bb;
    int16_t *memo;
    int8_t silu[256];
} xf1_slot;
static xf1_slot xf1_slots[XF1_SLOTS];

static void xf1_silu_table(int8_t *lut, float ssi, float sso, int smin, int smax)
{
    for (int v = 0; v < 256; v++) {
        int8_t bn_int8 = (int8_t)(v - 128);
        float fbv = (float)bn_int8 * ssi;
        float sy = fbv / (1.0f + expf(-fbv));
        int32_t q = (int32_t)roundf(sy / sso);
        if (q < smin) q = smin;
        if (q > smax) q = smax;
        lut[v] = (int8_t)q;
    }
}

static int xf1_slot_is(const xf1_slot *S, int OC, const float *bs, const float *bb, float bsi, float bso,
                       float ssi, float sso, int bmin, int bmax, int smin, int smax)
{
    return S->OC == OC && S->bmin == bmin && S->bmax == bmax && S->smin == smin && S->smax == smax
        && !memcmp(&S->bsi, &bsi, sizeof(float)) && !memcmp(&S->bso, &bso, sizeof(float))
        && !memcmp(&S->ssi, &ssi, sizeof(float)) && !memcmp(&S->sso, &sso, sizeof(float))
        && !memcmp(S->bs, bs, (size_t)OC * sizeof(float)) && !memcmp(S->bb, bb, (size_t)OC * sizeof(float));
}

/* This layer's kept tables, or NULL if there are none. */
static __attribute__((noinline)) const xf1_slot *xf1_tables(int OC, const float *bs, const float *bb, float bsi,
        float bso, float ssi, float sso, int bmin, int bmax, int smin, int smax)
{
    size_t h = ((uintptr_t)bs >> 4) * 0x9E3779B97F4A7C15ull >> 56;
    for (int i = 0; i < XF1_SLOTS; i++) {
        xf1_slot *S = &xf1_slots[(h + (size_t)i) % XF1_SLOTS];
        int st = __atomic_load_n(&S->state, __ATOMIC_ACQUIRE);
        if (st == 2) {
            if (xf1_slot_is(S, OC, bs, bb, bsi, bso, ssi, sso, bmin, bmax, smin, smax)) return S;
            continue;
        }
        if (st == 1) return NULL;
        int zero = 0;
        if (!__atomic_compare_exchange_n(&S->state, &zero, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { i--; continue; }
        int16_t *memo = malloc((size_t)OC * 256 * sizeof(int16_t));
        float *cp = malloc((size_t)OC * 2 * sizeof(float));
        if (!memo || !cp) {
            free(memo); free(cp);
            __atomic_store_n(&S->state, 0, __ATOMIC_RELEASE);
            return NULL;
        }
        memset(memo, 0x80, (size_t)OC * 256 * sizeof(int16_t));
        memcpy(cp, bs, (size_t)OC * sizeof(float));
        memcpy(cp + OC, bb, (size_t)OC * sizeof(float));
        xf1_silu_table(S->silu, ssi, sso, smin, smax);
        S->OC = OC; S->bsi = bsi; S->bso = bso; S->ssi = ssi; S->sso = sso;
        S->bmin = bmin; S->bmax = bmax; S->smin = smin; S->smax = smax;
        S->bs = cp; S->bb = cp + OC; S->memo = memo;
        __atomic_store_n(&S->state, 2, __ATOMIC_RELEASE);
        return S;
    }
    return NULL;
}

/* Bias add, Q0.31 requantize (vxrm = rnu), shift, offset, clamp, narrow to int8, for the
 * 16 int32 of one 4x4 tile in T (an m2 group); writes 16 bytes at OFF from %[o]. */
#define XF1_Q(T, B, OFF)                                             \
        "vsetivli t0, 16, e32, m2, ta, ma\n\t"                       \
        "vadd.vv " T ", " T ", " B "\n\t"                            \
        "vsmul.vx " T ", " T ", %[mu]\n\t"                           \
        "vsll.vx " T ", " T ", %[ls]\n\t"                            \
        "vssra.vx " T ", " T ", %[rs]\n\t"                           \
        "vadd.vx " T ", " T ", %[oo]\n\t"                            \
        "vmax.vx " T ", " T ", %[lo]\n\t"                            \
        "vmin.vx " T ", " T ", %[hi]\n\t"                            \
        "vsetivli t0, 16, e16, m1, ta, ma\n\t"                       \
        "vnsrl.wi v6, " T ", 0\n\t"                                  \
        "vsetivli t0, 16, e8, mf2, ta, ma\n\t"                       \
        "vnsrl.wi v7, v6, 0\n\t"                                     \
        "addi t1, %[o], " OFF "\n\t"                                 \
        "vse8.v v7, (t1)\n\t"

/* 2 m-tiles (a0, a1 = a0 + 32) x 4 panels (b .. b + 96), slabs strided by as / bs;
 * tile (i, p) -> 16 int8 at out + (i*4 + p)*16, row-major 4x4. */
static inline __attribute__((always_inline)) void xf1_mma_2x4(const int8_t *a0, const int8_t *a1, const int8_t *b,
        long slabs, long as, long bs, const int32_t *bias4, int8_t *out,
        long mult, long lsh, long rsh, long ooff, long amin, long amax)
{
    __asm__ volatile(
        "csrwi vxrm, 0\n\t"
        "vsetivli t0, 16, e32, m2, ta, ma\n\t"
        "vle32.v v24, (%[bi])\n\t"
        "addi t1, %[bi], 64\n\t"  "vle32.v v26, (t1)\n\t"
        "addi t1, %[bi], 128\n\t" "vle32.v v28, (t1)\n\t"
        "addi t1, %[bi], 192\n\t" "vle32.v v30, (t1)\n\t"
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "1:\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "vle8.v v1, (%[a1])\n\t"
        "vle8.v v2, (%[b])\n\t"
        "addi t1, %[b], 32\n\t" "vle8.v v3, (t1)\n\t"
        "addi t1, %[b], 64\n\t" "vle8.v v4, (t1)\n\t"
        "addi t1, %[b], 96\n\t" "vle8.v v5, (t1)\n\t"
        ".insn r 0x2b, 3, 0x71, x8,  x0, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x1, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x1, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x1, x5\n\t"
        "add %[a0], %[a0], %[as]\n\t"
        "add %[a1], %[a1], %[as]\n\t"
        "add %[b], %[b], %[bs]\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        XF1_Q("v8", "v24", "0")    XF1_Q("v10", "v26", "16")  XF1_Q("v12", "v28", "32")  XF1_Q("v14", "v30", "48")
        XF1_Q("v16", "v24", "64")  XF1_Q("v18", "v26", "80")  XF1_Q("v20", "v28", "96")  XF1_Q("v22", "v30", "112")
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs)
        : [bi] "r"(bias4), [o] "r"(out), [mu] "r"(mult), [ls] "r"(lsh), [rs] "r"(rsh),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax), [as] "r"(as), [bs] "r"(bs)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

/* One 4x4 tile (the team kernel's MAC loop) with the same vector tail. */
static inline __attribute__((always_inline)) void xf1_mma_1x1(const int8_t *a, const int8_t *b, long slabs,
        long as, long bs, const int32_t *bias4, int8_t *out,
        long mult, long lsh, long rsh, long ooff, long amin, long amax)
{
    __asm__ volatile(
        "csrwi vxrm, 0\n\t"
        "vsetivli t0, 16, e32, m2, ta, ma\n\t"
        "vle32.v v24, (%[bi])\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t"
        "1:\n\t"
        "vle8.v v0, (%[a])\n\t"
        "vle8.v v4, (%[b])\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        "add %[a], %[a], %[as]\n\t"
        "add %[b], %[b], %[bs]\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        XF1_Q("v8", "v24", "0")
        : [a] "+r"(a), [b] "+r"(b), [ks] "+r"(slabs)
        : [bi] "r"(bias4), [o] "r"(out), [mu] "r"(mult), [ls] "r"(lsh), [rs] "r"(rsh),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax), [as] "r"(as), [bs] "r"(bs)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v4", "v6", "v7", "v8", "v9", "v24", "v25");
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

    /* SiLU stage table (indexed by the BN stage's int8 output + 128) and the BN+SiLU
     * memo: this layer's kept tables (xf1_tables), else built for this call. */
    const xf1_slot *xf1_kept = xf1_tables(OC, bn_scale, bn_bias, bn_scale_in, bn_scale_out, silu_scale_in,
                                          silu_scale_out, bn_activation_min, bn_activation_max,
                                          silu_activation_min, silu_activation_max);
    int8_t silu_lut_l[xf1_kept ? 1 : 256];
    if (!xf1_kept) xf1_silu_table(silu_lut_l, silu_scale_in, silu_scale_out, silu_activation_min, silu_activation_max);
    const int8_t *silu_lut = xf1_kept ? xf1_kept->silu : silu_lut_l;

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
    const int pack_b_once = (b_all_bytes <= MB_XF1_B_PACK_BUDGET);
    const int b_panels_resident = pack_b_once ? n_panels : 1;
    const size_t b_slab_stride = (size_t)b_panels_resident * 32u;

    int8_t packed_a[(size_t)block_m_tiles * bytes_per_m_tile];
    int8_t packed_b[(size_t)k_slabs * b_slab_stride];
    int32_t xf1_bias[4][16] __attribute__((aligned(64)));
    int8_t xf1_q[128] __attribute__((aligned(64)));
    int16_t xf1_memo_l[xf1_kept ? 1 : (size_t)OC * 256];
    if (!xf1_kept) memset(xf1_memo_l, 0x80, sizeof xf1_memo_l);
    int16_t *xf1_memo = xf1_kept ? xf1_kept->memo : xf1_memo_l;
    const long xf1_lsh = conv_output_shift < 0 ? -conv_output_shift : 0;
    const long xf1_rsh = conv_output_shift > 0 ? conv_output_shift : 0;

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
    const int use_pad = (PH != 0 || PW != 0) && (pad_bytes <= MB_XF1_PAD_BUDGET);
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
            mb_xf1_interleave8(
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
                    mb_xf1_interleave8(
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
                        mb_xf1_interleave8(
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
                                mb_xf1_copy_strided(src, SW, dst, 8, ow1 - ow0 + 1);
                            }
                        }
                    }
                }
            }

            /* ---- xf1: 2 m-tiles x 4 panels per pass, vector tail, lazy BN+SiLU memo ---- */
            for (int p0 = 0; p0 < n_panels; p0 += 4) {
                int np = n_panels - p0; if (np > 4) np = 4;
                for (int p = 0; p < 4; p++)
                    for (int e = 0; e < 16; e++) {
                        int oc = (p0 + p) * 4 + (e & 3);
                        xf1_bias[p][e] = (bias && p < np && oc < OC) ? bias[oc] : 0;
                    }
                int mt = 0;
                if (pack_b_once && np == 4) {
                    for (; mt + 1 < current_m_tiles; mt += 2) {
                        const int8_t *a0 = packed_a + (size_t)mt * 32u;
                        xf1_mma_2x4(a0, a0 + 32, packed_b + (size_t)p0 * 32u, k_slabs,
                                    (long)a_slab_stride, (long)b_slab_stride, &xf1_bias[0][0], xf1_q,
                                    conv_output_multiplier, xf1_lsh, xf1_rsh, conv_output_offset,
                                    conv_activation_min, conv_activation_max);
                        int m0 = (mt_outer + mt) * 4;
                        for (int t = 0; t < 8; t++) {
                            int rbase = m0 + (t >> 2) * 4, cbase = (p0 + (t & 3)) * 4;
                            for (int i = 0; i < 4; i++) {
                                int row = rbase + i;
                                if (row >= M) break;
                                for (int j = 0; j < 4; j++) {
                                    int oc = cbase + j;
                                    if (oc >= OC) break;
                                    output[((size_t)n * OC + oc) * M + row] =
                                        xf1_map(xf1_memo, silu_lut, bn_scale, bn_bias, bn_scale_in, bn_scale_out,
                                                bn_activation_min, bn_activation_max, oc,
                                                xf1_q[t * 16 + i * 4 + j]);
                                }
                            }
                        }
                    }
                }
                /* the rest one 4x4 tile at a time: the team kernel's MAC, the vector tail */
                for (; mt < current_m_tiles; mt++) {
                    int m0 = (mt_outer + mt) * 4;
                    for (int p = 0; p < np; p++) {
                        int n0 = (p0 + p) * 4;
                        const int8_t *b_panel = packed_b + (pack_b_once ? (size_t)(p0 + p) * 32u : 0u);
                        if (!pack_b_once) {
                            memset(packed_b, 0, (size_t)k_slabs * 32u);
                            for (int k = 0; k < K; k++) {
                                int8_t *dst = packed_b + (size_t)(k >> 3) * 32u + (k & 7);
                                int cols = OC - n0; if (cols > 4) cols = 4;
                                mb_xf1_copy_strided(weight + (size_t)k * OC + n0, 1, dst, 8, cols);
                            }
                        }
                        xf1_mma_1x1(packed_a + (size_t)mt * 32u, b_panel, k_slabs, (long)a_slab_stride,
                                    (long)b_slab_stride, &xf1_bias[p][0], xf1_q,
                                    conv_output_multiplier, xf1_lsh, xf1_rsh, conv_output_offset,
                                    conv_activation_min, conv_activation_max);
                        for (int i = 0; i < 4; i++) {
                            int row = m0 + i;
                            if (row >= M) break;
                            for (int j = 0; j < 4; j++) {
                                int oc = n0 + j;
                                if (oc >= OC) break;
                                output[((size_t)n * OC + oc) * M + row] =
                                    xf1_map(xf1_memo, silu_lut, bn_scale, bn_bias, bn_scale_in, bn_scale_out,
                                            bn_activation_min, bn_activation_max, oc, xf1_q[i * 4 + j]);
                            }
                        }
                    }
                }
            }
        }
    }
}
