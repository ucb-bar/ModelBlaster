/* source: curated */
/* algorithm: x100_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* Backend ime_k3_x100 (K3 X100 only, cpus 0-7). conv2d_s8_pc: the per-channel ("per-filter")
 * twin of ime_k3_x100_conv2d_s8_x100_vmadot_4x4x8.c (k3-work TASK-019, 2026-11-17), which
 * stays unchanged. */
/* The same kernel except the final scaling step: one multiplier and one shift per output
 * channel (output_multiplier[oc], output_shift[oc]) instead of one each. The helpers, A and
 * B packers and indirection tables are the per-layer file's shared sections, verbatim and
 * under the same guards (MB_XF1_CORE, MB_XF1_CONVX: one copy per kernels.c, whichever file
 * comes first; they must stay identical). MB_XF1_PCX adds xf1x_core_pc -- xf1x_core with the
 * per-lane step -- and the per-lane MMA tails (vsmul.vv / vsll.vv / vssra.vv with
 * vectors built like the bias lanes: lane e of a 4x4 tile = channel panel*4 + (e & 3)).
 * Build-time "MBX100TW" / "MBX100TN" tiles or plain IHWOC weights, as the per-layer kernel.
 * A channel whose shift is >= 32 gives exactly 0 before the offset (the per-layer step
 * would read the shift mod 32). Symmetric int8 only (zero input and filter offsets); X100
 * cores (cpus 0-7) only. Written without the board; the board check is a byte compare
 * against rvv_conv2d_s8_pc_direct.c (k3-work experiments/vint-pc-kernels-20261117). */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <riscv_vector.h>
#ifndef MB_XF1_CORE
#define MB_XF1_CORE

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
#endif /* MB_XF1_CORE */

#ifndef MB_XF1_CONVX
#define MB_XF1_CONVX
/* Shared by the X100 conv and fused conv (TASK-014, 2026-11-13). The core below is xf1's
 * (2 m-tiles x 4 panels, vector tail, the installed per-element output step), with three
 * ways to pack A (the im2col operand, [k8 slab][m-tile][4 pixels][8 k]), chosen per
 * layer, and B (the weights, [k8 slab][panel][4 channels][8 k]) optionally packed at
 * build time. Measured with every variant under the same conditions on the 13 conv and
 * 35 + 35 fused YOLOv8-nano / DroNet shapes (k3-work experiments/conv-variant-matrix-20261113):
 *   IMP   implicit GEMM, NCHW: xf1's packer (per k-slab and output row, eight tap runs
 *         interleaved by one vsseg8 store) over a zero-padded copy made with inline
 *         8-byte moves (xf1 memset and refilled a stack copy per call and m-block).
 *   IND   indirect GEMM, NCHW: one pointer per (pixel, k) into the input as it is, built
 *         on the first call for a layer's input buffer and kept; padding points at zeros.
 *         Wins outputs at most 12 wide: 0.54-0.92x IMP (xf1's packer runs once per output
 *         row, and short rows waste it).
 *   NHWC  the input transposed into a zero-padded NHWC copy, k = (kh, kw, ic): a slab is
 *         one tap's 8 consecutive channels, one 8-byte move per pixel, no interleave.
 *         Selected by weights built in that k order ("MBX100TN").
 * Weights packed at build time: "MBX100TW" (k = (ic*KH + kh)*KW + kw) or "MBX100TN" (k =
 * (kh*KW + kw)*IC + ic), a 64-byte header (magic, OC, K as little-endian uint32, zeros)
 * then the B layout above, zero padded; plain IHWOC weights are packed per call as before.
 * The output stays NCHW in every case. */

static inline const int8_t *xf1x_tiles(const int8_t *w, int rows, int K, const char *magic)
{
    if (memcmp(w, magic, 8) != 0) return NULL;
    uint32_t r, k;
    memcpy(&r, w + 8, 4); memcpy(&k, w + 12, 4);
    if (r != (uint32_t)rows || k != (uint32_t)K) __builtin_trap();   /* packed for another shape */
    return w + 64;
}

typedef struct { char b[8]; } xf1x_b8;
#define xf1x_ld8(p) ({ uint64_t xf1x_v_; __asm__("ld %0, %1" : "=r"(xf1x_v_) : "m"(*(const xf1x_b8 *)(const void *)(p))); xf1x_v_; })
#define xf1x_sd8(p, v) __asm__("sd %1, %0" : "=m"(*(xf1x_b8 *)(void *)(p)) : "r"((uint64_t)(v)))
static inline void xf1x_zero(int8_t *d, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) xf1x_sd8(d + i, 0);
    for (; i < n; i++) d[i] = 0;
}
static inline void xf1x_copy(int8_t *d, const int8_t *s, size_t n)
{
    size_t i = 0;
    for (; i + 8 <= n; i += 8) xf1x_sd8(d + i, xf1x_ld8(s + i));
    for (; i < n; i++) d[i] = s[i];
}

/* One per-thread scratch buffer for the padded copies; it only grows. NULL if it cannot. */
static __thread int8_t *xf1x_buf;
static __thread size_t xf1x_cap;
static int8_t *xf1x_scratch(size_t bytes)
{
    size_t need = (bytes + 63) / 64 * 64;
    if (need > xf1x_cap) {
        free(xf1x_buf);
        xf1x_buf = aligned_alloc(64, need);
        xf1x_cap = xf1x_buf ? need : 0;
    }
    return xf1x_buf;
}

/* [IC][IH][IW] -> [IC][IH+2PH][IW+2PW] with the zero border, plus 16 zero bytes. */
static const int8_t *xf1x_pad_nchw(const int8_t *in, int IC, int IH, int IW, int PH, int PW)
{
    if (PH == 0 && PW == 0) return in;
    size_t hp = (size_t)IH + 2 * PH, wp = (size_t)IW + 2 * PW, plane = hp * wp;
    int8_t *b = xf1x_scratch((size_t)IC * plane + 16);
    if (!b) return NULL;
    for (int ic = 0; ic < IC; ic++) {
        int8_t *c = b + (size_t)ic * plane;
        xf1x_zero(c, (size_t)PH * wp);
        for (int ih = 0; ih < IH; ih++) {
            int8_t *row = c + ((size_t)PH + ih) * wp;
            xf1x_zero(row, (size_t)PW);
            xf1x_copy(row + PW, in + ((size_t)ic * IH + ih) * IW, (size_t)IW);
            xf1x_zero(row + PW + IW, (size_t)PW);
        }
        xf1x_zero(c + ((size_t)PH + IH) * wp, (size_t)PH * wp);
    }
    xf1x_zero(b + (size_t)IC * plane, 16);
    return b;
}

static inline void xf1x_tr8x8(uint64_t *x)
{
    uint64_t t;
    for (int r = 0; r < 8; r += 2) { t = ((x[r] >> 8) ^ x[r + 1]) & 0x00FF00FF00FF00FFull; x[r + 1] ^= t; x[r] ^= t << 8; }
    for (int r = 0; r < 8; r += (r & 1) ? 3 : 1) { t = ((x[r] >> 16) ^ x[r + 2]) & 0x0000FFFF0000FFFFull; x[r + 2] ^= t; x[r] ^= t << 16; }
    for (int r = 0; r < 4; r++) { t = ((x[r] >> 32) ^ x[r + 4]) & 0x00000000FFFFFFFFull; x[r + 4] ^= t; x[r] ^= t << 32; }
}

/* NCHW [IC][IH][IW] -> NHWC [IH+2PH][IW+2PW][IC] with the zero border, plus 16 zero
 * bytes: 8 channels x 8 pixels per 8x8 byte transpose, the rest one by one. */
static const int8_t *xf1x_pad_nhwc(const int8_t *in, int IC, int IH, int IW, int PH, int PW)
{
    size_t wp = (size_t)IW + 2 * PW, hp = (size_t)IH + 2 * PH, rowb = wp * IC, cs = (size_t)IH * IW;
    int8_t *b = xf1x_scratch(hp * rowb + 16);
    if (!b) return NULL;
    int c8 = IC & ~7, w8 = IW & ~7;
    xf1x_zero(b, (size_t)PH * rowb);
    xf1x_zero(b + ((size_t)PH + IH) * rowb, (size_t)PH * rowb);
    for (int ih = 0; ih < IH; ih++) {
        int8_t *row = b + ((size_t)PH + ih) * rowb, *dst = row + (size_t)PW * IC;
        const int8_t *src = in + (size_t)ih * IW;
        xf1x_zero(row, (size_t)PW * IC);
        xf1x_zero(row + ((size_t)PW + IW) * IC, (size_t)PW * IC);
        for (int c0 = 0; c0 < c8; c0 += 8) {
            for (int w0 = 0; w0 < w8; w0 += 8) {
                uint64_t x[8];
                for (int i = 0; i < 8; i++) x[i] = xf1x_ld8(src + (size_t)(c0 + i) * cs + w0);
                xf1x_tr8x8(x);
                for (int p = 0; p < 8; p++) xf1x_sd8(dst + (size_t)(w0 + p) * IC + c0, x[p]);
            }
            for (int w = w8; w < IW; w++)
                for (int i = 0; i < 8; i++) dst[(size_t)w * IC + c0 + i] = src[(size_t)(c0 + i) * cs + w];
        }
        for (int c = c8; c < IC; c++)
            for (int w = 0; w < IW; w++) dst[(size_t)w * IC + c] = src[(size_t)c * cs + w];
    }
    xf1x_zero(b + hp * rowb, 16);
    return b;
}

/* IND: one entry per (pixel, k), k up to the slab-rounded K: the tap's offset from the
 * start of the input image ((ic*IH + ih)*IW + iw), or -1 for a padding tap and for the
 * extra k (both read as zero). Offsets, not pointers, so a table depends only on the
 * layer's shape: built once per shape and shared by every input buffer (the walker's
 * per-instance buffer sets) and every row-split band of that shape (k3-work TASK-017;
 * keyed by input pointer before, a walker build needed 300-1100 tables and ran out of
 * the 128 slots). Slots are claimed once (0 free -> 1 building -> 2 ready) and never
 * freed; a call that finds the slot being built, no slot or no memory gets NULL and
 * packs the IMP way instead. */
#define XF1X_TABS 128
typedef struct { int state; int key[10]; const int32_t *tab; } xf1x_tab_t;
static xf1x_tab_t xf1x_tabs[XF1X_TABS];
static __attribute__((noinline)) const int32_t *xf1x_table(int IC, int IH, int IW,
        int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW, int K, int Kp)
{
    int key[10] = {IC, IH, IW, KH, KW, SH, SW, PH, PW, Kp};
    uint64_t hk = 1469598103934665603ull;
    for (int i = 0; i < 10; i++) hk = (hk ^ (uint32_t)key[i]) * 1099511628211ull;
    size_t h = (size_t)(hk >> 57);
    for (int i = 0; i < XF1X_TABS; i++) {
        xf1x_tab_t *S = &xf1x_tabs[(h + (size_t)i) % XF1X_TABS];
        int st = __atomic_load_n(&S->state, __ATOMIC_ACQUIRE);
        if (st == 2) {
            if (!memcmp(S->key, key, sizeof key)) return S->tab;
            continue;
        }
        if (st == 1) return NULL;
        int zero = 0;
        if (!__atomic_compare_exchange_n(&S->state, &zero, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { i--; continue; }
        size_t M = (size_t)OH * OW;
        int32_t *tab = malloc(sizeof(*tab) * M * (size_t)Kp);
        if (!tab) { __atomic_store_n(&S->state, 0, __ATOMIC_RELEASE); return NULL; }
        for (size_t m = 0; m < M; m++) {
            int oh = (int)(m / OW), ow = (int)(m % OW);
            for (int k = 0; k < Kp; k++) {
                int32_t o = -1;
                if (k < K) {
                    int ic = k / (KH * KW), rem = k % (KH * KW);
                    int ih = oh * SH - PH + rem / KW, iw = ow * SW - PW + rem % KW;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) o = (int32_t)(((size_t)ic * IH + ih) * IW + iw);
                }
                tab[m * Kp + k] = o;
            }
        }
        memcpy(S->key, key, sizeof key); S->tab = tab;
        __atomic_store_n(&S->state, 2, __ATOMIC_RELEASE);
        return tab;
    }
    return NULL;
}

/* IND or IMP for an NCHW layer, from the measured split (X100 cpu 0): IND for outputs at
 * most 12 wide; for 1x1 stride-1 unpadded layers (which IMP packs as the transposed input)
 * only up to 8 pixels. */
static inline int xf1x_use_ind(int OH, int OW, int KH, int KW, int SH, int SW, int PH, int PW)
{
    if (KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0) return OH * OW <= 8;
    return OW <= 12;
}

#define XF1X_IMP 0
#define XF1X_IND 1
#define XF1X_NHWC 2

/* The BN+SiLU step's parameters (fused only). */
typedef struct { int16_t *memo; const int8_t *silu; const float *bs, *bb; float bsi, bso; int bmin, bmax; } xf1x_fx;

/* The kernel body after the checks. `path` and `fused` are constants in each wrapper, so
 * each gets its own copy; wt == NULL means plain IHWOC weights, packed per call (NCHW
 * paths only). Fused: the step's parameters are copied into locals once, since through
 * the struct every int8 output store (which may alias anything) forces their reload. */
static inline __attribute__((always_inline)) void xf1x_core(const int path, const int fused, const int8_t *input,
        const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW,
        int OC, int KH, int KW, int SH, int SW, int PH, int PW, int ooff, int mult, int shift, int amin, int amax,
        const xf1x_fx *F)
{
    int16_t *const f_memo = fused ? F->memo : NULL;
    const int8_t *const f_silu = fused ? F->silu : NULL;
    const float *const f_bs = fused ? F->bs : NULL, *const f_bb = fused ? F->bb : NULL;
    const float f_bsi = fused ? F->bsi : 0.f, f_bso = fused ? F->bso : 0.f;
    const int f_bmin = fused ? F->bmin : 0, f_bmax = fused ? F->bmax : 0;
#define XF1X_OUT(oc, c) (fused ? xf1_map(f_memo, f_silu, f_bs, f_bb, f_bsi, f_bso, f_bmin, f_bmax, (oc), (c)) : (c))

    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW, K = IC * KH * KW;
    int m_tiles = (M + 3) / 4, k_slabs = (K + 7) / 8, n_panels = (OC + 3) / 4, Kp = k_slabs * 8;
    size_t bytes_per_m_tile = (size_t)k_slabs * 32u;
    int block_m_tiles = (int)((256u * 1024u) / bytes_per_m_tile);
    if (block_m_tiles < 1) block_m_tiles = 1;
    if (block_m_tiles > m_tiles) block_m_tiles = m_tiles;

    /* B: the build's tiles as they stand, or plain weights packed once per call when they
     * fit the stack budget, else one panel at a time (xf1, unchanged). */
    size_t b_all_bytes = (size_t)n_panels * (size_t)k_slabs * 32u;
    const int pack_b_once = wt || b_all_bytes <= MB_XF1_B_PACK_BUDGET;
    const size_t b_slab_stride = (size_t)(pack_b_once ? n_panels : 1) * 32u;
    int8_t b_buf[wt ? 1 : (size_t)k_slabs * b_slab_stride] __attribute__((aligned(64)));
    int8_t *packed_b = wt ? (int8_t *)(uintptr_t)wt : b_buf;
    int8_t packed_a[(size_t)block_m_tiles * bytes_per_m_tile] __attribute__((aligned(64)));
    int32_t xb[4][16] __attribute__((aligned(64)));
    int8_t xq[128] __attribute__((aligned(64)));
    const long lsh = shift < 0 ? -shift : 0, rsh = shift > 0 ? shift : 0;
    const int one_by_one = KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0;
    const int IHp = IH + 2 * PH, IWp = IW + 2 * PW;

    if (!wt && pack_b_once) {
        if (OC & 3) memset(b_buf, 0, (size_t)k_slabs * b_slab_stride);
        for (int ks = 0; ks < k_slabs; ks++) {
            int k0 = ks * 8;
            const int8_t *w0 = weight + (size_t)(k0 + 0) * OC;
            ptrdiff_t st = OC;
            mb_xf1_interleave8(
                (k0 + 0 < K) ? w0 : NULL, (k0 + 1 < K) ? w0 + st : NULL,
                (k0 + 2 < K) ? w0 + 2*st : NULL, (k0 + 3 < K) ? w0 + 3*st : NULL,
                (k0 + 4 < K) ? w0 + 4*st : NULL, (k0 + 5 < K) ? w0 + 5*st : NULL,
                (k0 + 6 < K) ? w0 + 6*st : NULL, (k0 + 7 < K) ? w0 + 7*st : NULL,
                1, b_buf + (size_t)ks * b_slab_stride, OC);
        }
    }

    /* per-layer offsets: tap_off[k] (IMP and IND's fallback) = offset of tap k's (oh=0, ow=0)
     * byte in the padded NCHW image, counted, never divided; soff[ks] / koffn[k] (NHWC) =
     * offsets in the padded NHWC image of slab ks (IC % 8 == 0) / of k (otherwise). */
    int32_t tap_off[path == XF1X_NHWC ? 1 : K];
    int32_t soff[path == XF1X_NHWC ? k_slabs : 1], koffn[path == XF1X_NHWC ? Kp : 1];
    if (path == XF1X_NHWC) {
        for (int ks = 0; ks < k_slabs; ks++) {
            int k0 = ks * 8, t = k0 / IC;
            soff[ks] = ((t / KW) * IWp + t % KW) * IC + k0 % IC;
        }
        for (int k = 0; k < Kp; k++) { int t = k / IC; koffn[k] = k < K ? ((t / KW) * IWp + t % KW) * IC + k % IC : 0; }
    } else {
        int t = 0;
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) tap_off[t++] = ((int32_t)ic * IHp + kh) * IWp + kw;
    }

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;
        int8_t *out = output + (size_t)n * OC * M;
        const int32_t *tab = path == XF1X_IND
            ? xf1x_table(IC, IH, IW, KH, KW, SH, SW, PH, PW, OH, OW, K, Kp) : NULL;
        const int8_t *base_in = NULL;
        if (path == XF1X_NHWC) {
            base_in = xf1x_pad_nhwc(in_n, IC, IH, IW, PH, PW);
            if (!base_in) __builtin_trap();   /* no memory for a ~100 KB copy */
        } else if (!tab) base_in = xf1x_pad_nchw(in_n, IC, IH, IW, PH, PW);   /* NULL: bounds-checked packer */

        for (int mt_outer = 0; mt_outer < m_tiles; mt_outer += block_m_tiles) {
            int current_m_tiles = m_tiles - mt_outer;
            if (current_m_tiles > block_m_tiles) current_m_tiles = block_m_tiles;
            const size_t a_slab_stride = (size_t)current_m_tiles * 32u;
            int m_lo = mt_outer * 4, m_hi = m_lo + current_m_tiles * 4;
            if (m_hi > M) m_hi = M;
            int rows = m_hi - m_lo;

            /* ---- pack A ---- */
            if (path == XF1X_IND && tab) {
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                for (int m = m_lo; m < m_hi; m++) {
                    const int32_t *tp = tab + (size_t)m * Kp;
                    int8_t *d = packed_a + (size_t)(m - m_lo) * 8u;
                    for (int ks = 0; ks < k_slabs; ks++) {
                        const int32_t *t = tp + ks * 8;
                        uint64_t v = 0;
                        /* scalar on purpose: GCC turns this 8-step loop into short
                         * vectors with an indexed load, 2x slower (TASK-017 bench) */
#pragma GCC unroll 8
#pragma GCC novector
                        for (int q = 0; q < 8; q++) {
                            /* o < 0 (padding / extra k): read byte 0, mask it to zero */
                            int32_t o = t[q], neg = o >> 31;
                            v |= (uint64_t)((uint8_t)in_n[o & ~neg] & (uint8_t)~neg) << (8 * q);
                        }
                        xf1x_sd8(d + (size_t)ks * a_slab_stride, v);
                    }
                }
            } else if (path == XF1X_NHWC) {
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh = m_lo / OW, ow = m_lo % OW;
                for (int m = m_lo; m < m_hi; m++) {
                    const int8_t *pbase = base_in + ((size_t)oh * SH * IWp + (size_t)ow * SW) * IC;
                    int8_t *d = packed_a + (size_t)(m - m_lo) * 8u;
                    if ((IC & 7) == 0) {
                        for (int ks = 0; ks < k_slabs; ks++) xf1x_sd8(d + (size_t)ks * a_slab_stride, xf1x_ld8(pbase + soff[ks]));
                    } else {
                        for (int ks = 0; ks < k_slabs; ks++) {
                            uint64_t v = 0;
                            for (int q = 0; q < 8; q++) { int k = ks * 8 + q; if (k < K) v |= (uint64_t)(uint8_t)pbase[koffn[k]] << (8 * q); }
                            xf1x_sd8(d + (size_t)ks * a_slab_stride, v);
                        }
                    }
                    if (++ow == OW) { ow = 0; oh++; }
                }
            } else if (one_by_one) {
                /* A[m][k] = in_n[k*M + m]: the im2col matrix is the input transposed, so a
                 * tap is one contiguous run of M pixels and a k-slab one segment store. */
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
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
            } else if (base_in) {
                /* every window lies inside the padded image: each tap of a k-slab covers
                 * the same run of output pixels, so a slab and output row is one vsseg8 */
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh_lo = m_lo / OW, oh_hi = (m_hi - 1) / OW;
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    int8_t *slab = packed_a + (size_t)ks * a_slab_stride;
                    for (int oh = oh_lo; oh <= oh_hi; oh++) {
                        int base = oh * OW, ow0 = 0, ow1 = OW - 1;
                        if (base + ow0 < m_lo) ow0 = m_lo - base;
                        if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                        if (ow0 > ow1) continue;
                        const int8_t *q0 = base_in + (ptrdiff_t)oh * SH * IWp + (ptrdiff_t)ow0 * SW;
                        mb_xf1_interleave8(
                            (k0+0 < K) ? q0 + tap_off[k0+0] : NULL, (k0+1 < K) ? q0 + tap_off[k0+1] : NULL,
                            (k0+2 < K) ? q0 + tap_off[k0+2] : NULL, (k0+3 < K) ? q0 + tap_off[k0+3] : NULL,
                            (k0+4 < K) ? q0 + tap_off[k0+4] : NULL, (k0+5 < K) ? q0 + tap_off[k0+5] : NULL,
                            (k0+6 < K) ? q0 + tap_off[k0+6] : NULL, (k0+7 < K) ? q0 + tap_off[k0+7] : NULL,
                            SW, slab + (size_t)(base + ow0 - m_lo) * 8u, ow1 - ow0 + 1);
                    }
                }
            } else {
                /* no memory for the padded copy: xf1's bounds-checked strided copies */
                memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh_lo = m_lo / OW, oh_hi = (m_hi - 1) / OW;
                int k = 0;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++, k++) {
                            int8_t *slab = packed_a + (size_t)(k >> 3) * a_slab_stride + (k & 7);
                            int ow_first = (PW - kw + SW - 1) / SW;
                            if (ow_first < 0) ow_first = 0;
                            int ow_last_num = IW - 1 - kw + PW;
                            if (ow_last_num < 0) continue;
                            int ow_last = ow_last_num / SW;
                            if (ow_last > OW - 1) ow_last = OW - 1;
                            if (ow_first > ow_last) continue;
                            for (int oh = oh_lo; oh <= oh_hi; oh++) {
                                int ih = oh * SH - PH + kh;
                                if (ih < 0 || ih >= IH) continue;
                                int base = oh * OW, ow0 = ow_first, ow1 = ow_last;
                                if (base + ow0 < m_lo) ow0 = m_lo - base;
                                if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                                if (ow0 > ow1) continue;
                                mb_xf1_copy_strided(in_n + ((size_t)ic * IH + ih) * IW + (size_t)(ow0 * SW - PW + kw), SW,
                                                    slab + (size_t)(base + ow0 - m_lo) * 8u, 8, ow1 - ow0 + 1);
                            }
                        }
            }

            /* ---- 2 m-tiles x 4 panels per pass, vector tail, then the output step ---- */
            for (int p0 = 0; p0 < n_panels; p0 += 4) {
                int np = n_panels - p0; if (np > 4) np = 4;
                for (int p = 0; p < 4; p++)
                    for (int e = 0; e < 16; e++) {
                        int oc = (p0 + p) * 4 + (e & 3);
                        xb[p][e] = (bias && p < np && oc < OC) ? bias[oc] : 0;
                    }
                int mt = 0;
                if (pack_b_once && np == 4) {
                    for (; mt + 1 < current_m_tiles; mt += 2) {
                        const int8_t *a0 = packed_a + (size_t)mt * 32u;
                        xf1_mma_2x4(a0, a0 + 32, packed_b + (size_t)p0 * 32u, k_slabs, (long)a_slab_stride,
                                    (long)b_slab_stride, &xb[0][0], xq, mult, lsh, rsh, ooff, amin, amax);
                        int m0 = (mt_outer + mt) * 4;
                        for (int t = 0; t < 8; t++) {
                            int rbase = m0 + (t >> 2) * 4, cbase = (p0 + (t & 3)) * 4;
                            for (int i = 0; i < 4; i++) {
                                int row = rbase + i;
                                if (row >= M) break;
                                for (int j = 0; j < 4; j++) {
                                    int oc = cbase + j;
                                    if (oc >= OC) break;
                                    out[(size_t)oc * M + row] = XF1X_OUT(oc, xq[t * 16 + i * 4 + j]);
                                }
                            }
                        }
                    }
                }
                /* the rest one 4x4 tile at a time */
                for (; mt < current_m_tiles; mt++) {
                    int m0 = (mt_outer + mt) * 4;
                    for (int p = 0; p < np; p++) {
                        int n0 = (p0 + p) * 4;
                        const int8_t *b_panel = packed_b + (pack_b_once ? (size_t)(p0 + p) * 32u : 0u);
                        if (!pack_b_once) {
                            memset(b_buf, 0, (size_t)k_slabs * 32u);
                            for (int k = 0; k < K; k++) {
                                int cols = OC - n0; if (cols > 4) cols = 4;
                                mb_xf1_copy_strided(weight + (size_t)k * OC + n0, 1, b_buf + (size_t)(k >> 3) * 32u + (k & 7), 8, cols);
                            }
                        }
                        xf1_mma_1x1(packed_a + (size_t)mt * 32u, b_panel, k_slabs, (long)a_slab_stride,
                                    (long)b_slab_stride, &xb[p][0], xq, mult, lsh, rsh, ooff, amin, amax);
                        for (int i = 0; i < 4; i++) {
                            int row = m0 + i;
                            if (row >= M) break;
                            for (int j = 0; j < 4; j++) {
                                int oc = n0 + j;
                                if (oc >= OC) break;
                                out[(size_t)oc * M + row] = XF1X_OUT(oc, xq[i * 4 + j]);
                            }
                        }
                    }
                }
            }
        }
    }
#undef XF1X_OUT
}
#endif /* MB_XF1_CONVX */


#ifndef MB_XF1_PCX
#define MB_XF1_PCX
/* ---- the per-channel step (conv2d_s8_pc, k3-work TASK-019) ------------------------
 * XF1_Q with the multiplier and the two shifts read per lane from v0-v1 / v2-v3 / v4-v5
 * (loaded by XF1P_PCV, after the MAC loop has freed them) instead of one scalar each:
 * vsmul.vv / vsll.vv / vssra.vv in place of the .vx forms. Lane e of a 4x4 tile is
 * channel (panel*4 + (e & 3)), as for the bias. */
#define XF1P_QEND(T, OFF) \
        "vadd.vx " T ", " T ", %[oo]\n\t" \
        "vmax.vx " T ", " T ", %[lo]\n\t" \
        "vmin.vx " T ", " T ", %[hi]\n\t" \
        "vsetivli t0, 16, e16, m1, ta, ma\n\t" \
        "vnsrl.wi v6, " T ", 0\n\t" \
        "vsetivli t0, 16, e8, mf2, ta, ma\n\t" \
        "vnsrl.wi v7, v6, 0\n\t" \
        "addi t1, %[o], " OFF "\n\t" \
        "vse8.v v7, (t1)\n\t"
#define XF1P_QPC(T, B, OFF) \
        "vsetivli t0, 16, e32, m2, ta, ma\n\t" \
        "vadd.vv " T ", " T ", " B "\n\t" \
        "vsmul.vv " T ", " T ", v0\n\t" \
        "vsll.vv " T ", " T ", v2\n\t" \
        "vssra.vv " T ", " T ", v4\n\t" \
        XF1P_QEND(T, OFF)
/* One panel's per-lane multiplier, left shift and right shift (int32 [16] each, at byte
 * offsets M, L, R from %[pv]) into v0-v1, v2-v3, v4-v5. */
#define XF1P_PCV(M, L, R) \
        "vsetivli t0, 16, e32, m2, ta, ma\n\t" \
        "addi t1, %[pv], " M "\n\t" "vle32.v v0, (t1)\n\t" \
        "addi t1, %[pv], " L "\n\t" "vle32.v v2, (t1)\n\t" \
        "addi t1, %[pv], " R "\n\t" "vle32.v v4, (t1)\n\t"
/* xf1_mma_2x4's MAC loop and xf1_mma_1x1's, unchanged. */
#define XF1P_LOOP_2x4 \
        "csrwi vxrm, 0\n\t" \
        "vsetivli t0, 16, e32, m2, ta, ma\n\t" \
        "vle32.v v24, (%[bi])\n\t" \
        "addi t1, %[bi], 64\n\t"  "vle32.v v26, (t1)\n\t" \
        "addi t1, %[bi], 128\n\t" "vle32.v v28, (t1)\n\t" \
        "addi t1, %[bi], 192\n\t" "vle32.v v30, (t1)\n\t" \
        "vsetvli t0, zero, e8, m8, ta, ma\n\t" \
        "vmv.v.i v8, 0\n\t" "vmv.v.i v16, 0\n\t" \
        "vsetvli t0, zero, e8, m1, ta, ma\n\t" \
        "1:\n\t" \
        "vle8.v v0, (%[a0])\n\t" \
        "vle8.v v1, (%[a1])\n\t" \
        "vle8.v v2, (%[b])\n\t" \
        "addi t1, %[b], 32\n\t" "vle8.v v3, (t1)\n\t" \
        "addi t1, %[b], 64\n\t" "vle8.v v4, (t1)\n\t" \
        "addi t1, %[b], 96\n\t" "vle8.v v5, (t1)\n\t" \
        ".insn r 0x2b, 3, 0x71, x8,  x0, x2\n\t" \
        ".insn r 0x2b, 3, 0x71, x10, x0, x3\n\t" \
        ".insn r 0x2b, 3, 0x71, x12, x0, x4\n\t" \
        ".insn r 0x2b, 3, 0x71, x14, x0, x5\n\t" \
        ".insn r 0x2b, 3, 0x71, x16, x1, x2\n\t" \
        ".insn r 0x2b, 3, 0x71, x18, x1, x3\n\t" \
        ".insn r 0x2b, 3, 0x71, x20, x1, x4\n\t" \
        ".insn r 0x2b, 3, 0x71, x22, x1, x5\n\t" \
        "add %[a0], %[a0], %[as]\n\t" \
        "add %[a1], %[a1], %[as]\n\t" \
        "add %[b], %[b], %[bs]\n\t" \
        "addi %[ks], %[ks], -1\n\t" \
        "bnez %[ks], 1b\n\t"
#define XF1P_LOOP_1x1 \
        "csrwi vxrm, 0\n\t" \
        "vsetivli t0, 16, e32, m2, ta, ma\n\t" \
        "vle32.v v24, (%[bi])\n\t" \
        "vsetvli t0, zero, e8, m1, ta, ma\n\t" \
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t" \
        "1:\n\t" \
        "vle8.v v0, (%[a])\n\t" \
        "vle8.v v4, (%[b])\n\t" \
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t" \
        "add %[a], %[a], %[as]\n\t" \
        "add %[b], %[b], %[bs]\n\t" \
        "addi %[ks], %[ks], -1\n\t" \
        "bnez %[ks], 1b\n\t"

/* xf1_mma_2x4 with the per-channel step: pv = int32 [3][4][16], the multiplier, left
 * shift and right shift of every lane of panels 0-3 (xf1x_core_pc's xs). */
static inline __attribute__((always_inline)) void xf1_mma_2x4_pc(const int8_t *a0, const int8_t *a1, const int8_t *b,
        long slabs, long as, long bs, const int32_t *bias4, int8_t *out,
        const int32_t *pv, long ooff, long amin, long amax)
{
    __asm__ volatile(
        XF1P_LOOP_2x4
        XF1P_PCV("0", "256", "512")    XF1P_QPC("v8", "v24", "0")   XF1P_QPC("v16", "v24", "64")
        XF1P_PCV("64", "320", "576")   XF1P_QPC("v10", "v26", "16") XF1P_QPC("v18", "v26", "80")
        XF1P_PCV("128", "384", "640")  XF1P_QPC("v12", "v28", "32") XF1P_QPC("v20", "v28", "96")
        XF1P_PCV("192", "448", "704")  XF1P_QPC("v14", "v30", "48") XF1P_QPC("v22", "v30", "112")
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs)
        : [bi] "r"(bias4), [o] "r"(out), [pv] "r"(pv),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax), [as] "r"(as), [bs] "r"(bs)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

/* xf1_mma_1x1 with the per-channel step: pv = &xs[0][p][0] for panel p (multiplier, then
 * left and right shifts 256 and 512 bytes on). */
static inline __attribute__((always_inline)) void xf1_mma_1x1_pc(const int8_t *a, const int8_t *b, long slabs,
        long as, long bs, const int32_t *bias4, int8_t *out,
        const int32_t *pv, long ooff, long amin, long amax)
{
    __asm__ volatile(
        XF1P_LOOP_1x1
        XF1P_PCV("0", "256", "512")
        XF1P_QPC("v8", "v24", "0")
        : [a] "+r"(a), [b] "+r"(b), [ks] "+r"(slabs)
        : [bi] "r"(bias4), [o] "r"(out), [pv] "r"(pv),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax), [as] "r"(as), [bs] "r"(bs)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v24", "v25");
}

/* xf1x_core with the per-channel step (pcm[oc], pcs[oc] instead of mult, shift);
 * everything else is xf1x_core's text. The kernel body after the checks. `path` and `fused` are constants in each wrapper, so
 * each gets its own copy; wt == NULL means plain IHWOC weights, packed per call (NCHW
 * paths only). Fused: the step's parameters are copied into locals once, since through
 * the struct every int8 output store (which may alias anything) forces their reload. */
static inline __attribute__((always_inline)) void xf1x_core_pc(const int path, const int fused, const int8_t *input,
        const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW,
        int OC, int KH, int KW, int SH, int SW, int PH, int PW, int ooff, const int32_t *pcm, const int32_t *pcs, int amin, int amax,
        const xf1x_fx *F)
{
    int16_t *const f_memo = fused ? F->memo : NULL;
    const int8_t *const f_silu = fused ? F->silu : NULL;
    const float *const f_bs = fused ? F->bs : NULL, *const f_bb = fused ? F->bb : NULL;
    const float f_bsi = fused ? F->bsi : 0.f, f_bso = fused ? F->bso : 0.f;
    const int f_bmin = fused ? F->bmin : 0, f_bmax = fused ? F->bmax : 0;
#define XF1X_OUT(oc, c) (fused ? xf1_map(f_memo, f_silu, f_bs, f_bb, f_bsi, f_bso, f_bmin, f_bmax, (oc), (c)) : (c))

    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW, K = IC * KH * KW;
    int m_tiles = (M + 3) / 4, k_slabs = (K + 7) / 8, n_panels = (OC + 3) / 4, Kp = k_slabs * 8;
    size_t bytes_per_m_tile = (size_t)k_slabs * 32u;
    int block_m_tiles = (int)((256u * 1024u) / bytes_per_m_tile);
    if (block_m_tiles < 1) block_m_tiles = 1;
    if (block_m_tiles > m_tiles) block_m_tiles = m_tiles;

    /* B: the build's tiles as they stand, or plain weights packed once per call when they
     * fit the stack budget, else one panel at a time (xf1, unchanged). */
    size_t b_all_bytes = (size_t)n_panels * (size_t)k_slabs * 32u;
    const int pack_b_once = wt || b_all_bytes <= MB_XF1_B_PACK_BUDGET;
    const size_t b_slab_stride = (size_t)(pack_b_once ? n_panels : 1) * 32u;
    int8_t b_buf[wt ? 1 : (size_t)k_slabs * b_slab_stride] __attribute__((aligned(64)));
    int8_t *packed_b = wt ? (int8_t *)(uintptr_t)wt : b_buf;
    int8_t packed_a[(size_t)block_m_tiles * bytes_per_m_tile] __attribute__((aligned(64)));
    int32_t xb[4][16] __attribute__((aligned(64)));
    int8_t xq[128] __attribute__((aligned(64)));
    int32_t xs[3][4][16] __attribute__((aligned(64)));   /* per lane: mult, lsh, rsh */
    const int one_by_one = KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0;
    const int IHp = IH + 2 * PH, IWp = IW + 2 * PW;

    if (!wt && pack_b_once) {
        if (OC & 3) memset(b_buf, 0, (size_t)k_slabs * b_slab_stride);
        for (int ks = 0; ks < k_slabs; ks++) {
            int k0 = ks * 8;
            const int8_t *w0 = weight + (size_t)(k0 + 0) * OC;
            ptrdiff_t st = OC;
            mb_xf1_interleave8(
                (k0 + 0 < K) ? w0 : NULL, (k0 + 1 < K) ? w0 + st : NULL,
                (k0 + 2 < K) ? w0 + 2*st : NULL, (k0 + 3 < K) ? w0 + 3*st : NULL,
                (k0 + 4 < K) ? w0 + 4*st : NULL, (k0 + 5 < K) ? w0 + 5*st : NULL,
                (k0 + 6 < K) ? w0 + 6*st : NULL, (k0 + 7 < K) ? w0 + 7*st : NULL,
                1, b_buf + (size_t)ks * b_slab_stride, OC);
        }
    }

    /* per-layer offsets: tap_off[k] (IMP and IND's fallback) = offset of tap k's (oh=0, ow=0)
     * byte in the padded NCHW image, counted, never divided; soff[ks] / koffn[k] (NHWC) =
     * offsets in the padded NHWC image of slab ks (IC % 8 == 0) / of k (otherwise). */
    int32_t tap_off[path == XF1X_NHWC ? 1 : K];
    int32_t soff[path == XF1X_NHWC ? k_slabs : 1], koffn[path == XF1X_NHWC ? Kp : 1];
    if (path == XF1X_NHWC) {
        for (int ks = 0; ks < k_slabs; ks++) {
            int k0 = ks * 8, t = k0 / IC;
            soff[ks] = ((t / KW) * IWp + t % KW) * IC + k0 % IC;
        }
        for (int k = 0; k < Kp; k++) { int t = k / IC; koffn[k] = k < K ? ((t / KW) * IWp + t % KW) * IC + k % IC : 0; }
    } else {
        int t = 0;
        for (int ic = 0; ic < IC; ic++)
            for (int kh = 0; kh < KH; kh++)
                for (int kw = 0; kw < KW; kw++) tap_off[t++] = ((int32_t)ic * IHp + kh) * IWp + kw;
    }

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;
        int8_t *out = output + (size_t)n * OC * M;
        const int32_t *tab = path == XF1X_IND
            ? xf1x_table(IC, IH, IW, KH, KW, SH, SW, PH, PW, OH, OW, K, Kp) : NULL;
        const int8_t *base_in = NULL;
        if (path == XF1X_NHWC) {
            base_in = xf1x_pad_nhwc(in_n, IC, IH, IW, PH, PW);
            if (!base_in) __builtin_trap();   /* no memory for a ~100 KB copy */
        } else if (!tab) base_in = xf1x_pad_nchw(in_n, IC, IH, IW, PH, PW);   /* NULL: bounds-checked packer */

        for (int mt_outer = 0; mt_outer < m_tiles; mt_outer += block_m_tiles) {
            int current_m_tiles = m_tiles - mt_outer;
            if (current_m_tiles > block_m_tiles) current_m_tiles = block_m_tiles;
            const size_t a_slab_stride = (size_t)current_m_tiles * 32u;
            int m_lo = mt_outer * 4, m_hi = m_lo + current_m_tiles * 4;
            if (m_hi > M) m_hi = M;
            int rows = m_hi - m_lo;

            /* ---- pack A ---- */
            if (path == XF1X_IND && tab) {
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                for (int m = m_lo; m < m_hi; m++) {
                    const int32_t *tp = tab + (size_t)m * Kp;
                    int8_t *d = packed_a + (size_t)(m - m_lo) * 8u;
                    for (int ks = 0; ks < k_slabs; ks++) {
                        const int32_t *t = tp + ks * 8;
                        uint64_t v = 0;
                        /* scalar on purpose: GCC turns this 8-step loop into short
                         * vectors with an indexed load, 2x slower (TASK-017 bench) */
#pragma GCC unroll 8
#pragma GCC novector
                        for (int q = 0; q < 8; q++) {
                            /* o < 0 (padding / extra k): read byte 0, mask it to zero */
                            int32_t o = t[q], neg = o >> 31;
                            v |= (uint64_t)((uint8_t)in_n[o & ~neg] & (uint8_t)~neg) << (8 * q);
                        }
                        xf1x_sd8(d + (size_t)ks * a_slab_stride, v);
                    }
                }
            } else if (path == XF1X_NHWC) {
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh = m_lo / OW, ow = m_lo % OW;
                for (int m = m_lo; m < m_hi; m++) {
                    const int8_t *pbase = base_in + ((size_t)oh * SH * IWp + (size_t)ow * SW) * IC;
                    int8_t *d = packed_a + (size_t)(m - m_lo) * 8u;
                    if ((IC & 7) == 0) {
                        for (int ks = 0; ks < k_slabs; ks++) xf1x_sd8(d + (size_t)ks * a_slab_stride, xf1x_ld8(pbase + soff[ks]));
                    } else {
                        for (int ks = 0; ks < k_slabs; ks++) {
                            uint64_t v = 0;
                            for (int q = 0; q < 8; q++) { int k = ks * 8 + q; if (k < K) v |= (uint64_t)(uint8_t)pbase[koffn[k]] << (8 * q); }
                            xf1x_sd8(d + (size_t)ks * a_slab_stride, v);
                        }
                    }
                    if (++ow == OW) { ow = 0; oh++; }
                }
            } else if (one_by_one) {
                /* A[m][k] = in_n[k*M + m]: the im2col matrix is the input transposed, so a
                 * tap is one contiguous run of M pixels and a k-slab one segment store. */
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
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
            } else if (base_in) {
                /* every window lies inside the padded image: each tap of a k-slab covers
                 * the same run of output pixels, so a slab and output row is one vsseg8 */
                if (rows & 3) memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh_lo = m_lo / OW, oh_hi = (m_hi - 1) / OW;
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    int8_t *slab = packed_a + (size_t)ks * a_slab_stride;
                    for (int oh = oh_lo; oh <= oh_hi; oh++) {
                        int base = oh * OW, ow0 = 0, ow1 = OW - 1;
                        if (base + ow0 < m_lo) ow0 = m_lo - base;
                        if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                        if (ow0 > ow1) continue;
                        const int8_t *q0 = base_in + (ptrdiff_t)oh * SH * IWp + (ptrdiff_t)ow0 * SW;
                        mb_xf1_interleave8(
                            (k0+0 < K) ? q0 + tap_off[k0+0] : NULL, (k0+1 < K) ? q0 + tap_off[k0+1] : NULL,
                            (k0+2 < K) ? q0 + tap_off[k0+2] : NULL, (k0+3 < K) ? q0 + tap_off[k0+3] : NULL,
                            (k0+4 < K) ? q0 + tap_off[k0+4] : NULL, (k0+5 < K) ? q0 + tap_off[k0+5] : NULL,
                            (k0+6 < K) ? q0 + tap_off[k0+6] : NULL, (k0+7 < K) ? q0 + tap_off[k0+7] : NULL,
                            SW, slab + (size_t)(base + ow0 - m_lo) * 8u, ow1 - ow0 + 1);
                    }
                }
            } else {
                /* no memory for the padded copy: xf1's bounds-checked strided copies */
                memset(packed_a, 0, (size_t)k_slabs * a_slab_stride);
                int oh_lo = m_lo / OW, oh_hi = (m_hi - 1) / OW;
                int k = 0;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++)
                        for (int kw = 0; kw < KW; kw++, k++) {
                            int8_t *slab = packed_a + (size_t)(k >> 3) * a_slab_stride + (k & 7);
                            int ow_first = (PW - kw + SW - 1) / SW;
                            if (ow_first < 0) ow_first = 0;
                            int ow_last_num = IW - 1 - kw + PW;
                            if (ow_last_num < 0) continue;
                            int ow_last = ow_last_num / SW;
                            if (ow_last > OW - 1) ow_last = OW - 1;
                            if (ow_first > ow_last) continue;
                            for (int oh = oh_lo; oh <= oh_hi; oh++) {
                                int ih = oh * SH - PH + kh;
                                if (ih < 0 || ih >= IH) continue;
                                int base = oh * OW, ow0 = ow_first, ow1 = ow_last;
                                if (base + ow0 < m_lo) ow0 = m_lo - base;
                                if (base + ow1 > m_hi - 1) ow1 = m_hi - 1 - base;
                                if (ow0 > ow1) continue;
                                mb_xf1_copy_strided(in_n + ((size_t)ic * IH + ih) * IW + (size_t)(ow0 * SW - PW + kw), SW,
                                                    slab + (size_t)(base + ow0 - m_lo) * 8u, 8, ow1 - ow0 + 1);
                            }
                        }
            }

            /* ---- 2 m-tiles x 4 panels per pass, vector tail, then the output step ---- */
            for (int p0 = 0; p0 < n_panels; p0 += 4) {
                int np = n_panels - p0; if (np > 4) np = 4;
                for (int p = 0; p < 4; p++)
                    for (int e = 0; e < 16; e++) {
                        int oc = (p0 + p) * 4 + (e & 3);
                        xb[p][e] = (bias && p < np && oc < OC) ? bias[oc] : 0;
                    }
                /* the same lanes' multiplier and shifts (lsh = -shift, rsh = shift, as the
                 * per-layer step computes them once). A shift >= 32 (a filter of all-zero or
                 * tiny weights) makes the exact result 0, which vssra cannot shift to (it
                 * reads 5 bits), so that lane gets multiplier 0 instead. */
                for (int p = 0; p < 4; p++)
                    for (int e = 0; e < 16; e++) {
                        int oc = (p0 + p) * 4 + (e & 3);
                        int ok = p < np && oc < OC;
                        int32_t m = ok ? pcm[oc] : 0, s = ok ? pcs[oc] : 0;
                        if (s >= 32) { m = 0; s = 0; }
                        xs[0][p][e] = m;
                        xs[1][p][e] = s < 0 ? -s : 0;
                        xs[2][p][e] = s > 0 ? s : 0;
                    }
                int mt = 0;
                if (pack_b_once && np == 4) {
                    for (; mt + 1 < current_m_tiles; mt += 2) {
                        const int8_t *a0 = packed_a + (size_t)mt * 32u;
                        xf1_mma_2x4_pc(a0, a0 + 32, packed_b + (size_t)p0 * 32u, k_slabs, (long)a_slab_stride,
                                       (long)b_slab_stride, &xb[0][0], xq, &xs[0][0][0], ooff, amin, amax);
                        int m0 = (mt_outer + mt) * 4;
                        for (int t = 0; t < 8; t++) {
                            int rbase = m0 + (t >> 2) * 4, cbase = (p0 + (t & 3)) * 4;
                            for (int i = 0; i < 4; i++) {
                                int row = rbase + i;
                                if (row >= M) break;
                                for (int j = 0; j < 4; j++) {
                                    int oc = cbase + j;
                                    if (oc >= OC) break;
                                    out[(size_t)oc * M + row] = XF1X_OUT(oc, xq[t * 16 + i * 4 + j]);
                                }
                            }
                        }
                    }
                }
                /* the rest one 4x4 tile at a time */
                for (; mt < current_m_tiles; mt++) {
                    int m0 = (mt_outer + mt) * 4;
                    for (int p = 0; p < np; p++) {
                        int n0 = (p0 + p) * 4;
                        const int8_t *b_panel = packed_b + (pack_b_once ? (size_t)(p0 + p) * 32u : 0u);
                        if (!pack_b_once) {
                            memset(b_buf, 0, (size_t)k_slabs * 32u);
                            for (int k = 0; k < K; k++) {
                                int cols = OC - n0; if (cols > 4) cols = 4;
                                mb_xf1_copy_strided(weight + (size_t)k * OC + n0, 1, b_buf + (size_t)(k >> 3) * 32u + (k & 7), 8, cols);
                            }
                        }
                        xf1_mma_1x1_pc(packed_a + (size_t)mt * 32u, b_panel, k_slabs, (long)a_slab_stride,
                                       (long)b_slab_stride, &xb[p][0], xq, &xs[0][p][0], ooff, amin, amax);
                        for (int i = 0; i < 4; i++) {
                            int row = m0 + i;
                            if (row >= M) break;
                            for (int j = 0; j < 4; j++) {
                                int oc = n0 + j;
                                if (oc >= OC) break;
                                out[(size_t)oc * M + row] = XF1X_OUT(oc, xq[i * 4 + j]);
                            }
                        }
                    }
                }
            }
        }
    }
#undef XF1X_OUT
}
#endif /* MB_XF1_PCX */

#define XF1X_ARGS const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int ooff, const int32_t *pcm, const int32_t *pcs, int amin, int amax, const xf1x_fx *F
#define XF1X_PASS(W) input, weight, W, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, ooff, pcm, pcs, amin, amax, F
static __attribute__((noinline)) void xf1p_imp_plain(XF1X_ARGS) { (void)wt; xf1x_core_pc(XF1X_IMP, 0, XF1X_PASS(NULL)); }
static __attribute__((noinline)) void xf1p_ind_plain(XF1X_ARGS) { (void)wt; xf1x_core_pc(XF1X_IND, 0, XF1X_PASS(NULL)); }
static __attribute__((noinline)) void xf1p_imp_tiled(XF1X_ARGS) { xf1x_core_pc(XF1X_IMP, 0, XF1X_PASS(wt)); }
static __attribute__((noinline)) void xf1p_ind_tiled(XF1X_ARGS) { xf1x_core_pc(XF1X_IND, 0, XF1X_PASS(wt)); }
static __attribute__((noinline)) void xf1p_nhwc_tiled(XF1X_ARGS) { xf1x_core_pc(XF1X_NHWC, 0, XF1X_PASS(wt)); }
#undef XF1X_ARGS
#undef XF1X_PASS

void kernel_conv2d_s8_pc(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output,
                         int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
                         int input_offset, int filter_offset, int output_offset,
                         const int32_t *output_multiplier, const int32_t *output_shift,
                         int activation_min, int activation_max)
{
    if (OC <= 0 || IC <= 0) return;
    /* Symmetric int8 only -- an asymmetric conv stays on RVV (the picker's job). */
    if (input_offset != 0 || filter_offset != 0) return;
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    if (OH <= 0 || OW <= 0) return;
    int K = IC * KH * KW;
    /* Weights packed at build time are B as it stands: "MBX100TW" (NCHW k order) or
     * "MBX100TN" (NHWC k order, which selects the NHWC packer); plain IHWOC otherwise. */
    const int8_t *wt = (long)OC * K >= 16 ? xf1x_tiles(weight, OC, K, "MBX100TW") : NULL;
    const int8_t *wn = (!wt && (long)OC * K >= 16) ? xf1x_tiles(weight, OC, K, "MBX100TN") : NULL;
    void (*core)(const int8_t *, const int8_t *, const int8_t *, const int32_t *, int8_t *, int, int, int, int, int,
                 int, int, int, int, int, int, int, const int32_t *, const int32_t *, int, int, const xf1x_fx *);
    if (wn) core = xf1p_nhwc_tiled;
    else {
        int ind = xf1x_use_ind(OH, OW, KH, KW, SH, SW, PH, PW);
        core = wt ? (ind ? xf1p_ind_tiled : xf1p_imp_tiled) : (ind ? xf1p_ind_plain : xf1p_imp_plain);
    }
    core(input, weight, wn ? wn : wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
         output_offset, output_multiplier, output_shift, activation_min, activation_max, NULL);
}
