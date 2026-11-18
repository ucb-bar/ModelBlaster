/* source: kb_cache[riscv] */
/* algorithm: x100_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* origin: k3-work TASK-016 step 7, KernelBlaster RISC-V cache winner for ime_k3_x100 conv2d_s8
 * on dronet's conv shapes: per shape the faster of the curated X100 IME conv
 * (x100_vmadot_4x4x8) and the curated RVV conv (rvv_vsmul_vnclip), both unchanged below,
 * measured on X100 cpu 0 (bench w1, median of 15, bit-exact): 1065.0 -> 967.6 us
 * over dronet's 10 convs. RVV-routed shapes: 3x112x112>32 k3 s2 p1, 32x27x27>32 k3 s2 p1, 32x27x27>32 k1 s2 p0.
 * The RVV path reads IHWOC: build-time "MBX100TW" tiles are unpacked once per weight
 * pointer (kept for the run); if that is not possible the IME path runs instead. */
/* Backend ime_k3_x100 (K3 X100 only, cpus 0-7). Copied unchanged from k3-work ModelBlaster k3-r dfd02d8 kernels/ime_x100/ (TASK-014); build-time "MBX100TW" tiles from pipeline/k3_weight_tiles.py. */
/* Named x100_vmadot_4x4x8 here (ime_vmadot_4x4x8 on k3-r): it reads IHWOC weights,
 * while the K1 kernels/ime conv of that name reads OIHW, and the curated layout
 * contract (generate_kernels) keys the weight layout on the algorithm entry. */
/* K3 X100 conv2d_s8 (ime_x100, K3-only backend), TASK-014 (2026-11-13). The X100 had no
 * matrix-engine plain conv of its own: ime_x100 builds fell back to the K1 team kernel
 * (kernels/ime/ime_conv2d_s8_ime_vmadot_4x4x8.c, one 4x4 tile at a time, two divisions
 * and a bounds check per packed byte), slower than the RVV conv on every bench shape.
 * This is the X100 fused conv's code (ime_x100_conv2d_batchnorm2d_silu_s8, "xf1") without
 * the BN+SiLU step: the same 2 m-tile x 4 panel vmadot core with the Q0.31 tail in vector
 * registers, and the same three ways to pack A, chosen per layer (implicit NCHW, indirect
 * NCHW for outputs at most 12 wide, NHWC from a transposed copy for "MBX100TN" weights);
 * see MB_XF1_CONVX. The core and helpers are shared with the fused file (guards MB_XF1_CORE,
 * MB_XF1_CONVX), so either can come first in a kernels.c.
 * Measured on X100 cpu 0, 13 DroNet / YOLOv8-nano conv2d_s8 shapes, all bit-exact
 * (k3-work experiments/conv-variant-matrix-20261113): implicit NCHW alone 26,431 ticks vs
 * 71,164 for the RVV conv and 113,476 for the K1 kernel.
 * Bit-exact: integer vmadot MAC, vsmul with vxrm=rnu == (acc*mult + 2^30) >> 31, the
 * reference's rounding shift, offset and clamp. Symmetric int8 only (zero input and filter
 * offsets), as the K1 matrix kernels; X100 cores (cpus 0-7) only. */
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

/* IND: one pointer per (pixel, k), k up to the slab-rounded K (the extra k point at
 * zeros). Built once per (input buffer, shape) and kept; slots are claimed once
 * (0 free -> 1 building -> 2 ready) and never freed, so threads of a split layer share
 * one table; a call that finds the slot being built, no slot or no memory gets NULL and
 * packs the IMP way instead. */
#define XF1X_TABS 128
typedef struct { int state; const int8_t *in; int key[10]; const int8_t **tab; } xf1x_tab_t;
static xf1x_tab_t xf1x_tabs[XF1X_TABS];
static const int8_t xf1x_zeros[64] __attribute__((aligned(64)));
static __attribute__((noinline)) const int8_t *const *xf1x_table(const int8_t *in, int IC, int IH, int IW,
        int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW, int K, int Kp)
{
    int key[10] = {IC, IH, IW, KH, KW, SH, SW, PH, PW, Kp};
    size_t h = ((uintptr_t)in >> 4) * 0x9E3779B97F4A7C15ull >> 57;
    for (int i = 0; i < XF1X_TABS; i++) {
        xf1x_tab_t *S = &xf1x_tabs[(h + (size_t)i) % XF1X_TABS];
        int st = __atomic_load_n(&S->state, __ATOMIC_ACQUIRE);
        if (st == 2) {
            if (S->in == in && !memcmp(S->key, key, sizeof key)) return (const int8_t *const *)S->tab;
            continue;
        }
        if (st == 1) return NULL;
        int zero = 0;
        if (!__atomic_compare_exchange_n(&S->state, &zero, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { i--; continue; }
        size_t M = (size_t)OH * OW;
        const int8_t **tab = malloc(sizeof(*tab) * M * (size_t)Kp);
        if (!tab) { __atomic_store_n(&S->state, 0, __ATOMIC_RELEASE); return NULL; }
        for (size_t m = 0; m < M; m++) {
            int oh = (int)(m / OW), ow = (int)(m % OW);
            for (int k = 0; k < Kp; k++) {
                const int8_t *p = xf1x_zeros;
                if (k < K) {
                    int ic = k / (KH * KW), rem = k % (KH * KW);
                    int ih = oh * SH - PH + rem / KW, iw = ow * SW - PW + rem % KW;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) p = in + ((size_t)ic * IH + ih) * IW + iw;
                }
                tab[m * Kp + k] = p;
            }
        }
        S->in = in; memcpy(S->key, key, sizeof key); S->tab = tab;
        __atomic_store_n(&S->state, 2, __ATOMIC_RELEASE);
        return (const int8_t *const *)tab;
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
        const int8_t *const *tab = path == XF1X_IND
            ? xf1x_table(in_n, IC, IH, IW, KH, KW, SH, SW, PH, PW, OH, OW, K, Kp) : NULL;
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
                    const int8_t *const *tp = tab + (size_t)m * Kp;
                    int8_t *d = packed_a + (size_t)(m - m_lo) * 8u;
                    for (int ks = 0; ks < k_slabs; ks++) {
                        const int8_t *const *t = tp + ks * 8;
                        uint64_t v = 0;
                        for (int q = 0; q < 8; q++) v |= (uint64_t)(uint8_t)*t[q] << (8 * q);
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

#define XF1X_ARGS const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int ooff, int mult, int shift, int amin, int amax, const xf1x_fx *F
#define XF1X_PASS(W) input, weight, W, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, ooff, mult, shift, amin, amax, F
static __attribute__((noinline)) void xf1c_imp_plain(XF1X_ARGS) { (void)wt; xf1x_core(XF1X_IMP, 0, XF1X_PASS(NULL)); }
static __attribute__((noinline)) void xf1c_ind_plain(XF1X_ARGS) { (void)wt; xf1x_core(XF1X_IND, 0, XF1X_PASS(NULL)); }
static __attribute__((noinline)) void xf1c_imp_tiled(XF1X_ARGS) { xf1x_core(XF1X_IMP, 0, XF1X_PASS(wt)); }
static __attribute__((noinline)) void xf1c_ind_tiled(XF1X_ARGS) { xf1x_core(XF1X_IND, 0, XF1X_PASS(wt)); }
static __attribute__((noinline)) void xf1c_nhwc_tiled(XF1X_ARGS) { xf1x_core(XF1X_NHWC, 0, XF1X_PASS(wt)); }
#undef XF1X_ARGS
#undef XF1X_PASS

static void mb_kbx_ime_conv2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset, int output_multiplier,
                      int output_shift, int activation_min, int activation_max)
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
                 int, int, int, int, int, int, int, int, int, int, int, const xf1x_fx *);
    if (wn) core = xf1c_nhwc_tiled;
    else {
        int ind = xf1x_use_ind(OH, OW, KH, KW, SH, SW, PH, PW);
        core = wt ? (ind ? xf1c_ind_tiled : xf1c_imp_tiled) : (ind ? xf1c_ind_plain : xf1c_imp_plain);
    }
    core(input, weight, wn ? wn : wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
         output_offset, output_multiplier, output_shift, activation_min, activation_max, NULL);
}


/*
 * ============================================================================
 * SHAPE-CONDITIONAL conv2d_s8: tiled-direct for KH*KW > 1, iGEMM for 1x1.
 * ============================================================================
 *
 * NOTE ON THE FILENAME (unchanged from before, still true): the
 * rvv_vsmul_vnclip ALGORITHM is retired. Measured on FPGA it is wrong on
 * yolov8n (untiled -> max_abs_err=3, tiled -> Store/AMO access fault) while
 * spike reports bit-exact, because it indexes the input with 32-bit int
 * arithmetic that wraps for BSS buffers above 0x80000000. This file carries
 * the production conv2d_s8 under that name because rvv_vsmul_vnclip sorts
 * before rvv_oc_blocked and rvv_igemm in spec.algorithms, so whatever lives
 * here is what every rvv model actually gets. Collapse the names when
 * convenient.
 *
 * WHY THE DISPATCH IS IN THE KERNEL AND NOT IN THE SELECTOR
 * --------------------------------------------------------
 * The kernel selector picks ONE algorithm per op for the whole model. It
 * cannot express "per-layer, by shape". yolov8n calls conv2d_s8 63 times with
 * two very different tap counts (39x 3x3, 24x 1x1), and the two algorithms
 * below have OPPOSITE FPGA verdicts on those two groups, so the choice has to
 * be made per call site -- i.e. here, as a runtime branch on KH/KW.
 *
 * THE MEASUREMENT THAT MOTIVATES IT (real FPGA, f2_dual_small_norose_tacit_
 * q31_60mhz, yolov8n, 63 conv layers; NOT spike -- spike reversed the sign of
 * this result and claimed iGEMM was a 1.58x win overall):
 *
 *     3x3 and larger : tiled-direct wins decisively (iGEMM 6-9x slower on the
 *                      small-channel C2f bottleneck layers). The indirection
 *                      buffer costs up to KH*KW=9 scattered pointer gathers
 *                      per tap, a DRAM/cache latency spike does not model.
 *     1x1            : the spatial gather disappears entirely (one tap, one
 *                      pointer, one ic-step per output pixel), leaving only
 *                      iGEMM's overhead-amortization benefit: the direct path
 *                      recomputes in_ic/w_ic/ih/row_in/iw0/iw_last and takes a
 *                      bounds branch on EVERY ic step, where iGEMM does one
 *                      pointer add.
 *
 * So: KH==1 && KW==1 -> iGEMM; everything else -> tiled-direct.
 *
 * The 1x1 iGEMM path is fully general in stride and padding (an out-of-bounds
 * tap gets the zero sentinel with ic-step 0, which still contributes
 * input_offset*(w+filter_offset) exactly as the conv2d_s8 spec requires), and
 * there is no dilation in this kernel's ABI, so no shape reaches it that it
 * cannot handle. Both paths are bit-exact against the scalar reference, and
 * they share one requantize/store sequence, so the dispatch changes no numeric
 * result whichever way it goes -- max_abs_err must stay 0.
 *
 * !!! THE 1x1 DISPATCH IS OFF BY DEFAULT AND MUST STAY OFF UNTIL THE NUMERIC
 * !!! DIVERGENCE BELOW IS ROOT-CAUSED. It is a real and large speedup that is
 * !!! NOT bit-exact on this FPGA. Build knobs:
 *
 *   (default)            tiled-direct for every shape. Compiles to code that
 *                        is instruction-for-instruction identical to the
 *                        pre-dispatch production kernel. max_abs_err = 0.
 *   -DMB_IGEMM_1X1=1     enable the 1x1 -> iGEMM dispatch. FAST (see table)
 *                        but max_abs_err = 5 on yolov8n on real hardware.
 *   -DMB_NO_IGEMM_1X1    hard opt-out; compiles the iGEMM body out entirely.
 *                        Same behaviour as the default, smaller binary.
 *   -DMB_FORCE_IGEMM     iGEMM for every shape (research arm; 1.55x on conv,
 *                        also not bit-exact -- max_abs_err = 1).
 *
 * WHAT WAS MEASURED (real FPGA, f2_dual_small_norose_tacit_q31_60mhz,
 * yolov8n, 63 conv2d_s8 layers, one matched IR + one matched curated kernel
 * set + one bitstream, all five arms submitted from the same session):
 *
 *   arm                              conv2d_s8 cycles   vs baseline  max_abs_err
 *   tiled-direct (baseline, job 188)      155,388,816       1.0000x        0
 *   1x1 dispatch (job 193)                135,300,478       1.1485x        5   <-- FAILS
 *   1x1 dispatch, GCC-cloned (job 189/190)134,714,913       1.1535x        6   <-- FAILS
 *   iGEMM everywhere (job 192)            100,031,882       1.5534x        1   <-- FAILS
 *   layout control (job 196)              155,395,165       1.0000x        0
 *
 *   split for the 1x1 dispatch arm: the 24 pointwise layers go 39,201,597 ->
 *   19,073,621 = 2.0553x and EVERY ONE of the 24 gets faster; the 39 non-1x1
 *   layers are 0.9997x, i.e. untouched, as the codegen identity guarantees.
 *
 * WHY IT IS OFF: the dispatch is not numerically neutral ON HARDWARE, and it
 * has to be. The two implementations are provably the same integer arithmetic
 * in the same order, spike agrees bit-for-bit at 27 shapes (every stride /
 * padding / OW-remainder / batch / no-bias combination plus the seven real
 * yolov8n pointwise shapes), and the whole-model spike verify is
 * max_abs_err=0 -- but yolov8n on the FPGA comes back max_abs_err=5,
 * deterministically (byte-identical reruns).
 *
 * WHAT THAT IS *NOT*: job 196 is the control that rules out the obvious
 * alternative. It links the iGEMM body in, so its .text/.rodata/BSS layout
 * and every model buffer address are identical to the enabled build's, but a
 * `volatile` gate keeps the dispatch from ever firing. It returns
 * max_abs_err=0 and baseline cycles. So the divergence is not the binary
 * moving and waking a latent layout-sensitive bug somewhere else in the
 * model -- it is EXECUTING the iGEMM body on the pointwise layers.
 *
 * WHERE TO LOOK NEXT: the iGEMM's indirection walk is the only thing the
 * direct path does not do. Note also that iGEMM-everywhere gives err=1 while
 * iGEMM-on-1x1-only gives err=5, so the size of the error is not monotone in
 * how much iGEMM runs -- consistent with a per-layer divergence whose effect
 * on the final tensor depends on which layers diverge. An isolated
 * kernel-vs-kernel unit test ON HARDWARE is the missing experiment
 * (the spike version of it is clean; the FPGA version hung in the harness
 * and was cancelled, jobs 191 and 195).
 *
 * The two implementations below are the two previously-shipped kernels moved
 * verbatim into static functions -- see their own doc blocks, which are kept
 * with them. NOTE: PATH B is a FORK of kernels/rvv/rvv_conv2d_s8_rvv_igemm.c
 * as of the point this file was written; that file is being evolved
 * independently (it has since been restructured to an ic-outer/tap-inner loop
 * for weight locality). For KH==KW==1 the two forms address exactly the same
 * weights and inputs, so the fork is behaviourally current for the only shape
 * class this file dispatches to it -- but re-sync before enabling
 * MB_FORCE_IGEMM for real work.
 *
 * WHY BOTH PATHS ARE __attribute__((noinline))
 * -------------------------------------------
 * It makes the A/B honest. With noinline, mb_conv2d_s8_tiled_direct compiles
 * to code that is instruction-for-instruction IDENTICAL to the previous
 * production kernel_conv2d_s8 (verified: same 96 basic blocks, same 1236
 * instructions, zero diffs), so every non-1x1 layer is bit-identical AND
 * cycle-identical to the baseline and the only measured delta is the 1x1
 * dispatch itself. Without it GCC re-schedules and re-allocates the merged
 * function and the baseline drifts by a couple of percent for no reason.
 * The wrapper it costs is 34 instructions, executed once per conv layer (63
 * times for the whole of yolov8n) against millions of cycles of inner loop.
 *
 * WHY THE iGEMM PATH IS ALSO __attribute__((noipa)) -- DO NOT REMOVE
 * ------------------------------------------------------------------
 * noipa blocks interprocedural constant propagation into the iGEMM body.
 * That looks like it throws away a free win (the dispatch condition proves
 * KH==KW==1, so GCC will happily clone the function as
 * mb_conv2d_s8_igemm.constprop.0 with taps folded to 1, the tap loop gone
 * and the indirection buffer demoted from a VLA to a fixed 4-entry array).
 * The clone is MEASURABLY WRONG ON HARDWARE.
 *
 * Measured, not theorised: with the clone enabled, yolov8n on
 * f2_dual_small_norose_tacit_q31_60mhz returns max_abs_err=6 against the
 * golden -- twice, cycle-for-cycle deterministic (fq jobs 189 and 190) --
 * while spike reports max_abs_err=0 for the same binary. With noipa the
 * same source is bit-exact on the same FPGA.
 *
 * What changes in the codegen is exactly one thing, and it is visible with
 * objdump: in the cloned function GCC schedules the bias load as
 *
 *     vsetvli zero,zero,e8,m1,ta,ma
 *     vle32.v v4,(t1)            <-- EEW=32 while SEW=8, so EMUL=4
 *
 * i.e. it reuses the e8/m1 vtype the following vle8 wants and relies on the
 * load's encoded EEW to imply EMUL=4. That is architecturally legal (vl and
 * VLMAX agree between e8/m1 and EEW=32/EMUL=4, and v4 is EMUL-aligned) and
 * spike executes it correctly. Every other build of this file -- the
 * tiled-direct path, -DMB_FORCE_IGEMM, -DMB_NO_IGEMM_1X1, and the original
 * standalone kernels/rvv/rvv_conv2d_s8_rvv_igemm.c -- emits the bias load
 * under a plain e32/m4 vtype and has zero such sites; the dispatch clone was
 * the only configuration that produced them, and the only one that failed.
 * Treat mixed-EEW unit-stride loads as a spike-invisible hazard on this
 * vector unit until the RTL is checked.
 *
 * Cost of the fix: the iGEMM body stays generic (a one-iteration tap loop
 * and a VLA of 4 pointers). Measured on FPGA that is still a 2.06x win on
 * the pointwise layers -- see the header table.
 */

/* Default-off gate. Unless the caller asks for the 1x1 iGEMM dispatch with
 * -DMB_IGEMM_1X1=1, this file behaves exactly like -DMB_NO_IGEMM_1X1, i.e.
 * exactly like the pre-dispatch production kernel. See the header for the
 * FPGA measurement that put it behind this gate. */
#if !defined(MB_FORCE_IGEMM) && !defined(MB_NO_IGEMM_1X1)
#  if !defined(MB_IGEMM_1X1) || (MB_IGEMM_1X1 == 0)
#    define MB_NO_IGEMM_1X1 1
#  endif
#endif

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_CONV_TILE
#define MB_CONV_TILE 4
#endif

static inline vint8m1_t mb_requant_i32m4(vint32m4_t vacc,
                                         int output_multiplier, int output_shift,
                                         int output_offset,
                                         int activation_min, int activation_max,
                                         size_t vl)
{
    vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
        vacc, output_multiplier, __RISCV_VXRM_RNU, vl);
    vint16m2_t vout16;
    if (output_shift < 0) {
        vint32m4_t vsh = __riscv_vsll_vx_i32m4(vscaled, (size_t)(-output_shift), vl);
        vout16 = __riscv_vnclip_wx_i16m2(vsh, 0, __RISCV_VXRM_RNU, vl);
    } else if (output_shift < 32) {
        vout16 = __riscv_vnclip_wx_i16m2(vscaled, (size_t)output_shift,
                                         __RISCV_VXRM_RNU, vl);
    } else {
        int sa2 = output_shift - 31;
        if (sa2 > 31) sa2 = 31;
        vint32m4_t v2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
        vout16 = __riscv_vnclip_wx_i16m2(v2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
    }
    vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
    vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
    vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);
    return __riscv_vnsra_wx_i8m1(vout16, 0, vl);
}

/* Store one output pixel's vl channels.
 *
 * V9: the vsse8 strided store is reinstated. Output is NCHW so a pixel's vl
 * channels sit OH*OW apart; the _obuf form spilled the vector to the stack
 * and then ran vl scalar load/stores (on dronet conv_modules.0 that is
 * 3136 px x 32 ch = 100k pairs). vsse8 expresses the same access pattern as
 * one instruction and measured 5.8% on dronet conv.
 *
 * The V8 port dropped it only to change one variable at a time while
 * chasing correctness. The root cause turned out to be 32-bit index
 * arithmetic wrapping for BSS buffers above 0x80000000 -- the STORE FORM was
 * never implicated, only the index arithmetic, which oc_blocked's size_t
 * hoisting (kept here) fixes. So vsse8 comes back and is re-measured. */
#define MB_STORE_PIX(ACC, OFF)                                                \
    do {                                                                      \
        vint8m1_t _v = mb_requant_i32m4((ACC), output_multiplier,             \
            output_shift, output_offset, activation_min, activation_max, vl); \
        __riscv_vsse8_v_i8m1((op) + (OFF), st, _v, vl);                       \
    } while (0)


/* ==========================================================================
 * PATH A -- tiled-direct (production default; used for every KH*KW > 1 shape)
 * Verbatim body of the previous kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c
 * kernel_conv2d_s8, now a static function. Its original doc block follows.
 * ==========================================================================
 */
/* NOTE: the rvv_vsmul_vnclip ALGORITHM is retired. Measured on FPGA it is
 * wrong on yolov8n (untiled -> max_abs_err=3, tiled -> Store/AMO access fault)
 * while spike reports bit-exact, because it indexes the input with 32-bit int
 * arithmetic that wraps for BSS buffers above 0x80000000. This file now carries
 * the rvv_oc_blocked-derived tiled port so that EVERY selector path is safe:
 * rvv_vsmul_vnclip sorts before rvv_oc_blocked in spec.algorithms, so leaving
 * the old code here would keep handing models the broken kernel. Collapse the
 * two names when convenient.
 */
/* origin: rvv_oc_blocked (OC cache blocking + size_t-hoisted indexing)
 *         WITH output-pixel register tiling cascaded 4 -> 2 -> 1.
 *
 * Why this file exists
 * --------------------
 * Two separate results forced this combination.
 *
 * 1. Tiling is the big win. Amortizing one weight vle8+vwadd over several
 *    output pixels took conv2d_s8 1.64x on dronet and 1.70x on fused_full,
 *    because conv was bound by per-reduction-index scalar overhead, not by
 *    vector width. (Vectorizing over OW instead was 1.7x SLOWER -- it makes
 *    the input the vector operand, and at SW=2 that is a vlse8 gather which
 *    this vector unit serialises.)
 *
 * 2. But tiling was built on rvv_vsmul_vnclip, and that kernel is WRONG on
 *    yolov8n: measured on FPGA, oc_blocked -> err=0, untiled vsmul_vnclip
 *    -> err=3, tiled vsmul_vnclip -> Store/AMO access fault. All three are
 *    bit-exact on spike, which cannot see it. The defect predates all
 *    tiling work.
 *
 * The two kernels have byte-identical MAC and requantize sequences. They
 * differ in exactly three things, and this file takes oc_blocked's side of
 * all three rather than guessing which one matters:
 *   - the oc_outer L1D blocking loop;
 *   - the size_t-hoisted row offset. vsmul_vnclip indexes the input as
 *     input[((n*IC+ic)*IH+ih)*IW+iw], entirely in int. oc_blocked hoists it
 *     to size_t with the comment that 32-bit index arithmetic wraps when a
 *     BSS-placed buffer's low 32 bits cross the int32 sign boundary --
 *     buffers live above 0x80000000, so their low word is negative as an
 *     int32. That is layout-dependent, which is exactly why the symptom
 *     moves between err=1 / err=3 / fault as the binary shifts;
 *   - the store form (V8 took oc_blocked's _obuf round-trip; V9 below
 *     reinstates vsse8 now that the index arithmetic is proven to be the
 *     actual defect).
 *
 * Padding semantics are preserved: an out-of-bounds tap is NOT skipped, it
 * contributes input_offset*(w+filter_offset), because the quantized pad
 * value 0 still carries the input zero point. Only the all-in-bounds fast
 * path elides the per-tap checks.
 */

static __attribute__((noinline)) void mb_conv2d_s8_tiled_direct(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;

    /* OC cache blocking, lifted from rvv_oc_blocked: keep a TILE_OC slab of
     * weights resident in L1D across the whole spatial sweep instead of
     * walking the entire weight tensor per output position. */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const int oc_slab_bytes = (int)oc_stride;
    int TILE_OC;
    if (oc_slab_bytes > 0 && oc_slab_bytes <= L1D_OC_BUDGET_BYTES) {
        TILE_OC = L1D_OC_BUDGET_BYTES / oc_slab_bytes;
        if (TILE_OC > vlmax_oc) TILE_OC = (TILE_OC / vlmax_oc) * vlmax_oc;
        else                    TILE_OC = vlmax_oc;
    } else {
        TILE_OC = vlmax_oc;
    }
    if (TILE_OC > OC) TILE_OC = OC;
    if (TILE_OC <= 0) TILE_OC = OC;

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ow = 0;

            /* ---- tiled strip: MB_CONV_TILE pixels share each weight load ---- */
            for (; ow + MB_CONV_TILE <= OW; ow += MB_CONV_TILE) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                    vint32m4_t a0, a1, a2, a3;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb; a2 = vb; a3 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz; a2 = vz; a3 = vz;
                    }

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            int iw_last = iw0 + (MB_CONV_TILE - 1) * SW + (KW - 1);
                            const int8_t *ip = in_ic + (size_t)ih * IW + iw0;

                            if (row_in && iw0 >= 0 && iw_last < IW) {
                                /* all taps of all TILE pixels in bounds */
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    const int8_t *q = ip + kw;
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)q[0]        + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)q[SW]       + input_offset), vw16, vl);
                                    a2 = __riscv_vwmacc_vx_i32m4(a2,
                                        (int16_t)((int32_t)q[2*SW]     + input_offset), vw16, vl);
                                    a3 = __riscv_vwmacc_vx_i32m4(a3,
                                        (int16_t)((int32_t)q[3*SW]     + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            } else {
                                const int8_t *in_row = in_ic + (size_t)ih * IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    int8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
                                    if (row_in) {
                                        int w0 = iw0 + kw;
                                        if (w0 >= 0 && w0 < IW)             b0 = in_row[w0];
                                        if (w0+SW >= 0 && w0+SW < IW)       b1 = in_row[w0+SW];
                                        if (w0+2*SW >= 0 && w0+2*SW < IW)   b2 = in_row[w0+2*SW];
                                        if (w0+3*SW >= 0 && w0+3*SW < IW)   b3 = in_row[w0+3*SW];
                                    }
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)b0 + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)b1 + input_offset), vw16, vl);
                                    a2 = __riscv_vwmacc_vx_i32m4(a2,
                                        (int16_t)((int32_t)b2 + input_offset), vw16, vl);
                                    a3 = __riscv_vwmacc_vx_i32m4(a3,
                                        (int16_t)((int32_t)b3 + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            }
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);
                    MB_STORE_PIX(a2, 2);
                    MB_STORE_PIX(a3, 3);

                    oc_base += (int)vl;
                }
            }

            /* ---- half-width strip: 2 pixels per weight load.
             * OW=7 shapes (dronet conv_modules.4/5/6) left 3 of every
             * 7 columns on the untiled path, which is why they gained
             * least from TILE=4 (19-33% of the vwmacc ceiling vs ~50%
             * for OW=56/14). A 2-pixel tier reclaims most of it. */
            for (; ow + 2 <= OW; ow += 2) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                    vint32m4_t a0, a1;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz;
                    }

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;

                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            int iw_last = iw0 + 1 * SW + (KW - 1);
                            const int8_t *ip = in_ic + (size_t)ih * IW + iw0;

                            if (row_in && iw0 >= 0 && iw_last < IW) {
                                /* all taps of all TILE pixels in bounds */
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    const int8_t *q = ip + kw;
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)q[0]        + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)q[SW]       + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            } else {
                                const int8_t *in_row = in_ic + (size_t)ih * IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                    vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                        vw8, (int16_t)filter_offset, vl);
                                    int8_t b0 = 0, b1 = 0;
                                    if (row_in) {
                                        int w0 = iw0 + kw;
                                        if (w0 >= 0 && w0 < IW)             b0 = in_row[w0];
                                        if (w0+SW >= 0 && w0+SW < IW)       b1 = in_row[w0+SW];
                                    }
                                    a0 = __riscv_vwmacc_vx_i32m4(a0,
                                        (int16_t)((int32_t)b0 + input_offset), vw16, vl);
                                    a1 = __riscv_vwmacc_vx_i32m4(a1,
                                        (int16_t)((int32_t)b1 + input_offset), vw16, vl);
                                    wq += OC;
                                }
                            }
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);

                    oc_base += (int)vl;
                }
            }

            /* ---- remainder columns: one pixel at a time ---- */
            for (; ow < OW; ow++) {
                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));
                    vint32m4_t vacc;
                    if (bias != NULL) vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else              vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_ic =
                            input + (size_t)(n*IC + ic) * IH * IW;
                        const int8_t *w_ic = weight
                            + (size_t)ic * KH * KW * OC + oc_base;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in = (ih >= 0 && ih < IH);
                            const int8_t *in_row = in_ic + (size_t)ih * IW;
                            const int8_t *wq = w_ic + (size_t)kh * KW * OC;
                            int iw0 = ow * SW - PW;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = iw0 + kw;
                                int8_t in_byte = 0;
                                if (row_in && iw >= 0 && iw < IW)
                                    in_byte = in_row[iw];
                                vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                                vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                    vw8, (int16_t)filter_offset, vl);
                                vacc = __riscv_vwmacc_vx_i32m4(vacc,
                                    (int16_t)((int32_t)in_byte + input_offset),
                                    vw16, vl);
                                wq += OC;
                            }
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(vacc, 0);
                    oc_base += (int)vl;
                }
            }
        }
    }
    }
}

#if !defined(MB_NO_IGEMM_1X1)

/* ==========================================================================
 * PATH B -- implicit GEMM with an indirection buffer (used for KH==KW==1).
 * Verbatim body of kernels/rvv/rvv_conv2d_s8_rvv_igemm.c's kernel_conv2d_s8,
 * now a static function. Its original doc block follows.
 * ==========================================================================
 */
/* weight_layout: ihwoc  (same pre-pack as rvv_oc_blocked / rvv_vsmul_vnclip:
 *  weight[((ic*KH+kh)*KW+kw)*OC+oc], OC innermost/contiguous.) */
/*
 * conv2d_s8 as implicit GEMM (XNNPACK igemm style), adapted to this SoC's
 * hardware constraints and to ModelBlaster's NCHW/planar tensor layout.
 *
 * WHY (diagnosis this file is trying to fix)
 * -------------------------------------------
 * rvv_vsmul_vnclip / rvv_oc_blocked already vectorize over OC (the natural
 * choice: weight is OC-contiguous, so an OC-wide vle8+vwmacc is the MAC),
 * and already register-tile 4 output pixels per weight load (1.64-1.88x
 * win, because that amortizes the weight vle8+vwadd across pixels). What
 * neither does is amortize the PADDING/BOUNDS BOOKKEEPING across the
 * reduction. Their loop nest is:
 *
 *   oc_base loop:
 *     for ic in IC:                      <-- redone per oc_base slice
 *       for kh in KH:                    <-- redone per ic (IC-fold waste)
 *         ih = oh*SH-PH+kh; row_in = ...; iw0 = ...; iw_last = ...
 *         if (row_in && in-bounds) { for kw: MAC }
 *         else                     { for kw: per-pixel-masked MAC }
 *
 * `ih`, `row_in`, `iw0`, `iw_last` do not depend on `ic` at all, yet they
 * are recomputed on every one of the IC iterations, and the whole thing is
 * redone again for every oc_base slice of the OC cache-blocking loop (up
 * to OC/vl times). For a 1x1 conv (KH=KW=1, dronet has one) that bookkeeping
 * IS the entire non-MAC cost of the tap. This is exactly the
 * "per-reduction-index overhead ... address arithmetic and a branch for
 * every vwmacc" diagnosis: the address/branch work already happens at
 * per-K-step granularity (well, per-(ic,kh) granularity after tiling), but
 * K = IC*KH*KW is not the loop the bookkeeping is over.
 *
 * THE FIX: an indirection buffer (XNNPACK's own trick), adapted to NCHW.
 * -------------------------------------------------------------------
 * XNNPACK's igemm builds one pointer per (output pixel, kernel tap) into
 * NHWC input, where a fixed pixel's IC channel values are contiguous, so
 * the K-loop over IC is a `*a0++` walk. Our input is NCHW (channel-planar):
 * a fixed pixel's channels are IH*IW bytes apart, not 1 byte apart. That
 * is fine for a *scalar* load (this kernel never turns it into a vector
 * strided load -- see the "measured and refuted" note below) -- it is
 * exactly as costly as a `*a0++` walk, just with a runtime stride instead
 * of 1. So the same trick applies with an added per-tap step size:
 *
 *   for a tile of MR=4 (then 2, then 1) consecutive output columns, build
 *   ONCE (before the oc_base loop, i.e. shared across the WHOLE OC sweep,
 *   not just the WHOLE IC reduction):
 *     aptr[pixel][tap]  = &input[n, 0, ih(tap), iw(tap,pixel)]   (ic=0 base)
 *                         or &mb_igemm_zero if that tap is out of bounds
 *     astep[pixel][tap] = IH*IW   (real data: walk to the next channel)
 *                         or 0    (padding: keep re-reading the same zero
 *                                  byte every ic step -- branch-free, and
 *                                  still contributes input_offset*(w+
 *                                  filter_offset) per ic, matching the
 *                                  conv2d_s8 spec's OOB semantics exactly)
 *
 *   Then the reduction becomes, per tap, per ic:
 *     vw8  = vle8(weight_tap_base + ic*(KH*KW*OC), vl)   // still OC-contig
 *     vw16 = vwadd_vx(vw8, filter_offset, vl)
 *     a[j] = vwmacc_vx(a[j], *p[j] + input_offset, vw16, vl)   for j in MR
 *     p[j] += astep[j][tap]                                    for j in MR
 *
 *   No branch, no ih/row_in/iw0 recompute, anywhere in that loop. The
 *   bookkeeping drops from O(TILE_OC/vl * IC * KH) occurrences per output
 *   tile down to O(MR * KH*KW) occurrences -- an IC-fold (up to 512x) and
 *   oc_base-fold (up to 8x, OC/vl) reduction, done ONCE per (n,oh,ow_tile)
 *   and reused by every oc_base slice.
 *
 *   The indirection buffer itself is a small stack array (<= MB_CONV_TILE *
 *   KH*KW pointers, e.g. 4*9=36 for a 3x3 -- nowhere near a materialised
 *   im2col buffer, which would be OH*OW*KH*KW*IC bytes). This matches
 *   XNNPACK's own microkernel, which also re-walks the SAME `a[]` pointer
 *   array once per nc-tile (rewinding `a -= ks`) rather than caching input
 *   data across the N/OC sweep -- redundant A-side re-reads across OC
 *   tiles is standard GEMM microkernel practice, not a defect; L1D makes
 *   the re-read cheap. This file does the same (rebuild-not-cache across
 *   oc_base is fine; what's fixed is the address ARITHMETIC, not the data
 *   re-reads).
 *
 * MEASURED AND REFUTED -- do not repeat:
 *   - This file never turns the input access into a *vector* strided load
 *     (no vlse8 on input). Every K-step input read is one scalar load per
 *     accumulator row (exactly as many scalar loads as vsmul_vnclip does
 *     today), because Saturn serialises strided vector loads to ~1
 *     elem/cycle and OW-vectorization already measured 1.7x slower for
 *     exactly that reason.
 *   - oc_blocked (this file's OC slab strategy, unchanged) is kept because
 *     dropping it was 1.66-1.81x slower previously.
 *
 * CORRECTNESS: all pointer/index arithmetic that scales with N, IC, IH, IW
 * or OC is size_t/ptrdiff_t (see rule: 32-bit index arithmetic wraps for
 * BSS buffers above 0x80000000 -- this kernel does MORE pointer arithmetic
 * than its predecessors, via the running `p[j] += astep` walk, so this is
 * enforced even more carefully here than in oc_blocked).
 *
 * mr x nr tile: mr=4 (register-tiled output pixels, cascading 4 -> 2 -> 1
 * for OW remainders) x nr=vlmax_e32m4 (32 output channels at VLEN=256,
 * LMUL=4). mr=4 matches fmaPipeDepth=4 -- four independent i32m4
 * accumulators (16 of 32 vector registers) keep the MAC pipeline full
 * without register pressure spilling the weight/temp registers.
 */

/* Build the indirection buffer for `mr` consecutive output columns starting
 * at (oh, ow0), for one batch element (in_n0 = input + n*IC*IH*IW, i.e. the
 * ic=0 base of that batch's channel-plane stack). One (pointer, ic-step)
 * pair per (pixel, tap); a tap out of [0,IH)x[0,IW) bounds for a pixel gets
 * the zero sentinel with step=0, so the caller's ic-loop reads the same
 * zero byte every step -- no per-element branch, and it still contributes
 * input_offset*(w+filter_offset) per ic exactly as the conv2d_s8 spec
 * requires for out-of-bounds taps.
 *
 * aptr/astep are flat [mr*taps] arrays, row-major as [pixel][tap] with row
 * stride `taps` (taps = KH*KW, constant for the whole kernel call, so the
 * caller can size/index a single buffer for every mr tier).
 */
static inline void mb_igemm_build_indir(
    const int8_t *restrict in_n0, int IH, int IW,
    int oh, int ow0, int SH, int SW, int PH, int PW, int KH, int KW,
    int mr, int taps,
    const int8_t **restrict aptr, ptrdiff_t *restrict astep)
{
    static const int8_t mb_igemm_zero = 0;
    const ptrdiff_t ic_stride = (ptrdiff_t)IH * (ptrdiff_t)IW;

    for (int j = 0; j < mr; j++) {
        const int ow_j = ow0 + j;
        int t = 0;
        for (int kh = 0; kh < KH; kh++) {
            const int ih = oh * SH - PH + kh;
            const int row_ok = (ih >= 0 && ih < IH);
            for (int kw = 0; kw < KW; kw++, t++) {
                const int iw = ow_j * SW - PW + kw;
                const size_t idx = (size_t)j * (size_t)taps + (size_t)t;
                if (row_ok && iw >= 0 && iw < IW) {
                    aptr[idx]  = in_n0 + (size_t)ih * (size_t)IW + (size_t)iw;
                    astep[idx] = ic_stride;
                } else {
                    aptr[idx]  = &mb_igemm_zero;
                    astep[idx] = 0;
                }
            }
        }
    }
}

static __attribute__((noinline, noipa)) void mb_conv2d_s8_igemm(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const int taps = KH * KW;

    const ptrdiff_t oc_stride = (ptrdiff_t)IC * KH * KW;
    const ptrdiff_t w_ic_stride = (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)OC;

    /* Indirection buffer, sized once for the largest tier (MB_CONV_TILE);
     * smaller tiers just use the first mr*taps entries. */
    const int8_t *aptr_buf[MB_CONV_TILE * (taps > 0 ? taps : 1)];
    ptrdiff_t     astep_buf[MB_CONV_TILE * (taps > 0 ? taps : 1)];

    /* OC cache blocking, unchanged from rvv_oc_blocked / rvv_vsmul_vnclip:
     * keep a TILE_OC slab of weights resident in L1D across the whole
     * spatial sweep instead of walking the entire weight tensor per
     * output position. */
    enum { L1D_OC_BUDGET_BYTES = 24 * 1024 };
    const int vlmax_oc = (int)__riscv_vsetvlmax_e32m4();
    const int oc_slab_bytes = (int)oc_stride;
    int TILE_OC;
    if (oc_slab_bytes > 0 && oc_slab_bytes <= L1D_OC_BUDGET_BYTES) {
        TILE_OC = L1D_OC_BUDGET_BYTES / oc_slab_bytes;
        if (TILE_OC > vlmax_oc) TILE_OC = (TILE_OC / vlmax_oc) * vlmax_oc;
        else                    TILE_OC = vlmax_oc;
    } else {
        TILE_OC = vlmax_oc;
    }
    if (TILE_OC > OC) TILE_OC = OC;
    if (TILE_OC <= 0) TILE_OC = OC;

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

    for (int n = 0; n < N; n++) {
        const int8_t *in_n0 = input + (size_t)n * (size_t)IC * (size_t)IH * (size_t)IW;

        for (int oh = 0; oh < OH; oh++) {
            int ow = 0;

            /* ---- tiled strip: MB_CONV_TILE pixels share the indirection
             * buffer AND every weight load across the whole oc_base sweep. */
            for (; ow + MB_CONV_TILE <= OW; ow += MB_CONV_TILE) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, MB_CONV_TILE, taps,
                                      aptr_buf, astep_buf);

                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                    vint32m4_t a0, a1, a2, a3;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb; a2 = vb; a3 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz; a2 = vz; a3 = vz;
                    }

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[0*(size_t)taps + t];
                        const int8_t *p1 = aptr_buf[1*(size_t)taps + t];
                        const int8_t *p2 = aptr_buf[2*(size_t)taps + t];
                        const int8_t *p3 = aptr_buf[3*(size_t)taps + t];
                        const ptrdiff_t s0 = astep_buf[0*(size_t)taps + t];
                        const ptrdiff_t s1 = astep_buf[1*(size_t)taps + t];
                        const ptrdiff_t s2 = astep_buf[2*(size_t)taps + t];
                        const ptrdiff_t s3 = astep_buf[3*(size_t)taps + t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);

                            a0 = __riscv_vwmacc_vx_i32m4(a0,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            a1 = __riscv_vwmacc_vx_i32m4(a1,
                                (int16_t)((int32_t)(*p1) + input_offset), vw16, vl);
                            a2 = __riscv_vwmacc_vx_i32m4(a2,
                                (int16_t)((int32_t)(*p2) + input_offset), vw16, vl);
                            a3 = __riscv_vwmacc_vx_i32m4(a3,
                                (int16_t)((int32_t)(*p3) + input_offset), vw16, vl);

                            p0 += s0; p1 += s1; p2 += s2; p3 += s3;
                            wq += w_ic_stride;
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);
                    MB_STORE_PIX(a2, 2);
                    MB_STORE_PIX(a3, 3);

                    oc_base += (int)vl;
                }
            }

            /* ---- half-width strip: 2 pixels. Same indirection scheme,
             * mr=2. Reclaims the OW=7-style shapes (dronet conv_modules
             * .4/5/6) that the 4-wide tile leaves 3-of-7 columns idle on. */
            for (; ow + 2 <= OW; ow += 2) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, 2, taps, aptr_buf, astep_buf);

                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));

                    vint32m4_t a0, a1;
                    if (bias != NULL) {
                        vint32m4_t vb = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                        a0 = vb; a1 = vb;
                    } else {
                        vint32m4_t vz = __riscv_vmv_v_x_i32m4(0, vl);
                        a0 = vz; a1 = vz;
                    }

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[0*(size_t)taps + t];
                        const int8_t *p1 = aptr_buf[1*(size_t)taps + t];
                        const ptrdiff_t s0 = astep_buf[0*(size_t)taps + t];
                        const ptrdiff_t s1 = astep_buf[1*(size_t)taps + t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);

                            a0 = __riscv_vwmacc_vx_i32m4(a0,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            a1 = __riscv_vwmacc_vx_i32m4(a1,
                                (int16_t)((int32_t)(*p1) + input_offset), vw16, vl);

                            p0 += s0; p1 += s1;
                            wq += w_ic_stride;
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(a0, 0);
                    MB_STORE_PIX(a1, 1);

                    oc_base += (int)vl;
                }
            }

            /* ---- remainder columns: one pixel at a time, mr=1. Still goes
             * through the same branch-free indirection scheme -- there is
             * no separate "slow path" in this kernel at all, unlike its
             * predecessors, because the padding sentinel makes the fast
             * and slow cases the same code. */
            for (; ow < OW; ow++) {
                mb_igemm_build_indir(in_n0, IH, IW, oh, ow, SH, SW, PH, PW,
                                      KH, KW, 1, taps, aptr_buf, astep_buf);

                int oc_base = oc_outer;
                while (oc_base < oc_end) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc_base));
                    vint32m4_t vacc;
                    if (bias != NULL) vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else              vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int t = 0; t < taps; t++) {
                        const int8_t *p0 = aptr_buf[t];
                        const ptrdiff_t s0 = astep_buf[t];
                        const int8_t *wq = weight + (size_t)t * (size_t)OC + (size_t)oc_base;

                        for (int ic = 0; ic < IC; ic++) {
                            vint8m1_t vw8 = __riscv_vle8_v_i8m1(wq, vl);
                            vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                vw8, (int16_t)filter_offset, vl);
                            vacc = __riscv_vwmacc_vx_i32m4(vacc,
                                (int16_t)((int32_t)(*p0) + input_offset), vw16, vl);
                            p0 += s0;
                            wq += w_ic_stride;
                        }
                    }

                    int8_t *op = output + ((size_t)n * OC + oc_base) * OH * OW
                                        + (size_t)oh * OW + ow;
                    ptrdiff_t st = (ptrdiff_t)(OH * OW);
                    MB_STORE_PIX(vacc, 0);
                    oc_base += (int)vl;
                }
            }
        }
    }
    }
}

#endif /* !MB_NO_IGEMM_1X1 */


/* ==========================================================================
 * DISPATCH
 * ========================================================================== */
static void mb_kbx_rvv_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
#if defined(MB_FORCE_IGEMM)
    const int use_igemm = 1;
#elif defined(MB_NO_IGEMM_1X1)
    const int use_igemm = 0;
#else
    /* Pointwise only. Stride/padding are handled correctly by the iGEMM
     * indirection buffer (OOB taps take the zero sentinel with ic-step 0), so
     * no further shape guard is needed -- but a degenerate shape that would
     * make OH/OW non-positive is sent to the direct path, which is the one
     * that has always handled it. */
    const int use_igemm = (KH == 1 && KW == 1 && SH > 0 && SW > 0 &&
                           IH > 0 && IW > 0 && IC > 0 && OC > 0);
#endif

#if !defined(MB_NO_IGEMM_1X1)
    if (use_igemm) {
        mb_conv2d_s8_igemm(input, weight, bias, output,
                           N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                           input_offset, filter_offset, output_offset,
                           output_multiplier, output_shift,
                           activation_min, activation_max);
        return;
    }
#else
    (void)use_igemm;
#endif

    mb_conv2d_s8_tiled_direct(input, weight, bias, output,
                              N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                              input_offset, filter_offset, output_offset,
                              output_multiplier, output_shift,
                              activation_min, activation_max);
}


/* ---- winner dispatch ---- */
static const short mb_kbx_rvv_shapes[][10] = {
    {3,112,112,32,3,3,2,2,1,1},
    {32,27,27,32,3,3,2,2,1,1},
    {32,27,27,32,1,1,2,2,0,0}
};
#define MB_KBX_UNTILE_SLOTS 64
typedef struct { int state; const int8_t *w; int8_t *ihwoc; } mb_kbx_slot_t;
static mb_kbx_slot_t mb_kbx_slots[MB_KBX_UNTILE_SLOTS];
/* IHWOC copy (w[k*OC + oc]) of X100 tiles [ks8][ceil(OC/4)][4][8] behind a 64-byte header. */
static const int8_t *mb_kbx_untile(const int8_t *t, int OC, int K)
{
    uint32_t r, k;
    memcpy(&r, t + 8, 4); memcpy(&k, t + 12, 4);
    if ((int)r != OC || (int)k != K) return NULL;
    size_t h = ((uintptr_t)t >> 6) % MB_KBX_UNTILE_SLOTS;
    for (int i = 0; i < MB_KBX_UNTILE_SLOTS; i++) {
        mb_kbx_slot_t *S = &mb_kbx_slots[(h + (size_t)i) % MB_KBX_UNTILE_SLOTS];
        int st = __atomic_load_n(&S->state, __ATOMIC_ACQUIRE);
        if (st == 2) { if (S->w == t) return S->ihwoc; continue; }
        if (st == 1) return NULL;                  /* being built by another thread */
        int zero = 0;
        if (!__atomic_compare_exchange_n(&S->state, &zero, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { i--; continue; }
        int8_t *d = (int8_t *)malloc((size_t)OC * (size_t)K);
        if (!d) { __atomic_store_n(&S->state, 0, __ATOMIC_RELEASE); return NULL; }
        int rt = (OC + 3) / 4;
        for (int kk = 0; kk < K; kk++)
            for (int oc = 0; oc < OC; oc++)
                d[(size_t)kk * OC + oc] = t[64 + ((size_t)(kk / 8) * rt + oc / 4) * 32 + (oc % 4) * 8 + kk % 8];
        S->w = t; S->ihwoc = d;
        __atomic_store_n(&S->state, 2, __ATOMIC_RELEASE);
        return d;
    }
    return NULL;
}

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset, int output_multiplier,
                      int output_shift, int activation_min, int activation_max)
{
    for (size_t i = 0; i < sizeof mb_kbx_rvv_shapes / sizeof mb_kbx_rvv_shapes[0]; i++) {
        const short *s = mb_kbx_rvv_shapes[i];
        if (s[0] != IC || s[1] != IH || s[2] != IW || s[3] != OC || s[4] != KH || s[5] != KW
            || s[6] != SH || s[7] != SW || s[8] != PH || s[9] != PW) continue;
        const int8_t *w = weight;
        long K = (long)IC * KH * KW;
        if ((long)OC * K >= 16 && memcmp(weight, "MBX100TW", 8) == 0)
            w = mb_kbx_untile(weight, OC, (int)K);
        if (w) {
            mb_kbx_rvv_conv2d_s8(input, w, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                                 input_offset, filter_offset, output_offset, output_multiplier,
                                 output_shift, activation_min, activation_max);
            return;
        }
        break;
    }
    mb_kbx_ime_conv2d_s8(input, weight, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                         input_offset, filter_offset, output_offset, output_multiplier,
                         output_shift, activation_min, activation_max);
}
