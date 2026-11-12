/* source: curated */
/* algorithm: ime2_vmadot_8x8x16 */
/* accuracy_class: bit_exact */
/* origin: A100 linear_s8, optimized 2026-10-28 from the 2026-10-19 hand-written
 *   1x1 kernel (kept as k3-work/docker/ime2_opt/variants/v0_baseline.c). Each step
 *   was measured on A100 cpu 8 (k3-work/docker/ime2_opt, bench_linear.c) and kept
 *   only if bit-exact on 63 cases and faster:
 *     - pack with 8-byte words / segment loads, no offset sums when symmetric
 *     - vector Q0.31 tail (vsmul/vssra with round-to-nearest-up = the reference)
 *     - 2x4 register blocking: 2 A x 4 B tiles per K slab, 8 accumulators
 *     - tail fused into the matrix asm; int8 tiles leave via a unit-stride buffer
 *     - B packed with strided 2-field segment loads (vlsseg2e64), 4 tiles at once
 *     - tiny outputs (M*N <= 16, e.g. dronet's heads) on a scalar path
 *   Measured (ticks of 24 MHz, one A100 core): ffn_block fc1 115,692 -> 11,822,
 *   fc2 93,701 -> 11,877, dronet 1x2048x1 2,707 -> 54. Tried and rejected:
 *   3x4 blocking, 2x K unroll, LMUL-8 tail, row-wise and scalar packing.
 *   Remaining cost is mostly packing the weights on every call; a build-time
 *   pre-packed weight layout would cut fc1 to about 8,800 (measured by simulation).
 *
 * A100 cores only: traps unless vlenb == 128. Helpers are shared with the other
 * ime2_a100 kernels through the MB_IME2_CORE guard (one translation unit). */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
/* A100 only: GCC's own short vector code (auto-vectorized loops, vl ~8) is slow
 * on this core (a reloaded-line vle8 costs ~17 cycles vs 2 on the X100), so it is
 * turned off for the A100 kernels alone; a global flag slows the X100 side 11%
 * (TASK-012). Popped at the end of the file. */
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize")

#ifndef MB_IME2_CORE
#define MB_IME2_CORE
/* One 8-byte scalar load / store at any alignment. The A100 does misaligned
 * ld/sd in hardware (ld 3.8 cycles vs 3.0 aligned, sd 2.0 either way;
 * micro_misalign.c, t013_micro_misalign_st.c), while GCC expands memcpy(.., 8)
 * or memset(.., 16) of unknown alignment into short vector code (slow on the
 * A100) or, with scalar string ops, byte loops (TASK-013).
 * Macros, not inline functions: GCC refuses to inline an always_inline helper
 * here when the file also holds a target("arch=...") function (mb_ai_task.h). */
typedef struct { char b[8]; } ime2c_b8;
#define ime2c_ld8(p) ({ uint64_t ime2c_v_; \
    __asm__("ld %0, %1" : "=r"(ime2c_v_) : "m"(*(const ime2c_b8 *)(const void *)(p))); ime2c_v_; })
#define ime2c_sd8(p, v) __asm__("sd %1, %0" : "=m"(*(ime2c_b8 *)(void *)(p)) : "r"((uint64_t)(v)))
static inline void ime2c_copy16(int8_t *restrict dst, const int8_t *restrict src)
{
    ime2c_sd8(dst, ime2c_ld8(src)); ime2c_sd8(dst + 8, ime2c_ld8(src + 8));
}

static inline __attribute__((always_inline)) void ime2c_pack_tile(int8_t *restrict dst, const int8_t *restrict src,
                               int r0, int R, int k0, int K)
{
    int full_k = k0 + 16 <= K;
    for (int i = 0; i < 8; i++) {
        int r = r0 + i;
        int8_t *d = dst + i * 16;
        if (r >= R) { ime2c_sd8(d, 0); ime2c_sd8(d + 8, 0); continue; }
        const int8_t *s = src + (size_t)r * (size_t)K + k0;
        if (full_k) { ime2c_copy16(d, s); continue; }
        for (int q = 0; q < 16; q++) d[q] = (k0 + q < K) ? s[q] : 0;
    }
}

static inline int8_t ime2c_requant(int32_t acc, int mult, int shift, int ooff, int amin, int amax)
{
    int64_t prod = (int64_t)acc * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t s = (int32_t)prod;
    if (shift > 0) s = (int32_t)(((int64_t)s + ((int64_t)1 << (shift - 1))) >> shift);
    else if (shift < 0) s = s << (-shift);
    s += ooff;
    if (s < amin) s = amin;
    if (s > amax) s = amax;
    return (int8_t)s;
}

static inline void ime2c_row_sums(const int8_t *m, int R, int K, int32_t *sums)
{
    for (int r = 0; r < R; r++) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += m[(size_t)r * K + k];
        sums[r] = s;
    }
}

static inline __attribute__((always_inline)) void ime2c_mma_1x1_s(const int8_t *a, const int8_t *b, long slabs, long bstride, int32_t *out)
{
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v9, 0\n\t"
        "1:\n\t"
        "vle8.v v0, (%[pa])\n\t"
        "vle8.v v4, (%[pb])\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        "addi %[pa], %[pa], 128\n\t"
        "add %[pb], %[pb], %[bs]\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v8, (%[out])\n\t"
        : [pa] "+r"(a), [pb] "+r"(b), [ks] "+r"(slabs)
        : [out] "r"(out), [bs] "r"(bstride)
        : "t0", "memory", "vl", "vtype", "v0", "v4", "v8", "v9");
}

static inline __attribute__((always_inline)) void ime2c_mma_2x4(const int8_t *a0, const int8_t *a1, const int8_t *b, long slabs, int32_t *out)
{
    __asm__ volatile(
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "1:\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "vle8.v v1, (%[a1])\n\t"
        "vle8.v v2, (%[b])\n\t"
        "addi t1, %[b], 128\n\t"
        "vle8.v v3, (t1)\n\t"
        "addi t1, %[b], 256\n\t"
        "vle8.v v4, (t1)\n\t"
        "addi t1, %[b], 384\n\t"
        "vle8.v v5, (t1)\n\t"
        ".insn r 0x2b, 3, 0x71, x8,  x0, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x1, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x1, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x1, x5\n\t"
        "addi %[a0], %[a0], 128\n\t"
        "addi %[a1], %[a1], 128\n\t"
        "addi %[b], %[b], 512\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vse32.v v8,  (%[o])\n\t"
        "addi t1, %[o], 256\n\t"  "vse32.v v10, (t1)\n\t"
        "addi t1, %[o], 512\n\t"  "vse32.v v12, (t1)\n\t"
        "addi t1, %[o], 768\n\t"  "vse32.v v14, (t1)\n\t"
        "addi t1, %[o], 1024\n\t" "vse32.v v16, (t1)\n\t"
        "addi t1, %[o], 1280\n\t" "vse32.v v18, (t1)\n\t"
        "addi t1, %[o], 1536\n\t" "vse32.v v20, (t1)\n\t"
        "addi t1, %[o], 1792\n\t" "vse32.v v22, (t1)\n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs)
        : [o] "r"(out)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
}

static inline __attribute__((always_inline)) void ime2c_mma_2x4_qb(const int8_t *a0, const int8_t *a1, const int8_t *b, long slabs,
                               const int32_t *bias4, int8_t *buf, long ldo_unused,
                               long mult, long lsh, long rsh, long ooff, long amin, long amax)
{
    (void)ldo_unused;
    __asm__ volatile(
        "csrwi vxrm, 0\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vle32.v v24, (%[bi])\n\t"
        "addi t1, %[bi], 256\n\t" "vle32.v v26, (t1)\n\t"
        "addi t1, %[bi], 512\n\t" "vle32.v v28, (t1)\n\t"
        "addi t1, %[bi], 768\n\t" "vle32.v v30, (t1)\n\t"
        "vsetvli t0, zero, e8, m8, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t"
        "vmv.v.i v16, 0\n\t"
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "1:\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "vle8.v v1, (%[a1])\n\t"
        "vle8.v v2, (%[b])\n\t"
        "addi t1, %[b], 128\n\t" "vle8.v v3, (t1)\n\t"
        "addi t1, %[b], 256\n\t" "vle8.v v4, (t1)\n\t"
        "addi t1, %[b], 384\n\t" "vle8.v v5, (t1)\n\t"
        ".insn r 0x2b, 3, 0x71, x8,  x0, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x5\n\t"
        ".insn r 0x2b, 3, 0x71, x16, x1, x2\n\t"
        ".insn r 0x2b, 3, 0x71, x18, x1, x3\n\t"
        ".insn r 0x2b, 3, 0x71, x20, x1, x4\n\t"
        ".insn r 0x2b, 3, 0x71, x22, x1, x5\n\t"
        "addi %[a0], %[a0], 128\n\t"
        "addi %[a1], %[a1], 128\n\t"
        "addi %[b], %[b], 512\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
#define MB_Q_TILE(T, B, OFF)                                         \
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"                      \
        "vadd.vv " T ", " T ", " B "\n\t"                            \
        "vsmul.vx " T ", " T ", %[mu]\n\t"                           \
        "vsll.vx " T ", " T ", %[ls]\n\t"                            \
        "vssra.vx " T ", " T ", %[rs]\n\t"                           \
        "vadd.vx " T ", " T ", %[oo]\n\t"                            \
        "vmax.vx " T ", " T ", %[lo]\n\t"                            \
        "vmin.vx " T ", " T ", %[hi]\n\t"                            \
        "vsetvli t0, zero, e16, m1, ta, ma\n\t"                      \
        "vnsrl.wi v6, " T ", 0\n\t"                                  \
        "vsetvli t0, zero, e8, mf2, ta, ma\n\t"                      \
        "vnsrl.wi v7, v6, 0\n\t"                                     \
        "addi t1, %[o], " OFF "\n\t"                                 \
        "vse8.v v7, (t1)\n\t"
        MB_Q_TILE("v8",  "v24", "0")
        MB_Q_TILE("v10", "v26", "64")
        MB_Q_TILE("v12", "v28", "128")
        MB_Q_TILE("v14", "v30", "192")
        MB_Q_TILE("v16", "v24", "256")
        MB_Q_TILE("v18", "v26", "320")
        MB_Q_TILE("v20", "v28", "384")
        MB_Q_TILE("v22", "v30", "448")
#undef MB_Q_TILE
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs), [o] "+r"(buf)
        : [bi] "r"(bias4), [mu] "r"(mult), [ls] "r"(lsh), [rs] "r"(rsh),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

static inline __attribute__((always_inline)) void ime2c_scatter_block(const int8_t *buf, int8_t *out, size_t ldo)
{
    for (int i = 0; i < 2; i++)
        for (int r = 0; r < 8; r++) {
            uint64_t *d = (uint64_t *)(out + (size_t)(8 * i + r) * ldo);
            const uint64_t *s = (const uint64_t *)(buf + i * 256 + r * 8);
            d[0] = s[0]; d[1] = s[8]; d[2] = s[16]; d[3] = s[24];
        }
}

static inline __attribute__((always_inline)) void ime2c_pack_rows_seg(int8_t *dst, const int8_t *src, long K, long rows)
{
    __asm__ volatile(
        "vsetvli zero, %[n], e64, m2, ta, ma\n\t"
        "vlsseg2e64.v v24, (%[s]), %[k]\n\t"
        "vsseg2e64.v v24, (%[d])\n\t"
        :
        : [n] "r"(rows), [s] "r"(src), [k] "r"(K), [d] "r"(dst)
        : "memory", "vl", "vtype", "v24", "v25", "v26", "v27");
}

static inline __attribute__((always_inline)) void ime2c_tail_vec(const int32_t *tile, const int32_t *bias64, int8_t *out, size_t ldo,
                              int mult, int shift, int ooff, int amin, int amax)
{
    size_t vl = __riscv_vsetvl_e32m2(64);
    vint32m2_t a = __riscv_vle32_v_i32m2(tile, vl);
    a = __riscv_vadd_vv_i32m2(a, __riscv_vle32_v_i32m2(bias64, vl), vl);
    a = __riscv_vsmul_vx_i32m2(a, mult, __RISCV_VXRM_RNU, vl);
    if (shift > 0) a = __riscv_vssra_vx_i32m2(a, (size_t)shift, __RISCV_VXRM_RNU, vl);
    else if (shift < 0) a = __riscv_vsll_vx_i32m2(a, (size_t)(-shift), vl);
    a = __riscv_vadd_vx_i32m2(a, ooff, vl);
    a = __riscv_vmax_vx_i32m2(a, amin, vl);
    a = __riscv_vmin_vx_i32m2(a, amax, vl);
    vint16m1_t h = __riscv_vncvt_x_x_w_i16m1(a, vl);
    vint8mf2_t b = __riscv_vncvt_x_x_w_i8mf2(h, vl);
    int8_t tmp[64] __attribute__((aligned(64)));
    __riscv_vse8_v_i8mf2(tmp, b, vl);
    for (int i = 0; i < 8; i++) ime2c_sd8(out + (size_t)i * ldo, ime2c_ld8(tmp + i * 8));
}

#endif /* MB_IME2_CORE */

#ifndef MB_IME2_TW
#define MB_IME2_TW
/* Weights packed at build time (TASK-002). In an ime2_a100 build ModelBlaster's
 * skeleton emits a linear, conv or fused-conv weight array already in the matrix
 * engine's tile order, behind a 64-byte header: "MBIME2TW", then rows (N for
 * linear, OC for conv) and K as little-endian uint32, then zeros. The build
 * aligns the array to 64 bytes, so the tiles start on a cache line: they are
 * loaded in place, and 8-byte-aligned tiles made the linear 2.2x slower
 * (TASK-008). Tiles, zero padded:
 *   linear: [ceil(N/32)][k_slabs][4][8][16]  (row n = 32g + 8p + r, k = 16ks + q)
 *   conv:   [ceil(OC/8)][k_slabs][8][16]     (k = (ic*KH + kh)*KW + kw)
 * which is exactly what these kernels otherwise pack on every call. Plain
 * weights have no header and are packed per call as before. */
static inline const int8_t *ime2tw_tiles(const int8_t *w, int rows, int K)
{
    if (memcmp(w, "MBIME2TW", 8) != 0) return NULL;
    uint32_t r, k;
    memcpy(&r, w + 8, 4); memcpy(&k, w + 12, 4);
    if (r != (uint32_t)rows || k != (uint32_t)K) __builtin_trap();   /* packed for another shape */
    return w + 64;
}

/* Tiles back to a plain copy (linear: [n][k]; conv: IHWOC [k][oc]) for the rare
 * paths that index weights one by one. The caller frees it. */
static int8_t *ime2tw_unpack(const int8_t *t, int rows, int K, int linear)
{
    int ksn = (K + 15) / 16;
    int8_t *p = malloc((size_t)rows * (size_t)K);
    if (!p) __builtin_trap();
    for (int r = 0; r < rows; r++)
        for (int k = 0; k < K; k++) {
            size_t i = linear
                ? ((((size_t)(r / 32) * ksn + k / 16) * 4 + (r % 32) / 8) * 128 + (r % 8) * 16 + k % 16)
                : (((size_t)(r / 8) * ksn + k / 16) * 128 + (r % 8) * 16 + k % 16);
            p[linear ? (size_t)r * K + k : (size_t)k * rows + r] = t[i];
        }
    return p;
}
#endif /* MB_IME2_TW */

/* The one place GCC's vectorizer pays on the A100: a long dot product (K up to
 * 2048) runs at full vector length, so it stays on here (1x2048x1: 52 ticks with
 * it, 305 without; TASK-013). */
static __attribute__((noinline, optimize("tree-vectorize"))) void ime2c_tiny_linear(const int8_t *input, const int8_t *weight, const int32_t *bias,
        int8_t *output, int M, int K, int N, int ioff, int foff, int ooff, int mult, int shift, int amin, int amax)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            const int8_t *x = input + (size_t)m * K, *w = weight + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += ((int32_t)x[k] + ioff) * ((int32_t)w[k] + foff);
            output[(size_t)m * N + n] = ime2c_requant(acc, mult, shift, ooff, amin, amax);
        }
}

static inline __attribute__((always_inline)) void ime2c_linear_tail(const int32_t *t, int m0, int n0, int M, int N, const int32_t *bias,
                           int asym, const int32_t *isum, const int32_t *wsum, int K, int ioff, int foff,
                           int vec_ok, const int32_t *bias64, int8_t *output,
                           int mult, int shift, int ooff, int amin, int amax)
{
    if (vec_ok && m0 + 8 <= M && n0 + 8 <= N) {
        ime2c_tail_vec(t, bias64, output + (size_t)m0 * N + n0, (size_t)N, mult, shift, ooff, amin, amax);
        return;
    }
    for (int i = 0; i < 8 && m0 + i < M; i++)
        for (int j = 0; j < 8 && n0 + j < N; j++) {
            int32_t acc = t[i * 8 + j];
            if (bias) acc += bias[n0 + j];
            if (asym)
                acc = (int32_t)((int64_t)acc + (int64_t)ioff * wsum[n0 + j] + (int64_t)foff * isum[m0 + i]
                                + (int64_t)K * ioff * foff);
            output[(size_t)(m0 + i) * N + n0 + j] = ime2c_requant(acc, mult, shift, ooff, amin, amax);
        }
}

/* The kernel body, once. Called with wt == NULL (plain weights, packed per call:
 * the code GCC generated before build-time tiles existed) and, from a separate
 * non-inlined function, with the tiles. Keeping the two apart matters: one body
 * handling both at run time compiled to 1.8x the code and made the plain path
 * 4-8% slower on ffn_block. */
static inline __attribute__((always_inline)) void ime2c_linear_core(const int8_t *input, const int8_t *weight,
        const int8_t *wt, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset,
        int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min,
        int activation_max) {
    int m_tiles = (M + 7) / 8, k_slabs = K > 0 ? (K + 15) / 16 : 1;
    size_t panel = (size_t)k_slabs * 128;
    int8_t *pa = aligned_alloc(64, (size_t)m_tiles * panel);
    int8_t *pb = wt ? NULL : aligned_alloc(64, 4 * panel);     /* 4 B panels, interleaved per slab */
    int32_t *ct = aligned_alloc(64, 8 * 64 * sizeof(int32_t)); /* 8 result tiles */
    int asym = input_offset != 0 || filter_offset != 0;
    int vec_ok = !asym && output_shift >= -31 && output_shift <= 31;
    int32_t *isum = NULL, *wsum = NULL;
    if (asym) {
        isum = malloc(sizeof(int32_t) * (size_t)M); ime2c_row_sums(input, M, K, isum);
        wsum = malloc(sizeof(int32_t) * (size_t)N); ime2c_row_sums(weight, N, K, wsum);
    }
    int32_t bias64[4][64] __attribute__((aligned(64)));
    int8_t qbuf[512] __attribute__((aligned(64)));
    int fuse_ok = vec_ok && ((uintptr_t)output & 7) == 0 && (N & 7) == 0;
    long lsh = output_shift < 0 ? -output_shift : 0, rsh = output_shift > 0 ? output_shift : 0;

    int seg_ok_a = (K & 7) == 0 && ((uintptr_t)input & 7) == 0;
    int seg_ok_b = (K & 7) == 0 && ((uintptr_t)weight & 7) == 0;
    for (int mt = 0; mt < m_tiles; mt++)
        for (int ks = 0; ks < k_slabs; ks++) {
            int8_t *d = pa + ((size_t)mt * k_slabs + ks) * 128;
            if (seg_ok_a && mt * 8 + 8 <= M && ks * 16 + 16 <= K)
                ime2c_pack_rows_seg(d, input + (size_t)mt * 8 * K + ks * 16, K, 8);
            else
                ime2c_pack_tile(d, input, mt * 8, M, ks * 16, K);
        }

    for (int n0 = 0; n0 < N; n0 += 32) {
        int np = (N - n0 + 7) / 8; if (np > 4) np = 4;          /* B panels in this group */
        const int8_t *pbg = wt ? wt + (size_t)(n0 / 32) * k_slabs * 512 : pb;   /* this group's B panels */
        if (!wt)
        for (int ks = 0; ks < k_slabs; ks++) {
            if (seg_ok_b && n0 + 32 <= N && ks * 16 + 16 <= K) {
                ime2c_pack_rows_seg(pb + (size_t)ks * 512, weight + (size_t)n0 * K + ks * 16, K, 32);
                continue;
            }
            for (int p = 0; p < np; p++)
                ime2c_pack_tile(pb + (size_t)ks * 512 + p * 128, weight, n0 + p * 8, N, ks * 16, K);
        }
        if (vec_ok)
            for (int p = 0; p < np; p++)
                for (int e = 0; e < 64; e++) {
                    int n = n0 + p * 8 + (e & 7);
                    bias64[p][e] = (bias && n < N) ? bias[n] : 0;
                }
        int mt = 0;
        if (np == 4)
            for (; mt + 1 < m_tiles; mt += 2) {
                if (fuse_ok && (mt + 2) * 8 <= M && n0 + 32 <= N) {
                    ime2c_mma_2x4_qb(pa + (size_t)mt * panel, pa + (size_t)(mt + 1) * panel, pbg, k_slabs,
                                 &bias64[0][0], qbuf, 0, output_multiplier,
                                 lsh, rsh, output_offset, activation_min, activation_max);
                    ime2c_scatter_block(qbuf, output + (size_t)mt * 8 * N + n0, (size_t)N);
                    continue;
                }
                ime2c_mma_2x4(pa + (size_t)mt * panel, pa + (size_t)(mt + 1) * panel, pbg, k_slabs, ct);
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 4; j++)
                        ime2c_linear_tail(ct + (i * 4 + j) * 64, (mt + i) * 8, n0 + j * 8, M, N, bias, asym, isum, wsum, K,
                                input_offset, filter_offset, vec_ok, bias64[j], output,
                                output_multiplier, output_shift, output_offset, activation_min, activation_max);
            }
        for (; mt < m_tiles; mt++)
            for (int p = 0; p < np; p++) {
                ime2c_mma_1x1_s(pa + (size_t)mt * panel, pbg + p * 128, k_slabs, 512, ct);
                ime2c_linear_tail(ct, mt * 8, n0 + p * 8, M, N, bias, asym, isum, wsum, K,
                        input_offset, filter_offset, vec_ok, bias64[p], output,
                        output_multiplier, output_shift, output_offset, activation_min, activation_max);
            }
    }
    free(pa); free(pb); free(ct); free(isum); free(wsum);
}

static __attribute__((noinline)) void ime2c_linear_tiled(const int8_t *input, const int8_t *wt, const int32_t *bias,
        int8_t *output, int M, int K, int N, int output_offset, int output_multiplier, int output_shift,
        int activation_min, int activation_max) {
    ime2c_linear_core(input, NULL, wt, bias, output, M, K, N, 0, 0, output_offset, output_multiplier, output_shift,
                      activation_min, activation_max);
}

void kernel_linear_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) return;
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 128) __builtin_trap();
    /* Weights packed at build time (MB_IME2_TW): the tiles are the B panels as
     * they stand. The tiny and asymmetric paths read weights one by one, so a
     * tiled weight reaching them is unpacked first (the build never does that). */
    const int8_t *wt = (long)N * K >= 16 ? ime2tw_tiles(weight, N, K) : NULL;
    int8_t *wplain = NULL;
    if (wt) {
        if ((long)M * N > 16 && input_offset == 0 && filter_offset == 0) {
            ime2c_linear_tiled(input, wt, bias, output, M, K, N, output_offset, output_multiplier, output_shift,
                               activation_min, activation_max);
            return;
        }
        weight = wplain = ime2tw_unpack(wt, N, K, 1);
    }

    if ((long)M * N <= 16) {                      /* e.g. dronet's 1x2048x1 heads */
        ime2c_tiny_linear(input, weight, bias, output, M, K, N, input_offset, filter_offset, output_offset,
                      output_multiplier, output_shift, activation_min, activation_max);
        free(wplain);
        return;
    }
    ime2c_linear_core(input, weight, NULL, bias, output, M, K, N, input_offset, filter_offset, output_offset,
                      output_multiplier, output_shift, activation_min, activation_max);
    free(wplain);
}

#pragma GCC pop_options
