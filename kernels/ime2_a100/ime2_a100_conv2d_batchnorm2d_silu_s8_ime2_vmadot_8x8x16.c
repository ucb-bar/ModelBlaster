/* source: curated */
/* algorithm: ime2_vmadot_8x8x16 */
/* accuracy_class: bit_exact */
/* origin: A100 conv2d_batchnorm2d_silu_s8, 2026-10-28. Replaces Kimi K2.5's first
 *   version (LLM flow, 2026-10-28). The conv part is conv2d_s8's (weights
 *   transposed into tiles, fast im2col, 2x4 core, Q0.31 tail fused, bias per
 *   channel). Batchnorm + silu are the reference's own float expressions, cached:
 *   silu has per-tensor scales, so it is ONE int8 -> int8 table per call (at most
 *   256 expf); batchnorm is per channel, so the final value is cached per
 *   (channel, conv int8). Bit-exact because each cache entry is the reference's
 *   function value. Measured on A100 cpu 8 over the 35 yolov8_nano fused shapes
 *   (k3-work/docker/ime2_opt/bench_fused.c): 1,934,974 ticks vs 4,933,941 for the
 *   LLM version and 2,080,081 for the RVV fused kernel; up to 2.4x faster than RVV
 *   on large layers, slower on 5x5 / 10x10 ones (per-call weight transposition).
 *   2026-11-02 (TASK-010): im2col reads a zero-padded copy of the input
 *   (ime2cv_padded), so no bounds checks or per-pixel divisions; 1x1 stride-1
 *   tiles take the 8-byte path across row ends. With build-time tiles the 35
 *   YOLOv8-nano shapes take 0.69x the ticks (0.51x RVV, faster on 32/35).
 *   Nonzero offsets use a scalar loop. A100 cores only (vlenb == 128). */
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

#ifndef MB_IME2_CONV_CORE
#define MB_IME2_CONV_CORE
static inline __attribute__((always_inline)) void ime2cv_mma_2x4_qb_rowbias(const int8_t *a0, const int8_t *a1, const int8_t *b, long slabs,
                               const int32_t *bias4, int8_t *buf, long ldo_unused,
                               long mult, long lsh, long rsh, long ooff, long amin, long amax)
{
    (void)ldo_unused;
    __asm__ volatile(
        "csrwi vxrm, 0\n\t"
        "vsetvli t0, zero, e32, m2, ta, ma\n\t"
        "vle32.v v24, (%[bi])\n\t"
        "addi t1, %[bi], 256\n\t" "vle32.v v26, (t1)\n\t"
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
        MB_Q_TILE("v10", "v24", "64")
        MB_Q_TILE("v12", "v24", "128")
        MB_Q_TILE("v14", "v24", "192")
        MB_Q_TILE("v16", "v26", "256")
        MB_Q_TILE("v18", "v26", "320")
        MB_Q_TILE("v20", "v26", "384")
        MB_Q_TILE("v22", "v26", "448")
#undef MB_Q_TILE
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs), [o] "+r"(buf)
        : [bi] "r"(bias4), [mu] "r"(mult), [ls] "r"(lsh), [rs] "r"(rsh),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax)
        : "t0", "t1", "memory", "vl", "vtype", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}


/* 8x8 byte transpose in 64-bit words: byte c of x[r] -> byte r of x[c]. */
static inline __attribute__((always_inline)) void ime2cv_tr8x8(uint64_t *x)
{
    uint64_t t;
    for (int r = 0; r < 8; r += 2) { t = ((x[r] >> 8) ^ x[r + 1]) & 0x00FF00FF00FF00FFull; x[r + 1] ^= t; x[r] ^= t << 8; }
    for (int r = 0; r < 8; r += (r & 1) ? 3 : 1) { t = ((x[r] >> 16) ^ x[r + 2]) & 0x0000FFFF0000FFFFull; x[r + 2] ^= t; x[r] ^= t << 16; }
    for (int r = 0; r < 4; r++) { t = ((x[r] >> 32) ^ x[r + 4]) & 0x00000000FFFFFFFFull; x[r + 4] ^= t; x[r] ^= t << 32; }
}

/* 8 channels x 16 taps of IHWOC weights (w = &weight[k0*OC + oc0], 8-aligned,
 * OC multiple of 8) -> tile rows = channels: two 8x8 byte transposes. */
static inline __attribute__((always_inline)) void ime2cv_pack_w_tile64(int8_t *dst, const int8_t *w, size_t OC)
{
    uint64_t a[8], b[8];
    for (int k = 0; k < 8; k++) { a[k] = *(const uint64_t *)(w + k * OC); b[k] = *(const uint64_t *)(w + (k + 8) * OC); }
    ime2cv_tr8x8(a); ime2cv_tr8x8(b);
    uint64_t *d = (uint64_t *)dst;
    for (int r = 0; r < 8; r++) { d[2 * r] = a[r]; d[2 * r + 1] = b[r]; }
}


/* Even bytes of 16 consecutive bytes at p (stride-2 pixels) packed into 8.
 * Reads p[0..14] only: p[15] is odd, never used, and on the input's last row
 * it can be one byte past the end of the buffer. The second load starts one
 * byte early and is shifted down, so it holds p[8..14] in bytes 0..6. */
static inline __attribute__((always_inline)) uint64_t ime2cv_even8(const int8_t *p)
{
    uint64_t lo, hi;
    lo = ime2c_ld8(p); hi = ime2c_ld8(p + 7) >> 8;
    lo &= 0x00FF00FF00FF00FFull; lo = (lo | (lo >> 8)) & 0x0000FFFF0000FFFFull; lo = (lo | (lo >> 16)) & 0xFFFFFFFFull;
    hi &= 0x00FF00FF00FF00FFull; hi = (hi | (hi >> 8)) & 0x0000FFFF0000FFFFull; hi = (hi | (hi >> 16)) & 0xFFFFFFFFull;
    return lo | (hi << 32);
}

/* Like ime2c_scatter_block but any alignment of out/ldo. */
static inline __attribute__((always_inline)) void ime2cv_scatter_any(const int8_t *buf, int8_t *out, size_t ldo)
{
    if ((((uintptr_t)out | ldo) & 7) == 0) { ime2c_scatter_block(buf, out, ldo); return; }
    for (int i = 0; i < 2; i++)
        for (int r = 0; r < 8; r++) {
            int8_t *d = out + (size_t)(8 * i + r) * ldo;
            const int8_t *src = buf + i * 256 + r * 8;
            for (int j = 0; j < 4; j++) ime2c_sd8(d + 8 * j, ime2c_ld8(src + 64 * j));
        }
}

/* Zero-padded copy of one input image (TASK-010): [IC][IH][IW] ->
 * [IC][IH+2PH][IW+2PW] with the zero border in place, plus 16 zero bytes of
 * slack, so every conv window lies inside it and im2col needs no bounds checks
 * (TASK-006 measured the per-element checks and divisions as most of the
 * A100 conv's tile building). Returns the input itself when there is no
 * padding, NULL if the buffer cannot grow. The buffer is per thread and only
 * grows (about 110 KB at most for YOLOv8-nano 160x160). */
static __thread int8_t *ime2cv_pad_buf;
static __thread size_t ime2cv_pad_cap;
static const int8_t *ime2cv_padded(const int8_t *in, int IC, int IH, int IW, int PH, int PW)
{
    if (PH == 0 && PW == 0) return in;
    size_t hp = (size_t)IH + 2 * PH, wp = (size_t)IW + 2 * PW;
    size_t need = ((size_t)IC * hp * wp + 16 + 63) / 64 * 64;
    if (need > ime2cv_pad_cap) {
        free(ime2cv_pad_buf);
        ime2cv_pad_buf = aligned_alloc(64, need);
        ime2cv_pad_cap = ime2cv_pad_buf ? need : 0;
        if (!ime2cv_pad_buf) return NULL;
    }
    int8_t *b = ime2cv_pad_buf;
    for (int ic = 0; ic < IC; ic++) {
        int8_t *c = b + (size_t)ic * hp * wp;
        memset(c, 0, (size_t)PH * wp);
        memset(c + ((size_t)PH + IH) * wp, 0, (size_t)PH * wp);
        for (int ih = 0; ih < IH; ih++) {
            int8_t *row = c + ((size_t)PH + ih) * wp;
            memset(row, 0, PW);
            memcpy(row + PW, in + ((size_t)ic * IH + ih) * IW, IW);
            memset(row + PW + IW, 0, PW);
        }
    }
    ime2c_sd8(b + (size_t)IC * hp * wp, 0); ime2c_sd8(b + (size_t)IC * hp * wp + 8, 0);
    return b;
}

#endif /* MB_IME2_CONV_CORE */


#define IME2CF_EMPTY 0x7fff

/* The reference's two float stages, verbatim, split: batchnorm depends on the
 * channel, silu only on its int8 input (its scales are per tensor). */
static __attribute__((noinline)) int8_t ime2cf_bn(int8_t conv_int8, float bn_s, float bn_b,
        float bn_scale_in, float bn_scale_out, int bn_amin, int bn_amax)
{
    float fv = (float)conv_int8 * bn_scale_in;
    float y = bn_s * fv + bn_b;
    int32_t bv = (int32_t)roundf(y / bn_scale_out);
    if (bv < bn_amin) bv = bn_amin;
    if (bv > bn_amax) bv = bn_amax;
    return (int8_t)bv;
}

static __attribute__((noinline)) int8_t ime2cf_silu(int8_t bn_int8, float silu_scale_in, float silu_scale_out,
        int silu_amin, int silu_amax)
{
    float fbv = (float)bn_int8 * silu_scale_in;
    float sy = fbv / (1.0f + expf(-fbv));
    int32_t v = (int32_t)roundf(sy / silu_scale_out);
    if (v < silu_amin) v = silu_amin;
    if (v > silu_amax) v = silu_amax;
    return (int8_t)v;
}

typedef struct { const float *bs, *bb; float bsi, bso, ssi, sso; int bmin, bmax, smin, smax; int16_t *memo; int16_t *silu; } ime2cf_post;

/* memo[oc][c] = final value for conv int8 c in channel oc; silu[b] = silu(b). Both lazy.
 * The tables can be shared by threads running the same layer (ime2cf_tables), so
 * entries are read and written as relaxed atomics (plain lh / sh); every writer
 * stores the same value, the reference's function of the entry's inputs. */
static inline __attribute__((always_inline)) int8_t ime2cf_map(const ime2cf_post *P, int oc, int8_t c)
{
    int16_t *e = P->memo + (size_t)oc * 256 + (uint8_t)(c + 128);
    int16_t v = __atomic_load_n(e, __ATOMIC_RELAXED);
    if (v == IME2CF_EMPTY) {
        int8_t b = ime2cf_bn(c, P->bs[oc], P->bb[oc], P->bsi, P->bso, P->bmin, P->bmax);
        int16_t *sv = P->silu + (uint8_t)(b + 128);
        v = __atomic_load_n(sv, __ATOMIC_RELAXED);
        if (v == IME2CF_EMPTY) {
            v = ime2cf_silu(b, P->ssi, P->sso, P->smin, P->smax);
            __atomic_store_n(sv, v, __ATOMIC_RELAXED);
        }
        __atomic_store_n(e, v, __ATOMIC_RELAXED);
    }
    return (int8_t)v;
}

/* BN+SiLU tables kept across calls (TASK-013). Rebuilding them on every call was
 * ~36% of this kernel on the A100 (TASK-012): each entry costs a roundf, and each
 * silu entry an expf. A slot holds one layer's tables, keyed by every parameter
 * the entries depend on: OC, the scales and limits, and a copy of the per-channel
 * bn_scale / bn_bias values, compared on every call (a few hundred cycles), so a
 * reused buffer with new values never sees old entries. Slots are claimed once
 * (0 free -> 1 claiming -> 2 ready) and never freed. A call that finds no slot (all
 * taken, another thread claiming this one, no memory) uses per-call tables as
 * before. The tables live until the process exits: OC * 512 + 512 bytes a layer. */
#define IME2CF_SLOTS 256
typedef struct {
    int state, OC;
    float bsi, bso, ssi, sso;
    int bmin, bmax, smin, smax;
    float *bs, *bb;
    int16_t *memo, *silu;
} ime2cf_slot;
static ime2cf_slot ime2cf_slots[IME2CF_SLOTS];

static int ime2cf_slot_is(const ime2cf_slot *S, const ime2cf_post *P, int OC)
{
    return S->OC == OC && S->bmin == P->bmin && S->bmax == P->bmax && S->smin == P->smin && S->smax == P->smax
        && !memcmp(&S->bsi, &P->bsi, sizeof(float)) && !memcmp(&S->bso, &P->bso, sizeof(float))
        && !memcmp(&S->ssi, &P->ssi, sizeof(float)) && !memcmp(&S->sso, &P->sso, sizeof(float))
        && !memcmp(S->bs, P->bs, (size_t)OC * sizeof(float)) && !memcmp(S->bb, P->bb, (size_t)OC * sizeof(float));
}

/* Points P->memo / P->silu at this layer's kept tables; 0 if there are none. */
static __attribute__((noinline)) int ime2cf_tables(ime2cf_post *P, int OC)
{
    size_t h = ((uintptr_t)P->bs >> 4) * 0x9E3779B97F4A7C15ull >> 56;
    for (int i = 0; i < IME2CF_SLOTS; i++) {
        ime2cf_slot *S = &ime2cf_slots[(h + (size_t)i) % IME2CF_SLOTS];
        int st = __atomic_load_n(&S->state, __ATOMIC_ACQUIRE);
        if (st == 2) {
            if (!ime2cf_slot_is(S, P, OC)) continue;
            P->memo = S->memo; P->silu = S->silu;
            return 1;
        }
        if (st == 1) return 0;
        int zero = 0;
        if (!__atomic_compare_exchange_n(&S->state, &zero, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) { i--; continue; }
        size_t n = (size_t)OC * 256 + 256;
        int16_t *memo = malloc(n * sizeof(int16_t));
        float *bs = malloc((size_t)OC * 2 * sizeof(float));
        if (!memo || !bs) {
            free(memo); free(bs);
            __atomic_store_n(&S->state, 0, __ATOMIC_RELEASE);
            return 0;
        }
        for (size_t e = 0; e < n; e++) memo[e] = IME2CF_EMPTY;
        memcpy(bs, P->bs, (size_t)OC * sizeof(float));
        memcpy(bs + OC, P->bb, (size_t)OC * sizeof(float));
        S->OC = OC; S->bsi = P->bsi; S->bso = P->bso; S->ssi = P->ssi; S->sso = P->sso;
        S->bmin = P->bmin; S->bmax = P->bmax; S->smin = P->smin; S->smax = P->smax;
        S->bs = bs; S->bb = bs + OC; S->memo = memo; S->silu = memo + (size_t)OC * 256;
        __atomic_store_n(&S->state, 2, __ATOMIC_RELEASE);
        P->memo = S->memo; P->silu = S->silu;
        return 1;
    }
    return 0;
}

static __attribute__((noinline)) void ime2cf_scalar(const int8_t *input, const int8_t *weight, const int32_t *bias,
        int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
        int ioff, int foff, int ooff, int mult, int shift, int amin, int amax, const ime2cf_post *P)
{
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++)
                        for (int kh = 0; kh < KH; kh++)
                            for (int kw = 0; kw < KW; kw++) {
                                int ih = oh * SH - PH + kh, iw = ow * SW - PW + kw;
                                int32_t x = (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? ioff
                                          : (int32_t)input[(((size_t)n * IC + ic) * IH + ih) * IW + iw] + ioff;
                                acc += x * ((int32_t)weight[(size_t)((ic * KH + kh) * KW + kw) * OC + oc] + foff);
                            }
                    output[(((size_t)n * OC + oc) * OH + oh) * OW + ow] =
                        ime2cf_map(P, oc, ime2c_requant(acc, mult, shift, ooff, amin, amax));
                }
}

/* The kernel body after the checks, once. Called with wt == NULL (plain weights,
 * packed per call: the code GCC generated before build-time tiles existed) and,
 * from a separate non-inlined function, with the tiles. Keeping the two apart
 * matters: one body handling both at run time made the plain path 2-3% slower. */
static inline __attribute__((always_inline)) void ime2cf_fused_core(const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int K, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max, const ime2cf_post *P) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW;
    int o_tiles = (OC + 7) / 8, k_slabs = (K + 15) / 16, k_full = K / 16;
    int IHp = IH + 2 * PH, IWp = IW + 2 * PW;   /* the padded image (ime2cv_padded) */
    size_t panel = (size_t)k_slabs * 128;
    int8_t pa_buf[wt ? 1 : (size_t)o_tiles * panel] __attribute__((aligned(64)));
    const int8_t *pa = wt ? wt : pa_buf;
    int8_t pb[4 * panel] __attribute__((aligned(64)));
    int32_t koff[k_slabs * 16];
    int32_t ct[64] __attribute__((aligned(64)));
    int32_t bias2[2][64] __attribute__((aligned(64)));
    int8_t qbuf[512] __attribute__((aligned(64)));
    long lsh = output_shift < 0 ? -output_shift : 0, rsh = output_shift > 0 ? output_shift : 0;
    int w64 = (OC & 7) == 0 && ((uintptr_t)weight & 7) == 0;
    /* 1x1, stride 1, no padding: output rows are contiguous in the input too, so any
     * 8 consecutive pixels are 8 consecutive bytes, even across a row end. */
    int lin = KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0;

    /* A: weights IHWOC weight[k*OC + oc] -> rows oc, 16 k per slab (already done at build time if tiled). */
    if (!wt)
    for (int ot = 0; ot < o_tiles; ot++)
        for (int ks = 0; ks < k_slabs; ks++) {
            int8_t *d = pa_buf + ((size_t)ot * k_slabs + ks) * 128;
            if (w64 && ot * 8 + 8 <= OC && ks * 16 + 16 <= K) {
                ime2cv_pack_w_tile64(d, weight + (size_t)ks * 16 * OC + ot * 8, (size_t)OC);
                continue;
            }
            for (int r = 0; r < 8; r++) {
                int oc = ot * 8 + r;
                for (int q = 0; q < 16; q++) {
                    int k = ks * 16 + q;
                    d[r * 16 + q] = (oc < OC && k < K) ? weight[(size_t)k * OC + oc] : 0;
                }
            }
        }
    /* im2col table: k -> input offset from the window's top-left in the padded image. */
    for (int k = 0; k < k_slabs * 16; k++) {
        if (k < K) {
            int ic = k / (KH * KW), rem = k % (KH * KW);
            koff[k] = ic * IHp * IWp + (rem / KW) * IWp + rem % KW;
        } else koff[k] = 0;
    }

    for (int n = 0; n < N; n++) {
        const int8_t *in = ime2cv_padded(input + (size_t)n * IC * IH * IW, IC, IH, IW, PH, PW);
        int8_t *out = output + (size_t)n * OC * M;
        if (!in) {   /* no memory for the padded copy: the exact scalar path */
            int8_t *wplain = wt ? ime2tw_unpack(wt, OC, K, 0) : NULL;
            if (wt && !wplain) __builtin_trap();
            ime2cf_scalar(input + (size_t)n * IC * IH * IW, wplain ? wplain : weight, bias, out, 1, IC, IH, IW, OC,
                          KH, KW, SH, SW, PH, PW, 0, 0, output_offset, output_multiplier, output_shift,
                          activation_min, activation_max, P);
            free(wplain);
            continue;
        }
        for (int m0 = 0; m0 < M; m0 += 32) {
            int np = (M - m0 + 7) / 8; if (np > 4) np = 4;
            /* B: 32 pixels x K, im2col on the fly; rows past M are zero. */
            for (int t8 = 0; t8 < 4; t8++) {
                int mt0 = m0 + t8 * 8;
                int8_t *tile0 = pb + t8 * 128;
                int oh = mt0 / OW, ow = mt0 % OW;
                /* fast: 8 pixels in one output row (any 8 if lin): 8-byte loads and transposes */
                if (mt0 + 8 <= M && (ow + 8 <= OW || lin) && (SW == 1 || SW == 2)) {
                    const int8_t *base = in + (size_t)oh * SH * IWp + (size_t)ow * SW;
                    for (int ks = 0; ks < k_slabs; ks++) {
                        uint64_t a[8], b[8];
                        for (int q = 0; q < 8; q++) {
                            int k0 = ks * 16 + q, k1 = k0 + 8;
                            if (SW == 1) {
                                a[q] = k0 < K ? ime2c_ld8(base + koff[k0]) : 0;
                                b[q] = k1 < K ? ime2c_ld8(base + koff[k1]) : 0;
                            } else {
                                a[q] = k0 < K ? ime2cv_even8(base + koff[k0]) : 0;
                                b[q] = k1 < K ? ime2cv_even8(base + koff[k1]) : 0;
                            }
                        }
                        ime2cv_tr8x8(a); ime2cv_tr8x8(b);
                        uint64_t *d = (uint64_t *)(tile0 + (size_t)ks * 512);
                        for (int r = 0; r < 8; r++) { d[2 * r] = a[r]; d[2 * r + 1] = b[r]; }
                    }
                    continue;
                }
                /* any other tile: every window is inside the padded image, so no checks;
                 * pixels walked without a division each */
                int poh = oh, pow_ = ow;
                for (int p = 0; p < 8; p++) {
                    int8_t *d = tile0 + p * 16;
                    if (mt0 + p >= M) { for (int ks = 0; ks < k_slabs; ks++) { ime2c_sd8(d + (size_t)ks * 512, 0); ime2c_sd8(d + (size_t)ks * 512 + 8, 0); } continue; }
                    const int8_t *pbase = in + (size_t)poh * SH * IWp + (size_t)pow_ * SW;
                    for (int ks = 0; ks < k_full; ks++) {
                        int8_t *dd = d + (size_t)ks * 512;
                        const int32_t *ko = koff + ks * 16;
                        for (int q = 0; q < 16; q++) dd[q] = pbase[ko[q]];
                    }
                    if (k_full < k_slabs) {
                        int8_t *dd = d + (size_t)k_full * 512;
                        for (int q = 0; q < 16; q++) {
                            int k = k_full * 16 + q;
                            dd[q] = k < K ? pbase[koff[k]] : 0;
                        }
                    }
                    if (++pow_ == OW) { pow_ = 0; poh++; }
                }
            }
            int ot = 0;
            if (np == 4 && m0 + 32 <= M)
                for (; ot + 1 < o_tiles && (ot + 2) * 8 <= OC; ot += 2) {
                    for (int i = 0; i < 2; i++)
                        for (int e = 0; e < 64; e++) bias2[i][e] = bias ? bias[(ot + i) * 8 + (e >> 3)] : 0;
                    ime2cv_mma_2x4_qb_rowbias(pa + (size_t)ot * panel, pa + (size_t)(ot + 1) * panel, pb, k_slabs,
                                              &bias2[0][0], qbuf, 0, output_multiplier, lsh, rsh,
                                              output_offset, activation_min, activation_max);
                    for (int i = 0; i < 2; i++)
                        for (int r = 0; r < 8; r++) {
                            int oc = (ot + i) * 8 + r;
                            int8_t *o = out + (size_t)oc * M + m0;
                            for (int j = 0; j < 4; j++)
                                for (int c = 0; c < 8; c++)
                                    o[8 * j + c] = ime2cf_map(P, oc, (int8_t)qbuf[(i * 4 + j) * 64 + r * 8 + c]);
                        }
                }
            for (; ot < o_tiles; ot++)
                for (int p = 0; p < np; p++) {
                    ime2c_mma_1x1_s(pa + (size_t)ot * panel, pb + p * 128, k_slabs, 512, ct);
                    for (int r = 0; r < 8 && ot * 8 + r < OC; r++)
                        for (int c = 0; c < 8 && m0 + p * 8 + c < M; c++) {
                            int oc = ot * 8 + r;
                            int32_t acc = ct[r * 8 + c] + (bias ? bias[oc] : 0);
                            out[(size_t)oc * M + m0 + p * 8 + c] =
                                ime2cf_map(P, oc, ime2c_requant(acc, output_multiplier, output_shift, output_offset,
                                                                 activation_min, activation_max));
                        }
                }
        }
    }
}

static __attribute__((noinline)) void ime2cf_fused_core_tiled(const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int K, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max, const ime2cf_post *P) {
    ime2cf_fused_core(input, weight, wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max, P);
}

void kernel_conv2d_batchnorm2d_silu_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, const float *bn_scale, const float *bn_bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int input_offset, int filter_offset, int conv_output_offset, int conv_output_multiplier, int conv_output_shift, int conv_activation_min, int conv_activation_max, float bn_scale_in, float bn_scale_out, int bn_activation_min, int bn_activation_max, float silu_scale_in, float silu_scale_out, int silu_activation_min, int silu_activation_max) {
    if (N <= 0 || OC <= 0) return;
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 128) __builtin_trap();
    ime2cf_post P = {bn_scale, bn_bias, bn_scale_in, bn_scale_out, silu_scale_in, silu_scale_out,
                     bn_activation_min, bn_activation_max, silu_activation_min, silu_activation_max, NULL, NULL};
    int kept = ime2cf_tables(&P, OC);
    int16_t memo[kept ? 1 : (size_t)OC * 256], silu[256];   /* per-call tables when none are kept */
    if (!kept) {
        for (size_t e = 0; e < (size_t)OC * 256; e++) memo[e] = IME2CF_EMPTY;
        for (int e = 0; e < 256; e++) silu[e] = IME2CF_EMPTY;
        P.memo = memo; P.silu = silu;
    }
    int output_multiplier = conv_output_multiplier, output_shift = conv_output_shift, output_offset = conv_output_offset;
    int activation_min = conv_activation_min, activation_max = conv_activation_max;
    int K = IC * KH * KW;
    /* Weights packed at build time (MB_IME2_TW) are the A operand as they stand. */
    const int8_t *wt = (long)OC * K >= 16 ? ime2tw_tiles(weight, OC, K) : NULL;
    if (input_offset != 0 || filter_offset != 0 || output_shift < -31 || output_shift > 31) {
        int8_t *wplain = wt ? ime2tw_unpack(wt, OC, K, 0) : NULL;
        ime2cf_scalar(input, wplain ? wplain : weight, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                      input_offset, filter_offset, output_offset, output_multiplier, output_shift,
                      activation_min, activation_max, &P);
        free(wplain);
        return;
    }
    if (wt) {
        ime2cf_fused_core_tiled(input, weight, wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max, &P);
        return;
    }
    ime2cf_fused_core(input, weight, NULL, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max, &P);
}

#pragma GCC pop_options
