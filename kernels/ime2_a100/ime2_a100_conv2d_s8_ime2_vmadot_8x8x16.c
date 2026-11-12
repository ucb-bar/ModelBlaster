/* source: curated */
/* algorithm: ime2_vmadot_8x8x16 */
/* accuracy_class: bit_exact */
/* origin: A100 conv2d_s8, 2026-10-28. Replaces Kimi K2.5's first version (LLM
 *   flow, 2026-10-27). Conv as GEMM with the operands swapped: A = weights
 *   (rows = output channels, transposed from IHWOC with 8x8 byte transposes in
 *   64-bit words), B = im2col (rows = pixels; 8 pixels of one output row and
 *   one tap = one 8-byte load, even bytes for stride 2, then 8x8 transposes;
 *   edge tiles per byte), so result tile rows are NCHW output rows. The 2x4
 *   blocked core and fused Q0.31 tail are the linear_s8 kernel's (MB_IME2_CORE),
 *   with the bias per row (channel).
 *   Measured on A100 cpu 8, 13 dronet/yolov8_nano conv2d_s8 shapes
 *   (k3-work/docker/ime2_opt/bench_conv.c, bit-exact on all): total 177,618
 *   ticks vs 352,752 for the LLM version and 179,088 for the RVV conv; faster
 *   than RVV on 7 of 13 (up to 1.82x on yolov8 1x1 20x20), slower on small
 *   spatial sizes where transposing the weights every call dominates. With
 *   build-time packed weights (timing simulation) the total would be 135,081.
 *   2026-11-02 (TASK-010): im2col reads a zero-padded copy of the input
 *   (ime2cv_padded), so no bounds checks or per-pixel divisions; 1x1 stride-1
 *   tiles take the 8-byte path across row ends. With build-time tiles the 13
 *   bench shapes take 0.64x the ticks (0.52x RVV, faster than RVV on 12/13).
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

/* Edge tiles: scalar Q0.31, channel-major output. */
static inline void ime2cv_tail(const int32_t *t, int oc0, int m0, int OC, int M, const int32_t *bias,
                               int8_t *out, int mult, int shift, int ooff, int amin, int amax)
{
    for (int r = 0; r < 8 && oc0 + r < OC; r++)
        for (int c = 0; c < 8 && m0 + c < M; c++) {
            int32_t acc = t[r * 8 + c] + (bias ? bias[oc0 + r] : 0);
            out[(size_t)(oc0 + r) * M + m0 + c] = ime2c_requant(acc, mult, shift, ooff, amin, amax);
        }
}

static __attribute__((noinline)) void ime2cv_scalar_conv(const int8_t *input, const int8_t *weight, const int32_t *bias,
        int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW,
        int ioff, int foff, int ooff, int mult, int shift, int amin, int amax)
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
                    output[(((size_t)n * OC + oc) * OH + oh) * OW + ow] = ime2c_requant(acc, mult, shift, ooff, amin, amax);
                }
}

/* The kernel body after the checks, once. Called with wt == NULL (plain weights,
 * packed per call: the code GCC generated before build-time tiles existed) and,
 * from a separate non-inlined function, with the tiles. Keeping the two apart
 * matters: one body handling both at run time made the plain path 2-3% slower. */
static inline __attribute__((always_inline)) void ime2cv_conv_core(const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int K, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW;
    int o_tiles = (OC + 7) / 8, k_slabs = (K + 15) / 16, k_full = K / 16;
    int IHp = IH + 2 * PH, IWp = IW + 2 * PW;   /* the padded image (ime2cv_padded) */
    size_t panel = (size_t)k_slabs * 128;
    int8_t pa_buf[wt ? 1 : (size_t)o_tiles * panel] __attribute__((aligned(64)));   /* weights, [ot][ks][8][16] */
    const int8_t *pa = wt ? wt : pa_buf;
    int8_t pb[4 * panel] __attribute__((aligned(64)));                 /* 32 pixels, [ks][4][8][16] */
    int32_t koff[k_slabs * 16];
    int32_t ct[64] __attribute__((aligned(64)));
    int32_t bias2[2][64] __attribute__((aligned(64)));
    int8_t qbuf[512] __attribute__((aligned(64)));
    long lsh = output_shift < 0 ? -output_shift : 0, rsh = output_shift > 0 ? output_shift : 0;
    int fuse_ok = 1;
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
            ime2cv_scalar_conv(input + (size_t)n * IC * IH * IW, wplain ? wplain : weight, bias, out, 1, IC, IH, IW, OC,
                               KH, KW, SH, SW, PH, PW, 0, 0, output_offset, output_multiplier, output_shift,
                               activation_min, activation_max);
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
            if (np == 4 && fuse_ok && m0 + 32 <= M)
                for (; ot + 1 < o_tiles && (ot + 2) * 8 <= OC; ot += 2) {
                    for (int i = 0; i < 2; i++)
                        for (int e = 0; e < 64; e++) bias2[i][e] = bias ? bias[(ot + i) * 8 + (e >> 3)] : 0;
                    ime2cv_mma_2x4_qb_rowbias(pa + (size_t)ot * panel, pa + (size_t)(ot + 1) * panel, pb, k_slabs,
                                              &bias2[0][0], qbuf, 0, output_multiplier, lsh, rsh,
                                              output_offset, activation_min, activation_max);
                    ime2cv_scatter_any(qbuf, out + (size_t)ot * 8 * M + m0, (size_t)M);
                }
            for (; ot < o_tiles; ot++)
                for (int p = 0; p < np; p++) {
                    ime2c_mma_1x1_s(pa + (size_t)ot * panel, pb + p * 128, k_slabs, 512, ct);
                    ime2cv_tail(ct, ot * 8, m0 + p * 8, OC, M, bias, out, output_multiplier, output_shift,
                                output_offset, activation_min, activation_max);
                }
        }
    }
}

static __attribute__((noinline)) void ime2cv_conv_core_tiled(const int8_t *input, const int8_t *weight, const int8_t *wt, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int K, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    ime2cv_conv_core(input, weight, wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max);
}

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    if (N <= 0 || OC <= 0) return;
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 128) __builtin_trap();
    int K = IC * KH * KW;
    /* Weights packed at build time (MB_IME2_TW) are the A operand as they stand. */
    const int8_t *wt = (long)OC * K >= 16 ? ime2tw_tiles(weight, OC, K) : NULL;
    if (input_offset != 0 || filter_offset != 0 || output_shift < -31 || output_shift > 31) {
        int8_t *wplain = wt ? ime2tw_unpack(wt, OC, K, 0) : NULL;
        ime2cv_scalar_conv(input, wplain ? wplain : weight, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                           input_offset, filter_offset, output_offset, output_multiplier, output_shift,
                           activation_min, activation_max);
        free(wplain);
        return;
    }
    if (wt) {
        ime2cv_conv_core_tiled(input, weight, wt, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max);
        return;
    }
    ime2cv_conv_core(input, weight, NULL, bias, output, N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, K, output_offset, output_multiplier, output_shift, activation_min, activation_max);
}

#pragma GCC pop_options
