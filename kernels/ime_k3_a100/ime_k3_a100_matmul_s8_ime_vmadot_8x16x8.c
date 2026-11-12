/* source: curated */
/* algorithm: ime_vmadot_8x16x8 */
/* accuracy_class: bit_exact */
/* origin: hand-written for the SpaceMiT K3 A100 (VLEN 1024); companion of
 *   ime_k3_a100_linear_s8_ime_vmadot_8x16x8.c, read its header for the measured
 *   A100 facts (IME tile 8(M)x16(K)x8(N), `.insn r 0x2b,3,0x71, vd, A=vs1, B=vs2`,
 *   C += A.B^T, C int32 row-major over (vd,vd+1); loads are the bottleneck).
 *
 * Structure: per block of M rows, A is packed to [m-tile][k-slab][8 rows x 16 B];
 * per block of 32 output columns, B is packed to [k-slab][4 n-tiles][8 cols x 16 B]
 * (packed ONCE, reused by every m-tile), and a 2(m) x 4(n) tile micro-kernel keeps 8
 * accumulator pairs in flight (vmadot latency 4.5 ns, throughput 0.7 ns) with one
 * e8m4 load for the four B tiles and two e8m1 loads for the A tiles per slab.
 *
 * B packing:  transpose_b != 0 -> b[j][k] rows are already B-tile rows (16-byte vector
 *   copies). transpose_b == 0 -> b is [K][N]; the 16(k) x 32(n) block is transposed by
 *   loading 16 contiguous 32-byte rows (vle8) and writing them with two strided
 *   segment stores (vssseg8e8, stride 16): segment j holds the 8 k-values of column j,
 *   i.e. exactly bytes [j*16 .. j*16+8) of the B tile. No scalar byte loops, no vlse8.
 *
 * Requantize is vectorized and bit-exact vs the oracle's roundf((float)acc * total):
 * vfcvt.f.x, vfmul.vf by the same float `total`, then vfcvt.x.f with static rounding
 * mode RMM (round to nearest, ties AWAY from zero == roundf), clamp, truncating narrow.
 * (The K1 kernels used a scalar tail because RNE != roundf; RMM removes the reason.)
 */

/* n <= 128 byte copy / zero with one vector load+store (libc memcpy is a call + byte loop). */
/* Opaque zero: stops GCC hoisting a zero vector out of a loop and spilling it (vs1r.v) around inline asm. */
static volatile int ime_zero_src;

static inline void ime_mm_cpv(int8_t *d, const int8_t *s, size_t n) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    __riscv_vse8_v_i8m1(d, __riscv_vle8_v_i8m1(s, vl), vl);
}
static inline void ime_mm_zv(int8_t *d, size_t n) {
    size_t vl = __riscv_vsetvl_e8m1(n);
    __riscv_vse8_v_i8m1(d, __riscv_vmv_v_x_i8m1((int8_t)ime_zero_src, vl), vl);
}

static __attribute__((noinline)) void ime_mm_requant(const int32_t *acc, int8_t *out, int n, float total,
                           int activation_min, int activation_max) {
    while (n > 0) {
        size_t vl = __riscv_vsetvl_e32m1((size_t)n);
        vint32m1_t v = __riscv_vle32_v_i32m1(acc, vl);
        vfloat32m1_t f = __riscv_vfcvt_f_x_v_f32m1(v, vl);
        f = __riscv_vfmul_vf_f32m1(f, total, vl);
        v = __riscv_vfcvt_x_f_v_i32m1_rm(f, __RISCV_FRM_RMM, vl);
        v = __riscv_vmax_vx_i32m1(v, activation_min, vl);
        v = __riscv_vmin_vx_i32m1(v, activation_max, vl);
        vint16mf2_t h = __riscv_vncvt_x_x_w_i16mf2(v, vl);
        vint8mf4_t bq = __riscv_vncvt_x_x_w_i8mf4(h, vl);
        __riscv_vse8_v_i8mf4(out, bq, vl);
        acc += vl; out += vl; n -= (int)vl;
    }
}

/* ---- path "packed": 2(m) x 4(n) tiles over `slabs` k-slabs --------------- */
/* a0,a1: packed A tiles [slab][128]; bp: packed B [slab][4 tiles][128].
 * c: out, 8 pairs stored as [mi][t][64] ints. */
static void ime_mm_packed(const int8_t *a0, const int8_t *a1, const int8_t *bp, int slabs,
                           int32_t *c) {
    size_t n128 = 128, n512 = 512, n32 = 32;
    __asm__ volatile(
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t" "vmv.v.i v10, 0\n\t" "vmv.v.i v11, 0\n\t"
        "vmv.v.i v12, 0\n\t" "vmv.v.i v13, 0\n\t" "vmv.v.i v14, 0\n\t" "vmv.v.i v15, 0\n\t"
        "vmv.v.i v24, 0\n\t" "vmv.v.i v25, 0\n\t" "vmv.v.i v26, 0\n\t" "vmv.v.i v27, 0\n\t"
        "vmv.v.i v28, 0\n\t" "vmv.v.i v29, 0\n\t" "vmv.v.i v30, 0\n\t" "vmv.v.i v31, 0\n\t"
        "1:\n\t"
        "vsetvli t0, %[n512], e8, m4, ta, ma\n\t"
        "vle8.v v16, (%[bp])\n\t"
        "vsetvli t0, %[n128], e8, m1, ta, ma\n\t"
        "vle8.v v0, (%[a0])\n\t"
        "vle8.v v1, (%[a1])\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x16\n\t"
        ".insn r 0x2b, 3, 0x71, x10, x0, x17\n\t"
        ".insn r 0x2b, 3, 0x71, x12, x0, x18\n\t"
        ".insn r 0x2b, 3, 0x71, x14, x0, x19\n\t"
        ".insn r 0x2b, 3, 0x71, x24, x1, x16\n\t"
        ".insn r 0x2b, 3, 0x71, x26, x1, x17\n\t"
        ".insn r 0x2b, 3, 0x71, x28, x1, x18\n\t"
        ".insn r 0x2b, 3, 0x71, x30, x1, x19\n\t"
        "addi %[a0], %[a0], 128\n\t" "addi %[a1], %[a1], 128\n\t" "addi %[bp], %[bp], 512\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        "vsetvli t0, %[n32], e32, m1, ta, ma\n\t"
        "addi t1, %[c], 128\n\t"
        "vse32.v v8, (%[c])\n\t"      "vse32.v v9, (t1)\n\t"
        "addi t2, %[c], 256\n\t" "addi t1, %[c], 384\n\t"
        "vse32.v v10, (t2)\n\t"       "vse32.v v11, (t1)\n\t"
        "addi t2, %[c], 512\n\t" "addi t1, %[c], 640\n\t"
        "vse32.v v12, (t2)\n\t"       "vse32.v v13, (t1)\n\t"
        "addi t2, %[c], 768\n\t" "addi t1, %[c], 896\n\t"
        "vse32.v v14, (t2)\n\t"       "vse32.v v15, (t1)\n\t"
        "addi t2, %[c], 1024\n\t" "addi t1, %[c], 1152\n\t"
        "vse32.v v24, (t2)\n\t"       "vse32.v v25, (t1)\n\t"
        "addi t2, %[c], 1280\n\t" "addi t1, %[c], 1408\n\t"
        "vse32.v v26, (t2)\n\t"       "vse32.v v27, (t1)\n\t"
        "addi t2, %[c], 1536\n\t" "addi t1, %[c], 1664\n\t"
        "vse32.v v28, (t2)\n\t"       "vse32.v v29, (t1)\n\t"
        "addi t2, %[c], 1792\n\t" "addi t1, %[c], 1920\n\t"
        "vse32.v v30, (t2)\n\t"       "vse32.v v31, (t1)\n\t"
        : [a0] "+r"(a0), [a1] "+r"(a1), [bp] "+r"(bp), [ks] "+r"(slabs)
        : [c] "r"(c), [n128] "r"(n128), [n512] "r"(n512), [n32] "r"(n32)
        : "t0", "t1", "t2", "memory", "v0", "v1", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31");
}

/* Transpose one 16(k) x nvl(n) block of b ([K][N] row-major, row stride ldb) into
 * `nvl` consecutive 16-byte B-tile rows (dst[j*16 + k]). nvl <= 32. All 16 source rows
 * must be readable (the caller stages a partial last slab in a zero-padded buffer). */
static void ime_mm_transpose16(const int8_t *src, size_t ldb, size_t nvl, int8_t *dst) {
    size_t stride16 = 16;
    const int8_t *d2 = dst + 8;
    __asm__ volatile(
        "vsetvli t0, %[nvl], e8, m1, ta, ma\n\t"
        "vle8.v v8, (%[p])\n\t"  "add %[p], %[p], %[ld]\n\t"
        "vle8.v v9, (%[p])\n\t"  "add %[p], %[p], %[ld]\n\t"
        "vle8.v v10, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v11, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v12, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v13, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v14, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v15, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v16, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v17, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v18, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v19, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v20, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v21, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v22, (%[p])\n\t" "add %[p], %[p], %[ld]\n\t"
        "vle8.v v23, (%[p])\n\t"
        "vssseg8e8.v v8, (%[d1]), %[s]\n\t"
        "vssseg8e8.v v16, (%[d2]), %[s]\n\t"
        : [p] "+r"(src)
        : [nvl] "r"(nvl), [ld] "r"(ldb), [d1] "r"(dst), [d2] "r"(d2), [s] "r"(stride16)
        : "t0", "memory", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17",
          "v18", "v19", "v20", "v21", "v22", "v23");
}

#ifndef IME_MM_TB0_MIN_M
#define IME_MM_TB0_MIN_M 32   /* transpose_b == 0 below this M: the B transpose (~4 ns/byte of b) costs more than IME saves; measured crossover M~34 */
#endif

/* transpose_b == 0, small M: vectorize over N (b[k][:] is contiguous), 4 rows share every
 * b load, vector requant. Same arithmetic as rvv_k_reduce_n_lanes' second path. */
static void ime_mm_rvv_small_tb0(const int8_t *a, const int8_t *b, int8_t *output, int M, int K,
                                 int N, float total, int activation_min, int activation_max) {
    int32_t lane_acc[4 * 128] __attribute__((aligned(16)));
    for (int i0 = 0; i0 < M; i0 += 4) {
        int r = M - i0 < 4 ? M - i0 : 4;
        const int8_t *a0 = a + (size_t)i0 * K;
        const int8_t *a1 = a0 + (r > 1 ? K : 0);
        const int8_t *a2 = a0 + (r > 2 ? 2 * (size_t)K : 0);
        const int8_t *a3 = a0 + (r > 3 ? 3 * (size_t)K : 0);
        for (int j0 = 0; j0 < N;) {
            size_t vl = __riscv_vsetvl_e8m1((size_t)(N - j0));
            vint32m4_t c0 = __riscv_vmv_v_x_i32m4(ime_zero_src, vl), c1 = c0, c2 = c0, c3 = c0;
            const int8_t *bp = b + j0;
            for (int k = 0; k < K; k++, bp += N) {
                vint8m1_t vb = __riscv_vle8_v_i8m1(bp, vl);
                c0 = __riscv_vwadd_wv_i32m4(c0, __riscv_vwmul_vx_i16m2(vb, a0[k], vl), vl);
                if (r > 1) c1 = __riscv_vwadd_wv_i32m4(c1, __riscv_vwmul_vx_i16m2(vb, a1[k], vl), vl);
                if (r > 2) c2 = __riscv_vwadd_wv_i32m4(c2, __riscv_vwmul_vx_i16m2(vb, a2[k], vl), vl);
                if (r > 3) c3 = __riscv_vwadd_wv_i32m4(c3, __riscv_vwmul_vx_i16m2(vb, a3[k], vl), vl);
            }
            __riscv_vse32_v_i32m4(lane_acc, c0, vl);
            __riscv_vse32_v_i32m4(lane_acc + 128, c1, vl);
            __riscv_vse32_v_i32m4(lane_acc + 256, c2, vl);
            __riscv_vse32_v_i32m4(lane_acc + 384, c3, vl);
            for (int q = 0; q < r; q++)
                ime_mm_requant(lane_acc + q * 128, output + (size_t)(i0 + q) * N + j0, (int)vl, total,
                               activation_min, activation_max);
            j0 += (int)vl;
        }
    }
}

#ifndef IME_MM_TB1_MIN_M
#define IME_MM_TB1_MIN_M 3    /* transpose_b != 0 below this M: B packing (~1.3 ns/B of b) is not amortized */
#endif

/* transpose_b != 0, tiny M: dot products straight from b[j][:] (K-contiguous), vwmul +
 * vwredsum with 4 b rows interleaved, vector requant. No packing. */
static void ime_mm_rvv_small_tb1(const int8_t *input, const int8_t *weight, int8_t *output,
                                 int M, int K, int N, float total, int activation_min,
                                 int activation_max) {
    int32_t sums[32];
    for (int m = 0; m < M; m++) {
        const int8_t *x = input + (size_t)m * K;
        for (int n0 = 0; n0 < N; n0 += 32) {
            int nb = N - n0 < 32 ? N - n0 : 32;
            int j = 0;
            for (; j + 4 <= nb; j += 4) {
                const int8_t *w0 = weight + (size_t)(n0 + j) * K;
                const int8_t *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
                vint32m1_t a0 = __riscv_vmv_s_x_i32m1(ime_zero_src, 1), a1 = a0, a2 = a0, a3 = a0;
                for (int k = 0; k < K;) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                    vint8m1_t vx = __riscv_vle8_v_i8m1(x + k, vl);
                    a0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w0 + k, vl), vl), a0, vl);
                    a1 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w1 + k, vl), vl), a1, vl);
                    a2 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w2 + k, vl), vl), a2, vl);
                    a3 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(vx, __riscv_vle8_v_i8m1(w3 + k, vl), vl), a3, vl);
                    k += (int)vl;
                }
                sums[j] = __riscv_vmv_x_s_i32m1_i32(a0); sums[j + 1] = __riscv_vmv_x_s_i32m1_i32(a1);
                sums[j + 2] = __riscv_vmv_x_s_i32m1_i32(a2); sums[j + 3] = __riscv_vmv_x_s_i32m1_i32(a3);
            }
            for (; j < nb; j++) {
                const int8_t *w0 = weight + (size_t)(n0 + j) * K;
                vint32m1_t a0 = __riscv_vmv_s_x_i32m1(ime_zero_src, 1);
                for (int k = 0; k < K;) {
                    size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                    a0 = __riscv_vwredsum_vs_i16m2_i32m1(__riscv_vwmul_vv_i16m2(__riscv_vle8_v_i8m1(x + k, vl), __riscv_vle8_v_i8m1(w0 + k, vl), vl), a0, vl);
                    k += (int)vl;
                }
                sums[j] = __riscv_vmv_x_s_i32m1_i32(a0);
            }
            ime_mm_requant(sums, output + (size_t)m * N + n0, nb, total, activation_min, activation_max);
        }
    }
}

void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output, int M, int K, int N, float scale_a, float scale_b, float scale_out, int transpose_b, float scale_div, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) return;
    const float total = (scale_a * scale_b) / (scale_out * scale_div);
    if (K <= 0) {
        int32_t v = (int32_t)roundf(0.0f * total);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        for (long i = 0; i < (long)M * N; i++) output[i] = (int8_t)v;
        return;
    }

    if (transpose_b && M < IME_MM_TB1_MIN_M) {
        ime_mm_rvv_small_tb1(a, b, output, M, K, N, total, activation_min, activation_max);
        return;
    }
    if (!transpose_b && M < IME_MM_TB0_MIN_M) {
        ime_mm_rvv_small_tb0(a, b, output, M, K, N, total, activation_min, activation_max);
        return;
    }

    int slabs = (K + 15) >> 4;
    size_t slab_stride = 128;
    int mt_blk = (int)((96u * 1024u) / ((size_t)slabs * slab_stride));
    if (mt_blk < 2) mt_blk = 2;
    mt_blk &= ~1;
    int mtiles = (M + 7) >> 3;
    if (mt_blk > mtiles + (mtiles & 1)) mt_blk = mtiles + (mtiles & 1);
    int8_t *apk = (int8_t *)__builtin_alloca((size_t)mt_blk * slabs * slab_stride + 16);
    int8_t *bp = (int8_t *)__builtin_alloca((size_t)slabs * 512 + 16);
    int32_t *ct = (int32_t *)__builtin_alloca(2 * 4 * 64 * sizeof(int32_t) + 16);
    int8_t *obuf = (int8_t *)__builtin_alloca(64 + 16);
    int8_t *stage = (int8_t *)__builtin_alloca(16 * 32 + 16);
    apk = (int8_t *)(((uintptr_t)apk + 15) & ~(uintptr_t)15);
    bp = (int8_t *)(((uintptr_t)bp + 15) & ~(uintptr_t)15);
    ct = (int32_t *)(((uintptr_t)ct + 15) & ~(uintptr_t)15);

    for (int mt0 = 0; mt0 < mtiles; mt0 += mt_blk) {
        int nmt = mtiles - mt0 < mt_blk ? mtiles - mt0 : mt_blk;
        for (int mt = 0; mt < nmt; mt++) {
            for (int s = 0; s < slabs; s++) {
                int8_t *dst = apk + ((size_t)mt * slabs + s) * 128;
                int kk = s * 16;
                int take = K - kk < 16 ? K - kk : 16;
                for (int i = 0; i < 8; i++) {
                    int row = (mt0 + mt) * 8 + i;
                    if (row < M) {
                        ime_mm_cpv(dst + i * 16, a + (size_t)row * K + kk, (size_t)take);
                        if (take < 16) ime_mm_zv(dst + i * 16 + take, (size_t)(16 - take));
                    } else {
                        ime_mm_zv(dst + i * 16, 16);
                    }
                }
            }
        }
        for (int n0 = 0; n0 < N; n0 += 32) {
            int nb = N - n0 < 32 ? N - n0 : 32;
            if (transpose_b) {
                for (int s = 0; s < slabs; s++) {
                    int kk = s * 16;
                    int take = K - kk < 16 ? K - kk : 16;
                    for (int t = 0; t < 4; t++) {
                        int8_t *dst = bp + ((size_t)s * 4 + t) * 128;
                        for (int j = 0; j < 8; j++) {
                            int col = n0 + t * 8 + j;
                            if (col < N) {
                                ime_mm_cpv(dst + j * 16, b + (size_t)col * K + kk, (size_t)take);
                                if (take < 16) ime_mm_zv(dst + j * 16 + take, (size_t)(16 - take));
                            } else {
                                ime_mm_zv(dst + j * 16, 16);
                            }
                        }
                    }
                }
            } else {
                for (int s = 0; s < slabs; s++) {
                    int kk = s * 16;
                    int take = K - kk < 16 ? K - kk : 16;
                    int8_t *dst = bp + (size_t)s * 512;
                    if (take == 16) {
                        ime_mm_transpose16(b + (size_t)kk * N + n0, (size_t)N, (size_t)nb, dst);
                    } else {
                        for (int r = 0; r < take; r++)
                            ime_mm_cpv(stage + r * 32, b + (size_t)(kk + r) * N + n0, (size_t)nb);
                        ime_mm_transpose16(stage, 32, (size_t)nb, dst);
                    }
                }
            }
            int nt = (nb + 7) >> 3;
            for (int mt = 0; mt < nmt; mt += 2) {
                const int8_t *a0 = apk + (size_t)mt * slabs * 128;
                int two = mt + 1 < nmt;
                const int8_t *a1 = two ? a0 + (size_t)slabs * 128 : a0;
                ime_mm_packed(a0, a1, bp, slabs, ct);
                for (int u = 0; u < (two ? 2 : 1); u++) {
                    int m0 = (mt0 + mt + u) * 8;
                    int mr = M - m0 < 8 ? M - m0 : 8;
                    for (int t = 0; t < nt; t++) {
                        int nr = nb - t * 8 < 8 ? nb - t * 8 : 8;
                        ime_mm_requant(ct + (u * 4 + t) * 64, obuf, 64, total, activation_min, activation_max);
                        for (int i = 0; i < mr; i++)
                            ime_mm_cpv(output + (size_t)(m0 + i) * N + n0 + t * 8, obuf + i * 8, (size_t)nr);
                    }
                }
            }
        }
    }
}
