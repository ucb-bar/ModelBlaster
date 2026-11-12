/* source: curated */
/* algorithm: ime_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* K3 X100 linear_s8 (ime_x100, K3-only backend; Director 2026-10-29). The K1-compatible
 * smt.vmadot on the X100 with the A100 recipe: tile M x K x N = 4 x 8 x 4 (A, B = 32 bytes
 * = one register at e8/m1 vl=32, C = 4x4 int32 row-major across (vd, vd+1)); 2x4 blocking;
 * Q0.31 tail fused in registers; tiles packed with strided 8-byte vector loads; tiny,
 * asymmetric and out-of-range-shift cases on a scalar path. Measured 2026-10-28 on X100
 * cpu 0 (docker/ime2_opt/x100/x1_x100_opt.c): ffn_block fc1/fc2 4.8-6x the K1 kernel
 * (kernels/ime), 63 bit-exact bench cases. X100 only: traps unless vlenb == 32. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

static inline int8_t x1_requant(int32_t acc, int mult, int shift, int ooff, int amin, int amax)
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

/* rows [r0, r0+4) x cols [k0, k0+8) of row-major [R][K] -> 32-byte tile, zero outside */
static inline __attribute__((always_inline)) void x1_pack_tile(int8_t *dst, const int8_t *src, int r0, int R, int k0, int K)
{
    for (int i = 0; i < 4; i++) {
        int r = r0 + i;
        for (int q = 0; q < 8; q++) dst[i * 8 + q] = (r < R && k0 + q < K) ? src[(size_t)r * K + k0 + q] : 0;
    }
}

/* `rows` consecutive rows' 8 bytes at column k0 (8-aligned) -> rows*8 contiguous bytes. */
static inline __attribute__((always_inline)) void x1_pack_rows(int8_t *dst, const int8_t *src, long K, long rows)
{
    __asm__ volatile("vsetvli zero, %[n], e64, m4, ta, ma\n\t"
                     "vlse64.v v24, (%[s]), %[k]\n\t"
                     "vse64.v v24, (%[d])\n\t"
                     : : [n] "r"(rows), [s] "r"(src), [k] "r"(K), [d] "r"(dst)
                     : "memory", "vl", "vtype", "v24", "v25", "v26", "v27");
}

static inline __attribute__((always_inline)) void x1_mma_1x1(const int8_t *a, const int8_t *b, long slabs, long bstride, int32_t *out)
{
    __asm__ volatile(
        "vsetvli t0, zero, e8, m1, ta, ma\n\t"
        "vmv.v.i v8, 0\n\t" "vmv.v.i v9, 0\n\t"
        "1:\n\t"
        "vle8.v v0, (%[pa])\n\t"
        "vle8.v v4, (%[pb])\n\t"
        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
        "addi %[pa], %[pa], 32\n\t"
        "add %[pb], %[pb], %[bs]\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
        "vsetivli t0, 16, e32, m2, ta, ma\n\t"
        "vse32.v v8, (%[out])\n\t"
        : [pa] "+r"(a), [pb] "+r"(b), [ks] "+r"(slabs)
        : [out] "r"(out), [bs] "r"(bstride)
        : "t0", "memory", "vl", "vtype", "vxrm", "v0", "v4", "v8", "v9");
}

/* 2x4 block (8 rows x 16 cols), fused tail; tile (i,j) int8 -> buf + (i*4+j)*16 (4x4 row-major). */
static inline __attribute__((always_inline)) void x1_mma_2x4_qb(const int8_t *a0, const int8_t *a1, const int8_t *b, long slabs,
        const int32_t *bias4, int8_t *buf, long mult, long lsh, long rsh, long ooff, long amin, long amax)
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
        "addi %[a0], %[a0], 32\n\t"
        "addi %[a1], %[a1], 32\n\t"
        "addi %[b], %[b], 128\n\t"
        "addi %[ks], %[ks], -1\n\t"
        "bnez %[ks], 1b\n\t"
#define X1_Q(T, B, OFF)                                              \
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
        X1_Q("v8", "v24", "0")    X1_Q("v10", "v26", "16")  X1_Q("v12", "v28", "32")  X1_Q("v14", "v30", "48")
        X1_Q("v16", "v24", "64")  X1_Q("v18", "v26", "80")  X1_Q("v20", "v28", "96")  X1_Q("v22", "v30", "112")
#undef X1_Q
        : [a0] "+r"(a0), [a1] "+r"(a1), [b] "+r"(b), [ks] "+r"(slabs)
        : [bi] "r"(bias4), [o] "r"(buf), [mu] "r"(mult), [ls] "r"(lsh), [rs] "r"(rsh),
          [oo] "r"(ooff), [lo] "r"(amin), [hi] "r"(amax)
        : "t0", "t1", "memory", "vl", "vtype", "vxrm", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
          "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
}

static __attribute__((noinline)) void x1_tiny(const int8_t *input, const int8_t *weight, const int32_t *bias,
        int8_t *output, int M, int K, int N, int ioff, int foff, int ooff, int mult, int shift, int amin, int amax)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            int32_t acc = bias ? bias[n] : 0;
            const int8_t *x = input + (size_t)m * K, *w = weight + (size_t)n * K;
            for (int k = 0; k < K; k++) acc += ((int32_t)x[k] + ioff) * ((int32_t)w[k] + foff);
            output[(size_t)m * N + n] = x1_requant(acc, mult, shift, ooff, amin, amax);
        }
}

void kernel_linear_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) return;
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb != 32) __builtin_trap();                    /* X100 / K1 tile only */
    int asym = input_offset != 0 || filter_offset != 0;
    if ((long)M * N <= 16 || asym || output_shift < -31 || output_shift > 31) {
        x1_tiny(input, weight, bias, output, M, K, N, input_offset, filter_offset, output_offset,
                output_multiplier, output_shift, activation_min, activation_max);
        return;
    }
    int m_tiles = (M + 3) / 4, k_slabs = K > 0 ? (K + 7) / 8 : 1, full = K / 8;
    size_t panel = (size_t)k_slabs * 32;
    int8_t pa[(size_t)m_tiles * panel] __attribute__((aligned(64)));
    int8_t pb[4 * panel] __attribute__((aligned(64)));
    int32_t ct[16] __attribute__((aligned(64)));
    int32_t bias4[4][16] __attribute__((aligned(64)));
    int8_t qbuf[128] __attribute__((aligned(64)));
    long lsh = output_shift < 0 ? -output_shift : 0, rsh = output_shift > 0 ? output_shift : 0;
    int fast_a = (K & 7) == 0 && ((uintptr_t)input & 7) == 0, fast_b = (K & 7) == 0 && ((uintptr_t)weight & 7) == 0;

    for (int mt = 0; mt < m_tiles; mt++)
        for (int ks = 0; ks < k_slabs; ks++) {
            int8_t *d = pa + ((size_t)mt * k_slabs + ks) * 32;
            if (fast_a && mt * 4 + 4 <= M && ks < full) x1_pack_rows(d, input + (size_t)mt * 4 * K + ks * 8, K, 4);
            else x1_pack_tile(d, input, mt * 4, M, ks * 8, K);
        }
    for (int n0 = 0; n0 < N; n0 += 16) {
        int np = (N - n0 + 3) / 4; if (np > 4) np = 4;
        for (int ks = 0; ks < k_slabs; ks++) {
            if (fast_b && n0 + 16 <= N && ks < full) { x1_pack_rows(pb + (size_t)ks * 128, weight + (size_t)n0 * K + ks * 8, K, 16); continue; }
            for (int p = 0; p < np; p++) x1_pack_tile(pb + (size_t)ks * 128 + p * 32, weight, n0 + p * 4, N, ks * 8, K);
        }
        for (int p = 0; p < np; p++)
            for (int e = 0; e < 16; e++) { int n = n0 + p * 4 + (e & 3); bias4[p][e] = (bias && n < N) ? bias[n] : 0; }
        int mt = 0;
        if (np == 4 && n0 + 16 <= N)
            for (; mt + 1 < m_tiles && (mt + 2) * 4 <= M; mt += 2) {
                x1_mma_2x4_qb(pa + (size_t)mt * panel, pa + (size_t)(mt + 1) * panel, pb, k_slabs, &bias4[0][0], qbuf,
                              output_multiplier, lsh, rsh, output_offset, activation_min, activation_max);
                for (int i = 0; i < 2; i++)
                    for (int r = 0; r < 4; r++) {
                        int8_t *o = output + (size_t)((mt + i) * 4 + r) * N + n0;
                        for (int j = 0; j < 4; j++) memcpy(o + 4 * j, qbuf + (i * 4 + j) * 16 + r * 4, 4);
                    }
            }
        for (; mt < m_tiles; mt++)
            for (int p = 0; p < np; p++) {
                x1_mma_1x1(pa + (size_t)mt * panel, pb + p * 32, k_slabs, 128, ct);
                for (int r = 0; r < 4 && mt * 4 + r < M; r++)
                    for (int c = 0; c < 4 && n0 + p * 4 + c < N; c++) {
                        int n = n0 + p * 4 + c;
                        output[(size_t)(mt * 4 + r) * N + n] = x1_requant(ct[r * 4 + c] + (bias ? bias[n] : 0),
                            output_multiplier, output_shift, output_offset, activation_min, activation_max);
                    }
            }
    }
}
