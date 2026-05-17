/* source: curated */
/* algorithm: outerprod */
/* origin: Saturn OPU outer-product i8 matmul. Ported from
 *   hw/chipyard/generators/saturn/benchmarks/opu-gemm/kernel.h
 *   (branch origin/opu-fp8), function `i8_mm_bme_sq`, by Miles Rusch.
 *
 * The upstream kernel operates on a TRANSPOSED left input `at` laid
 * out as [K, M] (row-major in K, then M). The OPU's VOPACC computes
 *   m_acc[r, c] += at[k*M + r] * b[k*N + c]
 * for each k, where r and c are the active lanes of the loaded i8
 * vectors. We preserve that core loop verbatim in `opu_i8_mm_tiled`
 * below.
 *
 * The agents-flow matmul_s8 signature accepts `a` in [M, K] row-major.
 * We convert to the upstream `at` layout by allocating a small stack
 * scratch buffer and transposing on entry (O(M*K) i8 reads/writes,
 * trivial relative to the K*M*N MAC). For shapes where transpose_b
 * is set, b's layout is [N, K] which we similarly transpose to [K, N]
 * before feeding the OPU.
 *
 * The requantize tail (i32 -> rescale * scale_a*scale_b/scale_out/scale_div
 * -> round -> clamp -> i8) is applied scalar-row at a time in the
 * drain loop; the heavy MAC work is OPU-resident.
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "saturn_opu.h"

/* Stack-allocated scratch caps. mlmax tile size is VLEN/8 (max 64 for
 * V512). Two transposed scratch buffers: at[K_TILE*M_TILE] and the
 * optional b_transposed[K_TILE*N_TILE]. Cap chosen for ViNT attention
 * (M=N=7, K=64..512) and FFN (M=7, K=512, N=512) — kernel falls back
 * via the runtime check if shapes exceed. */
#define OPU_MAX_TILE   64
#define OPU_MAX_K     1024

/* Core OPU MAC + drain — faithful port of upstream `i8_mm_bme_sq` with:
 *   - input bias path removed (matmul_s8 has no bias; m1 seeded to 0)
 *   - i32 output rows drained into the caller-provided c_out buffer
 *     directly (caller applies the requantize tail per row).
 * The tile coverage matches upstream: square mlmax×mlmax inner tiles
 * walked over (i, j), with a per-row narrow-N fallback. We only call
 * this when the FULL matrix fits in a single OPU tile — the agents
 * adapter does explicit shape checks first and bails to the scalar
 * fallback when M > mlmax || N > mlmax. */
static inline void opu_i8_mm_single_tile(int32_t *c_out,
                                         const int8_t *at,
                                         const int8_t *b,
                                         size_t M, size_t N, size_t K) {
    /* Seed m1 to zero. Same shape as the upstream OPMVINBCAST(m1, v0)
     * but v0 is "broadcast" of int32(0) — OPU initializes accumulator
     * with that vector. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(N));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m1, v0);

    /* MAC loop — verbatim from upstream `i8_sq_loop_k`. Two-way unroll
     * preserved; the OPU's pipelined sequencer keeps a 2-deep window of
     * outer-products in flight, so unrolling here is a perf hint to the
     * compiler / sequencer. */
    size_t k = 0;
    while (k + 2 <= K) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M));
        asm volatile("vle8.v v16, (%0)" : : "r"(&at[k * M]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b[k * N]));
        VOPACC(m1, v18, v16);

        k++;
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M));
        asm volatile("vle8.v v20, (%0)" : : "r"(&at[k * M]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N));
        asm volatile("vle8.v v22, (%0)" : : "r"(&b[k * N]));
        VOPACC(m1, v22, v20);

        k++;
    }
    if (k < K) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M));
        asm volatile("vle8.v v16, (%0)" : : "r"(&at[k * M]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b[k * N]));
        VOPACC(m1, v18, v16);
    }

    /* Drain m1 rows into c_out as i32. Caller applies requantize. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(N));
    for (size_t r = 0; r < M; r++) {
        VMV_VR(v0, r, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(&c_out[r * N]));
    }
}

/* Scalar reference fallback — copy of reference_kernels.py::MATMUL_S8's
 * reference_impl. Used when the OPU tile constraint isn't met or when
 * shape limits force fallback. Keeps the curated kernel self-contained
 * (no external link to the reference). */
static void matmul_s8_scalar_fallback(const int8_t *a, const int8_t *b,
                                      int8_t *output,
                                      int M, int K, int N,
                                      float scale_a, float scale_b,
                                      float scale_out,
                                      int transpose_b, float scale_div,
                                      int activation_min, int activation_max) {
    float total = (scale_a * scale_b) / (scale_out * scale_div);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t av = a[i * K + k];
                int8_t bv = transpose_b ? b[j * K + k] : b[k * N + j];
                acc += (int32_t)av * (int32_t)bv;
            }
            int32_t v = (int32_t)roundf((float)acc * total);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i * N + j] = (int8_t)v;
        }
    }
}

void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output,
                      int M, int K, int N,
                      float scale_a, float scale_b, float scale_out,
                      int transpose_b, float scale_div,
                      int activation_min, int activation_max) {
    /* Single-tile OPU requires the full M×N to fit in mlmax×mlmax,
     * where mlmax = vlmax_e8m1 = VLEN/8 is the runtime vector length
     * exposed by this Saturn bitstream / spike VLEN. Read it once
     * here; anything that doesn't fit falls back to the scalar
     * reference. Tiled OPU coverage for M>mlmax or N>mlmax is a
     * follow-up curation — the picker will use this kernel for
     * in-tile shapes (the dominant case in ViNT attention: M=N=7,
     * K=64..512 on V256+) and leave large gemms to scalar/RVV. */
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));
    if ((size_t)M > mlmax || (size_t)N > mlmax ||
        M > OPU_MAX_TILE || N > OPU_MAX_TILE || K > OPU_MAX_K) {
        matmul_s8_scalar_fallback(a, b, output, M, K, N,
                                  scale_a, scale_b, scale_out,
                                  transpose_b, scale_div,
                                  activation_min, activation_max);
        return;
    }

    /* Transpose a [M,K] -> at [K,M]. The OPU wants at[k,i] = a[i,k]. */
    int8_t at_buf[OPU_MAX_K * OPU_MAX_TILE];
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            at_buf[k * M + i] = a[i * K + k];
        }
    }

    /* For transpose_b=0, b is already [K,N] which is OPU-friendly.
     * For transpose_b=1, b is [N,K]; we transpose to [K,N]. */
    const int8_t *b_kn;
    int8_t b_buf[OPU_MAX_K * OPU_MAX_TILE];
    if (transpose_b) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++) {
                b_buf[k * N + j] = b[j * K + k];
            }
        }
        b_kn = b_buf;
    } else {
        b_kn = b;
    }

    /* OPU MAC into stack-allocated i32 output buffer. */
    int32_t c_i32[OPU_MAX_TILE * OPU_MAX_TILE];
    opu_i8_mm_single_tile(c_i32, at_buf, b_kn,
                          (size_t)M, (size_t)N, (size_t)K);

    /* Requantize tail — matches matmul_s8 reference: float scale, round
     * to nearest, clamp to activation_min..max, narrow to i8. */
    float total = (scale_a * scale_b) / (scale_out * scale_div);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = c_i32[i * N + j];
            int32_t v = (int32_t)roundf((float)acc * total);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i * N + j] = (int8_t)v;
        }
    }
}
