/* source: curated */
/* algorithm: k_reduce */
/* origin: RVV+Zvfh batched fp16 GEMM with B transposed (Q @ K.T per head),
 * fp32-accumulator dot product via vfwmacc.
 *
 * With B stored [batch, N, K], BOTH operands run contiguously along K, so
 * the K-reduction is the axis to vectorize -- this is
 * rvv_f16_linear_f16_widening one batch level up, with B's row index playing
 * the part of the output channel.
 *
 * The fp32 accumulator matches the reference impl's
 * `float acc = ...; C[...] = (_Float16)acc`, so the only divergence is
 * summation ORDER: the vector reduction is a pairwise tree, the reference is
 * left-to-right. Both stay in fp32 to the final cast, so the worst case is
 * ~1 ulp of fp16 -- declared NUMERIC_DRIFT, not BIT_EXACT.
 *
 * Why this op exists at all: without it a multi-head Q @ K.T has to
 * materialize the transpose, which on octo-small is 12 x 2.9 MB of copies
 * and buffers for data the kernel can index in place.
 */

#include <riscv_vector.h>

void kernel_bmm_tb_f16(const _Float16 *A, const _Float16 *B,
                       _Float16 *C, int batch, int M, int K, int N) {
    /* vlmax for the wide fp32 accumulator (LMUL=4); the matching narrow
     * fp16 LMUL is m2, since vfwmacc widens m2 -> m4. */
    const size_t vlmax_e32m4 = __riscv_vsetvlmax_e32m4();

    for (int b = 0; b < batch; b++) {
        const _Float16 *Ab = A + (size_t)b * (size_t)M * (size_t)K;
        const _Float16 *Bb = B + (size_t)b * (size_t)N * (size_t)K;
        _Float16 *Cb = C + (size_t)b * (size_t)M * (size_t)N;

        for (int m = 0; m < M; m++) {
            const _Float16 *a_row = Ab + (size_t)m * (size_t)K;

            for (int n = 0; n < N; n++) {
                const _Float16 *b_row = Bb + (size_t)n * (size_t)K;

                vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_e32m4);

                int k = 0;
                while (k < K) {
                    size_t vl = __riscv_vsetvl_e16m2((size_t)(K - k));
                    vfloat16m2_t va = __riscv_vle16_v_f16m2(a_row + k, vl);
                    vfloat16m2_t vb = __riscv_vle16_v_f16m2(b_row + k, vl);
                    /* vacc[i] += (float)va[i] * (float)vb[i] */
                    vacc = __riscv_vfwmacc_vv_f32m4(vacc, va, vb, vl);
                    k += (int)vl;
                }

                vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                vfloat32m1_t vred = __riscv_vfredusum_vs_f32m4_f32m1(
                    vacc, vzero, vlmax_e32m4);
                float acc = __riscv_vfmv_f_s_f32m1_f32(vred);

                Cb[(size_t)m * (size_t)N + (size_t)n] = (_Float16)acc;
            }
        }
    }
}
