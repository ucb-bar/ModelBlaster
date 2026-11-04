/* source: curated */
/* algorithm: n_lanes */
/* origin: RVV+Zvfh batched fp16 GEMM vectorized over N (attention weights @ V
 * per head), fp32 accumulator via vfwmacc.vf.
 *
 * B is stored [batch, K, N], so the reference's inner k-loop strides B by N --
 * vectorizing THAT axis would need a strided load per element, and a strided
 * vlse16 over the reduction axis measured no faster than scalar on this unit
 * (see notes/curated_rvv_kernels.md). Vectorize over N instead: hold vl fp32
 * accumulators for a strip of B's row, and for each k broadcast the scalar
 * A[m,k] against a UNIT-STRIDE fp16 load of B[k, n:n+vl].
 *
 * Every load is contiguous, and each output element still accumulates over k
 * in the reference's own left-to-right order in fp32 -- so unlike the
 * K-reduction form this is BIT-EXACT to the reference, not merely
 * reduction-order equivalent.
 */

#include <riscv_vector.h>

void kernel_bmm_f16(const _Float16 *A, const _Float16 *B,
                    _Float16 *C, int batch, int M, int K, int N) {
    for (int b = 0; b < batch; b++) {
        const _Float16 *Ab = A + (size_t)b * (size_t)M * (size_t)K;
        const _Float16 *Bb = B + (size_t)b * (size_t)K * (size_t)N;
        _Float16 *Cb = C + (size_t)b * (size_t)M * (size_t)N;

        for (int m = 0; m < M; m++) {
            const _Float16 *a_row = Ab + (size_t)m * (size_t)K;
            _Float16 *c_row = Cb + (size_t)m * (size_t)N;

            int n = 0;
            while (n < N) {
                size_t vl = __riscv_vsetvl_e16m2((size_t)(N - n));
                /* fp32 accumulators for this strip of N. vfwmacc widens
                 * f16m2 -> f32m4, so the accumulator LMUL is m4. */
                vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);

                for (int k = 0; k < K; k++) {
                    /* Unit-stride load of B's k-th row, N-strip. */
                    vfloat16m2_t vb = __riscv_vle16_v_f16m2(
                        Bb + (size_t)k * (size_t)N + (size_t)n, vl);
                    /* vacc[i] += (float)a_row[k] * (float)vb[i] */
                    vacc = __riscv_vfwmacc_vf_f32m4(vacc, a_row[k], vb, vl);
                }

                /* Narrow the fp32 strip back to fp16 and store. */
                vfloat16m2_t vout = __riscv_vfncvt_f_f_w_f16m2(vacc, vl);
                __riscv_vse16_v_f16m2(c_row + n, vout, vl);
                n += (int)vl;
            }
        }
    }
}
