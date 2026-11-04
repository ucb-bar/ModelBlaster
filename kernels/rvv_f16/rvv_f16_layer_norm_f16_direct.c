/* source: curated */
/* algorithm: direct */
/* origin: RVV+Zvfh fp16 LayerNorm, fp32-accumulator moments via vfwadd/vfwmacc.
 *
 * Every pass of the reference is vectorizable and there is no transcendental
 * in the element loop -- one sqrtf per ROW, which stays scalar. So this is a
 * pure structural win with no approximation anywhere:
 *
 *   pass 1  sum   += v          -> vfwadd.wv into an fp32 accumulator
 *           sqsum += v * v      -> vfwmacc.vv into a second one
 *   pass 2  the affine + store  -> widen, fma, narrow, store
 *
 * The reference accumulates mean/variance in fp32 and casts at the store, and
 * so does this, so the only divergence is reduction ORDER (pairwise tree vs
 * left-to-right) -- NUMERIC_DRIFT, not BIT_EXACT.
 *
 * gamma/beta are optional: a GroupNorm expressed as reshape + layer_norm +
 * per-channel affine passes NULL for both.
 */

#include <math.h>
#include <riscv_vector.h>

void kernel_layer_norm_f16(const _Float16 *input, const _Float16 *gamma,
                           const _Float16 *beta, _Float16 *output,
                           int M, int K, float eps) {
    const size_t vlmax_e32m4 = __riscv_vsetvlmax_e32m4();

    for (int m = 0; m < M; m++) {
        const _Float16 *row_in = input + (size_t)m * (size_t)K;
        _Float16 *row_out = output + (size_t)m * (size_t)K;

        /* Pass 1: sum and sum-of-squares, both fp32. */
        vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_e32m4);
        vfloat32m4_t vsq = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_e32m4);
        int k = 0;
        while (k < K) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(K - k));
            vfloat16m2_t v = __riscv_vle16_v_f16m2(row_in + k, vl);
            /* vfwadd.wv: fp32 accumulator += widened fp16 lane. */
            vsum = __riscv_vfwadd_wv_f32m4(vsum, v, vl);
            vsq = __riscv_vfwmacc_vv_f32m4(vsq, v, v, vl);
            k += (int)vl;
        }
        vfloat32m1_t vz = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        float sum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(vsum, vz, vlmax_e32m4));
        float sqsum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(vsq, vz, vlmax_e32m4));

        float mean = sum / (float)K;
        float var = sqsum / (float)K - mean * mean;
        float inv_sigma = 1.0f / sqrtf(var + eps);

        /* Pass 2: (v - mean) * inv_sigma * gamma + beta, fp32, store fp16. */
        k = 0;
        while (k < K) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(K - k));
            vfloat16m2_t v = __riscv_vle16_v_f16m2(row_in + k, vl);
            vfloat32m4_t vf = __riscv_vfwcvt_f_f_v_f32m4(v, vl);
            /* n = (v - mean) * inv_sigma */
            vf = __riscv_vfsub_vf_f32m4(vf, mean, vl);
            vf = __riscv_vfmul_vf_f32m4(vf, inv_sigma, vl);
            if (gamma) {
                vfloat16m2_t vg = __riscv_vle16_v_f16m2(gamma + k, vl);
                vf = __riscv_vfmul_vv_f32m4(
                    vf, __riscv_vfwcvt_f_f_v_f32m4(vg, vl), vl);
            }
            if (beta) {
                vfloat16m2_t vb = __riscv_vle16_v_f16m2(beta + k, vl);
                vf = __riscv_vfadd_vv_f32m4(
                    vf, __riscv_vfwcvt_f_f_v_f32m4(vb, vl), vl);
            }
            __riscv_vse16_v_f16m2(row_out + k,
                                  __riscv_vfncvt_f_f_w_f16m2(vf, vl), vl);
            k += (int)vl;
        }
    }
}
