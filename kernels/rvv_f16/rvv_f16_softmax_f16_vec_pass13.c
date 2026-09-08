/* source: curated */
/* algorithm: vec_pass13 */
/* origin: RVV+Zvfh row softmax. Two of the three passes vectorize EXACTLY;
 * the middle one is expf-bound and there is no vector exp in V+Zvfh, so it
 * stays scalar. Deliberately not a polynomial exp: that would trade a
 * measurable speedup for an approximation in the one op whose output feeds
 * every attention weight, and the two exact passes are worth having on
 * their own.
 *
 *   pass 1  row max     -> vfredmax over the fp16 row, then ONE scalar
 *                          multiply by input_scale. For input_scale >= 0 the
 *                          multiply is monotonic, so the max of the scaled
 *                          values IS the scaled max of the values -- the
 *                          same fp32 expression on the same element, hence
 *                          bit-exact. Seeded with -65504.0f to reproduce the
 *                          reference's initializer, which matters only when
 *                          input_scale > 1 pushes every value below it.
 *                          input_scale < 0 falls back to the scalar loop.
 *   pass 2  exp + sum   -> scalar expf, as the reference.
 *   pass 3  * 1/sum     -> vfwcvt + vfmul.vf + vfncvt over the fp16 values
 *                          pass 2 already stored, which is exactly what the
 *                          reference re-reads. Bit-exact.
 */

#include <math.h>
#include <riscv_vector.h>

void kernel_softmax_f16(const _Float16 *input, _Float16 *output,
                        int M, int K, float input_scale) {
    for (int m = 0; m < M; m++) {
        const _Float16 *row_in = input + (size_t)m * (size_t)K;
        _Float16 *row_out = output + (size_t)m * (size_t)K;

        /* --- pass 1: row max --- */
        float maxv;
        if (input_scale >= 0.0f) {
            vfloat16m1_t vseed = __riscv_vfmv_v_f_f16m1((_Float16)-65504.0f, 1);
            vfloat16m8_t vmax = __riscv_vfmv_v_f_f16m8(
                (_Float16)-65504.0f, __riscv_vsetvlmax_e16m8());
            int k = 0;
            while (k < K) {
                size_t vl = __riscv_vsetvl_e16m8((size_t)(K - k));
                vmax = __riscv_vfmax_vv_f16m8(
                    vmax, __riscv_vle16_v_f16m8(row_in + k, vl), vl);
                k += (int)vl;
            }
            _Float16 hmax = __riscv_vfmv_f_s_f16m1_f16(
                __riscv_vfredmax_vs_f16m8_f16m1(
                    vmax, vseed, __riscv_vsetvlmax_e16m8()));
            maxv = (float)hmax * input_scale;
            if (maxv < -65504.0f) maxv = -65504.0f;
        } else {
            maxv = -65504.0f;
            for (int k = 0; k < K; k++) {
                float v = (float)row_in[k] * input_scale;
                if (v > maxv) maxv = v;
            }
        }

        /* --- pass 2: exp + sum (scalar: no vector exp) --- */
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = (float)row_in[k] * input_scale;
            float e = expf(v - maxv);
            row_out[k] = (_Float16)e;
            sum += e;
        }

        /* --- pass 3: normalize --- */
        const float inv_sum = 1.0f / sum;
        int k = 0;
        while (k < K) {
            size_t vl = __riscv_vsetvl_e16m4((size_t)(K - k));
            vfloat16m4_t v = __riscv_vle16_v_f16m4(row_out + k, vl);
            vfloat32m8_t vf = __riscv_vfwcvt_f_f_v_f32m8(v, vl);
            vf = __riscv_vfmul_vf_f32m8(vf, inv_sum, vl);
            __riscv_vse16_v_f16m4(row_out + k,
                                  __riscv_vfncvt_f_f_w_f16m4(vf, vl), vl);
            k += (int)vl;
        }
    }
}
