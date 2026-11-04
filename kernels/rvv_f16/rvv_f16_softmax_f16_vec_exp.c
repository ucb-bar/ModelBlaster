/* source: curated */
/* algorithm: vec_exp */
/* origin: RVV+Zvfh row softmax, all three passes vectorized.
 *
 *   pass 1  row max   vfredmax over the fp16 row, then ONE scalar multiply
 *                     by input_scale. For input_scale >= 0 that multiply is
 *                     monotonic, so the max of the scaled values IS the
 *                     scaled max of the values -- the same fp32 expression
 *                     on the same element, hence exact. Seeded with
 *                     -65504.0f to reproduce the reference's initializer,
 *                     which matters only when input_scale > 1 pushes every
 *                     value below it. Negative scale falls back to scalar.
 *   pass 2  exp+sum   vectorized exp (see below) with the fp32 results
 *                     accumulated in a vector and reduced at the end. The
 *                     reference sums the fp32 exp, not the fp16-rounded
 *                     store, and so does this.
 *   pass 3  1/sum     vfwcvt + vfmul.vf + vfncvt over the fp16 values pass 2
 *                     stored, which is what the reference re-reads. Exact.
 *
 * VECTOR EXP: z = a*log2(e), n = rint(z), r = z - n, degree-5 minimax
 * polynomial for 2^r on |r| <= 0.5, then 2^n built directly as float bits
 * ((n + 127) << 23). The argument is a - maxv, so it is always <= 0; it is
 * clamped at -80 so that n + 127 >= 12 and the exponent is always normal.
 * exp(-80) = 1.8e-35, which is zero in fp16 many times over, so the clamp
 * is invisible in the output and shifts the row sum by at most that much.
 *
 * WHY AN APPROXIMATE EXP IS RIGHT HERE, measured rather than argued. The
 * reference ALREADY rounds every exp to fp16 on the way out
 * (`output[k] = (_Float16)e`, ~5e-4 relative), so a 1.6e-6 fp32 exp is
 * three orders below error the reference itself introduces. Over 200 random
 * rows of K=690: fp16 output max|d| 7.6e-06, row-sum max rel err 6.2e-07,
 * and 0.04% of elements differing by one fp16 ulp -- against an fp16 verify
 * atol of 1e-2. Leaving pass 2 scalar cost 72.6 of 83.3 Mcycles on
 * octo-small's fp16 profile, i.e. it made softmax the largest remaining
 * scalar op by a wide margin, for an accuracy margin that was never in
 * doubt.
 *
 * The exp helper is duplicated from rvv_f16_gelu_f16_vec_tanh.c: each
 * curated kernel is its own translation unit, so there is nowhere shared to
 * put it.
 */

#include <math.h>
#include <riscv_vector.h>

/* exp(v) for v <= 0, vectorized. See the header for the range argument. */
static inline vfloat32m4_t _mb_vexp_f32m4(vfloat32m4_t v, size_t vl) {
    const float LOG2E = 1.4426950408889634f;
    const float C0 = 1.0f;
    const float C1 = 0.6931471805599453f;
    const float C2 = 0.2402265069591007f;
    const float C3 = 0.05550410866482158f;
    const float C4 = 0.009618129107628477f;
    const float C5 = 0.0013333558146428443f;

    v = __riscv_vfmax_vf_f32m4(v, -80.0f, vl);
    vfloat32m4_t z = __riscv_vfmul_vf_f32m4(v, LOG2E, vl);
    vint32m4_t ni = __riscv_vfcvt_x_f_v_i32m4(z, vl);        /* rint */
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(ni, vl);
    vfloat32m4_t r = __riscv_vfsub_vv_f32m4(z, nf, vl);

    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(C5, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C4, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C3, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C2, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C1, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C0, vl);

    vint32m4_t bits = __riscv_vsll_vx_i32m4(
        __riscv_vadd_vx_i32m4(ni, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(
        p, __riscv_vreinterpret_v_i32m4_f32m4(bits), vl);
}

void kernel_softmax_f16(const _Float16 *input, _Float16 *output,
                        int M, int K, float input_scale) {
    const size_t vlmax_e16m8 = __riscv_vsetvlmax_e16m8();
    const size_t vlmax_e32m4 = __riscv_vsetvlmax_e32m4();

    for (int m = 0; m < M; m++) {
        const _Float16 *row_in = input + (size_t)m * (size_t)K;
        _Float16 *row_out = output + (size_t)m * (size_t)K;

        /* --- pass 1: row max --- */
        float maxv;
        if (input_scale >= 0.0f) {
            vfloat16m1_t vseed = __riscv_vfmv_v_f_f16m1((_Float16)-65504.0f, 1);
            vfloat16m8_t vmax = __riscv_vfmv_v_f_f16m8((_Float16)-65504.0f,
                                                       vlmax_e16m8);
            int k = 0;
            while (k < K) {
                size_t vl = __riscv_vsetvl_e16m8((size_t)(K - k));
                vmax = __riscv_vfmax_vv_f16m8(
                    vmax, __riscv_vle16_v_f16m8(row_in + k, vl), vl);
                k += (int)vl;
            }
            _Float16 hmax = __riscv_vfmv_f_s_f16m1_f16(
                __riscv_vfredmax_vs_f16m8_f16m1(vmax, vseed, vlmax_e16m8));
            maxv = (float)hmax * input_scale;
            if (maxv < -65504.0f) maxv = -65504.0f;
        } else {
            maxv = -65504.0f;
            for (int k = 0; k < K; k++) {
                float v = (float)row_in[k] * input_scale;
                if (v > maxv) maxv = v;
            }
        }

        /* --- pass 2: exp + sum --- */
        vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vlmax_e32m4);
        int k = 0;
        while (k < K) {
            size_t vl = __riscv_vsetvl_e16m2((size_t)(K - k));
            vfloat16m2_t vh = __riscv_vle16_v_f16m2(row_in + k, vl);
            vfloat32m4_t v = __riscv_vfwcvt_f_f_v_f32m4(vh, vl);
            v = __riscv_vfmul_vf_f32m4(v, input_scale, vl);
            v = __riscv_vfsub_vf_f32m4(v, maxv, vl);
            vfloat32m4_t e = _mb_vexp_f32m4(v, vl);
            /* store the fp16-rounded exp, sum the fp32 one -- as the
             * reference does */
            __riscv_vse16_v_f16m2(row_out + k,
                                  __riscv_vfncvt_f_f_w_f16m2(e, vl), vl);
            vsum = __riscv_vfadd_vv_f32m4(vsum, e, vl);
            k += (int)vl;
        }
        vfloat32m1_t vz = __riscv_vfmv_v_f_f32m1(0.0f, 1);
        float sum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(vsum, vz, vlmax_e32m4));

        /* --- pass 3: normalize --- */
        const float inv_sum = 1.0f / sum;
        k = 0;
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
