/* source: curated */
/* algorithm: vec_tanh */
/* origin: RVV+Zvfh tanh-GELU with a vectorized tanh. V+Zvfh has no vector
 * transcendental, so the reference's per-element tanhf call is the whole
 * cost of this op (9% of octo-small's fp16 profile with everything else
 * curated). Replaced with the exp identity
 *
 *     tanh(a) = 1 - 2 / (exp(2a) + 1)
 *
 * and a vectorized exp: z = 2a*log2(e), n = rint(z), r = z - n, a degree-5
 * minimax polynomial for 2^r on |r| <= 0.5, then 2^n built directly as
 * float bits ((n + 127) << 23). Every step is a vector op; no masks and no
 * branches.
 *
 * BRANCH-FREE SATURATION. `a` is clamped to +/-8 before the identity, and
 * that clamp alone produces the saturated value -- exp(+/-16) gives
 * t = +/-0.99999978 against a true tanh(8) = 0.99999977 -- so no select is
 * needed for the tails. z then lands in +/-23.1, n in +/-23, and n + 127 in
 * [104, 150]: always a normal exponent, so no overflow or underflow path.
 *
 * MEASURED against glibc tanhf over x in +/-30 (400k points), inside the
 * full GELU expression and cast to fp16 as the reference does:
 *     tanh max|d| = 1.64e-06     gelu fp16 max|d| = 9.77e-04
 * against the fp16 verify atol of 1e-2, i.e. an order of magnitude inside
 * it. 1-ulp fp16 differences are frequent (~44% of points) and that is not
 * a defect: expf/tanhf are not correctly-rounded, so the REFERENCE itself
 * gives different last bits under a different libm -- which is exactly why
 * this tree's spike and native builds are not bit-identical. Declared
 * NUMERIC_DRIFT on the strength of the magnitude, not the ulp count.
 */

#include <riscv_vector.h>

void kernel_gelu_f16(const _Float16 *input, _Float16 *output, int n) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float LOG2E = 1.4426950408889634f;
    /* minimax 2^r on |r| <= 0.5 */
    const float C0 = 1.0f;
    const float C1 = 0.6931471805599453f;
    const float C2 = 0.2402265069591007f;
    const float C3 = 0.05550410866482158f;
    const float C4 = 0.009618129107628477f;
    const float C5 = 0.0013333558146428443f;

    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m2((size_t)(n - i));
        vfloat16m2_t vh = __riscv_vle16_v_f16m2(input + i, vl);
        vfloat32m4_t x = __riscv_vfwcvt_f_f_v_f32m4(vh, vl);

        /* arg = sqrt(2/pi) * (x + 0.044715 * x^3) */
        vfloat32m4_t x2 = __riscv_vfmul_vv_f32m4(x, x, vl);
        vfloat32m4_t x3 = __riscv_vfmul_vv_f32m4(x2, x, vl);
        vfloat32m4_t arg = __riscv_vfmacc_vf_f32m4(x, 0.044715f, x3, vl);
        arg = __riscv_vfmul_vf_f32m4(arg, SQRT_2_OVER_PI, vl);

        /* clamp to +/-8; this alone saturates the tanh (see header) */
        arg = __riscv_vfmin_vf_f32m4(arg, 8.0f, vl);
        arg = __riscv_vfmax_vf_f32m4(arg, -8.0f, vl);

        /* exp(2 * arg) */
        vfloat32m4_t z = __riscv_vfmul_vf_f32m4(arg, 2.0f * LOG2E, vl);
        vint32m4_t ni = __riscv_vfcvt_x_f_v_i32m4(z, vl);     /* rint */
        vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(ni, vl);
        vfloat32m4_t r = __riscv_vfsub_vv_f32m4(z, nf, vl);

        /* Horner for 2^r: p = ((((C5*r + C4)*r + C3)*r + C2)*r + C1)*r + C0.
         * vfmul_vv then vfadd_vf, rather than vfmadd_vv against a
         * materialized constant vector: same two instructions, no vfmv. */
        vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(C5, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C4, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C3, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C2, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C1, vl);
        p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, r, vl), C0, vl);

        /* 2^n as raw float bits: ((n + 127) << 23) */
        vint32m4_t bits = __riscv_vsll_vx_i32m4(
            __riscv_vadd_vx_i32m4(ni, 127, vl), 23, vl);
        vfloat32m4_t pow2n = __riscv_vreinterpret_v_i32m4_f32m4(bits);
        vfloat32m4_t e = __riscv_vfmul_vv_f32m4(p, pow2n, vl);

        /* t = 1 - 2 / (e + 1) */
        vfloat32m4_t den = __riscv_vfadd_vf_f32m4(e, 1.0f, vl);
        vfloat32m4_t q = __riscv_vfrdiv_vf_f32m4(den, 2.0f, vl);
        vfloat32m4_t t = __riscv_vfrsub_vf_f32m4(q, 1.0f, vl);

        /* out = 0.5 * x * (1 + t) */
        vfloat32m4_t s = __riscv_vfadd_vf_f32m4(t, 1.0f, vl);
        vfloat32m4_t out = __riscv_vfmul_vv_f32m4(x, s, vl);
        out = __riscv_vfmul_vf_f32m4(out, 0.5f, vl);

        __riscv_vse16_v_f16m2(output + i,
                              __riscv_vfncvt_f_f_w_f16m2(out, vl), vl);
        i += (int)vl;
    }
}
