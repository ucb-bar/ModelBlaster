/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER int8->int8 sigmoid, fully fixed-point. See
 *         rvv_sigmoid_s8_rvv_memo_lut_gather.c for the math and accuracy
 *         (<= 1 LSB vs expf reference; err=0 on dronet). This "direct" variant
 *         is the tiny-tensor form: no table, per-element fixed-point logistic,
 *         which is exactly what dronet's n=1 output head needs. No floating
 *         point beyond two one-time float->fixed conversions of the scalar
 *         quant scales at entry (soft-float, non-trapping on misa.F=0). */

#include <stdint.h>
#include <stddef.h>
#include <math.h>

/* e^{-a} in Q16 for a_q16 >= 0, result in (0, 65536]. Pure integer. */
#ifndef MB_EXP_NEG_Q16_
#define MB_EXP_NEG_Q16_
static inline int32_t mb_exp_neg_q16(int32_t a_q16) {
    if (a_q16 <= 0) return 65536;
    int64_t t = ((int64_t)a_q16 * 94548) >> 16;      /* a/ln2, Q16 (1/ln2=1.4427) */
    int32_t k = (int32_t)(t >> 16);
    int32_t f = (int32_t)(t & 0xFFFF);
    if (k >= 31) return 0;
    int64_t f2 = ((int64_t)f * f) >> 16;
    int64_t f3 = (f2 * f) >> 16;
    int64_t p = 65536
              - (45426LL * f  >> 16)
              + (15743LL * f2 >> 16)
              - ( 3638LL * f3 >> 16);
    if (k > 0) p = (p + ((int64_t)1 << (k - 1))) >> k;
    if (p < 0) p = 0;
    if (p > 65536) p = 65536;
    return (int32_t)p;
}
#endif

#ifndef MB_SIGMOID_S8_FX_
#define MB_SIGMOID_S8_FX_
static inline int8_t mb_sigmoid_s8_fx(int8_t x, int32_t si_q16, int32_t inv_so_q16,
                                      int activation_min, int activation_max) {
    int32_t z = (int32_t)x * si_q16;
    int32_t e, sig_q16;
    if (z >= 0) {
        e = mb_exp_neg_q16(z);
        sig_q16 = (int32_t)(((int64_t)65536 * 65536) / (65536 + e));
    } else {
        e = mb_exp_neg_q16(-z);
        sig_q16 = (int32_t)(((int64_t)e * 65536) / (65536 + e));
    }
    int64_t prod = (int64_t)sig_q16 * inv_so_q16;
    int32_t v = (int32_t)((prod + ((int64_t)1 << 31)) >> 32);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}
#endif

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max) {
    if (n <= 0) return;
    const int32_t si_q16 = (int32_t)lrintf(scale_in * 65536.0f);
    const int32_t inv_so_q16 = (int32_t)lrintf((1.0f / scale_out) * 65536.0f);
    for (int i = 0; i < n; i++)
        output[i] = mb_sigmoid_s8_fx(input[i], si_q16, inv_so_q16,
                                     activation_min, activation_max);
}
