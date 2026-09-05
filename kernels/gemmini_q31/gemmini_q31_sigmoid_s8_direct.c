/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER fixed-point sigmoid_s8 for the Gemmini target.
 *         Fixed-point logistic (Q16 exp2 poly, no expf); scalar, no floating
 *         point in the loop (only one-time float->fixed conversions of the two
 *         scalar scales at entry, soft-float, non-trapping on misa.F=0). See
 *         kernels/rvv/rvv_sigmoid_s8_rvv_memo_lut_gather.c for the math and the
 *         accuracy characterisation (<= 1 LSB vs the expf reference across
 *         scales; err=0 on dronet, whose scale_out saturates every code to
 *         127). dronet's sigmoid is n=1 at the output head. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max)
{
    if (n <= 0) return;
    const int32_t si_q16 = (int32_t)lrintf(scale_in * 65536.0f);
    const int32_t inv_so_q16 = (int32_t)lrintf((1.0f / scale_out) * 65536.0f);
    for (int i = 0; i < n; i++) {
        int32_t z = (int32_t)input[i] * si_q16;      /* x*scale_in, Q16 */
        int32_t a = z < 0 ? -z : z, e;
        if (a == 0) e = 65536;
        else {
            int64_t t = ((int64_t)a * 94548) >> 16;  /* a/ln2, Q16 */
            int32_t k = (int32_t)(t >> 16), f = (int32_t)(t & 0xFFFF);
            if (k >= 31) e = 0;
            else {
                int64_t f2 = ((int64_t)f * f) >> 16, f3 = (f2 * f) >> 16;
                int64_t p = 65536 - (45426LL*f>>16) + (15743LL*f2>>16) - (3638LL*f3>>16);
                if (k > 0) p = (p + ((int64_t)1 << (k-1))) >> k;
                if (p < 0) p = 0;
                if (p > 65536) p = 65536;
                e = (int32_t)p;
            }
        }
        int32_t sig = z >= 0 ? (int32_t)(((int64_t)65536*65536)/(65536+e))
                             : (int32_t)(((int64_t)e*65536)/(65536+e));
        int64_t prod = (int64_t)sig * inv_so_q16;
        int32_t v = (int32_t)((prod + ((int64_t)1 << 31)) >> 32);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}
