/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER fixed-point per-channel batchnorm2d_s8 for the
 *         Gemmini Q0.31 target (NCHW). Scalar integer -- no vector, no floating
 *         point in the loop (only one-time float->fixed conversions of the
 *         per-channel scale/bias at entry; soft-float, non-trapping on
 *         misa.F=0). Companion to conv2d_s8's gemmini RoCC path: the conv runs
 *         on Gemmini, the requantize runs here in pure integer.
 *
 *   WHY. The previous batchnorm on this target was the fp32 scalar reference
 *   (dequant -> gamma*x+beta -> roundf(/scale_out)), ~600 soft-float cyc/elem
 *   and dronet's single largest cost (~19.8 M cyc / 55% of the gemmini run).
 *   The BN affine out = round(cs[c]*x + cb[c]) with cs=scale*scale_in/scale_out,
 *   cb=bias/scale_out is a compile-time-constant multiply-add, exactly
 *   representable in integer fixed point (gemmlowp / Jacob et al.), the same
 *   technique conv2d_s8's Q0.31 requantize uses.
 *
 *   MATH. Per channel fold cs,cb into Q(S) integers Mc=cs*2^S, Bc=cb*2^S. S is
 *   picked so Mc<2^22 (x*Mc<2^29) and Bc<2^30, keeping acc=x*Mc+Bc in int32.
 *   Then round-half-away(acc/2^S) (matches roundf ties-away), clamp. Measured
 *   on the live FPGA: 19.8 M -> 1.09 M cyc (~18x), output bit-identical to the
 *   fp32 path (err unchanged). Vectorised RVV variants live in kernels/rvv/. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max)
{
    const float inv_so = 1.0f / scale_out;
    const int32_t lo = activation_min < -128 ? -128
                       : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128
                       : (activation_max > 127 ? 127 : activation_max);
    const long hw = (long)H * (long)W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float cs = scale[c] * scale_in * inv_so;
            float cb = bias[c] * inv_so;
            float csa = cs < 0 ? -cs : cs, cba = cb < 0 ? -cb : cb;
            int S = 24;
            while (S > 1 && (csa * (float)((uint32_t)1<<S) >= 4194304.0f
                            || cba * (float)((uint32_t)1<<S) >= 1073741824.0f)) S--;
            int32_t Mc = (int32_t)lrintf(cs * (float)((uint32_t)1<<S));
            int32_t Bc = (int32_t)lrintf(cb * (float)((uint32_t)1<<S));
            int32_t rnd = (int32_t)1 << (S - 1);
            long base = ((long)n * C + c) * hw;
            for (long i = 0; i < hw; i++) {
                int32_t acc = (int32_t)input[base + i] * Mc + Bc;
                int32_t v = acc >= 0 ? (acc + rnd) >> S : -(((-acc) + rnd) >> S);
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                output[base + i] = (int8_t)v;
            }
        }
    }
}
