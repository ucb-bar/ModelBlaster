/* source: curated */
/* algorithm: scalar_rows_rmm */
/* accuracy_class: bit_exact */
/* origin: hand-written. The same window sum as the reference, with the
 *         four-term index expression replaced by a row pointer and the
 *         roundf call replaced by the fcvt rounding mode that means roundf.
 *
 *   WHY THIS FILE EXISTS. adaptive_avg_pool2d_s8's only AlgorithmCandidate
 *   (rvv_window_sum) is rvv-affined, so no Gemmini target had a file to
 *   probe for. Measured baseline, spike, vint, gemmini_q31: 19,855,989
 *   cycles over 17 dispatches and 1,446,528 INPUT elements = 13.7 cycles to
 *   load one int8 and add it to an accumulator.
 *
 *   WHERE THOSE 13.7 CYCLES GO. ViNT's pools are global (OH=OW=1), so there
 *   are 192 outputs for 258,048 inputs on the largest dispatch -- the
 *   requantize tail is nothing and the accumulate loop is everything. The
 *   reference recomputes `((n*C+c)*IH+ih)*IW+iw` -- three multiplies -- for
 *   every input element of that loop. Hoisting the plane base out of the
 *   (oh,ow) loops and the row base out of the iw loop leaves one add.
 *
 *   The roundf call is replaced by a single fcvt.w.s under the rmm rounding
 *   mode, which is exactly roundf's round-to-nearest-ties-away, and which
 *   saturates rather than being UB out of range (the clamp then gives the
 *   op's intended answer). It is worth little here -- one call per OUTPUT,
 *   and the outputs are few -- and is applied for consistency with the other
 *   int8 kernels rather than for the cycles.
 *
 *   BIT-EXACT: the window bounds, the int32 accumulate order, the
 *   `(float)acc * scale_in / (float)win` expression and the clamp are the
 *   reference's, unchanged. MB_DRIFT_ATOL must NOT be set for this op. */

#include <stddef.h>
#include <stdint.h>

#ifndef MB_SCALAR_RMM_
#define MB_SCALAR_RMM_
static inline int32_t mb_cvt_rmm(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#endif /* MB_SCALAR_RMM_ */

void kernel_adaptive_avg_pool2d_s8(const int8_t *input, int8_t *output,
                                   int N, int C, int IH, int IW,
                                   int OH, int OW,
                                   float scale_in, float scale_out,
                                   int activation_min, int activation_max)
{
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *ip = input + (size_t)(n*C + c) * IH * IW;
            int8_t *op = output + (size_t)(n*C + c) * OH * OW;
            for (int oh = 0; oh < OH; oh++) {
                int ih0 = (oh * IH) / OH;
                int ih1 = ((oh + 1) * IH + OH - 1) / OH;
                if (ih1 > IH) ih1 = IH;
                for (int ow = 0; ow < OW; ow++) {
                    int iw0 = (ow * IW) / OW;
                    int iw1 = ((ow + 1) * IW + OW - 1) / OW;
                    if (iw1 > IW) iw1 = IW;
                    int win = (ih1 - ih0) * (iw1 - iw0);
                    if (win <= 0) win = 1;
                    int32_t acc = 0;
                    for (int ih = ih0; ih < ih1; ih++) {
                        const int8_t *ir = ip + (size_t)ih * IW;
                        for (int iw = iw0; iw < iw1; iw++)
                            acc += (int32_t)ir[iw];
                    }
                    float mean = (float)acc * scale_in / (float)win;
                    int32_t v = mb_cvt_rmm(mean / scale_out);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    op[(size_t)oh * OW + ow] = (int8_t)v;
                }
            }
        }
    }
}
