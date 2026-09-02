/* source: curated */
/* algorithm: row_replicate */
/* accuracy_class: bit_exact */
/* origin: hand-written. upsample_nearest_s8 is pure data movement; the
 *         reference does an integer DIVIDE per output element to find the
 *         source index.
 *
 *   WHY THIS FILE EXISTS. upsample_nearest_s8 had no AlgorithmCandidate on
 *   any target. Measured baseline, spike, yolov8_nano, gemmini_q31:
 *   894,058 cycles over 2 dispatches -- 1.3% of the model AFTER
 *   gemmini_q31_silu_s8_scalar_lut.c landed, and the last op in yolov8_nano
 *   still running the scalar reference on this target. Small, and reported
 *   as small.
 *
 *   Two divisions (`oh / scale`, `ow / scale`) per output byte become none:
 *   the source row is a loop counter, each source pixel is written `scale`
 *   times in a run, and rows 1..scale-1 of each group are a memcpy of row 0.
 *
 *   BIT-EXACT: a gather with no arithmetic. MB_DRIFT_ATOL must NOT be set. */

#include <stdint.h>
#include <string.h>

void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                int N, int C, int IH, int IW, int scale)
{
    if (scale <= 0) return;
    int OH = IH * scale, OW = IW * scale;
    size_t OHW = (size_t)OH * (size_t)OW;
    size_t IHW = (size_t)IH * (size_t)IW;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *ip = input + (size_t)(n*C + c) * IHW;
            int8_t *op = output + (size_t)(n*C + c) * OHW;
            for (int ih = 0; ih < IH; ih++) {
                int8_t *row0 = op + (size_t)(ih * scale) * (size_t)OW;
                const int8_t *src = ip + (size_t)ih * (size_t)IW;
                int8_t *d = row0;
                for (int iw = 0; iw < IW; iw++) {
                    int8_t v = src[iw];
                    for (int s = 0; s < scale; s++) *d++ = v;
                }
                for (int r = 1; r < scale; r++)
                    memcpy(row0 + (size_t)r * (size_t)OW, row0, (size_t)OW);
            }
        }
    }
}
