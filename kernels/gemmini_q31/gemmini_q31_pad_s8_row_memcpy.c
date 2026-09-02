/* source: curated */
/* algorithm: row_memcpy */
/* accuracy_class: bit_exact */
/* origin: hand-written. pad_s8 is pure data movement, so the only question
 *         is how many cycles per byte moved -- and the reference answers
 *         16.8.
 *
 *   WHY THIS FILE EXISTS. pad_s8 had no AlgorithmCandidate on any target.
 *   Measured baseline, spike, vint, gemmini_q31: 22,800,506 cycles over 5
 *   dispatches and 1,358,208 output elements = 16.8 cycles per BYTE
 *   WRITTEN, because the reference tests `ih >= 0 && ih < IH && iw >= 0 &&
 *   iw < IW` and rebuilds two four-term index expressions for every single
 *   output byte, interior bytes included.
 *
 *   FIRST VERSION AND WHY IT ONLY GOT 1.43x. The obvious rewrite -- memset
 *   the whole output plane to the pad value, then memcpy IW bytes per input
 *   row into place -- measured 15,972,778 cycles, 11.8 cycles per byte. It
 *   WRITES THE INTERIOR TWICE (once as pad, once as data), and on this
 *   target that is not free: picolibc's memset/memcpy here cost several
 *   cycles per byte, so doubling the byte traffic is most of what was left.
 *
 *   THIS VERSION writes every output byte exactly once. A padding row is one
 *   memset of OW. A data row is memset(pad_left) + memcpy(IW) +
 *   memset(pad_right), and the two memsets are skipped entirely when the
 *   corresponding pad is zero -- which is the common case in ViNT, where
 *   four of the five dispatches pad only right and bottom.
 *
 *   BIT-EXACT: a permutation plus a constant fill. There is no arithmetic in
 *   this op, so anything other than max_abs_err=0 is a bug, and
 *   MB_DRIFT_ATOL must NOT be set for it -- the same argument the relayout
 *   kernels' headers make. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

void kernel_pad_s8(const int8_t *input, int8_t *output,
                   int N, int C, int IH, int IW,
                   int pad_left, int pad_right, int pad_top, int pad_bottom,
                   int pad_value)
{
    int OH = IH + pad_top + pad_bottom;
    int OW = IW + pad_left + pad_right;
    size_t OHW = (size_t)OH * (size_t)OW;
    size_t IHW = (size_t)IH * (size_t)IW;
    unsigned char pad_v = (unsigned char)(int8_t)pad_value;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            int8_t *op = output + (size_t)(n*C + c) * OHW;
            const int8_t *ip = input + (size_t)(n*C + c) * IHW;

            if (pad_top > 0)
                memset(op, pad_v, (size_t)pad_top * (size_t)OW);
            for (int h = 0; h < IH; h++) {
                int8_t *row = op + (size_t)(h + pad_top) * (size_t)OW;
                if (pad_left > 0)  memset(row, pad_v, (size_t)pad_left);
                if (IW > 0)        memcpy(row + pad_left,
                                          ip + (size_t)h * (size_t)IW,
                                          (size_t)IW);
                if (pad_right > 0) memset(row + pad_left + IW, pad_v,
                                          (size_t)pad_right);
            }
            if (pad_bottom > 0)
                memset(op + (size_t)(pad_top + IH) * (size_t)OW, pad_v,
                       (size_t)pad_bottom * (size_t)OW);
        }
    }
}
