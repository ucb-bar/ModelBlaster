/* source: curated */
/* algorithm: scalar_tap_ranges */
/* accuracy_class: bit_exact */
/* origin: hand-written. Same integer arithmetic as the reference, same
 *         accumulation order, with the per-tap BOUNDS TESTS turned into
 *         clamped loop ranges and the per-tap address math lifted out.
 *
 *   WHY THIS FILE EXISTS. depthwise_conv2d_s8's only AlgorithmCandidate
 *   (rvv_ow_lanes_taps) is rvv-affined, so no Gemmini target had a file to
 *   probe for. Measured baseline, spike, vint, gemmini_q31: 459,599,735
 *   cycles over 16 dispatches -- 5.87% of the model and the largest int8 op
 *   after the convolutions.
 *
 *   AND WHY GEMMINI ITSELF DOES NOT HELP. A depthwise conv has one filter
 *   per channel, so its "GEMM" is a batch of C independent 1-row matmuls;
 *   feeding a 16x16 systolic array one useful row at a time wastes 15/16 of
 *   it. The realistic target is a better scalar loop.
 *
 *   WHAT THE COST ACTUALLY IS -- AND A FIRST ATTEMPT THAT MADE IT WORSE.
 *   The first version of this file hoisted the channel's KH*KW taps into a
 *   `tap[]` scratch to save one filter_offset add per tap. It measured
 *   495,843,479 cycles: 7.9% SLOWER than the reference it replaced. The
 *   reason is in ViNT's shapes. Its depthwise layers are
 *   N=6,C=1152,IH=IW=2,KH=KW=5,PH=PW=2 and N=6,C=672,IH=4,IW=5,KH=KW=5 --
 *   tiny spatial extents with large kernels and full padding, so 21 of every
 *   25 taps are OUT OF BOUNDS. Precomputing all 25 taps per channel (6912
 *   channels x 25) cost more than the four real MACs per output it was
 *   meant to accelerate, and the `use_tap ?` ternary left in the inner loop
 *   blocked the compiler besides. Logged as a negative result before this
 *   rewrite; the tap scratch is GONE.
 *
 *   WHAT ACTUALLY PAYS is not touching the out-of-bounds taps at all. The
 *   valid kh range for an output row is [max(0,-ih0), min(KH, IH-ih0)) and
 *   depends only on oh; the kw range likewise depends only on ow. Clamping
 *   the loops to those ranges replaces 25 bounds tests per output with 4
 *   iterations, and lets the input row pointer and the weight row pointer
 *   be formed once per kh instead of five index multiplies per tap.
 *
 *   The visited taps and the order they are visited in are identical to the
 *   reference's -- clamping the bounds and `continue`-ing inside them
 *   enumerate the same set. The Q0.31 requantize tail is the reference's,
 *   copied unchanged.
 *
 *   BIT-EXACT: integer arithmetic throughout, accumulated in the same order
 *   over the same taps. MB_DRIFT_ATOL must NOT be set for this op. */

#include <stddef.h>
#include <stdint.h>

void kernel_depthwise_conv2d_s8(const int8_t *input, const int8_t *weight,
                                const int32_t *bias, int8_t *output,
                                int N, int C, int IH, int IW,
                                int KH, int KW, int SH, int SW, int PH, int PW,
                                int input_offset, int filter_offset,
                                int output_offset,
                                int output_multiplier, int output_shift,
                                int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_c = input + (size_t)(n*C + c) * IH * IW;
            int8_t *out_c = output + (size_t)(n*C + c) * OH * OW;
            const int8_t *w_c = weight + (size_t)c * KH * KW;
            int32_t bias_v = bias ? bias[c] : 0;

            for (int oh = 0; oh < OH; oh++) {
                int ih0 = oh * SH - PH;
                int kh_lo = ih0 < 0 ? -ih0 : 0;
                int kh_hi = IH - ih0; if (kh_hi > KH) kh_hi = KH;
                int8_t *orow = out_c + (size_t)oh * OW;
                for (int ow = 0; ow < OW; ow++) {
                    int iw0 = ow * SW - PW;
                    int kw_lo = iw0 < 0 ? -iw0 : 0;
                    int kw_hi = IW - iw0; if (kw_hi > KW) kw_hi = KW;
                    int32_t acc = bias_v;
                    for (int kh = kh_lo; kh < kh_hi; kh++) {
                        const int8_t *ir = in_c + (size_t)(ih0 + kh) * IW + iw0;
                        const int8_t *wr = w_c + (size_t)kh * KW;
                        for (int kw = kw_lo; kw < kw_hi; kw++) {
                            acc += ((int32_t)ir[kw] + input_offset)
                                 * ((int32_t)wr[kw] + filter_offset);
                        }
                    }
                    /* Requantize: the reference tail, unchanged. */
                    int64_t prod = ((int64_t)acc * (int64_t)output_multiplier
                                    + (1LL << 30)) >> 31;
                    int32_t v;
                    if (output_shift > 0) {
                        int32_t r = 1 << (output_shift - 1);
                        v = ((int32_t)prod + r) >> output_shift;
                    } else {
                        v = ((int32_t)prod) << (-output_shift);
                    }
                    v += output_offset;
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    orow[ow] = (int8_t)v;
                }
            }
        }
    }
}
