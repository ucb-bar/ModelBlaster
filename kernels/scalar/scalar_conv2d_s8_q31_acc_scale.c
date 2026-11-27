/* source: curated */
/* algorithm: q31_acc_scale */
/* accuracy_class: numeric_drift */
/* origin: hand-written. NCHW / OIHW scalar conv2d_s8 with the requantize a Gemmini build
 *         runs (MODELBLASTER_GEMMINI_Q31_ACC_SCALE, gemmini_q31_rvv_conv2d_s8_gemmini_tiled_conv*):
 *           sq  = output_shift == 0 ? M : (M + 2^(s-1)) >> s          (one folded Q0.31 scale)
 *           acc = (bias + 1) + sum x*w                                 (beta = 1 compensation)
 *           y   = sat8((acc*sq + 2^30) >> 31), then clamp [activation_min, activation_max]
 *         and the Gemmini kernel's own fallback (KH != KW, SH != SW, PH != PW, nonzero offsets, shift
 *         outside [0, 30]) to the reference two-stage requantize. Opt-in (pin_only): it exists so a
 *         CPU-only image computes the same tensors as such a Gemmini build. */
#include <stdint.h>
#include <stddef.h>

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias,
                      int8_t *output, int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    const int OH = (IH + 2 * PH - KH) / SH + 1, OW = (IW + 2 * PW - KW) / SW + 1;
    const int gem = !(KH != KW || SH != SW || PH != PW || input_offset != 0 || filter_offset != 0 ||
                      output_offset != 0 || output_shift < 0 || output_shift > 30);
    const int64_t sq = output_shift == 0 ? (int64_t)output_multiplier
        : (((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
            for (int oh = 0; oh < OH; oh++)
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++)
                        for (int kh = 0; kh < KH; kh++) {
                            const int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                const int iw = ow * SW - PW + kw;
                                const int32_t x = (ih < 0 || ih >= IH || iw < 0 || iw >= IW) ? input_offset
                                    : (int32_t)input[(((size_t)n * IC + ic) * IH + ih) * IW + iw] + input_offset;
                                acc += x * ((int32_t)weight[(((size_t)oc * IC + ic) * KH + kh) * KW + kw] + filter_offset);
                            }
                        }
                    int64_t v;
                    if (gem) {
                        v = ((int64_t)(acc + 1) * sq + ((int64_t)1 << 30)) >> 31;
                        if (v < -128) v = -128;
                        if (v > 127) v = 127;
                    } else {
                        int64_t prod = ((int64_t)acc * (int64_t)output_multiplier + (1LL << 30)) >> 31;
                        v = (int32_t)prod;
                        if (output_shift > 0) v = (v + ((int64_t)1 << (output_shift - 1))) >> output_shift;
                        else if (output_shift < 0) v = v << (-output_shift);
                        v += output_offset;
                    }
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    output[(((size_t)n * OC + oc) * OH + oh) * OW + ow] = (int8_t)v;
                }
}
