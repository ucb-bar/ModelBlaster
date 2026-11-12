/* source: curated */
/* algorithm: ime_vmadot_8x16x8 */
/* accuracy_class: bit_exact */
/* origin: hand-written for the SpaceMiT K3 A100 (IME smt.vmadot int8, 8x16x8
 *         tile). Implicit-GEMM conv + BN; see ime_k3_a100_common.h for the
 *         design. A100 ONLY (CPUs 8-15); scalar fallback if vlenb != 128 or
 *         input_offset/filter_offset != 0. */
#include "ime_k3_a100_common.h"

void kernel_conv2d_batchnorm2d_s8(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max)
{
    mb_ime_conv_driver(input, weight, bias, bn_scale, bn_bias, output,
        N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
        input_offset, filter_offset, conv_output_offset,
        conv_output_multiplier, conv_output_shift,
        conv_activation_min, conv_activation_max,
        bn_scale_in, bn_scale_out, bn_activation_min, bn_activation_max,
        1.0f, 1.0f, -128, 127, 1);
}
