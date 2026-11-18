/* source: curated */
/* algorithm: ime_vmadot_8x16x8 */
/* accuracy_class: bit_exact */
/* origin: hand-written for the SpaceMiT K3 A100 (IME smt.vmadot int8, 8x16x8
 *         tile). conv2d_s8_pc, the per-channel ("per-filter") twin of
 *         ime_k3_a100_conv2d_s8_ime_vmadot_8x16x8.c (k3-work TASK-019): the same
 *         implicit-GEMM conv with one multiplier and one shift per output channel in
 *         the requantize tail; see ime_k3_a100_conv_pc.h. A100 ONLY (CPUs 8-15);
 *         scalar fallback if vlenb != 128 or input_offset/filter_offset != 0. Written
 *         without the board; the board check is a byte compare against
 *         rvv_conv2d_s8_pc_direct.c (k3-work experiments/vint-pc-kernels-20261117). */
#include "ime_k3_a100_conv_pc.h"

void kernel_conv2d_s8_pc(
    const int8_t *input, const int8_t *weight, const int32_t *bias,
    int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int output_offset,
    const int32_t *output_multiplier, const int32_t *output_shift,
    int activation_min, int activation_max)
{
    mb_ime_conv_driver_pc(input, weight, bias, NULL, NULL, output,
        N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
        input_offset, filter_offset, output_offset,
        output_multiplier, output_shift, activation_min, activation_max,
        1.0f, 1.0f, -128, 127, 1.0f, 1.0f, -128, 127, 0);
}
