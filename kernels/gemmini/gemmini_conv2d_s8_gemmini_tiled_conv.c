/* source: curated */
/* algorithm: gemmini_tiled_conv */
/* origin: tiled_conv_auto — gemmini hardware im2col + GEMM + float-scale
 *         requantize in one call.  Validated on Saturn FireSim May 2026.
 *         Square kernels only (KH==KW, SH==SW, PH==PW); asymmetric offsets
 *         must be zero (symmetric per-tensor int8 from extract_int8).
 *         Float-scale introduces ≤1 LSB drift vs Q0.31 golden. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

/* 512 KB covers all square conv layers in dronet and yolov8_nano.
 *   dronet max weight: IC=128, KH=3, KW=3, OC=128 →  144 KB
 *   yolov8 max weight: IC=128, KH=3, KW=3, OC=256 →  288 KB
 *   yolov8 max input:  IC=3,   IH=160, IW=160      →   75 KB
 *   yolov8 max output: IC=16,  OH=80,  OW=80        →  100 KB
 * All three fit well within 512 KB. */
enum { GEMMINI_WS_BYTES = 512 * 1024 };

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    static elem_t ws_input  [GEMMINI_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_weight [GEMMINI_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output [GEMMINI_WS_BYTES] __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    /* Fall back to scalar for non-square kernels, non-zero offsets, tensors
     * that exceed the static workspace, or weight tiles that overflow the
     * gemmini scratchpad (B_rows = (OC/DIM)*KH*KW*IC > BANK_NUM*BANK_ROWS/2).
     * tiled_conv_stride_auto's reduction loop mis-behaves when B_rows alone
     * exceeds max_spad_rows, producing a degenerate tile and a null-PC trap. */
    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0
            || output_offset != 0
            || (size_t)(N * IH * IW * IC) > GEMMINI_WS_BYTES
            || (size_t)(OC * KH * KW * IC) > GEMMINI_WS_BYTES
            || (size_t)(N * OH * OW * OC)  > GEMMINI_WS_BYTES
            || (OC / DIM) * KH * KW * IC   > BANK_NUM * BANK_ROWS / 2) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        in_v = input_offset;
                                    else
                                        in_v = (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                                             + input_offset;
                                    acc += in_v * ((int32_t)weight[((oc*IC+ic)*KH+kh)*KW+kw]
                                                   + filter_offset);
                                }
                            }
                        }
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Transpose input NCHW → NHWC into ws_input. */
    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

    /* Transpose weights OIHW → patch-major [KH*KW*IC, OC] into ws_weight.
     * tiled_conv_auto reads this as a flattened [K*K*IC, OC] matrix —
     * rows are (kh, kw, ic) in patch order, columns are oc. */
    for (int oc = 0; oc < OC; oc++)
        for (int kh = 0; kh < KH; kh++)
            for (int kw = 0; kw < KW; kw++)
                for (int ic = 0; ic < IC; ic++)
                    ws_weight[((kh*KW + kw)*IC + ic)*OC + oc] =
                        weight[((oc*IC + ic)*KH + kh)*KW + kw];

    /* Q0.31 multiplier+shift → float scale for tiled_conv_auto's requantize.
     * effective_scale = output_multiplier * 2^(-(31 + output_shift)) */
    float scale = ldexpf((float)output_multiplier, -(31 + output_shift));

    /* RELU (1) clamps to [0, INT8_MAX]; NO_ACTIVATION (0) allows full range. */
    int act_kind = (activation_min == 0) ? 1 : 0;

    /* Drain CPU store buffer before gemmini mvin reads ws_input/ws_weight. */
    asm volatile("fence" ::: "memory");

    /* tiled_conv_auto: gemmini does im2col + GEMM + requantize in hardware.
     * WS = weight-stationary dataflow. */
    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, ws_weight, bias, ws_output,
        act_kind, scale,
        0, 0, 0,
        WS
    );

    /* tiled_conv_auto's body (tiled_conv) does NOT end with a
     * gemmini_fence — unlike tiled_matmul_outer_eigen.  Without this
     * explicit drain, the post-conv NHWC->NCHW read and the next op's
     * gemmini_flush race with in-flight mvout DMAs and corrupt memory
     * (FireSim Saturn: mcause=1, mepc=0). */
    gemmini_fence();
    gemmini_flush(0);

    /* Transpose output NHWC → NCHW. */
    for (int n = 0; n < N; n++)
        for (int c = 0; c < OC; c++)
            for (int h = 0; h < OH; h++)
                for (int w = 0; w < OW; w++)
                    output[((n*OC + c)*OH + h)*OW + w] =
                        ws_output[((n*OH + h)*OW + w)*OC + c];

    /* Post-clamp for activation_max < 127 (gemmini RELU only handles min==0). */
    if (activation_max < 127) {
        int n_out = N * OC * OH * OW;
        for (int i = 0; i < n_out; i++) {
            int v = output[i];
            if (v > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
