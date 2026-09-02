/* source: curated */
/* algorithm: gemmini_tiled_conv_bn_epilogue */
/* accuracy_class: numeric_drift */
/* origin: FAST fused Conv2d+BatchNorm(+ReLU) for gemmini_q31. Conv MAC +
 *         HW im2col + HW Q0.31 requantize on the systolic array via
 *         tiled_conv_auto; BN per-channel affine + activation clamp applied
 *         as a scalar epilogue during the NHWC->NCHW write-back. Keeps the
 *         MAC/im2col on HW (single-digit-M cycles) vs the ~40M-cycle CPU
 *         im2col of the im2col_full_C variant.
 *
 *         ACCURACY: conv uses Gemmini's SINGLE-STAGE Q0.31 mvout (<=1 LSB/
 *         layer drift vs two-stage golden) -> numeric_drift, gemmini_q31
 *         atol=128 envelope. HW im2col couples to HW requantize (conv API
 *         has no raw-int32/full_C output), so a bit-exact HW-im2col conv is
 *         not achievable without a Gemmini change; the bit-exact fallback is
 *         the sibling im2col_full_C_bn kernel.
 *
 *         WEIGHT LAYOUT CONTRACT: `weight` is pre-packed flat HWIO. */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

/* Workspace slot selector -- mirrors gemmini_conv2d_s8_gemmini_tiled_conv.c's
 * MB_GEM_WS_SLOT. Named distinctly because kernels.c concatenates every
 * selected kernel into one translation unit and identically-named macros or
 * enums would collide. */
#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_GEM_CBNT_WS_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_GEM_CBNT_WS_SLOT ((int)arch_proc_id())
#else
enum { MB_GEM_CBNT_WS_SLOTS = 1 };
#define MB_GEM_CBNT_WS_SLOT 0
#endif

enum { CBNT_WS_BYTES = 512 * 1024 };

void kernel_conv2d_batchnorm2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, const float *bn_scale, const float *bn_bias, int8_t *output,
    int N, int IC, int IH, int IW, int OC,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int input_offset, int filter_offset, int conv_output_offset,
    int conv_output_multiplier, int conv_output_shift,
    int conv_activation_min, int conv_activation_max,
    float bn_scale_in, float bn_scale_out,
    int bn_activation_min, int bn_activation_max)
{
    /* ONE WORKSPACE SLOT PER HART. These buffers hold THIS call's activation
     * data / partial sums, so two harts inside this function at once
     * overwrite each other's image. Same defect, same fix, as
     * gemmini_conv2d_s8_gemmini_im2col_full_C.c, where it was REACHED and
     * measured: yolov8_nano over a gemmini pair ran 14 concurrent conv pairs
     * and reported max_abs_err=89 against 0 on a single gemmini hart.
     *
     * NOT REACHED HERE, and not measurable from this campaign: no curated
     * pick in dronet / yolov8_nano / vint / mlp_control selects this kernel,
     * so it contributes no .bss to any binary measured for that fix and no
     * FPGA run in this campaign exercises it. It is corrected because the
     * defect class is proven, not because a run showed it. */
    static elem_t ws_input_all  [MB_GEM_CBNT_WS_SLOTS][CBNT_WS_BYTES] __attribute__((aligned(64)));
    static elem_t ws_output_all [MB_GEM_CBNT_WS_SLOTS][CBNT_WS_BYTES] __attribute__((aligned(64)));
    elem_t *const ws_input  = ws_input_all [MB_GEM_CBNT_WS_SLOT];
    elem_t *const ws_output = ws_output_all[MB_GEM_CBNT_WS_SLOT];

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0 || conv_output_offset != 0
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
            || conv_output_shift < 0 || conv_output_shift > 30
#endif
            || (size_t)(N * IH * IW * IC) > CBNT_WS_BYTES
            || (size_t)(N * OH * OW * OC) > CBNT_WS_BYTES) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                float bn_s = bn_scale[oc], bn_b = bn_bias[oc];
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++)
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        ? input_offset
                                        : (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw] + input_offset;
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc] + filter_offset);
                                }
                            }
                        int64_t prod = (int64_t)acc * (int64_t)conv_output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (conv_output_shift > 0)
                            scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (conv_output_shift-1))) >> conv_output_shift);
                        else if (conv_output_shift < 0)
                            scaled <<= (-conv_output_shift);
                        scaled += conv_output_offset;
                        if (scaled < conv_activation_min) scaled = conv_activation_min;
                        if (scaled > conv_activation_max) scaled = conv_activation_max;
                        float fv = (float)(int8_t)scaled * bn_scale_in;
                        float y = bn_s * fv + bn_b;
                        int32_t bv = (int32_t)roundf(y / bn_scale_out);
                        if (bv < bn_activation_min) bv = bn_activation_min;
                        if (bv > bn_activation_max) bv = bn_activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)bv;
                    }
                }
            }
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    int32_t scale_q31 = conv_output_shift == 0
        ? conv_output_multiplier
        : (int32_t)(((int64_t)conv_output_multiplier + ((int64_t)1 << (conv_output_shift - 1))) >> conv_output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;
#else
    float scale = ldexpf((float)conv_output_multiplier, -(31 + conv_output_shift));
#endif

    asm volatile("fence" ::: "memory");

    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        ws_input, weight, bias, ws_output,
        NO_ACTIVATION, scale,
        0, 0, 0,
        WS
    );

    gemmini_fence();
    gemmini_flush(0);

    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float bn_s = bn_scale[oc], bn_b = bn_bias[oc];
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int8_t conv_int8 = ws_output[((n*OH + oh)*OW + ow)*OC + oc];
                    float fv = (float)conv_int8 * bn_scale_in;
                    float y = bn_s * fv + bn_b;
                    int32_t bv = (int32_t)roundf(y / bn_scale_out);
                    if (bv < bn_activation_min) bv = bn_activation_min;
                    if (bv > bn_activation_max) bv = bn_activation_max;
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)bv;
                }
            }
        }
    }
}
