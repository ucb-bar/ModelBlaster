/* source: curated */
/* algorithm: gemmini_tiled_conv_pool_nhwc */
/* accuracy_class: numeric_drift */
/* act_layouts: nhwc */
/* NHWC fused conv2d + maxpool for the Gemmini targets.
 *
 * This is the NHWC counterpart of
 * gemmini_q31_rvv_conv2d_pool_s8_gemmini_tiled_conv_pool.c (the NCHW fused
 * kernel from the conv+maxpool fusion), and the pool-tail counterpart of
 * gemmini_q31_rvv_conv2d_s8_gemmini_tiled_conv_nhwc.c (the NHWC conv). It folds
 * conv0->maxpool1 into ONE gemmini call: tiled_conv_auto's native pool tail
 * (LoopConv has_max_pool) takes the running max over the SAME accumulator the
 * conv produced, so the full-resolution conv output is never materialised and
 * the standalone maxpool dispatch (plus its two boundary relayouts) disappears.
 *
 * Two wins stacked vs the plain NCHW path:
 *   - conv side: NHWC-native, so NO NCHW<->NHWC transpose on entry/exit
 *     (the ~87% of gemmini conv time that is layout conversion is gone).
 *   - pool side: HW mvout pool tail, so the ~630K standalone NHWC maxpool and
 *     the maxpool<->bn relayouts are gone.
 *
 * EXACTNESS: identical to running the NHWC conv then a maxpool -- max-pool never
 * rescales (commutes with the conv's monotonic round+clamp), same argument as
 * the NCHW conv2d_pool kernel. numeric_drift is inherited from the fast-conv
 * Q0.31 mvout, same class as conv2d_s8's tiled_conv_nhwc.
 *
 * CONSTRAINTS (fast path): square conv kernel/stride, symmetric pad, zero
 * offsets, foldable shift; pool square with zero pad and no dilation (gemmini's
 * pool params are scalar and its OOB pool fill is 0, so pool_P==0 keeps it
 * exact). Anything else -> scalar NHWC conv-then-pool fallback.
 *
 * ACTIVATION LAYOUT: input/output are [N,H,W,C]. Enforced by act_layouts=("nhwc",)
 * + the deny-by-default gate in generate_kernels.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_CPN_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_CPN_SLOT ((int)arch_proc_id())
#else
enum { MB_CPN_SLOTS = 1 };
#define MB_CPN_SLOT 0
#endif
enum { MB_CPN_BIAS_ELEMS = 4096, MB_CPN_TMP_BYTES = 512 * 1024 };

void kernel_conv2d_pool_s8(const int8_t *input, const int8_t *weight,
                           const int32_t *bias, int8_t *output,
                           int N, int IC, int IH, int IW, int OC,
                           int KH, int KW, int SH, int SW, int PH, int PW,
                           int input_offset, int filter_offset, int output_offset,
                           int output_multiplier, int output_shift,
                           int activation_min, int activation_max,
                           int pool_KH, int pool_KW, int pool_SH, int pool_SW,
                           int pool_PH, int pool_PW, int pool_DH, int pool_DW)
{
    const int OH = (IH + 2*PH - KH) / SH + 1;
    const int OW = (IW + 2*PW - KW) / SW + 1;
    const int OHp = (OH + 2*pool_PH - pool_DH*(pool_KH-1) - 1) / pool_SH + 1;
    const int OWp = (OW + 2*pool_PW - pool_DW*(pool_KW-1) - 1) / pool_SW + 1;

    const int fast =
        (KH == KW && SH == SW && PH == PW
         && input_offset == 0 && filter_offset == 0 && output_offset == 0
         && pool_KH == pool_KW && pool_SH == pool_SW
         && pool_PH == 0 && pool_PW == 0 && pool_DH == 1 && pool_DW == 1
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
         && output_shift >= 0 && output_shift <= 30
#endif
         && (size_t)(N * OH * OW * OC) <= MB_CPN_TMP_BYTES);

    if (!fast) {
        /* Scalar NHWC conv -> NHWC maxpool. Bit-identical to conv2d_s8's scalar
         * path followed by maxpool2d_s8's. Unreachable for dronet's conv0. */
        static int8_t tmp_all[MB_CPN_SLOTS][MB_CPN_TMP_BYTES];
        int8_t *const tmp = tmp_all[MB_CPN_SLOT];
        for (int n = 0; n < N; n++)
        for (int oh = 0; oh < OH; oh++)
        for (int ow = 0; ow < OW; ow++)
        for (int oc = 0; oc < OC; oc++) {
            int32_t acc = bias ? bias[oc] : 0;
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh * SH - PH + kh;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow * SW - PW + kw;
                    int oob = (ih < 0 || ih >= IH || iw < 0 || iw >= IW);
                    for (int ic = 0; ic < IC; ic++) {
                        int32_t in_v = oob ? 0 : input[
                            (((size_t)n*IH + ih)*IW + iw)*IC + ic];
                        in_v += input_offset;
                        acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                       + filter_offset);
                    }
                }
            }
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + ((int64_t)1 << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0)
                scaled = (int32_t)(((int64_t)scaled
                    + ((int64_t)1 << (output_shift - 1))) >> output_shift);
            else if (output_shift < 0)
                scaled <<= (-output_shift);
            scaled += output_offset;
            if (scaled < activation_min) scaled = activation_min;
            if (scaled > activation_max) scaled = activation_max;
            tmp[(((size_t)n*OH + oh)*OW + ow)*OC + oc] = (int8_t)scaled;
        }
        for (int n = 0; n < N; n++)
        for (int ohp = 0; ohp < OHp; ohp++)
        for (int owp = 0; owp < OWp; owp++)
        for (int oc = 0; oc < OC; oc++) {
            int8_t m = INT8_MIN;
            for (int kh = 0; kh < pool_KH; kh++) {
                int oh = ohp*pool_SH - pool_PH + kh*pool_DH;
                if (oh < 0 || oh >= OH) continue;
                for (int kw = 0; kw < pool_KW; kw++) {
                    int ow = owp*pool_SW - pool_PW + kw*pool_DW;
                    if (ow < 0 || ow >= OW) continue;
                    int8_t v = tmp[(((size_t)n*OH + oh)*OW + ow)*OC + oc];
                    if (v > m) m = v;
                }
            }
            output[(((size_t)n*OHp + ohp)*OWp + owp)*OC + oc] = m;
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    int32_t scale_q31 = output_shift == 0
        ? output_multiplier
        : (int32_t)(((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;
    static acc_t ws_bias_all[MB_CPN_SLOTS][MB_CPN_BIAS_ELEMS];
    acc_t *const ws_bias = ws_bias_all[MB_CPN_SLOT];
    const acc_t *bias_used;
    if (OC <= (int)MB_CPN_BIAS_ELEMS) {
        for (int oc = 0; oc < OC; oc++)
            ws_bias[oc] = (bias ? bias[oc] : 0) + 1;   /* beta=1, as the conv */
        bias_used = ws_bias;
    } else {
        bias_used = bias;
    }
#else
    float scale = ldexpf((float)output_multiplier, -(31 + output_shift));
    const acc_t *bias_used = bias;
#endif

    int act_kind = (activation_min == 0) ? 1 : 0;

    asm volatile("fence" ::: "memory");

    /* tiled_conv_auto with the pool tail (pool_size, pool_stride, pool_padding).
     * input/output are NHWC -> no transpose; output holds the POOLED
     * [N,OHp,OWp,OC] directly (the pre-pool tensor is never written). */
    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        input, weight, bias_used, output,
        act_kind, scale,
        pool_KH, pool_SH, pool_PH,
        WS
    );

    gemmini_fence();
    gemmini_flush(0);

    if (activation_max < 127) {
        size_t total = (size_t)N * OHp * OWp * OC;
        for (size_t i = 0; i < total; i++)
            if (output[i] > activation_max) output[i] = (int8_t)activation_max;
    }
}
