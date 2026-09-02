/* source: curated */
/* algorithm: gemmini_tiled_conv_nhwc */
/* accuracy_class: numeric_drift */
/* act_layouts: nhwc */
/* NHWC entry point for gemmini's tiled_conv_auto.
 *
 * This is gemmini_conv2d_s8_gemmini_tiled_conv.c with the two activation
 * transposes DELETED, and nothing else changed. It is not an optimisation of
 * the convolution -- the gemmini call below is byte-for-byte the same call --
 * it is the same convolution handed activations that are already in the layout
 * the accelerator wanted all along.
 *
 * WHY, in measured cycles (AWS F2, all-gemmini dronet int8, rdcycle phase
 * brackets in the NCHW kernel, fq job 475): input transpose 34.3%, gemmini
 * itself 12.8%, output transpose 52.9%. 87% of "gemmini conv time" was layout
 * conversion, and it was invisible -- welded inside the kernel, so the offline
 * scheduler charged it to the convolution and could neither move it nor split
 * it. See docs/IR_TENSOR_LAYOUT_DESIGN.md.
 *
 * WHAT THIS KERNEL ASSUMES, and what enforces it. `input` is
 * [N, IH, IW, IC] and `output` is [N, OH, OW, OC]. Nothing about the POINTERS
 * says so -- NCHW and NHWC of one tensor hold the same number of bytes -- so
 * handing this kernel an NCHW activation is not a crash or a size mismatch, it
 * is a plausible wrong answer. The declaration that keeps that from happening
 * is `act_layouts=("nhwc",)` on this algorithm in
 * pipeline/reference_kernels.py, and the deny-by-default gate
 * pipeline/generate_kernels.assert_act_layout_contract that refuses to select
 * it for a tensor that does not declare nhwc.
 *
 * WEIGHT LAYOUT CONTRACT: unchanged from the NCHW kernel. `weight` is already
 * flat HWIO ([KH*KW*IC, OC]), permuted at codegen time by
 * generate_skeleton._backend_pack_weight. Activation layout and weight layout
 * are independent axes: this file changes only the first.
 *
 * NOT CHANGED, DELIBERATELY: the gemmini_fence() / gemmini_flush(0) pair after
 * tiled_conv_auto. tiled_conv's body does not end with a fence, and without
 * this drain the next op's gemmini_flush races with in-flight mvout DMAs and
 * corrupts memory (FireSim Saturn: mcause=1, mepc=0). The NCHW kernel's comment
 * attached that fence to "the post-conv NHWC->NCHW read", and that read is gone
 * here -- but the race was never with the read, it was with the next
 * gemmini_flush, so the fence stays exactly where it was.
 *
 * NO WORKSPACES. ws_input / ws_output existed only to hold the transposed
 * copies. tiled_conv_auto now reads and writes the caller's buffers directly.
 * Those are buffers.c arrays, which carry no alignment attribute, where the
 * workspaces were __attribute__((aligned(64))) -- see the alignment note in
 * generate_skeleton's buffer emitter and MB_BUF_ALIGN64.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <gemmini.h>
#include <gemmini_params.h>

#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_GEMN_WS_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_GEMN_WS_SLOT ((int)arch_proc_id())
#else
enum { MB_GEMN_WS_SLOTS = 1 };
#define MB_GEMN_WS_SLOT 0
#endif
enum { MB_GEMN_BIAS_SLOT_ELEMS = 4096 };

/* Phase attribution, same CSV shape as the NCHW kernel so one parser reads
 * both. The two transpose columns are structurally zero here: that is the
 * point of the file, and printing them keeps an A/B diff honest rather than
 * making the NHWC arm look like a different measurement. */
#if defined(MB_GEM_PHASE_TRACE) && defined(CONFIG_PRINTK)
#include <zephyr/sys/printk.h>
static inline uint64_t mb_gemn_rdc(void)
{
    uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c;
}
#define MB_GEMN_PH(v) uint64_t v = mb_gemn_rdc()
#define MB_GEMN_PH_EMIT(IC,IH,IW,OC,a,b,c,d)                                  \
    printk("MB_GEMPHASE,%d,%d,%d,%d,%d,%llu,%llu,%llu\n",                     \
           MB_GEMN_WS_SLOT, (IC), (IH), (IW), (OC),                            \
           (unsigned long long)((b) - (a)), (unsigned long long)((c) - (b)),   \
           (unsigned long long)((d) - (c)))
#else
#define MB_GEMN_PH(v)            do { } while (0)
#define MB_GEMN_PH_EMIT(...)     do { } while (0)
#endif


void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    /* Same fallback predicate as the NCHW kernel minus the workspace-capacity
     * tests, which no longer exist to overflow. */
    if (KH != KW || SH != SW || PH != PW
            || input_offset != 0 || filter_offset != 0
            || output_offset != 0
#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
            || output_shift < 0 || output_shift > 30
#endif
       ) {
        /* Scalar reference, NHWC-indexed on both activation surfaces. The
         * arithmetic is copied verbatim from the NCHW kernel -- only the two
         * index expressions differ, which is the whole delta of this file. */
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    for (int oc = 0; oc < OC; oc++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int oob = (ih < 0 || ih >= IH
                                           || iw < 0 || iw >= IW);
                                for (int ic = 0; ic < IC; ic++) {
                                    int32_t in_v = oob ? 0 : input[
                                        (((size_t)n * IH + ih) * IW + iw) * IC + ic];
                                    in_v += input_offset;
                                    acc += in_v *
                                        ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
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
                        output[(((size_t)n * OH + oh) * OW + ow) * OC + oc] =
                            (int8_t)scaled;
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

    MB_GEMN_PH(mb_ph_t0);
    /* (input transpose: deleted -- `input` is already NHWC) */
    MB_GEMN_PH(mb_ph_t1);

#ifdef MODELBLASTER_GEMMINI_Q31_ACC_SCALE
    int32_t scale_q31 = output_shift == 0
        ? output_multiplier
        : (int32_t)(((int64_t)output_multiplier + ((int64_t)1 << (output_shift - 1))) >> output_shift);
    acc_scale_t scale = (acc_scale_t)scale_q31;

    /* beta=1 rounding-bias compensation, identical to the NCHW kernel. Per-hart
     * because two Gemmini-bearing harts can be inside this function at once. */
    static acc_t ws_bias_all[MB_GEMN_WS_SLOTS][MB_GEMN_BIAS_SLOT_ELEMS];
    acc_t *const ws_bias = ws_bias_all[MB_GEMN_WS_SLOT];
    const acc_t *bias_used;
    if (OC <= (int)MB_GEMN_BIAS_SLOT_ELEMS) {
        for (int oc = 0; oc < OC; oc++)
            ws_bias[oc] = (bias ? bias[oc] : 0) + 1;
        bias_used = ws_bias;
    } else {
        bias_used = bias;
    }
#else
    float scale = ldexpf((float)output_multiplier, -(31 + output_shift));
    const acc_t *bias_used = bias;
#endif

    /* RELU (1) clamps to [0, INT8_MAX]; NO_ACTIVATION (0) allows full range. */
    int act_kind = (activation_min == 0) ? 1 : 0;

    /* Drain CPU store buffer before gemmini mvin reads the input buffer. */
    asm volatile("fence" ::: "memory");

    tiled_conv_auto(
        N, IH, IW, IC,
        OC, OH, OW,
        SH, 1, 1, PH, KH,
        false, false, false, false, false,
        input, weight, bias_used, output,
        act_kind, scale,
        0, 0, 0,
        WS
    );

    /* KEEP. See the file header: the race is with the NEXT op's
     * gemmini_flush, not with the transpose this file deleted. */
    gemmini_fence();
    gemmini_flush(0);

    MB_GEMN_PH(mb_ph_t2);
    /* (output transpose: deleted -- tiled_conv_auto wrote NHWC in place) */
    MB_GEMN_PH(mb_ph_t3);
    MB_GEMN_PH_EMIT(IC, IH, IW, OC, mb_ph_t0, mb_ph_t1, mb_ph_t2, mb_ph_t3);

    /* Post-clamp for activation_max < 127 (gemmini RELU only handles min==0).
     * Under NHWC the whole output is one contiguous run, so unlike the NCHW
     * kernel there is no per-plane walk to get right. */
    if (activation_max < 127) {
        size_t total = (size_t)N * OH * OW * OC;
        for (size_t i = 0; i < total; i++)
            if (output[i] > activation_max) output[i] = (int8_t)activation_max;
    }
}
