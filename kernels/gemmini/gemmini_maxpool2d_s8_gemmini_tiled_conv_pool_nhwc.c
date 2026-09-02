/* source: curated */
/* algorithm: gemmini_tiled_conv_pool_nhwc */
/* accuracy_class: bit_exact */
/* act_layouts: nhwc */
/* NHWC entry point for gemmini's depthwise-passthrough + mvout-pool maxpool.
 *
 * This is gemmini_q31_maxpool2d_s8_gemmini_tiled_conv_pool.c with the two
 * activation transposes deleted. The gemmini datapath was ALREADY NHWC
 * internally -- the NCHW kernel transposed into ws_input, called
 * tiled_conv_dw_auto, and transposed ws_output back -- so this file is a
 * deletion, not new arithmetic.
 *
 * The +2 passthrough weight and the ACC_SCALE_IDENTITY reasoning are unchanged
 * and are what make this bit_exact; see the original file's header for the
 * exhaustive int8 argument (ACC_SCALE(2*v, ACC_SCALE_IDENTITY) == v for every
 * v, while weight=+1 silently halves). Nothing here touches that path.
 *
 * ACTIVATION LAYOUT: `input` is [N, IH, IW, C] and `output` is [N, OH, OW, C].
 * Enforced by act_layouts=("nhwc",) on this algorithm plus the deny-by-default
 * gate in pipeline/generate_kernels.assert_act_layout_contract, because an
 * NCHW tensor handed to this kernel is size-identical and therefore silent.
 */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_MPN_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_MPN_SLOT ((int)arch_proc_id())
#else
enum { MB_MPN_SLOTS = 1 };
#define MB_MPN_SLOT 0
#endif

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW)
{
    enum { MAXPOOL_MAX_CHANNELS = 1024 };
    /* Per-hart, unlike the NCHW kernel's single shared table: two harts can be
     * inside this function at once on a multi-Gemmini SoC, and a shared
     * lazily-initialised table is a read-before-write race that only appears
     * under a schedule that actually overlaps them. C entries of a constant,
     * so re-filling per call is cheaper than the flag it replaces. */
    static elem_t ws_weights_all[MB_MPN_SLOTS][MAXPOOL_MAX_CHANNELS]
        __attribute__((aligned(64)));
    elem_t *const ws_weights = ws_weights_all[MB_MPN_SLOT];

    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;

    bool gemmini_ok =
           KH == KW && SH == SW && PH == PW
        && DH == 1 && DW == 1
        && PH == 0
        && C <= MAXPOOL_MAX_CHANNELS;

    if (!gemmini_ok) {
        /* Scalar fallback, NHWC-indexed. Same sliding-window max as the NCHW
         * kernel's fallback; only the two index expressions differ. */
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    for (int c = 0; c < C; c++) {
                        int8_t m = INT8_MIN;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh*SH - PH + kh*DH;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow*SW - PW + kw*DW;
                                if (iw < 0 || iw >= IW) continue;
                                int8_t v = input[
                                    (((size_t)n * IH + ih) * IW + iw) * C + c];
                                if (v > m) m = v;
                            }
                        }
                        output[(((size_t)n * OH + oh) * OW + ow) * C + c] = m;
                    }
                }
            }
        }
        return;
    }

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    for (int i = 0; i < C; i++) ws_weights[i] = 2;

    gemmini_flush(0);

    asm volatile("fence" ::: "memory");

    tiled_conv_dw_auto(
        N, IH, IW,
        C, IH, IW,
        /* stride       = */ 1,
        /* padding      = */ 0,
        /* kernel_dim   = */ 1,
        (elem_t *)input, ws_weights,
        /* bias         = */ NULL,
        output,
        /* act          = */ 0,             /* NO_ACTIVATION */
        /* scale        = */ ACC_SCALE_IDENTITY,
        /* pool_size    = */ KH,
        /* pool_stride  = */ SH,
        /* pool_padding = */ PH,
        WS
    );

    gemmini_fence();
    gemmini_flush(0);
}
