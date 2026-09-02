/* source: curated */
/* algorithm: scalar_chan_lut_nhwc */
/* accuracy_class: bit_exact */
/* act_layouts: nhwc */
/* NHWC batchnorm2d_s8 for the Gemmini targets.
 *
 * Unlike the conv and maxpool NHWC kernels beside it, this one is genuinely new
 * C rather than a deletion: batchnorm has no gemmini datapath at all (it is a
 * per-channel float affine, and gemmini is an int8 MAC array), so both layouts
 * are plain loops on the scalar core.
 *
 * It is the NHWC counterpart of gemmini_q31_batchnorm2d_s8_scalar_chan_lut.c
 * and keeps both of that kernel's exact tricks -- the per-channel 256-entry
 * table and mb_cvt_rmm in place of a roundf call -- so an NCHW/NHWC A/B prices
 * the LAYOUT and not a change of algorithm.
 *
 * WHERE THE TWO LAYOUTS GENUINELY DIFFER, and it is not in this op's favour.
 * Under NCHW the channel is loop-invariant over a whole H*W plane, so ONE
 * 256-entry table is live at a time and it can be filled LAZILY -- an entry
 * costs work only when its byte value actually appears, so the build cost is
 * `distinct <= min(256, H*W)`. Under NHWC the channel changes every element, so
 * every channel's table has to be live at once ([C][256]) and lazy filling
 * would need a [C][256] `seen` array whose memset alone can exceed the plane.
 * So this kernel fills EAGERLY, which costs a flat C*256 evaluations, and takes
 * the table path only when H*W is long enough to amortise that:
 *
 *   dronet, the three batchnorms:   C=32 H*W=729 -> table  (8192 build, 23328 lookups)
 *                                   C=32 H*W=196 -> direct (a table would lose)
 *                                   C=64 H*W=49  -> direct (a table would lose badly)
 *
 * The NCHW kernel memoises all three of those (3.03x / 1.85x / direct). So the
 * middle layer is a real, small regression under NHWC and is reported as one --
 * layout does not move every cost in the same direction, and section 6 of
 * docs/IR_TENSOR_LAYOUT_DESIGN.md says the same thing about split axes.
 *
 * BIT-EXACT BY CONSTRUCTION, and that is a requirement here rather than a
 * nicety: the whole NHWC island is a permutation, so the model output must be
 * bit-identical to the NCHW build. mb_bn_s8n_one is the reference's per-element
 * body verbatim, every output element is an independent function of one input
 * element and its channel's (scale, bias), and a table entry is that same body
 * evaluated on that same byte. MB_DRIFT_ATOL must NOT be set.
 *
 * ACTIVATION LAYOUT: `input` and `output` are [N, H, W, C]. Enforced by
 * act_layouts=("nhwc",) plus the deny-by-default gate in
 * pipeline/generate_kernels.assert_act_layout_contract -- an NCHW tensor handed
 * here is size-identical and would be silently wrong.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* Positions per image below which the eager table cannot pay for itself. The
 * table costs C*256 evaluations and saves (H*W - 256) per channel, so it wins
 * exactly when H*W > 256; 256 is the break-even and is used as the threshold
 * rather than tuned, because dronet's three layers (729, 196, 49) are nowhere
 * near it and tuning on them would be fitting noise. */
#ifndef MB_BN_S8_NHWC_LUT_MIN
#define MB_BN_S8_NHWC_LUT_MIN 256
#endif

/* Channels above which the table would not fit its static budget. dronet's
 * widest batchnorm is C=64 (16 KB of table); yolov8 goes to C=256, which would
 * be 64 KB per hart and is left on the direct path deliberately. */
#ifndef MB_BN_S8_NHWC_MAX_C
#define MB_BN_S8_NHWC_MAX_C 128
#endif

#if defined(CONFIG_SMP) && defined(CONFIG_MP_MAX_NUM_CPUS) && CONFIG_MP_MAX_NUM_CPUS > 1
#include <zephyr/kernel.h>
enum { MB_BNN_SLOTS = CONFIG_MP_MAX_NUM_CPUS };
#define MB_BNN_SLOT ((int)arch_proc_id())
#else
enum { MB_BNN_SLOTS = 1 };
#define MB_BNN_SLOT 0
#endif

/* (int32_t)roundf(x) as ONE instruction. rmm is "round to nearest, ties to Max
 * Magnitude" = roundf's ties-away-from-zero, so this is not an approximation of
 * (int32_t)roundf -- it IS it. Guarded because every curated body lands in one
 * kernels.c and the NCHW batchnorm defines the same helper. */
#ifndef MB_SCALAR_RMM_
#define MB_SCALAR_RMM_
static inline int32_t mb_cvt_rmm(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#endif /* MB_SCALAR_RMM_ */

static inline int8_t mb_bn_s8n_one(int8_t x, float s, float b,
                                   float scale_in, float scale_out,
                                   int activation_min, int activation_max)
{
    float fv = (float)x * scale_in;
    float y = s * fv + b;
    int32_t v = mb_cvt_rmm(y / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max)
{
    static int8_t mb_bnn_tab_all[MB_BNN_SLOTS][MB_BN_S8_NHWC_MAX_C][256];
    const size_t HW = (size_t)H * (size_t)W;
    const int use_tab = (HW >= (size_t)MB_BN_S8_NHWC_LUT_MIN
                         && C <= MB_BN_S8_NHWC_MAX_C);

    if (!use_tab) {
        for (int n = 0; n < N; n++) {
            const int8_t *in  = input  + (size_t)n * HW * (size_t)C;
            int8_t       *out = output + (size_t)n * HW * (size_t)C;
            for (size_t p = 0; p < HW; p++) {
                const int8_t *ip = in  + p * (size_t)C;
                int8_t       *op = out + p * (size_t)C;
                for (int c = 0; c < C; c++)
                    op[c] = mb_bn_s8n_one(ip[c], scale[c], bias[c],
                                          scale_in, scale_out,
                                          activation_min, activation_max);
            }
        }
        return;
    }

    int8_t (*tab)[256] = mb_bnn_tab_all[MB_BNN_SLOT];
    for (int c = 0; c < C; c++) {
        float s = scale[c], b = bias[c];
        int8_t *t = tab[c];
        for (int u = 0; u < 256; u++)
            t[u] = mb_bn_s8n_one((int8_t)(u - 128), s, b, scale_in, scale_out,
                                 activation_min, activation_max);
    }
    for (int n = 0; n < N; n++) {
        const int8_t *in  = input  + (size_t)n * HW * (size_t)C;
        int8_t       *out = output + (size_t)n * HW * (size_t)C;
        for (size_t p = 0; p < HW; p++) {
            const int8_t *ip = in  + p * (size_t)C;
            int8_t       *op = out + p * (size_t)C;
            for (int c = 0; c < C; c++)
                op[c] = tab[c][(unsigned char)ip[c] ^ 0x80u];
        }
    }
}
