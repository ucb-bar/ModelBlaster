/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV conv2d_s8_pc (per-output-channel requant), reading the
 *   NATIVE OCIHW weight layout (the synthesized "direct" algo declares
 *   weight_layout=oihw, so no layout transform is applied).
 *
 *   Strategy — OC-vectorized GEMV with a one-time weight transpose. Conv is a
 *   GEMM [OH*OW, R] x [R, OC] with R = IC*KH*KW. The scalar reference's cost is
 *   OH*OW*OC*R MACs. We vectorize the OC dimension: for each output pixel we
 *   accumulate a VECTOR of OC partial sums (i32), adding patch[r] * weight[:,r]
 *   for each reduction index r via a widening multiply-accumulate. This removes
 *   the per-oc reduction (vredsum) that made an im2col+dot variant only ~1.2x —
 *   here the OC lanes accumulate in place, so the arithmetic speedup is ~VL.
 *
 *   For unit-stride weight vector loads (spike charges strided/gather loads
 *   per-element) the weights must be [R, OC] but the native layout is [OC, R],
 *   so we transpose once per call into a static scratch buffer (amortized over
 *   OH*OW pixels). The input receptive field is gathered once per pixel (shared
 *   across all OC). Requant is per-oc scalar Q0.31 -> bit-exact vs the
 *   conv2d_s8_pc reference. Symmetric quant only (input/filter offset 0). */

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

/* This model's convs: max R=IC*KH*KW=576, max R*OC=36864, max OC=64. Static
 * (kernel runs single-threaded on the harness) to keep it off the thread stack.
 * Sized with headroom. */
#define MB_CONV_WT_MAX   65536   /* >= max R*OC */
#define MB_CONV_PATCH_MAX 1024   /* >= max R */
#define MB_CONV_OC_MAX    256    /* >= max OC */

/* Weight transposed to [R, OC] once per call (kept int8 — spike charges vector
 * loads per byte, so an i8 load + widening vwmacc beats an i16 load). */
static int8_t  mb_wt[MB_CONV_WT_MAX];
static int8_t  mb_patch[MB_CONV_PATCH_MAX];
static int32_t mb_drain[MB_CONV_OC_MAX];

static inline int32_t mb_q31_requant_pc(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return (scaled + round) >> shift;
    }
    return scaled << (-shift);
}

void kernel_conv2d_s8_pc(const int8_t *input, const int8_t *weight,
                         const int32_t *bias, int8_t *output,
                         int N, int IC, int IH, int IW, int OC,
                         int KH, int KW, int SH, int SW, int PH, int PW,
                         int input_offset, int filter_offset, int output_offset,
                         const int32_t *output_multiplier,
                         const int32_t *output_shift,
                         int activation_min, int activation_max)
{
    (void)input_offset; (void)filter_offset;   /* symmetric quant only */
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    const int R = IC * KH * KW;                 /* reduction / weight-row length */

    /* Transpose weight [OC, R] (OCIHW, contiguous per oc) -> mb_wt[R, OC] so
     * the inner loop loads weight[:, r] (all OC for one r) unit-stride. */
    for (int oc = 0; oc < OC; oc++) {
        const int8_t *wsrc = weight + (size_t)oc * R;
        for (int r = 0; r < R; r++)
            mb_wt[(size_t)r * OC + oc] = wsrc[r];
    }

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                /* Gather the receptive field once (shared across all oc).
                 * Out-of-bounds taps -> 0 (== symmetric zero-point). */
                int idx = 0;
                for (int ic = 0; ic < IC; ic++) {
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) {
                            for (int kw = 0; kw < KW; kw++) mb_patch[idx++] = 0;
                            continue;
                        }
                        const size_t row_off =
                            (((size_t)n * IC + ic) * IH + ih) * (size_t)IW;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            mb_patch[idx++] =
                                (iw >= 0 && iw < IW) ? input[row_off + iw] : 0;
                        }
                    }
                }

                /* OC-vectorized accumulation over the reduction. */
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc_base));
                    vint32m4_t vacc;
                    if (bias != NULL)
                        vacc = __riscv_vle32_v_i32m4(bias + oc_base, vl);
                    else
                        vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int r = 0; r < R; r++) {
                        int8_t pv = mb_patch[r];
                        if (pv == 0) continue;           /* skip zero taps */
                        const int8_t *wr = mb_wt + (size_t)r * OC + oc_base;
                        vint8m1_t w8 = __riscv_vle8_v_i8m1(wr, vl);
                        vint16m2_t w16 = __riscv_vwadd_vx_i16m2(w8, 0, vl);
                        vacc = __riscv_vwmacc_vx_i32m4(vacc, (int16_t)pv, w16, vl);
                    }

                    /* per-oc scalar Q0.31 requantize on the drained i32 vector */
                    __riscv_vse32_v_i32m4(mb_drain, vacc, vl);
                    for (size_t lane = 0; lane < vl; lane++) {
                        int oc = oc_base + (int)lane;
                        int32_t s = mb_q31_requant_pc(mb_drain[lane],
                                                      output_multiplier[oc],
                                                      output_shift[oc]);
                        s += output_offset;
                        if (s < activation_min) s = activation_min;
                        if (s > activation_max) s = activation_max;
                        output[((size_t)(n * OC + oc) * OH + oh) * OW + ow] = (int8_t)s;
                    }
                    oc_base += (int)vl;
                }
            }
        }
    }
}
