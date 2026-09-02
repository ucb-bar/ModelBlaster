/* source: curated */
/* algorithm: scalar_chan_lut */
/* accuracy_class: bit_exact */
/* origin: hand-written. batchnorm2d_s8's input is int8 and its scale/bias
 *         are per CHANNEL, so each channel's whole H*W plane has at most 256
 *         distinct outputs -- the activation-LUT argument, one channel at a
 *         time.
 *
 *   WHY THIS FILE EXISTS. batchnorm2d_s8's only AlgorithmCandidate
 *   (per_channel_lut) is affined to rvv_opu, and the curated RVV kernel
 *   occupies the universal `direct` slot under kernels/rvv/. So no Gemmini
 *   target had an (op, algorithm) file to probe for, and it ran the scalar
 *   reference -- expf-free, but with a roundf CALL and a float divide per
 *   element.
 *
 *   MEASURED, spike, dronet, target gemmini_q31, 3 dispatches:
 *   1,881,267 cycles = 32.4% OF THE WHOLE MODEL, more than either the ten
 *   convolutions (2,168,407 across all of them) or the maxpool. dronet's
 *   BN planes are C=32 H*W=729, C=32 H*W=196 and C=64 H*W=49.
 *
 *   MEMOIZED, NOT EAGER, and that is load-bearing here rather than a
 *   nicety: two of dronet's three BN layers have planes SMALLER than 256
 *   elements, so an eagerly built 256-entry table per channel would be a
 *   straight loss on them (196 and 49 elements against 256 evaluations).
 *   Filling an entry the first time its byte value appears costs
 *   distinct*work instead, and distinct <= min(256, H*W) by construction.
 *   Below MB_BN_S8_LUT_MIN elements per plane even the 256-byte `seen`
 *   memset is not worth it and the reference expression runs per element.
 *
 *   TWO CHANGES, BOTH EXACT. (1) memoize the table; (2) replace
 *   `(int32_t)roundf(...)` -- a libgcc CALL followed by a truncating fcvt --
 *   with a single fcvt.w.s under the rmm rounding mode, which is what roundf
 *   means. The second is what makes the SHORT planes win too: memoization
 *   alone measured 3.03x on dronet's H*W=729 layer, 1.85x on its H*W=196
 *   layer and 0.95x -- a loss -- on its H*W=49 layer, because a plane
 *   shorter than the table cannot amortize it.
 *
 *   BIT-EXACT BY CONSTRUCTION. mb_bn_s8_one is the reference's per-element
 *   body -- same scale_in multiply, same s*fv + b (contracted or not,
 *   identically, since it is the same expression compiled with the same
 *   flags), the same rounding, same clamp. MB_DRIFT_ATOL must NOT be set. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Planes smaller than this run the per-element expression: with the roundf
 * call already gone (below), a table cannot pay back its 256-byte `seen`
 * memset plus a per-element branch on a plane this short. Set from
 * measurement, not taste -- at 32 the C=64,H*W=49 layer of dronet went the
 * table way and measured 0.95x, i.e. a LOSS (194,510 vs 184,690 cycles). */
#ifndef MB_BN_S8_LUT_MIN
#define MB_BN_S8_LUT_MIN 128
#endif

/* (int32_t)roundf(x) as ONE instruction instead of a libgcc call plus a
 * truncating convert. RISC-V's fcvt.w.s takes a static rounding mode and
 * rmm ("round to nearest, ties to Max Magnitude") is exactly roundf's
 * ties-away-from-zero, so this is not an approximation of the reference's
 * (int32_t)roundf -- it IS it, and it saturates rather than being UB out of
 * range, which the clamp below then turns into the right answer. Same
 * argument as kernels/rvv/rvv_add_s8_rvv_frm_rmm.c makes for vfcvt, minus
 * the csrw, because the mode is in the instruction. Guarded: every curated
 * body lands in one kernels.c. */
#ifndef MB_SCALAR_RMM_
#define MB_SCALAR_RMM_
static inline int32_t mb_cvt_rmm(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#endif /* MB_SCALAR_RMM_ */

static inline int8_t mb_bn_s8_one(int8_t x, float s, float b,
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
    size_t hw = (size_t)H * (size_t)W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            size_t base = (size_t)(n*C + c) * hw;
            const int8_t *ip = input + base;
            int8_t *op = output + base;

            if (hw < MB_BN_S8_LUT_MIN) {
                for (size_t i = 0; i < hw; i++)
                    op[i] = mb_bn_s8_one(ip[i], s, b, scale_in, scale_out,
                                         activation_min, activation_max);
                continue;
            }

            int8_t table[256];
            unsigned char seen[256];
            memset(seen, 0, sizeof(seen));
            for (size_t i = 0; i < hw; i++) {
                unsigned u = (unsigned char)ip[i] ^ 0x80u;
                if (!seen[u]) {
                    table[u] = mb_bn_s8_one((int8_t)((int)u - 128), s, b,
                                            scale_in, scale_out,
                                            activation_min, activation_max);
                    seen[u] = 1;
                }
                op[i] = table[u];
            }
        }
    }
}
