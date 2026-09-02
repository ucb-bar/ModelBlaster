/* source: curated */
/* algorithm: scalar_lut */
/* accuracy_class: bit_exact */
/* origin: hand-written. Structurally identical to
 *         gemmini_q31_silu_s8_scalar_lut.c -- read that file's header for
 *         why the memoized table is the right shape for this op on a
 *         target with no vector unit, and why the table is stack-local.
 *
 *   WHY THIS FILE EXISTS. sigmoid_s8's only AlgorithmCandidate was
 *   rvv-affined, so no Gemmini target had an (op, algorithm) pair to probe
 *   for. The note in that candidate's description -- "its one appearance in
 *   these models is DroNet's output head at n=1, 0.0% of the run" -- is
 *   true of DroNet and is not true of ViNT, which dispatches sigmoid_s8 65
 *   times over real tensors.
 *
 *   MEASURED, spike, vint, target gemmini_q31, 65 dispatches:
 *   416,508,975 cycles, 5.2% of the 7.99-billion-cycle model and the
 *   third-largest int8 op in it, all of it expf.
 *
 *   BIT-EXACT BY CONSTRUCTION. mb_sigmoid_s8_one_g is the
 *   KernelSpec.reference_impl body for one element, unchanged. Do NOT set
 *   MB_DRIFT_ATOL for this op. */

#include <math.h>
#include <stdint.h>
#include <string.h>

/* The reference expression, verbatim, for one input byte. Suffixed _g
 * because every curated body lands in one kernels.c and kernels/rvv/
 * already defines mb_sigmoid_s8_one there. */
static inline int8_t mb_sigmoid_s8_one_g(int8_t x, float scale_in,
                                         float scale_out,
                                         int activation_min,
                                         int activation_max)
{
    float fv = (float)x * scale_in;
    float sig = 1.0f / (1.0f + expf(-fv));
    int32_t v = (int32_t)roundf(sig / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max)
{
    if (n <= 0) return;

    int8_t table[256];
    unsigned char seen[256];
    memset(seen, 0, sizeof(seen));

    for (int i = 0; i < n; i++) {
        unsigned b = (unsigned char)input[i] ^ 0x80u;
        if (!seen[b]) {
            table[b] = mb_sigmoid_s8_one_g((int8_t)((int)b - 128), scale_in,
                                           scale_out, activation_min,
                                           activation_max);
            seen[b] = 1;
        }
        output[i] = table[b];
    }
}
