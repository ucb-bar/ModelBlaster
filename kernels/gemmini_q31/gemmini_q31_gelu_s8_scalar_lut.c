/* source: curated */
/* algorithm: scalar_lut */
/* accuracy_class: bit_exact */
/* origin: hand-written. Structurally identical to
 *         gemmini_q31_silu_s8_scalar_lut.c -- read that file's header.
 *         Only the per-element expression differs: erff, not expf.
 *
 *   WHY THIS FILE EXISTS. gelu_s8's only AlgorithmCandidate was
 *   rvv-affined. ViNT dispatches it 4 times, measured at 14,995,775 spike
 *   cycles on gemmini_q31 -- 0.19% of that model. Small, and reported as
 *   small: this closes the coverage hole so the op is not the thing that
 *   makes a shard boundary look expensive, and it is not claimed to move
 *   ViNT's makespan.
 *
 *   BIT-EXACT BY CONSTRUCTION, including the kInvSqrt2 constant, which is
 *   the reference's 0.70710678118f and not a recomputed 1/sqrtf(2). Do NOT
 *   set MB_DRIFT_ATOL for this op. */

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline int8_t mb_gelu_s8_one_g(int8_t x, float scale_in,
                                      float scale_out,
                                      int activation_min, int activation_max)
{
    const float kInvSqrt2 = 0.70710678118f;
    float f = (float)x * scale_in;
    float y = 0.5f * f * (1.0f + erff(f * kInvSqrt2));
    int32_t v = (int32_t)roundf(y / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_gelu_s8(const int8_t *input, int8_t *output, int n,
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
            table[b] = mb_gelu_s8_one_g((int8_t)((int)b - 128), scale_in,
                                        scale_out, activation_min,
                                        activation_max);
            seen[b] = 1;
        }
        output[i] = table[b];
    }
}
