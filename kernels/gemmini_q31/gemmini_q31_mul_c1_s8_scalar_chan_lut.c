/* source: curated */
/* algorithm: scalar_chan_lut */
/* accuracy_class: bit_exact */
/* origin: hand-written. mul_c1_s8's gate is per CHANNEL and its other
 *         operand is int8, so each channel's plane has at most 256 distinct
 *         outputs -- the same structure as
 *         gemmini_q31_batchnorm2d_s8_scalar_chan_lut.c.
 *
 *   WHY THIS FILE EXISTS. mul_c1_s8 had no AlgorithmCandidate at all on any
 *   target -- only the universal `direct` synthesized by
 *   KernelSpec.__post_init__ -- and no Gemmini file. Measured baseline,
 *   spike, vint, gemmini_q31: 29,569,126 cycles over 16 dispatches and
 *   1,415,808 elements = 20.9 cycles per element, of which the float DIVIDE
 *   by scale_out is the dominant term (unpipelined fdiv.s on this core).
 *   The table is what removes the divide from the per-element path.
 *
 *   MEMOIZED, NOT EAGER, and the first version of this file got that wrong.
 *   It built all 256 entries whenever HW >= 256 and measured 30,191,022
 *   cycles -- 2% SLOWER than the reference. ViNT's mul_c1_s8 planes are
 *   HW=4, 20, 80, 336 and 1344: at HW=336 an eager table pays 256
 *   evaluations to save 336, which is a wash once the lookup loop is added,
 *   and only the single HW=1344 dispatch was ever ahead. Filling on demand
 *   costs distinct*work with distinct <= min(256, HW), so it cannot lose
 *   that way. Recorded rather than quietly fixed because "LUT the int8
 *   operand" is the obvious move and the eager form of it is a trap at
 *   these plane sizes.
 *
 *   NOT `roundf`, AND NOT fcvt-rmm EITHER. This op's reference does NOT use
 *   roundf: it uses (int32_t)(v >= 0 ? v + 0.5f : v - 0.5f), which rounds
 *   the ADDITION in float before truncating and therefore differs from
 *   roundf wherever that sum is inexact. The fcvt.w.s rmm substitution that
 *   gemmini_q31_mul_s8_scalar_dq_tables.c is built on would be a real 1-LSB
 *   change here, so _mb_mul_c1_s8_round below is the reference helper
 *   copied character for character instead.
 *
 *   BIT-EXACT BY CONSTRUCTION. MB_DRIFT_ATOL must NOT be set for this op. */

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef MB_MUL_C1_S8_LUT_MIN
#define MB_MUL_C1_S8_LUT_MIN 32
#endif

/* The reference's rounding helper, verbatim. Renamed only because every
 * curated body is concatenated into one kernels.c alongside the reference
 * impls of the ops that did NOT get a curated kernel. */
static inline int32_t _mb_mul_c1_s8_round(float v)
{
    int32_t i = (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
    if (i < -128) i = -128;
    if (i > 127)  i = 127;
    return i;
}

void kernel_mul_c1_s8(const int8_t *gate, const int8_t *x, int8_t *output,
                      int N, int C, int HW,
                      float scale_gate, float scale_x, float scale_out)
{
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float g_real = (float)gate[c] * scale_gate;
            size_t base = (size_t)(n*C + c) * (size_t)HW;
            const int8_t *xp = x + base;
            int8_t *op = output + base;

            if (HW < MB_MUL_C1_S8_LUT_MIN) {
                for (int i = 0; i < HW; i++) {
                    float prod = g_real * ((float)xp[i] * scale_x);
                    op[i] = (int8_t)_mb_mul_c1_s8_round(prod / scale_out);
                }
                continue;
            }

            int8_t table[256];
            unsigned char seen[256];
            memset(seen, 0, sizeof(seen));
            for (int i = 0; i < HW; i++) {
                unsigned u = (unsigned char)xp[i] ^ 0x80u;
                if (!seen[u]) {
                    float xv = (float)(int8_t)((int)u - 128);
                    float prod = g_real * (xv * scale_x);
                    table[u] = (int8_t)_mb_mul_c1_s8_round(prod / scale_out);
                    seen[u] = 1;
                }
                op[i] = table[u];
            }
        }
    }
}
