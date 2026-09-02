/* source: curated */
/* algorithm: scalar_lut */
/* accuracy_class: bit_exact */
/* origin: hand-written. The silu_s8 reference expression, unchanged, but
 *         evaluated at most 256 times per call instead of n times.
 *
 *   WHY THIS FILE EXISTS. silu_s8 had two AlgorithmCandidates and both were
 *   affined away from Gemmini (rvv_lut_gather -> rvv_opu; the curated
 *   kernels/rvv/rvv_silu_s8_direct.c occupies the universal `direct` slot
 *   and lives under kernels/rvv/). So on a Gemmini target the curated probe
 *   had no (op, algorithm) file to find and every SiLU ran the scalar
 *   reference's expf per element inside a build labelled gemmini_q31.
 *
 *   MEASURED, spike, yolov8_nano, target gemmini_q31, 57 dispatches:
 *   99,691,648 cycles -- 64.1% of the entire 155,628,733-cycle model, more
 *   than the 63 convolutions put together (46,504,004). An activation
 *   function was the dominant cost of an object detector because nothing
 *   pointed a curated kernel at it.
 *
 *   WHY A LUT, AND WHY NOT THE RVV ONE. Gemmini is a systolic int8 MAC
 *   array; it has no functional unit that evaluates a transcendental, so
 *   there is no "Gemmini algorithm" for this op to write. The Gemmini
 *   targets compile kernels at -march=rv64imafdc (backends.GEMMINI_Q31: no
 *   `v`, by construction -- the pure-Gemmini bitstream's hart has no vector
 *   unit), so kernels/rvv/rvv_silu_s8_direct.c's vrgather-out-of-a-
 *   register-resident-table trick has no analogue either. What survives is
 *   the half of that idea that was never about the vector unit: the input
 *   is int8, so for a fixed (scale_in, scale_out, activation_min,
 *   activation_max) the op has at most 256 distinct outputs, and a
 *   quantized activation tensor repeats values heavily.
 *
 *   MEMOIZED, NOT EAGER. Building all 256 entries up front costs 256 expf
 *   regardless of n, which loses outright for the small tensors these
 *   models also contain. Filling an entry the first time its byte value is
 *   seen never loses by more than one predictable not-taken branch per
 *   element, and needs no small-n guard at all.
 *
 *   STACK-LOCAL, NOT STATIC. kernels/rvv/rvv_silu_s8_direct.c caches its
 *   table in `static` storage keyed on the quant tuple. That is safe for a
 *   single-threaded model walk and is NOT safe here: the sharding sweeps
 *   run two harts through one kernels.c, and two dispatches of the same op
 *   with different scales would race on the table and the cache key. 512
 *   bytes of stack per call is the cost of not having that bug.
 *
 *   BIT-EXACT BY CONSTRUCTION, not by tolerance. mb_silu_s8_one is the
 *   KernelSpec.reference_impl body for one element, character for
 *   character: same (float) cast, same expf, same divide, same roundf,
 *   same clamp order. The lookup loop performs no arithmetic. There is no
 *   rounding mode to match and MB_DRIFT_ATOL must NOT be set for this op --
 *   if it ever fails verify, the table is wrong, not imprecise. */

#include <math.h>
#include <stdint.h>
#include <string.h>

/* The reference expression, verbatim, for one input byte. */
static inline int8_t mb_silu_s8_one(int8_t x, float scale_in, float scale_out,
                                    int activation_min, int activation_max)
{
    float f = (float)x * scale_in;
    float y = f / (1.0f + expf(-f));
    int32_t v = (int32_t)roundf(y / scale_out);
    if (v < activation_min) v = activation_min;
    if (v > activation_max) v = activation_max;
    return (int8_t)v;
}

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max)
{
    if (n <= 0) return;

    int8_t table[256];
    unsigned char seen[256];
    memset(seen, 0, sizeof(seen));

    for (int i = 0; i < n; i++) {
        /* ^0x80 biases int8 -> [0,255] with one xor; (int)x+128 needs a
         * sign-extending load plus an add. Same map, one instruction. */
        unsigned b = (unsigned char)input[i] ^ 0x80u;
        if (!seen[b]) {
            table[b] = mb_silu_s8_one((int8_t)((int)b - 128), scale_in,
                                      scale_out, activation_min,
                                      activation_max);
            seen[b] = 1;
        }
        output[i] = table[b];
    }
}
