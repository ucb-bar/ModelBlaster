/* source: curated */
/* algorithm: scalar_dq_tables */
/* accuracy_class: bit_exact */
/* origin: hand-written. Two things the mul_s8 reference does per element
 *         that it only has to do 256 times, plus the roundf call.
 *
 *   WHY THIS FILE EXISTS. mul_s8's only AlgorithmCandidate (rvv_frm_rmm) is
 *   rvv-affined, so no Gemmini target had an (op, algorithm) pair to probe
 *   for. Measured baseline, spike, vint, gemmini_q31: 167,191,069 cycles
 *   over 49 dispatches and 3,817,320 elements = 43.8 cycles PER ELEMENT for
 *   one multiply.
 *
 *   WHERE THOSE 43.8 CYCLES GO, and the two fixes:
 *
 *   1. `(int32_t)roundf(fout)` is a FUNCTION CALL. Confirmed in the emitted
 *      assembly for this toolchain: `call roundf` followed by
 *      `fcvt.w.s a6,fa0,rtz`. But roundf is round-to-nearest ties-AWAY-from-
 *      zero, and RISC-V's fcvt has exactly that as a static rounding mode:
 *      rmm, "round to nearest, ties to Max Magnitude". So the call plus the
 *      truncating convert collapse into ONE instruction,
 *      `fcvt.w.s rd, rs1, rmm`, which is not an approximation of
 *      (int32_t)roundf -- it IS it. This is the scalar counterpart of the
 *      argument kernels/rvv/rvv_add_s8_rvv_frm_rmm.c makes for vfcvt, and
 *      it needs no csrw at all because the mode is encoded in the
 *      instruction rather than read from frm, so nothing else in the kernel
 *      changes rounding behaviour.
 *
 *      Out-of-range is BETTER defined, not worse: the reference's
 *      `(int32_t)roundf(huge)` is undefined behaviour, while fcvt saturates
 *      to INT32_MIN/MAX and the clamp that follows then produces exactly
 *      activation_min/activation_max, which is what the op means.
 *
 *   2. `(float)a[i] * scale_a` has only 256 possible values, and so does
 *      `(float)b[i] * scale_b`. Both are pure functions of one int8 operand
 *      and a loop-invariant scale. Two 256-entry float tables replace two
 *      int-to-float converts and two multiplies per element with two loads.
 *      Below MB_MUL_S8_TABLE_MIN elements the 512 table evaluations cost
 *      more than they save, and the per-element expression runs instead --
 *      the same guard shape kernels/rvv/rvv_sigmoid_s8_rvv_memo_lut_gather.c
 *      uses, for the same reason.
 *
 *   WHAT IS DELIBERATELY LEFT ALONE: the divide. `(fa*fb)/scale_out` is one
 *   fdiv.s per element and multiplying by a precomputed 1.0f/scale_out is
 *   NOT the same float, so it would be a silent 1-ULP drift in the stage
 *   that is supposed to be exact. Same reason the two dequantizes stay two
 *   separate multiplies rather than being folded into (a*b)*(scale_a*scale_b).
 *
 *   BIT-EXACT BY CONSTRUCTION. Every float operation, and their order, is
 *   the reference's. MB_DRIFT_ATOL must NOT be set for this op. */

#include <math.h>
#include <stdint.h>

#ifndef MB_MUL_S8_TABLE_MIN
#define MB_MUL_S8_TABLE_MIN 512
#endif

/* (int32_t)roundf(x) as one instruction. Guarded: every curated body lands
 * in one kernels.c. */
#ifndef MB_SCALAR_RMM_
#define MB_SCALAR_RMM_
static inline int32_t mb_cvt_rmm(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#endif /* MB_SCALAR_RMM_ */

void kernel_mul_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    if (n <= 0) return;

    if (n < MB_MUL_S8_TABLE_MIN) {
        for (int i = 0; i < n; i++) {
            float fa = (float)a[i] * scale_a;
            float fb = (float)b[i] * scale_b;
            int32_t v = mb_cvt_rmm((fa * fb) / scale_out);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i] = (int8_t)v;
        }
        return;
    }

    float ta[256], tb[256];
    for (int u = 0; u < 256; u++) {
        float x = (float)(int8_t)((int)u - 128);
        ta[u] = x * scale_a;
        tb[u] = x * scale_b;
    }

    for (int i = 0; i < n; i++) {
        unsigned ia = (unsigned char)a[i] ^ 0x80u;
        unsigned ib = (unsigned char)b[i] ^ 0x80u;
        int32_t v = mb_cvt_rmm((ta[ia] * tb[ib]) / scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}
