/* source: curated */
/* algorithm: scalar_cephes_exp */
/* accuracy_class: numeric_drift */
/* origin: hand-written. The SAME Cephes minimax expf that
 *         kernels/rvv/rvv_elu_direct.c evaluates on the vector unit,
 *         transcribed to the scalar unit -- same clamp, same constants,
 *         same operation order.
 *
 *   WHY THIS FILE EXISTS, AND WHAT IT DOES AND DOES NOT FIX. mlp_control
 *   has 7 dispatches: 4 linear and 3 elu. Measured on spike, all-reference
 *   where noted:
 *
 *              gemmini_q31        rvv        ratio
 *     elu           23,334      1,093        21.4x
 *     linear       370,025     31,519        11.7x
 *     TOTAL        393,359     32,612        12.1x
 *
 *   elu was 5.9% of the model on gemmini_q31 and 3.4% on rvv, i.e. the
 *   backend gap is not concentrated in elu -- but 21.4x on an op the other
 *   arm has a curated kernel for is exactly the asymmetry that makes a
 *   2-backend sweep's gemmini arm uninterpretable, so it is worth closing
 *   as far as it can be closed.
 *
 *   WHY THERE IS NO BIT-EXACT VERSION OF THIS. The int8 activations in this
 *   tree (silu_s8, sigmoid_s8, gelu_s8) get a 256-entry memoized LUT that is
 *   bit-exact by construction, because their input domain is 256 values.
 *   elu is fp32: its domain is continuous, nothing memoizes, and the entire
 *   cost is picolibc's expf. The only way past expf is to not call it, and
 *   any replacement returns a different float. So NUMERIC_DRIFT is not a
 *   concession here, it is the honest class for the only available move --
 *   and it is the same class kernels/rvv/rvv_elu_direct.c already declares
 *   for the same reason, with the same polynomial, so this does not widen
 *   the accuracy envelope the rvv arm of a sweep already sits in.
 *
 *   THE EMPTY ASM BARRIERS ARE LOAD-BEARING. GCC defaults to
 *   -ffp-contract=fast and would fuse each `y*x + c` of the Horner chain
 *   into an fmadd, dropping the intermediate rounding. The RVV kernel writes
 *   the chain as separate vfmul/vfadd and therefore does NOT fuse, so a
 *   contracted scalar version would drift AWAY from the rvv arm rather than
 *   toward it. mb_elu_mul is a plain multiply with a zero-instruction
 *   barrier on the result, which is what makes the two arms produce the same
 *   float elementwise by construction. (Claimed by construction from the op
 *   sequence -- the vector kernel is purely elementwise -- not from a
 *   side-by-side hardware run.)
 *
 *   Accuracy of the polynomial itself: Cephes single-precision expf,
 *   < 1e-6 relative, as documented in the RVV kernel's header. */

#include <stdint.h>
#include <string.h>

/* Guarded: every curated body lands in one kernels.c. */
#ifndef MB_ELU_CEPHES_
#define MB_ELU_CEPHES_
/* Multiply that the compiler may not fold into an FMA with a following
 * add. Costs nothing; see the header for why it is required. */
static inline float mb_elu_mul(float a, float b)
{
    float r = a * b;
    __asm__("" : "+f"(r));
    return r;
}

/* fcvt.w.s under the RNE rounding mode -- what vfcvt.x.f does with the
 * default frm, which is what the RVV kernel's exponent split uses. */
static inline int32_t mb_elu_cvt_rne(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rne" : "=r"(r) : "f"(x));
    return r;
}

/* Cephes single-precision exp. Constants and order copied from
 * kernels/rvv/rvv_elu_direct.c:rvv_exp_ps_elu. */
static inline float mb_elu_exp(float x)
{
    if (x > 88.3762626647949f)  x = 88.3762626647949f;
    if (x < -88.3762626647949f) x = -88.3762626647949f;

    float fx = mb_elu_mul(x, 1.44269504088896341f);
    int32_t n = mb_elu_cvt_rne(fx);
    fx = (float)n;

    x = x - mb_elu_mul(fx, 0.693359375f);
    x = x - mb_elu_mul(fx, -2.12194440e-4f);

    float z = mb_elu_mul(x, x);
    float y = 1.9875691500E-4f;
    y = mb_elu_mul(y, x); y = y + 1.3981999507E-3f;
    y = mb_elu_mul(y, x); y = y + 8.3334519073E-3f;
    y = mb_elu_mul(y, x); y = y + 4.1665795894E-2f;
    y = mb_elu_mul(y, x); y = y + 1.6666665459E-1f;
    y = mb_elu_mul(y, x); y = y + 5.0000001201E-1f;
    y = mb_elu_mul(y, z);
    y = y + x;
    y = y + 1.0f;

    uint32_t bits = (uint32_t)(n + 127) << 23;
    float pow2n;
    memcpy(&pow2n, &bits, sizeof pow2n);
    return mb_elu_mul(y, pow2n);
}
#endif /* MB_ELU_CEPHES_ */

void kernel_elu(const float *input, float *output, int n, float alpha)
{
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : mb_elu_mul(mb_elu_exp(v) - 1.0f, alpha);
    }
}
