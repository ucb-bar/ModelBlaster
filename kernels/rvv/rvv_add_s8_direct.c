/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER fixed-point residual add, vectorised on the
 *         RVV INTEGER datapath -- no floating point in the loop (and none at
 *         all beyond a few one-time float->fixed conversions of the scalar
 *         quant scales at entry, which do not trap on a core with misa.F=0).
 *
 *   WHY INTEGER. The previous curated add ran the reference expression on the
 *   vector FPU (vfmul/vfadd/vfdiv + a vfcvt in RMM mode). On a Saturn-class
 *   core built with a fp16-only scalar FPU (misa.F=0, WithRocketFPU16) the
 *   `.vf` scalar-operand broadcast needs an f-register move that TRAPS, so the
 *   whole op falls back to all-scalar soft-float (~80 cyc/elem). add_s8 is a
 *   two-scale rescale-then-sum, which is exactly representable in integer
 *   fixed point (gemmlowp / Jacob et al.), the same technique conv2d_s8's
 *   Q0.31 requantize and Gemmini's acc_scale already use.
 *
 *   MATH (mirrors kernels/gemmini_q31/gemmini_q31_add_s8_gemmini_resadd.c,
 *   which verifies err=0 vs the dronet golden). Fold scale_a/scale_out and
 *   scale_b/scale_out into two Q(S) integer multipliers ma, mb once at entry.
 *   S is chosen so ma,mb stay < 2^23 (then a[i]*ma < 2^30 and the two-operand
 *   sum < 2^31, i.e. the whole element loop runs in int32 -- 4 lanes/cycle at
 *   DLEN=128, twice int64's rate). Per element:
 *       acc = a[i]*ma + b[i]*mb            (int32)
 *       v   = round_half_away(acc / 2^S)   (matches roundf's ties-away)
 *   round-half-away is issued branch-free in the vector unit via the
 *   abs/shift/re-sign trick. Then saturate to [activation_min, activation_max]
 *   (already clamped into int8) and narrow i32 -> i16 -> i8.
 *
 *   ACCURACY. Bit-identical to the gemmini_resadd integer path; ties-away
 *   matches roundf. Differs from the old fp32 vector kernel only by the
 *   fixed-point quantisation of the two scale constants (< 1 LSB), so it stays
 *   inside numeric_drift and measures err=0 on dronet's residual adds. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    if (n <= 0) return;

    /* Fold the two per-tensor rescales into Q(S) integer multipliers. */
    float a_ratio = scale_a / scale_out;
    float b_ratio = scale_b / scale_out;
    float a_abs = a_ratio < 0 ? -a_ratio : a_ratio;
    float b_abs = b_ratio < 0 ? -b_ratio : b_ratio;
    int S = 24;
    float mx = a_abs > b_abs ? a_abs : b_abs;
    while (S > 1 && mx * (float)((uint32_t)1 << S) >= 8388608.0f /* 2^23 */) S--;

    const int32_t ma = (int32_t)lrintf(a_ratio * (float)((uint32_t)1 << S));
    const int32_t mb = (int32_t)lrintf(b_ratio * (float)((uint32_t)1 << S));
    const int32_t rnd = (int32_t)1 << (S - 1);

    const int32_t lo = activation_min < -128 ? -128
                       : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128
                       : (activation_max > 127 ? 127 : activation_max);

    int i = 0;
    size_t vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m2(n - i);

        vint8m2_t va8 = __riscv_vle8_v_i8m2(a + i, vl);
        vint8m2_t vb8 = __riscv_vle8_v_i8m2(b + i, vl);
        vint32m8_t va = __riscv_vsext_vf4_i32m8(va8, vl);
        vint32m8_t vb = __riscv_vsext_vf4_i32m8(vb8, vl);

        /* acc = a*ma + b*mb, all int32. */
        vint32m8_t acc = __riscv_vmul_vx_i32m8(va, ma, vl);
        acc = __riscv_vmacc_vx_i32m8(acc, mb, vb, vl);

        /* v = round_half_away(acc / 2^S), branch-free. */
        vint32m8_t sign = __riscv_vsra_vx_i32m8(acc, 31, vl);
        vint32m8_t absv = __riscv_vsub_vv_i32m8(
                              __riscv_vxor_vv_i32m8(acc, sign, vl), sign, vl);
        vint32m8_t r = __riscv_vsra_vx_i32m8(
                           __riscv_vadd_vx_i32m8(absv, rnd, vl), S, vl);
        vint32m8_t res = __riscv_vsub_vv_i32m8(
                             __riscv_vxor_vv_i32m8(r, sign, vl), sign, vl);

        res = __riscv_vmax_vx_i32m8(res, lo, vl);
        res = __riscv_vmin_vx_i32m8(res, hi, vl);

        vint16m4_t o16 = __riscv_vncvt_x_x_w_i16m4(res, vl);
        vint8m2_t o8 = __riscv_vncvt_x_x_w_i8m2(o16, vl);
        __riscv_vse8_v_i8m2(output + i, o8, vl);
    }
}
