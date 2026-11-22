/* source: curated */
/* algorithm: rvv */
/* accuracy_class: numeric_drift */
/* origin: hand-written. RVV two-scale asymmetric residual add -- the vectorised
 *         counterpart of gemmini_q31_rvv_add_s8_direct.c. BIT-EXACT to it:
 *         identical integer fixed-point rescale (fold scale_a/scale_out and
 *         scale_b/scale_out into Q(S) integer multipliers ma,mb; a one-time
 *         soft-float setup that does not trap on misa.F=0), and the SAME
 *         per-element acc = a*ma + b*mb ; round_half_away(acc >> S) ; clamp.
 *
 *   WHY. add_s8 was the last substantial SCALAR compute in dronet (the residual
 *   adds have asymmetric input-scale ratios, a_ratio~0.20, so gemmini/fp resadd
 *   traps on the no-FPU core and it ran as a scalar int loop). The whole thing
 *   is pure integer, so vectorise it on the Saturn RVV integer datapath -- NO
 *   FP in the element loop. Same round-half-away branch-free trick as the
 *   integer batchnorm/requant kernels; e32m8 for VLEN=128 (32 int32 lanes). */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    if (n <= 0) return;
    /* --- one-time scalar setup, identical to the direct kernel --- */
    float a_ratio = scale_a / scale_out;
    float b_ratio = scale_b / scale_out;
    float a_abs = a_ratio < 0 ? -a_ratio : a_ratio;
    float b_abs = b_ratio < 0 ? -b_ratio : b_ratio;
    int S = 24;
    float mx = a_abs > b_abs ? a_abs : b_abs;
    while (S > 1 && mx * (float)((uint32_t)1 << S) >= 8388608.0f) S--;
    const int32_t ma = (int32_t)lrintf(a_ratio * (float)((uint32_t)1 << S));
    const int32_t mb = (int32_t)lrintf(b_ratio * (float)((uint32_t)1 << S));
    const int32_t rnd = (int32_t)1 << (S - 1);
    const int32_t lo = activation_min < -128 ? -128
                       : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128
                       : (activation_max > 127 ? 127 : activation_max);

    /* --- vectorised element loop: e8m2 -> sext i32m8 (32 int32 lanes) --- */
    size_t i = 0, rem = (size_t)n;
    while (rem > 0) {
        size_t vl = __riscv_vsetvl_e8m2(rem);
        vint8m2_t  va8 = __riscv_vle8_v_i8m2(a + i, vl);
        vint8m2_t  vb8 = __riscv_vle8_v_i8m2(b + i, vl);
        vint32m8_t va  = __riscv_vsext_vf4_i32m8(va8, vl);
        vint32m8_t vb  = __riscv_vsext_vf4_i32m8(vb8, vl);

        /* acc = a*ma + b*mb  (int32, fits by construction of S) */
        vint32m8_t acc = __riscv_vmul_vx_i32m8(va, ma, vl);
        acc = __riscv_vmacc_vx_i32m8(acc, mb, vb, vl);      /* acc += mb*vb */

        /* v = round_half_away(acc / 2^S), branch-free (matches the scalar
         * ternary: acc>=0 ? (acc+rnd)>>S : -(((-acc)+rnd)>>S)). */
        vint32m8_t sign = __riscv_vsra_vx_i32m8(acc, 31, vl);
        vint32m8_t absv = __riscv_vsub_vv_i32m8(
                              __riscv_vxor_vv_i32m8(acc, sign, vl), sign, vl);
        vint32m8_t r = __riscv_vsra_vx_i32m8(
                           __riscv_vadd_vx_i32m8(absv, rnd, vl), S, vl);
        vint32m8_t v = __riscv_vsub_vv_i32m8(
                           __riscv_vxor_vv_i32m8(r, sign, vl), sign, vl);

        v = __riscv_vmax_vx_i32m8(v, lo, vl);
        v = __riscv_vmin_vx_i32m8(v, hi, vl);

        /* v is already clamped into int8 range -> plain narrow (no saturate) */
        vint16m4_t o16 = __riscv_vncvt_x_x_w_i16m4(v, vl);
        vint8m2_t  o8  = __riscv_vncvt_x_x_w_i8m2(o16, vl);
        __riscv_vse8_v_i8m2(output + i, o8, vl);
        i += vl; rem -= vl;
    }
}
