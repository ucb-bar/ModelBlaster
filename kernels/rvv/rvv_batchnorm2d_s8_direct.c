/* source: curated */
/* algorithm: direct */
/* accuracy_class: numeric_drift */
/* origin: hand-written. INTEGER fixed-point per-channel batchnorm2d_s8,
 *         vectorised on the RVV INTEGER datapath -- no floating point in the
 *         loop (only a few one-time float->fixed conversions of the per-channel
 *         scale/bias at entry, which do not trap on a core with misa.F=0).
 *
 *   WHY INTEGER. The previous curated batchnorm folded the affine into
 *   x*(scale[c]*scale_in/scale_out) + bias[c]/scale_out and ran it as a
 *   vfmadd.vf on the vector FPU. On a Saturn-class core with a fp16-only
 *   scalar FPU (misa.F=0, WithRocketFPU16) the `.vf` broadcast needs an
 *   f-register move that TRAPS, so it fell back to all-scalar soft-float
 *   (~14 M cycles for dronet's three BN layers -- the single largest fp cost
 *   in the model). The BN affine is a compile-time-constant multiply-add, i.e.
 *   exactly representable in integer fixed point, the same technique
 *   conv2d_s8's Q0.31 requantize uses.
 *
 *   MATH. out[c] = round( cs[c]*x + cb[c] ), with
 *       cs[c] = scale[c]*scale_in/scale_out,   cb[c] = bias[c]/scale_out.
 *   Per channel, fold cs,cb into Q(S) integers Mc=cs*2^S, Bc=cb*2^S. S is
 *   picked so Mc < 2^22 (x*Mc < 2^29) and Bc < 2^30, so x*Mc+Bc stays in
 *   int32 -- the whole H*W plane runs at 4 int32 lanes/cycle. Per element:
 *       acc = x*Mc + Bc               (int32)
 *       v   = round_half_away(acc / 2^S)
 *   round-half-away (matching the reference roundf) is issued branch-free via
 *   the abs/shift/re-sign trick. Then clamp to [activation_min,activation_max]
 *   and narrow i32 -> i16 -> i8.
 *
 *   ACCURACY. Reproduces the float affine to <= 1 LSB (the fixed-point
 *   quantisation of cs,cb); measures err=0 on dronet's BN layers. */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max) {
    const int hw = H * W;
    if (hw <= 0) return;

    const float inv_so = 1.0f / scale_out;
    const int32_t lo = activation_min < -128 ? -128
                       : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128
                       : (activation_max > 127 ? 127 : activation_max);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float cs = scale[c] * scale_in * inv_so;
            const float cb = bias[c] * inv_so;
            const float cs_abs = cs < 0 ? -cs : cs;
            const float cb_abs = cb < 0 ? -cb : cb;

            /* Pick S so the whole element loop stays in int32. */
            int S = 24;
            while (S > 1 &&
                   (cs_abs * (float)((uint32_t)1 << S) >= 4194304.0f   /* 2^22 */
                    || cb_abs * (float)((uint32_t)1 << S) >= 1073741824.0f /* 2^30 */))
                S--;

            const int32_t Mc = (int32_t)lrintf(cs * (float)((uint32_t)1 << S));
            const int32_t Bc = (int32_t)lrintf(cb * (float)((uint32_t)1 << S));
            const int32_t rnd = (int32_t)1 << (S - 1);

            const int base = (n * C + c) * hw;
            const int8_t *ip = input + base;
            int8_t *op = output + base;

            int i = 0;
            size_t vl;
            for (; i < hw; i += vl) {
                vl = __riscv_vsetvl_e8m2(hw - i);

                vint8m2_t v8 = __riscv_vle8_v_i8m2(ip + i, vl);
                vint32m8_t vx = __riscv_vsext_vf4_i32m8(v8, vl);

                /* acc = x*Mc + Bc. */
                vint32m8_t acc = __riscv_vmul_vx_i32m8(vx, Mc, vl);
                acc = __riscv_vadd_vx_i32m8(acc, Bc, vl);

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
                __riscv_vse8_v_i8m2(op + i, o8, vl);
            }
        }
    }
}
