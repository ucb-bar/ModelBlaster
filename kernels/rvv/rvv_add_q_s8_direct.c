/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: hand-written. Folded integer add (pipeline/fold_quant_consts.py): the
 *         rvv_frm_rmm add with its (ma, mb, S) computed at build time instead of
 *         in soft-float on every call, and a shorter rounding sequence.
 *
 *   ROUNDING. The reference computes v = sign(acc) * ((|acc| + rnd) >> S), rnd = 2^(S-1).
 *     acc >= 0:  v = floor((acc + rnd) / 2^S)
 *     acc <  0:  v = -floor((-acc + rnd) / 2^S) = ceil((acc - rnd) / 2^S)
 *                  = floor((acc - rnd + 2^S - 1) / 2^S) = floor((acc + rnd - 1) / 2^S)
 *   so v = (acc + rnd + (acc >> 31)) >> S with an arithmetic shift. vnclip with
 *   vxrm = rdn is that shift, saturating to int16; a second vnclip by 0 saturates
 *   to int8, and the clamp to [lo, hi] (inside int8) then equals the reference's
 *   clamp. 8 m8 operations per strip instead of the 13 of abs/shift/re-sign.
 *   fold_quant_consts guarantees 128*(|ma|+|mb|) + rnd < 2^31. */
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_add_q_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                     int ma, int mb, int S, int activation_min, int activation_max) {
    const int32_t lo = activation_min < -128 ? -128 : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128 : (activation_max > 127 ? 127 : activation_max);
    const int32_t rnd = (int32_t)1 << (S - 1);
    size_t vl;
    for (int i = 0; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e8m2((size_t)(n - i));
        vint32m8_t va = __riscv_vsext_vf4_i32m8(__riscv_vle8_v_i8m2(a + i, vl), vl);
        vint32m8_t vb = __riscv_vsext_vf4_i32m8(__riscv_vle8_v_i8m2(b + i, vl), vl);
        vint32m8_t acc = __riscv_vmul_vx_i32m8(va, ma, vl);
        acc = __riscv_vmacc_vx_i32m8(acc, mb, vb, vl);
        vint32m8_t sgn = __riscv_vsra_vx_i32m8(acc, 31, vl);
        acc = __riscv_vadd_vv_i32m8(__riscv_vadd_vx_i32m8(acc, rnd, vl), sgn, vl);
        vint16m4_t v16 = __riscv_vnclip_wx_i16m4(acc, (size_t)S, __RISCV_VXRM_RDN, vl);
        vint8m2_t v8 = __riscv_vnclip_wx_i8m2(v16, 0, __RISCV_VXRM_RDN, vl);
        v8 = __riscv_vmax_vx_i8m2(v8, (int8_t)lo, vl);
        v8 = __riscv_vmin_vx_i8m2(v8, (int8_t)hi, vl);
        __riscv_vse8_v_i8m2(output + i, v8, vl);
    }
}
