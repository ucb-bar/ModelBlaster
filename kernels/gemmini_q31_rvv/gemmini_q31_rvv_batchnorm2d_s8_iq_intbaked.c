/* source: curated */
/* algorithm: iq_intbaked */
/* accuracy_class: bit_exact */
/* origin: hand-written. INTEGER-BAKED per-channel batchnorm2d_s8. Identical
 *         element loop to gemmini_q31_rvv_batchnorm2d_s8_direct.c, but the
 *         per-channel fixed-point params (Mc, Bc, S) are BAKED AT CODEGEN and
 *         passed in as int arrays -- so there is ZERO floating point in the
 *         kernel, not even the one-time per-channel float->fixed setup.
 *
 *   WHY. The `direct` kernel is already integer in the element loop, but it
 *   still derives Mc[c]/Bc[c]/S[c] from the baked FLOAT scale/bias/scale_in/
 *   scale_out at entry (cs/cb multiplies, an iterative S-search, lrintf). On a
 *   misa.F=0 core those become soft-float libcalls -- TACIT measured the three
 *   dronet BN setups at ~2.2% of the whole inference (__mulsf3/__floatunsisf/
 *   lrintf/etc). Those constants are compile-time known, so the whole float->
 *   fixed decomposition is done once in Python at codegen (bit-identical: same
 *   float32 arithmetic, same round-to-nearest lrintf) and only Mc/Bc/S ship.
 *
 *   BIT-EXACT vs the `direct` kernel: same Mc/Bc/S values, same
 *   acc = x*Mc + Bc ; round_half_away(acc >> S) ; clamp ; narrow. */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_batchnorm2d_s8_iq(const int8_t *input,
                              const int32_t *Mc, const int32_t *Bc,
                              const int8_t *Sh, int8_t *output,
                              int N, int C, int H, int W,
                              int activation_min, int activation_max) {
    const int hw = H * W;
    if (hw <= 0) return;

    const int32_t lo = activation_min < -128 ? -128
                       : (activation_min > 127 ? 127 : activation_min);
    const int32_t hi = activation_max < -128 ? -128
                       : (activation_max > 127 ? 127 : activation_max);

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int32_t mc = Mc[c];
            const int32_t bc = Bc[c];
            const int      S = (int)Sh[c];
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
                vint32m8_t acc = __riscv_vmul_vx_i32m8(vx, mc, vl);
                acc = __riscv_vadd_vx_i32m8(acc, bc, vl);

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
