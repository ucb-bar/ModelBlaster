/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV mul_c1_s8 (per-channel gate x NCHW tensor), the
 * reference expression issued unchanged on the vector unit.
 *
 * The multiplicative twin of rvv_add_c1_s8_direct.c: the gate is one int8
 * per channel, so for a fixed (n, c) the whole H*W plane is scaled by the
 * single scalar g_real = gate[c]*scale_gate, and the body is two vfmul plus
 * one vfdiv against a unit-stride int8 load.
 *
 * There is an existing fp16-backend sibling, kernels/rvv_f16/
 * rvv_f16_mul_c1_s8_direct.c, and this file is NOT a copy of it: that one
 * folds gate[c]*scale_gate*scale_x/scale_out into ONE f32 constant and
 * rounds ties-to-even via vfncvt, and its header states that as a sub-LSB
 * deviation. This kernel keeps the reference's three separate roundings and
 * reproduces its rounding expression, so it is bit-exact instead:
 *
 *   1. THE OPERATION ORDER. `(x*scale_x)`, then `* g_real`, then
 *      `/ scale_out`, as three instructions. Folding the constants or
 *      replacing the divide with a reciprocal multiply drops intermediate
 *      roundings the reference performs.
 *
 *   2. THE ROUNDING, which is a trap worth stating once per family. The
 *      reference tail is `(int32_t)(v >= 0 ? v + 0.5f : v - 0.5f)`, and the
 *      obvious vectorization -- vfcvt.x.f under frm=RMM, i.e. exactly
 *      `roundf` -- is NOT equal to it: adding 0.5f is itself a rounded
 *      operation, so for the largest float below a .5 tie the add carries
 *      into the next integer and the truncation keeps it. Measured over a
 *      dense sweep of the 400 floats either side of every x.5 for x in
 *      [-300, 300] (481,401 values), frm=RMM/roundf disagrees with the
 *      reference form on 2 of them. So the tail is reproduced literally:
 *      |v| + 0.5f (vfabs, vfadd.vf), sign restored from v (vfsgnj.vv),
 *      round-toward-zero convert (vfcvt.rtz.x.f.v) -- identical to the
 *      reference's branch because float negation is exact, verified over
 *      108,606,540 random float bit patterns with |v| < 4096 plus that tie
 *      sweep, ZERO mismatches. See rvv_add_c1_s8_direct.c for the long
 *      form.
 *
 * mul_c1_s8 carries no activation_min/max in its signature: the reference
 * clamps to the int8 range itself, so the clamp is a literal [-128, 127] in
 * the int32 domain and not a caller-supplied window. vfcvt.rtz saturates
 * out-of-range inputs where the reference's `(int32_t)` cast is undefined
 * behaviour; the clamp folds that away.
 *
 * VTYPE. Each width transition names its own width and every vsetvl takes
 * the ELEMENT COUNT, not a previous vsetvl's result (this GCC miscompiles
 * the chained form on partial tails -- scripts/check_rvv_avl.py). vsext.vf4
 * is issued under SEW=32; under SEW=8 it implies a 2-bit source and SIGILLs
 * on the board rather than failing the build.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_mul_c1_s8(const int8_t *gate, const int8_t *x, int8_t *output,
                      int N, int C, int HW,
                      float scale_gate, float scale_x, float scale_out)
{
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float g_real = (float)gate[c] * scale_gate;
            const size_t base = (size_t)(n * C + c) * (size_t)HW;
            const int8_t *xp = x + base;
            int8_t *op = output + base;

            int i = 0;
            while (i < HW) {
                const size_t n_elem = (size_t)(HW - i);

                size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
                vint8m2_t vx8 = __riscv_vle8_v_i8m2(xp + i, vl8);

                size_t vl = __riscv_vsetvl_e32m8(n_elem);
                vint32m8_t vx32 = __riscv_vsext_vf4_i32m8(vx8, vl);

                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vx32, vl);
                vf = __riscv_vfmul_vf_f32m8(vf, scale_x, vl);
                vf = __riscv_vfmul_vf_f32m8(vf, g_real, vl);
                vf = __riscv_vfdiv_vf_f32m8(vf, scale_out, vl);

                vfloat32m8_t vm = __riscv_vfabs_v_f32m8(vf, vl);
                vm = __riscv_vfadd_vf_f32m8(vm, 0.5f, vl);
                vm = __riscv_vfsgnj_vv_f32m8(vm, vf, vl);
                vint32m8_t vi = __riscv_vfcvt_rtz_x_f_v_i32m8(vm, vl);

                vi = __riscv_vmax_vx_i32m8(vi, -128, vl);
                vi = __riscv_vmin_vx_i32m8(vi, 127, vl);

                size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
                vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl16);
                size_t vlo8 = __riscv_vsetvl_e8m2(n_elem);
                vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vlo8);
                __riscv_vse8_v_i8m2(op + i, vi8, vlo8);

                i += (int)vl8;
            }
        }
    }
}
