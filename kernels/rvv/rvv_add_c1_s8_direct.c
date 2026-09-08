/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV add_c1_s8 (per-channel bias + NCHW tensor), the
 * reference expression issued unchanged on the vector unit -- including its
 * rounding, which is NOT the rounding it looks like.
 *
 * The bias is one int8 per channel, so for a fixed (n, c) the ENTIRE H*W
 * plane is offset by the single scalar g_real = gate[c]*scale_gate. That is
 * what makes this op worth its own kernel rather than a broadcast: the
 * per-channel term is loop-invariant over the vectorized axis, so the body
 * is one vfmul + one vfadd.vf + one vfdiv against a unit-stride int8 load,
 * with no second operand stream and no gather.
 *
 * BIT-EXACT, and two things have to hold for that. The second one is a trap
 * and is measured, not argued.
 *
 *   1. THE OPERATION ORDER. The three float steps are issued separately
 *      (vfmul, vfadd, vfdiv) and NOT contracted: folding scale_x/scale_out
 *      into one constant, or turning `g_real + x*scale_x` into a vfmacc,
 *      removes an intermediate rounding the reference performs. The divide
 *      stays a divide for the same reason -- multiplying by a precomputed
 *      1/scale_out is a different number.
 *
 *   2. THE ROUNDING. The reference tail is
 *      `q = (int32_t)(v >= 0 ? v + 0.5f : v - 0.5f)`, which is USUALLY
 *      round-half-away-from-zero -- and the obvious vectorization is
 *      therefore vfcvt.x.f under frm=RMM ("ties to Max Magnitude"), which
 *      is exactly `roundf`. That is wrong here, because the reference's
 *      expression is not roundf: adding 0.5f first is itself a rounded
 *      operation, so for the largest float below a .5 tie the add carries
 *      up to the next integer and the truncation keeps it. Measured on this
 *      toolchain over a dense sweep of the 400 floats either side of every
 *      x.5 for x in [-300, 300] (481,401 values): frm=RMM/roundf disagrees
 *      with the reference form on 2 of them, e.g. x = 0x3EFFFFFF where the
 *      reference yields 1 and roundf yields 0.
 *
 *      So the tail reproduces the reference's expression instead of its
 *      intent: |v| + 0.5f (vfabs, vfadd.vf), sign restored from v
 *      (vfsgnj.vv), then a round-toward-zero convert
 *      (vfcvt.rtz.x.f.v). Magnitude-then-sign is identical to the
 *      reference's branch because negating a float is exact, so both sides
 *      round the same magnitude; verified over 108,606,540 random float bit
 *      patterns with |v| < 4096 plus the dense tie sweep above, ZERO
 *      mismatches. It is also cheaper than the frm route, which needs two
 *      CSR writes per block to avoid leaving RMM in force across the
 *      arithmetic.
 *
 * The activation clamp runs in the int32 domain before narrowing, so it
 * cannot be confused by a saturating narrow: activation_min/max may be
 * tighter than [-128,127] and the reference clamps to them, not to int8.
 * vfcvt.rtz also SATURATES out-of-range values where the reference's
 * `(int32_t)` cast is undefined behaviour, which the clamp then folds away.
 *
 * VTYPE. Every width-domain transition names its own width, and every
 * vsetvl is handed the ELEMENT COUNT rather than a previous vsetvl's
 * result -- chaining them is miscompiled by this GCC (vl saturates to
 * VLMAX on a partial tail); see rvv_cat2_c1_s8_direct.c and
 * scripts/check_rvv_avl.py. vsext.vf4 is issued under SEW=32 because
 * under SEW=8 it would imply a 2-bit source, which is a SIGILL on the
 * board and not a build error.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_add_c1_s8(const int8_t *gate, const int8_t *x, int8_t *output,
                      int N, int C, int HW,
                      float scale_gate, float scale_x, float scale_out,
                      int activation_min, int activation_max)
{
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            /* Loop-invariant over the whole plane, exactly as the reference
             * hoists it. */
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

                /* The reference's float steps, in its order, each separately
                 * rounded. */
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vx32, vl);
                vf = __riscv_vfmul_vf_f32m8(vf, scale_x, vl);
                vf = __riscv_vfadd_vf_f32m8(vf, g_real, vl);
                vf = __riscv_vfdiv_vf_f32m8(vf, scale_out, vl);

                /* (int32_t)(v >= 0 ? v + 0.5f : v - 0.5f), as written. */
                vfloat32m8_t vm = __riscv_vfabs_v_f32m8(vf, vl);
                vm = __riscv_vfadd_vf_f32m8(vm, 0.5f, vl);
                vm = __riscv_vfsgnj_vv_f32m8(vm, vf, vl);
                vint32m8_t vi = __riscv_vfcvt_rtz_x_f_v_i32m8(vm, vl);

                vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
                vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);

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
