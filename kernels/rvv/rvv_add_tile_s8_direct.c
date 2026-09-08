/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV add_tile_s8 (a repeated trailing block added to a
 * larger tensor), the reference expression issued unchanged on the vector
 * unit.
 *
 * `tile` holds INNER elements and is re-read for every one of the OUTER
 * repeats; `x` and `output` are one contiguous OUTER*INNER run. So the
 * vectorized axis is INNER, and BOTH operand streams are unit-stride inside
 * it -- there is no gather anywhere in this kernel, which is the whole
 * reason the op exists (lowering the same add through add_s8 would walk
 * OUTER*INNER elements of an INNER-element buffer).
 *
 * The tile is NOT pre-dequantized into an f32 scratch buffer even though it
 * is re-read OUTER times. That would trade INNER int8 byte-loads per repeat
 * for INNER*4 f32 byte-loads per repeat, and spike charges vector loads
 * roughly per BYTE (notes/curated_rvv_kernels.md, variants #1/#3/#4) -- so
 * the "obvious" hoist is a 4x load-traffic regression on the operand that
 * dominates the loop. Re-converting from int8 each pass is cheaper AND it is
 * what keeps the arithmetic identical to the reference.
 *
 * BIT-EXACT, on the same two grounds as rvv_add_c1_s8_direct.c:
 *
 *   1. THE OPERATION ORDER. Two vfmul, one vfadd, one vfdiv, each issued
 *      separately. Folding scale_tile/scale_x/scale_out into two constants,
 *      or contracting the scale-and-add into a vfmacc, would drop
 *      intermediate roundings the reference performs; and the divide stays
 *      a divide rather than a reciprocal multiply for the same reason.
 *
 *   2. THE ROUNDING. The reference tail is
 *      `(int32_t)(v >= 0 ? v + 0.5f : v - 0.5f)`, which is NOT `roundf`:
 *      adding 0.5f is itself a rounded operation, so for the largest float
 *      below a .5 tie the add carries into the next integer and the
 *      truncation keeps it. Measured over the 400 floats either side of
 *      every x.5 for x in [-300, 300] (481,401 values), frm=RMM/roundf --
 *      the obvious vectorization -- disagrees with the reference form on 2
 *      of them. So the tail is reproduced literally: |v| + 0.5f (vfabs,
 *      vfadd.vf), sign restored from v (vfsgnj.vv), round-toward-zero
 *      convert (vfcvt.rtz.x.f.v); identical to the reference's branch
 *      because float negation is exact, verified over 108,606,540 random
 *      float bit patterns with |v| < 4096 plus that tie sweep, ZERO
 *      mismatches. See rvv_add_c1_s8_direct.c for the long form.
 *
 * The clamp runs in int32 against the caller's activation_min/max, which may
 * be tighter than the int8 range, so it cannot be left to a saturating
 * narrow; vfcvt.rtz saturates out-of-range values where the reference's cast
 * is undefined behaviour, and the clamp folds that away.
 *
 * VTYPE. Every width transition names its own width and every vsetvl gets
 * the ELEMENT COUNT, never a previous vsetvl's result (this GCC miscompiles
 * the chained form on partial tails -- scripts/check_rvv_avl.py). vsext.vf4
 * is issued under SEW=32; under SEW=8 it implies a 2-bit source and is a
 * SIGILL on the board, not a build error.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_add_tile_s8(const int8_t *tile, const int8_t *x, int8_t *output,
                        int OUTER, int INNER,
                        float scale_tile, float scale_x, float scale_out,
                        int activation_min, int activation_max)
{
    for (int o = 0; o < OUTER; o++) {
        const size_t base = (size_t)o * (size_t)INNER;
        const int8_t *xp = x + base;
        int8_t *op = output + base;

        int i = 0;
        while (i < INNER) {
            const size_t n_elem = (size_t)(INNER - i);

            size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
            vint8m2_t vt8 = __riscv_vle8_v_i8m2(tile + i, vl8);
            vint8m2_t vx8 = __riscv_vle8_v_i8m2(xp + i, vl8);

            size_t vl = __riscv_vsetvl_e32m8(n_elem);
            vint32m8_t vt32 = __riscv_vsext_vf4_i32m8(vt8, vl);
            vfloat32m8_t vft = __riscv_vfcvt_f_x_v_f32m8(vt32, vl);
            vft = __riscv_vfmul_vf_f32m8(vft, scale_tile, vl);

            vint32m8_t vx32 = __riscv_vsext_vf4_i32m8(vx8, vl);
            vfloat32m8_t vfx = __riscv_vfcvt_f_x_v_f32m8(vx32, vl);
            vfx = __riscv_vfmul_vf_f32m8(vfx, scale_x, vl);

            vfloat32m8_t vf = __riscv_vfadd_vv_f32m8(vft, vfx, vl);
            vf = __riscv_vfdiv_vf_f32m8(vf, scale_out, vl);

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
