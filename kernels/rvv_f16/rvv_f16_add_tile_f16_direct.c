/* source: curated */
/* algorithm: direct */
/* origin: RVV+Zvfh add of a repeated trailing block (a shared attention mask
 * against per-head scores). Both operands are unit-stride within the block,
 * so the whole thing is vfwcvt + vfadd.vv + vfncvt -- BIT_EXACT to the
 * reference's fp32 add and fp16 store.
 */

#include <riscv_vector.h>

void kernel_add_tile_f16(const _Float16 *tile, const _Float16 *x,
                         _Float16 *output, int OUTER, int INNER) {
    for (int o = 0; o < OUTER; o++) {
        const size_t base = (size_t)o * (size_t)INNER;
        int i = 0;
        while (i < INNER) {
            size_t vl = __riscv_vsetvl_e16m4((size_t)(INNER - i));
            vfloat16m4_t vt = __riscv_vle16_v_f16m4(tile + i, vl);
            vfloat16m4_t vx = __riscv_vle16_v_f16m4(x + base + i, vl);
            vfloat32m8_t vf = __riscv_vfwadd_vv_f32m8(vt, vx, vl);
            __riscv_vse16_v_f16m4(output + base + i,
                                  __riscv_vfncvt_f_f_w_f16m4(vf, vl), vl);
            i += (int)vl;
        }
    }
}
