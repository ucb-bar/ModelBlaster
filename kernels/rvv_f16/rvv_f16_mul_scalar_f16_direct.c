/* source: curated */
/* algorithm: direct */
/* origin: RVV+Zvfh multiply by a compile-time scalar (the 1/sqrt(d_head) in a
 * hand-written attention). vfwcvt + vfmul.vf + vfncvt, unit stride
 * throughout -- BIT_EXACT to the reference's fp32 multiply and fp16 store.
 */

#include <riscv_vector.h>

void kernel_mul_scalar_f16(const _Float16 *input, _Float16 *output,
                           int n, float s) {
    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e16m4((size_t)(n - i));
        vfloat16m4_t v = __riscv_vle16_v_f16m4(input + i, vl);
        vfloat32m8_t vf = __riscv_vfwcvt_f_f_v_f32m8(v, vl);
        vf = __riscv_vfmul_vf_f32m8(vf, s, vl);
        __riscv_vse16_v_f16m4(output + i,
                              __riscv_vfncvt_f_f_w_f16m4(vf, vl), vl);
        i += (int)vl;
    }
}
