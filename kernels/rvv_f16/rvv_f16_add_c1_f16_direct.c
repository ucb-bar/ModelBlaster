/* source: curated */
/* algorithm: direct */
/* origin: RVV+Zvfh channel-broadcast add. The per-channel bias is a SCALAR
 * for the whole H*W plane, so it goes in as vfadd.vf and the plane is a
 * unit-stride load/store -- BIT_EXACT to the reference, which does the same
 * add in fp32 and casts at the store (widen/narrow round-trips through fp32
 * are exact for a single add).
 */

#include <riscv_vector.h>

void kernel_add_c1_f16(const _Float16 *gate, const _Float16 *x,
                       _Float16 *output, int N, int C, int HW) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const float g = (float)gate[c];
            const size_t base = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW;
            int i = 0;
            while (i < HW) {
                size_t vl = __riscv_vsetvl_e16m4((size_t)(HW - i));
                vfloat16m4_t v = __riscv_vle16_v_f16m4(x + base + i, vl);
                vfloat32m8_t vf = __riscv_vfwcvt_f_f_v_f32m8(v, vl);
                vf = __riscv_vfadd_vf_f32m8(vf, g, vl);
                __riscv_vse16_v_f16m4(output + base + i,
                                      __riscv_vfncvt_f_f_w_f16m4(vf, vl), vl);
                i += (int)vl;
            }
        }
    }
}
