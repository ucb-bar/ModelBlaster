/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV fp16 three-way channel concat. Unlike the int8
 *   cat3_c1_s8 this does NOT requantize -- fp16 inputs and output share one
 *   representation, so the op is a pure copy and every element is preserved
 *   exactly. (That absence of a requantize is the whole reason the fused
 *   sensor net's fuse is promoted to fp16: the int8 version has to squeeze the
 *   512 vision features, max-abs 0.24, onto an output scale of 1.72 set by the
 *   optical-flow component of the low-dimensional state vector, which rounds
 *   every one of them to zero.)
 *
 *   Vectorized with unit-stride e16m4 load/store rather than memcpy so the op
 *   is a vector kernel under a vector-labelled build, and so the profile's
 *   `implementation` column records a curated kernel instead of the scalar
 *   reference (see scripts/check_kernel_coverage.py). Single SEW throughout,
 *   so no vtype-carry hazard. */

#include <stddef.h>
#include <riscv_vector.h>

static inline void mb_copy_f16(const _Float16 *src, _Float16 *dst, int n) {
    int i = 0;
    size_t vl;
    for (; i < n; i += (int)vl) {
        vl = __riscv_vsetvl_e16m4(n - i);
        __riscv_vse16_v_f16m4(dst + i, __riscv_vle16_v_f16m4(src + i, vl), vl);
    }
}

void kernel_cat3_c1_f16(const _Float16 *in0, int c0,
                        const _Float16 *in1, int c1,
                        const _Float16 *in2, int c2,
                        _Float16 *out, int N, int H, int W) {
    const int hw = H * W;
    const int c_total = c0 + c1 + c2;
    for (int n = 0; n < N; n++) {
        _Float16 *dst = out + (size_t)n * c_total * hw;
        mb_copy_f16(in0 + (size_t)n * c0 * hw, dst, c0 * hw);
        dst += c0 * hw;
        mb_copy_f16(in1 + (size_t)n * c1 * hw, dst, c1 * hw);
        dst += c1 * hw;
        mb_copy_f16(in2 + (size_t)n * c2 * hw, dst, c2 * hw);
    }
}
