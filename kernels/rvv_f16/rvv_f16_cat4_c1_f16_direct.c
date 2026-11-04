/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: RVV cat4_c1_f16 -- the same shape as rvv_f16_cat2_c1_f16_direct
 * with two more source planes. Each plane is a contiguous run, so this is
 * four vector copies per batch at eew=16 / LMUL=8 instead of four calls into
 * Zephyr's minimal-libc memcpy. Copies only, so bit-exact.
 *
 * Worth having even though the reference is already memcpy: cat2_c1_f16's
 * curated kernel measured 66x against exactly that memcpy on this target
 * (7.5M -> 0.1M cycles), so the library copy is not vector here. cat4 is one
 * dispatch and 1.6 of octo-small's 541 Mcycles, i.e. this closes the last
 * op in the model rather than moving the total much.
 */

#include <riscv_vector.h>

static inline void mb_cat4f16_copy(const _Float16 *src, _Float16 *dst,
                                   size_t n) {
    size_t i = 0, vl;
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m8(n - i);
        __riscv_vse16_v_f16m8(dst + i, __riscv_vle16_v_f16m8(src + i, vl), vl);
    }
}

void kernel_cat4_c1_f16(const _Float16 *in0, int c0,
                        const _Float16 *in1, int c1,
                        const _Float16 *in2, int c2,
                        const _Float16 *in3, int c3,
                        _Float16 *out, int N, int H, int W) {
    const size_t HW = (size_t)H * (size_t)W;
    const size_t Cout = (size_t)(c0 + c1 + c2 + c3);
    for (int n = 0; n < N; n++) {
        _Float16 *dst = out + (size_t)n * Cout * HW;
        mb_cat4f16_copy(in0 + (size_t)n * (size_t)c0 * HW, dst,
                        (size_t)c0 * HW);
        dst += (size_t)c0 * HW;
        mb_cat4f16_copy(in1 + (size_t)n * (size_t)c1 * HW, dst,
                        (size_t)c1 * HW);
        dst += (size_t)c1 * HW;
        mb_cat4f16_copy(in2 + (size_t)n * (size_t)c2 * HW, dst,
                        (size_t)c2 * HW);
        dst += (size_t)c2 * HW;
        mb_cat4f16_copy(in3 + (size_t)n * (size_t)c3 * HW, dst,
                        (size_t)c3 * HW);
    }
}
