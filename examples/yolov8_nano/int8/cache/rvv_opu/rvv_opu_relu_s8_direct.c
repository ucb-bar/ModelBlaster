/* source: curated (resurrected v25) */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: Vectorized RVV relu using vmax_vx i8m8. Restored from
 *         kernels/rvv/rvv_relu_s8_direct.c — there was no rvv_opu-
 *         named variant before, so dronet/yolov8 relu_s8 dispatches
 *         on rvv_opu were falling back to the scalar reference impl
 *         (~9 cycles per element). LMUL=8 gives ~256 elements per
 *         vector op, so n=23328 takes ~91 iterations vs 23328 scalar
 *         loops — expected ~50x kernel-cycle reduction. */
void kernel_relu_s8(const int8_t *input, int8_t *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m8(n - i);
        vint8m8_t v = __riscv_vle8_v_i8m8(input + i, vl);
        v = __riscv_vmax_vx_i8m8(v, 0, vl);
        __riscv_vse8_v_i8m8(output + i, v, vl);
    }
}
