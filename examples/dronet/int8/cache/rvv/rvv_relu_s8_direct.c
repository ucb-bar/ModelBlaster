void kernel_relu_s8(const int8_t *input, int8_t *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m1(n - i);
        vint8m1_t v = __riscv_vle8_v_i8m1(input + i, vl);
        v = __riscv_vmax_vx_i8m1(v, 0, vl);
        __riscv_vse8_v_i8m1(output + i, v, vl);
    }
}