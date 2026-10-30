void kernel_relu(const float *input, float *output, int n) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        v = __riscv_vfmax_vf_f32m1(v, 0.0f, vl);
        __riscv_vse32_v_f32m1(output + i, v, vl);
    }
}