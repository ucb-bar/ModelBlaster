void kernel_elu(const float *input, float *output, int n, float alpha) {
    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vbool32_t mask = __riscv_vmflt_vf_f32m1_b32(v, 0.0f, vl);
        for (int j = 0; j < vl; j++) {
            float x = input[i + j];
            output[i + j] = x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
        }
    }
}