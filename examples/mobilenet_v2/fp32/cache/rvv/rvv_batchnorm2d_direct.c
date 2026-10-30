void kernel_batchnorm2d(const float *input, const float *scale, const float *bias, float *output, int N, int C, int H, int W) {
    size_t vl;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            int idx_base = ((n*C)*H + h)*W;
            for (int w = 0; w < W; w += vl) {
                vl = __riscv_vsetvl_e32m4(W - w);
                vfloat32m4_t vin = __riscv_vle32_v_f32m4(input + idx_base + w, vl);
                for (int c = 0; c < C; c++) {
                    float s = scale[c];
                    float b = bias[c];
                    vfloat32m4_t vs = __riscv_vfmv_v_f_f32m4(s, vl);
                    vfloat32m4_t vb = __riscv_vfmv_v_f_f32m4(b, vl);
                    vfloat32m4_t vout = __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(vin, vs, vl), vb, vl);
                    __riscv_vse32_v_f32m4(output + idx_base + c*H*W + w, vout, vl);
                }
            }
        }
    }
}