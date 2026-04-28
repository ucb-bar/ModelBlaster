void kernel_adaptive_avg_pool2d(const float *input, float *output, int N, int C, int IH, int IW) {
    int n_per_chan = IH * IW;
    float inv = 1.0f / (float)n_per_chan;
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c += vlmax) {
            size_t vl = __riscv_vsetvl_e32m4(C - c);
            vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
            const float *src_base = input + (n*C + c) * IH * IW;
            int i = 0;
            for (; i < n_per_chan; i += 1) {
                const float *src = src_base + i;
                vfloat32m4_t v = __riscv_vle32_v_f32m4(src, vl);
                vacc = __riscv_vfadd_vv_f32m4(vacc, v, vl);
            }
            vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
            float acc = __riscv_vfmv_f_s_f32m1_f32(vsum) * inv;
            __riscv_vse32_v_f32m4(output + n*C + c, __riscv_vfmv_v_f_f32m4(acc, vl), vl);
        }
    }
}