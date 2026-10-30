void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            size_t vlmax = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t vacc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
            const float *in_row  = input  + m * K;
            const float *w_row   = weight + n * K;
            size_t k = 0;
            for (; k + vlmax <= K; k += vlmax) {
                vfloat32m4_t va = __riscv_vle32_v_f32m4(in_row + k, vlmax);
                vfloat32m4_t vb = __riscv_vle32_v_f32m4(w_row + k, vlmax);
                vacc = __riscv_vfmacc_vv_f32m4(vacc, va, vb, vlmax);
            }
            vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m4_f32m1(
                vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlmax);
            float acc = __riscv_vfmv_f_s_f32m1_f32(vsum);
            if (bias) acc += bias[n];
            output[m * N + n] = acc;
        }
    }
}