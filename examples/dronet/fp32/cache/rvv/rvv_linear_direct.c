void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            size_t vl;
            vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            const float *in_row  = input  + m * K;
            const float *w_row   = weight + n * K;
            for (int k = 0; k < K; k += vl) {
                vl = __riscv_vsetvl_e32m1(K - k);
                vfloat32m1_t va = __riscv_vle32_v_f32m1(in_row + k, vl);
                vfloat32m1_t vb = __riscv_vle32_v_f32m1(w_row + k, vl);
                vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vb, vl);
            }
            vfloat32m1_t vsum = __riscv_vfredusum_vs_f32m1_f32m1(
                vacc, __riscv_vfmv_s_f_f32m1(0.0f, 1), __riscv_vsetvlmax_e32m1());
            float acc = __riscv_vfmv_f_s_f32m1_f32(vsum);
            if (bias) acc += bias[n];
            output[m * N + n] = acc;
        }
    }
}