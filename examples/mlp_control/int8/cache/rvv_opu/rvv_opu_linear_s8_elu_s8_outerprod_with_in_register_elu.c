void kernel_linear_s8_elu_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int linear_output_offset, int output_multiplier, int output_shift, int linear_activation_min, int linear_activation_max, float scale_linear_out, float scale_final_out, int activation_min, int activation_max, float alpha) {
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));

    if ((size_t)M > mlmax || (size_t)N > mlmax ||
        input_offset != 0 || filter_offset != 0) {
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                int32_t acc = bias ? bias[n] : 0;
                for (int k = 0; k < K; k++) {
                    int32_t iv = (int32_t)input[m*K+k] + input_offset;
                    int32_t wv = (int32_t)weight[n*K+k] + filter_offset;
                    acc += iv * wv;
                }
                int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                prod = (prod + (1LL << 30)) >> 31;
                int32_t scaled = (int32_t)prod;
                if (output_shift > 0) {
                    scaled = (scaled + (1 << (output_shift - 1))) >> output_shift;
                } else {
                    scaled = scaled << (-output_shift);
                }
                scaled += linear_output_offset;
                if (scaled < linear_activation_min) scaled = linear_activation_min;
                if (scaled > linear_activation_max) scaled = linear_activation_max;
                int8_t lin8 = (int8_t)scaled;
                float f = (float)lin8 * scale_linear_out;
                float y = (f > 0.0f) ? f : alpha * (expf(f) - 1.0f);
                int32_t v = (int32_t)roundf(y / scale_final_out);
                if (v < activation_min) v = activation_min;
                if (v > activation_max) v = activation_max;
                output[m*N+n] = (int8_t)v;
            }
        }
        return;
    }

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)N));
    if (bias) {
        asm volatile("vle32.v v0, (%0)" : : "r"(bias));
    } else {
        asm volatile("vmv.v.i v0, 0");
    }
    OPMVINBCAST(m1, v0);

    const ptrdiff_t in_stride = (ptrdiff_t)K * sizeof(int8_t);
    const ptrdiff_t w_stride = (ptrdiff_t)K * sizeof(int8_t);
    for (int k = 0; k < K; k++) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)M));
        asm volatile("vlse8.v v16, (%0), %1"
                     : : "r"(&input[k]), "r"((unsigned long)in_stride));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"((size_t)N));
        asm volatile("vlse8.v v18, (%0), %1"
                     : : "r"(&weight[k]), "r"((unsigned long)w_stride));
        VOPACC(m1, v18, v16);
    }

    int32_t row_i32[64];
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"((size_t)N));
    for (int m = 0; m < M; m++) {
        VMV_VR(v0, m, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(row_i32));
        for (int n = 0; n < N; n++) {
            int32_t acc = row_i32[n];
            int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
            prod = (prod + (1LL << 30)) >> 31;
            int32_t scaled = (int32_t)prod;
            if (output_shift > 0) {
                scaled = (scaled + (1 << (output_shift - 1))) >> output_shift;
            } else {
                scaled = scaled << (-output_shift);
            }
            scaled += linear_output_offset;
            if (scaled < linear_activation_min) scaled = linear_activation_min;
            if (scaled > linear_activation_max) scaled = linear_activation_max;
            int8_t lin8 = (int8_t)scaled;
            float f = (float)lin8 * scale_linear_out;
            float y = (f > 0.0f) ? f : alpha * (expf(f) - 1.0f);
            int32_t v = (int32_t)roundf(y / scale_final_out);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[m*N+n] = (int8_t)v;
        }
    }
}