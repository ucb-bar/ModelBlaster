void kernel_linear(const float *input, const float *weight, const float *bias, float *output, int M, int K, int N) {
    int has_bias = bias != NULL;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n + 4 <= N; n += 4) {
            float acc0 = has_bias ? bias[n+0] : 0.0f;
            float acc1 = has_bias ? bias[n+1] : 0.0f;
            float acc2 = has_bias ? bias[n+2] : 0.0f;
            float acc3 = has_bias ? bias[n+3] : 0.0f;
            for (int k = 0; k < K; k++) {
                float x = input[m * K + k];
                acc0 += x * weight[(n+0) * K + k];
                acc1 += x * weight[(n+1) * K + k];
                acc2 += x * weight[(n+2) * K + k];
                acc3 += x * weight[(n+3) * K + k];
            }
            output[m * N + n+0] = acc0;
            output[m * N + n+1] = acc1;
            output[m * N + n+2] = acc2;
            output[m * N + n+3] = acc3;
        }
        for (int n = N & ~3; n < N; n++) {
            float acc = has_bias ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                acc += input[m * K + k] * weight[n * K + k];
            }
            output[m * N + n] = acc;
        }
    }
}