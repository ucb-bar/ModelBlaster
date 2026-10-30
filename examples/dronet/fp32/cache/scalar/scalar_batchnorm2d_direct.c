void kernel_batchnorm2d(const float *input, const float *scale, const float *bias, float *output, int N, int C, int H, int W) {
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    int idx = ((n*C + c)*H + h)*W + w;
                    output[idx] = s * input[idx] + b;
                }
            }
        }
    }
}