void kernel_sigmoid(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}