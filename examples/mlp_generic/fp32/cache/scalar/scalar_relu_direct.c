void kernel_relu(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}