void kernel_elu(const float *input, float *output, int n, float alpha) {
    for (int i = 0; i < n; i++) {
        float v = input[i];
        output[i] = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    }
}