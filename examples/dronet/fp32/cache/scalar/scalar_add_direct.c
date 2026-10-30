void kernel_add(const float *a, const float *b, float *output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = a[i] + b[i];
    }
}