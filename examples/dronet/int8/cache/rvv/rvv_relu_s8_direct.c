void kernel_relu_s8(const int8_t *input, int8_t *output, int n) {
    for (int i = 0; i < n; i++) {
        int8_t v = input[i];
        output[i] = v > 0 ? v : 0;
    }
}

