void kernel_sigmoid_s8(const int8_t *input, int8_t *output, int n,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max) {
    for (int i = 0; i < n; i++) {
        float fv = (float)input[i] * scale_in;
        float sig = 1.0f / (1.0f + expf(-fv));
        int32_t v = (int32_t)roundf(sig / scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}

