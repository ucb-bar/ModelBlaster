void kernel_elu_s8(const int8_t *input, int8_t *output, int n,
                   float scale_in, float scale_out,
                   int activation_min, int activation_max, float alpha) {
    for (int i = 0; i < n; i++) {
        float f = (float)input[i] * scale_in;
        float y = (f > 0.0f) ? f : alpha * (expf(f) - 1.0f);
        int32_t v = (int32_t)roundf(y / scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}