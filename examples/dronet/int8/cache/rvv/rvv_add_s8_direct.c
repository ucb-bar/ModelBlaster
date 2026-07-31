void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    for (int i = 0; i < n; i++) {
        float fa = (float)a[i] * scale_a;
        float fb = (float)b[i] * scale_b;
        float fout = (fa + fb) / scale_out;
        int32_t v = (int32_t)roundf(fout);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}

