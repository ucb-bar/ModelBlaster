void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    const float inv_scale_out = 1.0f / scale_out;
    const int8_t clamp_min = (int8_t)activation_min;
    const int8_t clamp_max = (int8_t)activation_max;
    
    size_t vl;
    int i = 0;
    
    for (; i + 8 <= n; i += 8) {
        float f0 = (float)input[i + 0] * scale_in;
        float f1 = (float)input[i + 1] * scale_in;
        float f2 = (float)input[i + 2] * scale_in;
        float f3 = (float)input[i + 3] * scale_in;
        float f4 = (float)input[i + 4] * scale_in;
        float f5 = (float)input[i + 5] * scale_in;
        float f6 = (float)input[i + 6] * scale_in;
        float f7 = (float)input[i + 7] * scale_in;
        
        float y0 = f0 / (1.0f + expf(-f0));
        float y1 = f1 / (1.0f + expf(-f1));
        float y2 = f2 / (1.0f + expf(-f2));
        float y3 = f3 / (1.0f + expf(-f3));
        float y4 = f4 / (1.0f + expf(-f4));
        float y5 = f5 / (1.0f + expf(-f5));
        float y6 = f6 / (1.0f + expf(-f6));
        float y7 = f7 / (1.0f + expf(-f7));
        
        int32_t v0 = (int32_t)roundf(y0 * inv_scale_out);
        int32_t v1 = (int32_t)roundf(y1 * inv_scale_out);
        int32_t v2 = (int32_t)roundf(y2 * inv_scale_out);
        int32_t v3 = (int32_t)roundf(y3 * inv_scale_out);
        int32_t v4 = (int32_t)roundf(y4 * inv_scale_out);
        int32_t v5 = (int32_t)roundf(y5 * inv_scale_out);
        int32_t v6 = (int32_t)roundf(y6 * inv_scale_out);
        int32_t v7 = (int32_t)roundf(y7 * inv_scale_out);
        
        if (v0 < activation_min) v0 = activation_min;
        if (v0 > activation_max) v0 = activation_max;
        if (v1 < activation_min) v1 = activation_min;
        if (v1 > activation_max) v1 = activation_max;
        if (v2 < activation_min) v2 = activation_min;
        if (v2 > activation_max) v2 = activation_max;
        if (v3 < activation_min) v3 = activation_min;
        if (v3 > activation_max) v3 = activation_max;
        if (v4 < activation_min) v4 = activation_min;
        if (v4 > activation_max) v4 = activation_max;
        if (v5 < activation_min) v5 = activation_min;
        if (v5 > activation_max) v5 = activation_max;
        if (v6 < activation_min) v6 = activation_min;
        if (v6 > activation_max) v6 = activation_max;
        if (v7 < activation_min) v7 = activation_min;
        if (v7 > activation_max) v7 = activation_max;
        
        output[i + 0] = (int8_t)v0;
        output[i + 1] = (int8_t)v1;
        output[i + 2] = (int8_t)v2;
        output[i + 3] = (int8_t)v3;
        output[i + 4] = (int8_t)v4;
        output[i + 5] = (int8_t)v5;
        output[i + 6] = (int8_t)v6;
        output[i + 7] = (int8_t)v7;
    }
    
    for (; i < n; i++) {
        float f = (float)input[i] * scale_in;
        float y = f / (1.0f + expf(-f));
        int32_t v = (int32_t)roundf(y * inv_scale_out);
        if (v < activation_min) v = activation_min;
        if (v > activation_max) v = activation_max;
        output[i] = (int8_t)v;
    }
}