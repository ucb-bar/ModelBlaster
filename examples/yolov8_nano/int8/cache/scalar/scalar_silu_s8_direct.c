void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    float inv_scale_out = 1.0f / scale_out;
    int i = 0;
    
    for (; i + 4 <= n; i += 4) {
        float f0 = (float)input[i + 0] * scale_in;
        float f1 = (float)input[i + 1] * scale_in;
        float f2 = (float)input[i + 2] * scale_in;
        float f3 = (float)input[i + 3] * scale_in;
        
        float exp0 = expf(-f0);
        float exp1 = expf(-f1);
        float exp2 = expf(-f2);
        float exp3 = expf(-f3);
        
        float y0 = f0 / (1.0f + exp0);
        float y1 = f1 / (1.0f + exp1);
        float y2 = f2 / (1.0f + exp2);
        float y3 = f3 / (1.0f + exp3);
        
        float scaled0 = y0 * inv_scale_out;
        float scaled1 = y1 * inv_scale_out;
        float scaled2 = y2 * inv_scale_out;
        float scaled3 = y3 * inv_scale_out;
        
        int32_t v0 = (int32_t)roundf(scaled0);
        int32_t v1 = (int32_t)roundf(scaled1);
        int32_t v2 = (int32_t)roundf(scaled2);
        int32_t v3 = (int32_t)roundf(scaled3);
        
        v0 = (v0 < activation_min) ? activation_min : v0;
        v0 = (v0 > activation_max) ? activation_max : v0;
        v1 = (v1 < activation_min) ? activation_min : v1;
        v1 = (v1 > activation_max) ? activation_max : v1;
        v2 = (v2 < activation_min) ? activation_min : v2;
        v2 = (v2 > activation_max) ? activation_max : v2;
        v3 = (v3 < activation_min) ? activation_min : v3;
        v3 = (v3 > activation_max) ? activation_max : v3;
        
        output[i + 0] = (int8_t)v0;
        output[i + 1] = (int8_t)v1;
        output[i + 2] = (int8_t)v2;
        output[i + 3] = (int8_t)v3;
    }
    
    for (; i < n; i++) {
        float f = (float)input[i] * scale_in;
        float y = f / (1.0f + expf(-f));
        int32_t v = (int32_t)roundf(y * inv_scale_out);
        v = (v < activation_min) ? activation_min : v;
        v = (v > activation_max) ? activation_max : v;
        output[i] = (int8_t)v;
    }
}