void kernel_cat4_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2, const int8_t *in3, int c3, float scale3,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    const int8_t *ins[4] = { in0, in1, in2, in3 };
    int cs[4] = { c0, c1, c2, c3 };
    float scales[4] = { scale0, scale1, scale2, scale3 };
    int c_total = c0 + c1 + c2 + c3;
    
    float inv_scale_out = 1.0f / scale_out;
    int act_min = activation_min;
    int act_max = activation_max;
    
    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 4; i++) {
            float combined_scale = scales[i] * inv_scale_out;
            int c_in = cs[i];
            const int8_t *in_ptr = ins[i] + n * c_in * stride;
            int8_t *out_ptr = output + (n * c_total + out_c) * stride;
            
            for (int c = 0; c < c_in; c++) {
                const int8_t *in_channel = in_ptr + c * stride;
                int8_t *out_channel = out_ptr + c * stride;
                
                int hw = 0;
                for (; hw + 8 <= stride; hw += 8) {
                    int8_t i0 = in_channel[hw + 0];
                    int8_t i1 = in_channel[hw + 1];
                    int8_t i2 = in_channel[hw + 2];
                    int8_t i3 = in_channel[hw + 3];
                    int8_t i4 = in_channel[hw + 4];
                    int8_t i5 = in_channel[hw + 5];
                    int8_t i6 = in_channel[hw + 6];
                    int8_t i7 = in_channel[hw + 7];
                    
                    float f0 = (float)i0 * combined_scale;
                    float f1 = (float)i1 * combined_scale;
                    float f2 = (float)i2 * combined_scale;
                    float f3 = (float)i3 * combined_scale;
                    float f4 = (float)i4 * combined_scale;
                    float f5 = (float)i5 * combined_scale;
                    float f6 = (float)i6 * combined_scale;
                    float f7 = (float)i7 * combined_scale;
                    
                    int32_t v0 = (int32_t)roundf(f0);
                    int32_t v1 = (int32_t)roundf(f1);
                    int32_t v2 = (int32_t)roundf(f2);
                    int32_t v3 = (int32_t)roundf(f3);
                    int32_t v4 = (int32_t)roundf(f4);
                    int32_t v5 = (int32_t)roundf(f5);
                    int32_t v6 = (int32_t)roundf(f6);
                    int32_t v7 = (int32_t)roundf(f7);
                    
                    v0 = (v0 < act_min) ? act_min : ((v0 > act_max) ? act_max : v0);
                    v1 = (v1 < act_min) ? act_min : ((v1 > act_max) ? act_max : v1);
                    v2 = (v2 < act_min) ? act_min : ((v2 > act_max) ? act_max : v2);
                    v3 = (v3 < act_min) ? act_min : ((v3 > act_max) ? act_max : v3);
                    v4 = (v4 < act_min) ? act_min : ((v4 > act_max) ? act_max : v4);
                    v5 = (v5 < act_min) ? act_min : ((v5 > act_max) ? act_max : v5);
                    v6 = (v6 < act_min) ? act_min : ((v6 > act_max) ? act_max : v6);
                    v7 = (v7 < act_min) ? act_min : ((v7 > act_max) ? act_max : v7);
                    
                    out_channel[hw + 0] = (int8_t)v0;
                    out_channel[hw + 1] = (int8_t)v1;
                    out_channel[hw + 2] = (int8_t)v2;
                    out_channel[hw + 3] = (int8_t)v3;
                    out_channel[hw + 4] = (int8_t)v4;
                    out_channel[hw + 5] = (int8_t)v5;
                    out_channel[hw + 6] = (int8_t)v6;
                    out_channel[hw + 7] = (int8_t)v7;
                }
                
                for (; hw < stride; hw++) {
                    float f = (float)in_channel[hw] * combined_scale;
                    int32_t v = (int32_t)roundf(f);
                    v = (v < act_min) ? act_min : ((v > act_max) ? act_max : v);
                    out_channel[hw] = (int8_t)v;
                }
            }
            out_c += c_in;
        }
    }
}