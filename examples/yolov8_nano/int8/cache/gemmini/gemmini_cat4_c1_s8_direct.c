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
                for (; hw + 16 <= stride; hw += 16) {
                    float f0 = (float)in_channel[hw + 0] * combined_scale;
                    float f1 = (float)in_channel[hw + 1] * combined_scale;
                    float f2 = (float)in_channel[hw + 2] * combined_scale;
                    float f3 = (float)in_channel[hw + 3] * combined_scale;
                    float f4 = (float)in_channel[hw + 4] * combined_scale;
                    float f5 = (float)in_channel[hw + 5] * combined_scale;
                    float f6 = (float)in_channel[hw + 6] * combined_scale;
                    float f7 = (float)in_channel[hw + 7] * combined_scale;
                    float f8 = (float)in_channel[hw + 8] * combined_scale;
                    float f9 = (float)in_channel[hw + 9] * combined_scale;
                    float f10 = (float)in_channel[hw + 10] * combined_scale;
                    float f11 = (float)in_channel[hw + 11] * combined_scale;
                    float f12 = (float)in_channel[hw + 12] * combined_scale;
                    float f13 = (float)in_channel[hw + 13] * combined_scale;
                    float f14 = (float)in_channel[hw + 14] * combined_scale;
                    float f15 = (float)in_channel[hw + 15] * combined_scale;
                    
                    int32_t v0 = (int32_t)(f0 + copysignf(0.5f, f0));
                    int32_t v1 = (int32_t)(f1 + copysignf(0.5f, f1));
                    int32_t v2 = (int32_t)(f2 + copysignf(0.5f, f2));
                    int32_t v3 = (int32_t)(f3 + copysignf(0.5f, f3));
                    int32_t v4 = (int32_t)(f4 + copysignf(0.5f, f4));
                    int32_t v5 = (int32_t)(f5 + copysignf(0.5f, f5));
                    int32_t v6 = (int32_t)(f6 + copysignf(0.5f, f6));
                    int32_t v7 = (int32_t)(f7 + copysignf(0.5f, f7));
                    int32_t v8 = (int32_t)(f8 + copysignf(0.5f, f8));
                    int32_t v9 = (int32_t)(f9 + copysignf(0.5f, f9));
                    int32_t v10 = (int32_t)(f10 + copysignf(0.5f, f10));
                    int32_t v11 = (int32_t)(f11 + copysignf(0.5f, f11));
                    int32_t v12 = (int32_t)(f12 + copysignf(0.5f, f12));
                    int32_t v13 = (int32_t)(f13 + copysignf(0.5f, f13));
                    int32_t v14 = (int32_t)(f14 + copysignf(0.5f, f14));
                    int32_t v15 = (int32_t)(f15 + copysignf(0.5f, f15));
                    
                    v0 = (v0 < activation_min) ? activation_min : (v0 > activation_max) ? activation_max : v0;
                    v1 = (v1 < activation_min) ? activation_min : (v1 > activation_max) ? activation_max : v1;
                    v2 = (v2 < activation_min) ? activation_min : (v2 > activation_max) ? activation_max : v2;
                    v3 = (v3 < activation_min) ? activation_min : (v3 > activation_max) ? activation_max : v3;
                    v4 = (v4 < activation_min) ? activation_min : (v4 > activation_max) ? activation_max : v4;
                    v5 = (v5 < activation_min) ? activation_min : (v5 > activation_max) ? activation_max : v5;
                    v6 = (v6 < activation_min) ? activation_min : (v6 > activation_max) ? activation_max : v6;
                    v7 = (v7 < activation_min) ? activation_min : (v7 > activation_max) ? activation_max : v7;
                    v8 = (v8 < activation_min) ? activation_min : (v8 > activation_max) ? activation_max : v8;
                    v9 = (v9 < activation_min) ? activation_min : (v9 > activation_max) ? activation_max : v9;
                    v10 = (v10 < activation_min) ? activation_min : (v10 > activation_max) ? activation_max : v10;
                    v11 = (v11 < activation_min) ? activation_min : (v11 > activation_max) ? activation_max : v11;
                    v12 = (v12 < activation_min) ? activation_min : (v12 > activation_max) ? activation_max : v12;
                    v13 = (v13 < activation_min) ? activation_min : (v13 > activation_max) ? activation_max : v13;
                    v14 = (v14 < activation_min) ? activation_min : (v14 > activation_max) ? activation_max : v14;
                    v15 = (v15 < activation_min) ? activation_min : (v15 > activation_max) ? activation_max : v15;
                    
                    out_channel[hw + 0] = (int8_t)v0;
                    out_channel[hw + 1] = (int8_t)v1;
                    out_channel[hw + 2] = (int8_t)v2;
                    out_channel[hw + 3] = (int8_t)v3;
                    out_channel[hw + 4] = (int8_t)v4;
                    out_channel[hw + 5] = (int8_t)v5;
                    out_channel[hw + 6] = (int8_t)v6;
                    out_channel[hw + 7] = (int8_t)v7;
                    out_channel[hw + 8] = (int8_t)v8;
                    out_channel[hw + 9] = (int8_t)v9;
                    out_channel[hw + 10] = (int8_t)v10;
                    out_channel[hw + 11] = (int8_t)v11;
                    out_channel[hw + 12] = (int8_t)v12;
                    out_channel[hw + 13] = (int8_t)v13;
                    out_channel[hw + 14] = (int8_t)v14;
                    out_channel[hw + 15] = (int8_t)v15;
                }
                
                for (; hw < stride; hw++) {
                    float f = (float)in_channel[hw] * combined_scale;
                    int32_t v = (int32_t)(f + copysignf(0.5f, f));
                    v = (v < activation_min) ? activation_min : (v > activation_max) ? activation_max : v;
                    out_channel[hw] = (int8_t)v;
                }
            }
            out_c += c_in;
        }
    }
}