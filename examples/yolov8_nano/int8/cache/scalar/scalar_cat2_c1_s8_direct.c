void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    float inv_scale_out = 1.0f / scale_out;
    float scale0_norm = scale0 * inv_scale_out;
    float scale1_norm = scale1 * inv_scale_out;
    
    int out_c_total = c0 + c1;
    
    for (int n = 0; n < N; n++) {
        const int8_t *in0_base = in0 + n * c0 * stride;
        const int8_t *in1_base = in1 + n * c1 * stride;
        int8_t *out_base = output + n * out_c_total * stride;
        
        for (int c = 0; c < c0; c++) {
            const int8_t *in_ptr = in0_base + c * stride;
            int8_t *out_ptr = out_base + c * stride;
            
            int hw = 0;
            for (; hw + 16 <= stride; hw += 16) {
                float f0 = (float)in_ptr[hw + 0] * scale0_norm;
                float f1 = (float)in_ptr[hw + 1] * scale0_norm;
                float f2 = (float)in_ptr[hw + 2] * scale0_norm;
                float f3 = (float)in_ptr[hw + 3] * scale0_norm;
                float f4 = (float)in_ptr[hw + 4] * scale0_norm;
                float f5 = (float)in_ptr[hw + 5] * scale0_norm;
                float f6 = (float)in_ptr[hw + 6] * scale0_norm;
                float f7 = (float)in_ptr[hw + 7] * scale0_norm;
                float f8 = (float)in_ptr[hw + 8] * scale0_norm;
                float f9 = (float)in_ptr[hw + 9] * scale0_norm;
                float f10 = (float)in_ptr[hw + 10] * scale0_norm;
                float f11 = (float)in_ptr[hw + 11] * scale0_norm;
                float f12 = (float)in_ptr[hw + 12] * scale0_norm;
                float f13 = (float)in_ptr[hw + 13] * scale0_norm;
                float f14 = (float)in_ptr[hw + 14] * scale0_norm;
                float f15 = (float)in_ptr[hw + 15] * scale0_norm;
                
                int32_t v0 = (int32_t)roundf(f0);
                int32_t v1 = (int32_t)roundf(f1);
                int32_t v2 = (int32_t)roundf(f2);
                int32_t v3 = (int32_t)roundf(f3);
                int32_t v4 = (int32_t)roundf(f4);
                int32_t v5 = (int32_t)roundf(f5);
                int32_t v6 = (int32_t)roundf(f6);
                int32_t v7 = (int32_t)roundf(f7);
                int32_t v8 = (int32_t)roundf(f8);
                int32_t v9 = (int32_t)roundf(f9);
                int32_t v10 = (int32_t)roundf(f10);
                int32_t v11 = (int32_t)roundf(f11);
                int32_t v12 = (int32_t)roundf(f12);
                int32_t v13 = (int32_t)roundf(f13);
                int32_t v14 = (int32_t)roundf(f14);
                int32_t v15 = (int32_t)roundf(f15);
                
                v0 = v0 < activation_min ? activation_min : (v0 > activation_max ? activation_max : v0);
                v1 = v1 < activation_min ? activation_min : (v1 > activation_max ? activation_max : v1);
                v2 = v2 < activation_min ? activation_min : (v2 > activation_max ? activation_max : v2);
                v3 = v3 < activation_min ? activation_min : (v3 > activation_max ? activation_max : v3);
                v4 = v4 < activation_min ? activation_min : (v4 > activation_max ? activation_max : v4);
                v5 = v5 < activation_min ? activation_min : (v5 > activation_max ? activation_max : v5);
                v6 = v6 < activation_min ? activation_min : (v6 > activation_max ? activation_max : v6);
                v7 = v7 < activation_min ? activation_min : (v7 > activation_max ? activation_max : v7);
                v8 = v8 < activation_min ? activation_min : (v8 > activation_max ? activation_max : v8);
                v9 = v9 < activation_min ? activation_min : (v9 > activation_max ? activation_max : v9);
                v10 = v10 < activation_min ? activation_min : (v10 > activation_max ? activation_max : v10);
                v11 = v11 < activation_min ? activation_min : (v11 > activation_max ? activation_max : v11);
                v12 = v12 < activation_min ? activation_min : (v12 > activation_max ? activation_max : v12);
                v13 = v13 < activation_min ? activation_min : (v13 > activation_max ? activation_max : v13);
                v14 = v14 < activation_min ? activation_min : (v14 > activation_max ? activation_max : v14);
                v15 = v15 < activation_min ? activation_min : (v15 > activation_max ? activation_max : v15);
                
                out_ptr[hw + 0] = (int8_t)v0;
                out_ptr[hw + 1] = (int8_t)v1;
                out_ptr[hw + 2] = (int8_t)v2;
                out_ptr[hw + 3] = (int8_t)v3;
                out_ptr[hw + 4] = (int8_t)v4;
                out_ptr[hw + 5] = (int8_t)v5;
                out_ptr[hw + 6] = (int8_t)v6;
                out_ptr[hw + 7] = (int8_t)v7;
                out_ptr[hw + 8] = (int8_t)v8;
                out_ptr[hw + 9] = (int8_t)v9;
                out_ptr[hw + 10] = (int8_t)v10;
                out_ptr[hw + 11] = (int8_t)v11;
                out_ptr[hw + 12] = (int8_t)v12;
                out_ptr[hw + 13] = (int8_t)v13;
                out_ptr[hw + 14] = (int8_t)v14;
                out_ptr[hw + 15] = (int8_t)v15;
            }
            
            for (; hw < stride; hw++) {
                float f = (float)in_ptr[hw] * scale0_norm;
                int32_t v = (int32_t)roundf(f);
                v = v < activation_min ? activation_min : (v > activation_max ? activation_max : v);
                out_ptr[hw] = (int8_t)v;
            }
        }
        
        for (int c = 0; c < c1; c++) {
            const int8_t *in_ptr = in1_base + c * stride;
            int8_t *out_ptr = out_base + (c0 + c) * stride;
            
            int hw = 0;
            for (; hw + 16 <= stride; hw += 16) {
                float f0 = (float)in_ptr[hw + 0] * scale1_norm;
                float f1 = (float)in_ptr[hw + 1] * scale1_norm;
                float f2 = (float)in_ptr[hw + 2] * scale1_norm;
                float f3 = (float)in_ptr[hw + 3] * scale1_norm;
                float f4 = (float)in_ptr[hw + 4] * scale1_norm;
                float f5 = (float)in_ptr[hw + 5] * scale1_norm;
                float f6 = (float)in_ptr[hw + 6] * scale1_norm;
                float f7 = (float)in_ptr[hw + 7] * scale1_norm;
                float f8 = (float)in_ptr[hw + 8] * scale1_norm;
                float f9 = (float)in_ptr[hw + 9] * scale1_norm;
                float f10 = (float)in_ptr[hw + 10] * scale1_norm;
                float f11 = (float)in_ptr[hw + 11] * scale1_norm;
                float f12 = (float)in_ptr[hw + 12] * scale1_norm;
                float f13 = (float)in_ptr[hw + 13] * scale1_norm;
                float f14 = (float)in_ptr[hw + 14] * scale1_norm;
                float f15 = (float)in_ptr[hw + 15] * scale1_norm;
                
                int32_t v0 = (int32_t)roundf(f0);
                int32_t v1 = (int32_t)roundf(f1);
                int32_t v2 = (int32_t)roundf(f2);
                int32_t v3 = (int32_t)roundf(f3);
                int32_t v4 = (int32_t)roundf(f4);
                int32_t v5 = (int32_t)roundf(f5);
                int32_t v6 = (int32_t)roundf(f6);
                int32_t v7 = (int32_t)roundf(f7);
                int32_t v8 = (int32_t)roundf(f8);
                int32_t v9 = (int32_t)roundf(f9);
                int32_t v10 = (int32_t)roundf(f10);
                int32_t v11 = (int32_t)roundf(f11);
                int32_t v12 = (int32_t)roundf(f12);
                int32_t v13 = (int32_t)roundf(f13);
                int32_t v14 = (int32_t)roundf(f14);
                int32_t v15 = (int32_t)roundf(f15);
                
                v0 = v0 < activation_min ? activation_min : (v0 > activation_max ? activation_max : v0);
                v1 = v1 < activation_min ? activation_min : (v1 > activation_max ? activation_max : v1);
                v2 = v2 < activation_min ? activation_min : (v2 > activation_max ? activation_max : v2);
                v3 = v3 < activation_min ? activation_min : (v3 > activation_max ? activation_max : v3);
                v4 = v4 < activation_min ? activation_min : (v4 > activation_max ? activation_max : v4);
                v5 = v5 < activation_min ? activation_min : (v5 > activation_max ? activation_max : v5);
                v6 = v6 < activation_min ? activation_min : (v6 > activation_max ? activation_max : v6);
                v7 = v7 < activation_min ? activation_min : (v7 > activation_max ? activation_max : v7);
                v8 = v8 < activation_min ? activation_min : (v8 > activation_max ? activation_max : v8);
                v9 = v9 < activation_min ? activation_min : (v9 > activation_max ? activation_max : v9);
                v10 = v10 < activation_min ? activation_min : (v10 > activation_max ? activation_max : v10);
                v11 = v11 < activation_min ? activation_min : (v11 > activation_max ? activation_max : v11);
                v12 = v12 < activation_min ? activation_min : (v12 > activation_max ? activation_max : v12);
                v13 = v13 < activation_min ? activation_min : (v13 > activation_max ? activation_max : v13);
                v14 = v14 < activation_min ? activation_min : (v14 > activation_max ? activation_max : v14);
                v15 = v15 < activation_min ? activation_min : (v15 > activation_max ? activation_max : v15);
                
                out_ptr[hw + 0] = (int8_t)v0;
                out_ptr[hw + 1] = (int8_t)v1;
                out_ptr[hw + 2] = (int8_t)v2;
                out_ptr[hw + 3] = (int8_t)v3;
                out_ptr[hw + 4] = (int8_t)v4;
                out_ptr[hw + 5] = (int8_t)v5;
                out_ptr[hw + 6] = (int8_t)v6;
                out_ptr[hw + 7] = (int8_t)v7;
                out_ptr[hw + 8] = (int8_t)v8;
                out_ptr[hw + 9] = (int8_t)v9;
                out_ptr[hw + 10] = (int8_t)v10;
                out_ptr[hw + 11] = (int8_t)v11;
                out_ptr[hw + 12] = (int8_t)v12;
                out_ptr[hw + 13] = (int8_t)v13;
                out_ptr[hw + 14] = (int8_t)v14;
                out_ptr[hw + 15] = (int8_t)v15;
            }
            
            for (; hw < stride; hw++) {
                float f = (float)in_ptr[hw] * scale1_norm;
                int32_t v = (int32_t)roundf(f);
                v = v < activation_min ? activation_min : (v > activation_max ? activation_max : v);
                out_ptr[hw] = (int8_t)v;
            }
        }
    }
}