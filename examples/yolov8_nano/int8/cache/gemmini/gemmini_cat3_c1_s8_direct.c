void kernel_cat3_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    int c_total = c0 + c1 + c2;
    
    float scale_ratio0 = scale0 / scale_out;
    float scale_ratio1 = scale1 / scale_out;
    float scale_ratio2 = scale2 / scale_out;
    
    for (int n = 0; n < N; n++) {
        const int8_t *src0 = in0 + n * c0 * stride;
        const int8_t *src1 = in1 + n * c1 * stride;
        const int8_t *src2 = in2 + n * c2 * stride;
        int8_t *dst = output + n * c_total * stride;
        
        for (int c = 0; c < c0; c++) {
            const int8_t *src_ptr = src0 + c * stride;
            int8_t *dst_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 16 <= stride; hw += 16) {
                int32_t v0 = lroundf((float)src_ptr[hw + 0] * scale_ratio0);
                int32_t v1 = lroundf((float)src_ptr[hw + 1] * scale_ratio0);
                int32_t v2 = lroundf((float)src_ptr[hw + 2] * scale_ratio0);
                int32_t v3 = lroundf((float)src_ptr[hw + 3] * scale_ratio0);
                int32_t v4 = lroundf((float)src_ptr[hw + 4] * scale_ratio0);
                int32_t v5 = lroundf((float)src_ptr[hw + 5] * scale_ratio0);
                int32_t v6 = lroundf((float)src_ptr[hw + 6] * scale_ratio0);
                int32_t v7 = lroundf((float)src_ptr[hw + 7] * scale_ratio0);
                int32_t v8 = lroundf((float)src_ptr[hw + 8] * scale_ratio0);
                int32_t v9 = lroundf((float)src_ptr[hw + 9] * scale_ratio0);
                int32_t v10 = lroundf((float)src_ptr[hw + 10] * scale_ratio0);
                int32_t v11 = lroundf((float)src_ptr[hw + 11] * scale_ratio0);
                int32_t v12 = lroundf((float)src_ptr[hw + 12] * scale_ratio0);
                int32_t v13 = lroundf((float)src_ptr[hw + 13] * scale_ratio0);
                int32_t v14 = lroundf((float)src_ptr[hw + 14] * scale_ratio0);
                int32_t v15 = lroundf((float)src_ptr[hw + 15] * scale_ratio0);
                
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
                
                dst_ptr[hw + 0] = (int8_t)v0;
                dst_ptr[hw + 1] = (int8_t)v1;
                dst_ptr[hw + 2] = (int8_t)v2;
                dst_ptr[hw + 3] = (int8_t)v3;
                dst_ptr[hw + 4] = (int8_t)v4;
                dst_ptr[hw + 5] = (int8_t)v5;
                dst_ptr[hw + 6] = (int8_t)v6;
                dst_ptr[hw + 7] = (int8_t)v7;
                dst_ptr[hw + 8] = (int8_t)v8;
                dst_ptr[hw + 9] = (int8_t)v9;
                dst_ptr[hw + 10] = (int8_t)v10;
                dst_ptr[hw + 11] = (int8_t)v11;
                dst_ptr[hw + 12] = (int8_t)v12;
                dst_ptr[hw + 13] = (int8_t)v13;
                dst_ptr[hw + 14] = (int8_t)v14;
                dst_ptr[hw + 15] = (int8_t)v15;
            }
            for (; hw < stride; hw++) {
                int32_t v = lroundf((float)src_ptr[hw] * scale_ratio0);
                v = (v < activation_min) ? activation_min : (v > activation_max) ? activation_max : v;
                dst_ptr[hw] = (int8_t)v;
            }
        }
        
        dst += c0 * stride;
        for (int c = 0; c < c1; c++) {
            const int8_t *src_ptr = src1 + c * stride;
            int8_t *dst_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 16 <= stride; hw += 16) {
                int32_t v0 = lroundf((float)src_ptr[hw + 0] * scale_ratio1);
                int32_t v1 = lroundf((float)src_ptr[hw + 1] * scale_ratio1);
                int32_t v2 = lroundf((float)src_ptr[hw + 2] * scale_ratio1);
                int32_t v3 = lroundf((float)src_ptr[hw + 3] * scale_ratio1);
                int32_t v4 = lroundf((float)src_ptr[hw + 4] * scale_ratio1);
                int32_t v5 = lroundf((float)src_ptr[hw + 5] * scale_ratio1);
                int32_t v6 = lroundf((float)src_ptr[hw + 6] * scale_ratio1);
                int32_t v7 = lroundf((float)src_ptr[hw + 7] * scale_ratio1);
                int32_t v8 = lroundf((float)src_ptr[hw + 8] * scale_ratio1);
                int32_t v9 = lroundf((float)src_ptr[hw + 9] * scale_ratio1);
                int32_t v10 = lroundf((float)src_ptr[hw + 10] * scale_ratio1);
                int32_t v11 = lroundf((float)src_ptr[hw + 11] * scale_ratio1);
                int32_t v12 = lroundf((float)src_ptr[hw + 12] * scale_ratio1);
                int32_t v13 = lroundf((float)src_ptr[hw + 13] * scale_ratio1);
                int32_t v14 = lroundf((float)src_ptr[hw + 14] * scale_ratio1);
                int32_t v15 = lroundf((float)src_ptr[hw + 15] * scale_ratio1);
                
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
                
                dst_ptr[hw + 0] = (int8_t)v0;
                dst_ptr[hw + 1] = (int8_t)v1;
                dst_ptr[hw + 2] = (int8_t)v2;
                dst_ptr[hw + 3] = (int8_t)v3;
                dst_ptr[hw + 4] = (int8_t)v4;
                dst_ptr[hw + 5] = (int8_t)v5;
                dst_ptr[hw + 6] = (int8_t)v6;
                dst_ptr[hw + 7] = (int8_t)v7;
                dst_ptr[hw + 8] = (int8_t)v8;
                dst_ptr[hw + 9] = (int8_t)v9;
                dst_ptr[hw + 10] = (int8_t)v10;
                dst_ptr[hw + 11] = (int8_t)v11;
                dst_ptr[hw + 12] = (int8_t)v12;
                dst_ptr[hw + 13] = (int8_t)v13;
                dst_ptr[hw + 14] = (int8_t)v14;
                dst_ptr[hw + 15] = (int8_t)v15;
            }
            for (; hw < stride; hw++) {
                int32_t v = lroundf((float)src_ptr[hw] * scale_ratio1);
                v = (v < activation_min) ? activation_min : (v > activation_max) ? activation_max : v;
                dst_ptr[hw] = (int8_t)v;
            }
        }
        
        dst += c1 * stride;
        for (int c = 0; c < c2; c++) {
            const int8_t *src_ptr = src2 + c * stride;
            int8_t *dst_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 16 <= stride; hw += 16) {
                int32_t v0 = lroundf((float)src_ptr[hw + 0] * scale_ratio2);
                int32_t v1 = lroundf((float)src_ptr[hw + 1] * scale_ratio2);
                int32_t v2 = lroundf((float)src_ptr[hw + 2] * scale_ratio2);
                int32_t v3 = lroundf((float)src_ptr[hw + 3] * scale_ratio2);
                int32_t v4 = lroundf((float)src_ptr[hw + 4] * scale_ratio2);
                int32_t v5 = lroundf((float)src_ptr[hw + 5] * scale_ratio2);
                int32_t v6 = lroundf((float)src_ptr[hw + 6] * scale_ratio2);
                int32_t v7 = lroundf((float)src_ptr[hw + 7] * scale_ratio2);
                int32_t v8 = lroundf((float)src_ptr[hw + 8] * scale_ratio2);
                int32_t v9 = lroundf((float)src_ptr[hw + 9] * scale_ratio2);
                int32_t v10 = lroundf((float)src_ptr[hw + 10] * scale_ratio2);
                int32_t v11 = lroundf((float)src_ptr[hw + 11] * scale_ratio2);
                int32_t v12 = lroundf((float)src_ptr[hw + 12] * scale_ratio2);
                int32_t v13 = lroundf((float)src_ptr[hw + 13] * scale_ratio2);
                int32_t v14 = lroundf((float)src_ptr[hw + 14] * scale_ratio2);
                int32_t v15 = lroundf((float)src_ptr[hw + 15] * scale_ratio2);
                
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
                
                dst_ptr[hw + 0] = (int8_t)v0;
                dst_ptr[hw + 1] = (int8_t)v1;
                dst_ptr[hw + 2] = (int8_t)v2;
                dst_ptr[hw + 3] = (int8_t)v3;
                dst_ptr[hw + 4] = (int8_t)v4;
                dst_ptr[hw + 5] = (int8_t)v5;
                dst_ptr[hw + 6] = (int8_t)v6;
                dst_ptr[hw + 7] = (int8_t)v7;
                dst_ptr[hw + 8] = (int8_t)v8;
                dst_ptr[hw + 9] = (int8_t)v9;
                dst_ptr[hw + 10] = (int8_t)v10;
                dst_ptr[hw + 11] = (int8_t)v11;
                dst_ptr[hw + 12] = (int8_t)v12;
                dst_ptr[hw + 13] = (int8_t)v13;
                dst_ptr[hw + 14] = (int8_t)v14;
                dst_ptr[hw + 15] = (int8_t)v15;
            }
            for (; hw < stride; hw++) {
                int32_t v = lroundf((float)src_ptr[hw] * scale_ratio2);
                v = (v < activation_min) ? activation_min : (v > activation_max) ? activation_max : v;
                dst_ptr[hw] = (int8_t)v;
            }
        }
    }
}