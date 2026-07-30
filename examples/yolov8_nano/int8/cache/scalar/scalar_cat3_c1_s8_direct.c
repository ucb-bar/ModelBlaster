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
    
    float min_f = (float)activation_min;
    float max_f = (float)activation_max;
    
    for (int n = 0; n < N; n++) {
        const int8_t *src0 = in0 + n * c0 * stride;
        const int8_t *src1 = in1 + n * c1 * stride;
        const int8_t *src2 = in2 + n * c2 * stride;
        int8_t *dst = output + n * c_total * stride;
        
        for (int c = 0; c < c0; c++) {
            const int8_t *in_ptr = src0 + c * stride;
            int8_t *out_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 8 <= stride; hw += 8) {
                float f0 = (float)in_ptr[hw + 0] * scale_ratio0;
                float f1 = (float)in_ptr[hw + 1] * scale_ratio0;
                float f2 = (float)in_ptr[hw + 2] * scale_ratio0;
                float f3 = (float)in_ptr[hw + 3] * scale_ratio0;
                float f4 = (float)in_ptr[hw + 4] * scale_ratio0;
                float f5 = (float)in_ptr[hw + 5] * scale_ratio0;
                float f6 = (float)in_ptr[hw + 6] * scale_ratio0;
                float f7 = (float)in_ptr[hw + 7] * scale_ratio0;
                
                f0 = roundf(f0);
                f1 = roundf(f1);
                f2 = roundf(f2);
                f3 = roundf(f3);
                f4 = roundf(f4);
                f5 = roundf(f5);
                f6 = roundf(f6);
                f7 = roundf(f7);
                
                f0 = (f0 < min_f) ? min_f : ((f0 > max_f) ? max_f : f0);
                f1 = (f1 < min_f) ? min_f : ((f1 > max_f) ? max_f : f1);
                f2 = (f2 < min_f) ? min_f : ((f2 > max_f) ? max_f : f2);
                f3 = (f3 < min_f) ? min_f : ((f3 > max_f) ? max_f : f3);
                f4 = (f4 < min_f) ? min_f : ((f4 > max_f) ? max_f : f4);
                f5 = (f5 < min_f) ? min_f : ((f5 > max_f) ? max_f : f5);
                f6 = (f6 < min_f) ? min_f : ((f6 > max_f) ? max_f : f6);
                f7 = (f7 < min_f) ? min_f : ((f7 > max_f) ? max_f : f7);
                
                out_ptr[hw + 0] = (int8_t)f0;
                out_ptr[hw + 1] = (int8_t)f1;
                out_ptr[hw + 2] = (int8_t)f2;
                out_ptr[hw + 3] = (int8_t)f3;
                out_ptr[hw + 4] = (int8_t)f4;
                out_ptr[hw + 5] = (int8_t)f5;
                out_ptr[hw + 6] = (int8_t)f6;
                out_ptr[hw + 7] = (int8_t)f7;
            }
            for (; hw < stride; hw++) {
                float f = (float)in_ptr[hw] * scale_ratio0;
                f = roundf(f);
                f = (f < min_f) ? min_f : ((f > max_f) ? max_f : f);
                out_ptr[hw] = (int8_t)f;
            }
        }
        
        dst += c0 * stride;
        for (int c = 0; c < c1; c++) {
            const int8_t *in_ptr = src1 + c * stride;
            int8_t *out_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 8 <= stride; hw += 8) {
                float f0 = (float)in_ptr[hw + 0] * scale_ratio1;
                float f1 = (float)in_ptr[hw + 1] * scale_ratio1;
                float f2 = (float)in_ptr[hw + 2] * scale_ratio1;
                float f3 = (float)in_ptr[hw + 3] * scale_ratio1;
                float f4 = (float)in_ptr[hw + 4] * scale_ratio1;
                float f5 = (float)in_ptr[hw + 5] * scale_ratio1;
                float f6 = (float)in_ptr[hw + 6] * scale_ratio1;
                float f7 = (float)in_ptr[hw + 7] * scale_ratio1;
                
                f0 = roundf(f0);
                f1 = roundf(f1);
                f2 = roundf(f2);
                f3 = roundf(f3);
                f4 = roundf(f4);
                f5 = roundf(f5);
                f6 = roundf(f6);
                f7 = roundf(f7);
                
                f0 = (f0 < min_f) ? min_f : ((f0 > max_f) ? max_f : f0);
                f1 = (f1 < min_f) ? min_f : ((f1 > max_f) ? max_f : f1);
                f2 = (f2 < min_f) ? min_f : ((f2 > max_f) ? max_f : f2);
                f3 = (f3 < min_f) ? min_f : ((f3 > max_f) ? max_f : f3);
                f4 = (f4 < min_f) ? min_f : ((f4 > max_f) ? max_f : f4);
                f5 = (f5 < min_f) ? min_f : ((f5 > max_f) ? max_f : f5);
                f6 = (f6 < min_f) ? min_f : ((f6 > max_f) ? max_f : f6);
                f7 = (f7 < min_f) ? min_f : ((f7 > max_f) ? max_f : f7);
                
                out_ptr[hw + 0] = (int8_t)f0;
                out_ptr[hw + 1] = (int8_t)f1;
                out_ptr[hw + 2] = (int8_t)f2;
                out_ptr[hw + 3] = (int8_t)f3;
                out_ptr[hw + 4] = (int8_t)f4;
                out_ptr[hw + 5] = (int8_t)f5;
                out_ptr[hw + 6] = (int8_t)f6;
                out_ptr[hw + 7] = (int8_t)f7;
            }
            for (; hw < stride; hw++) {
                float f = (float)in_ptr[hw] * scale_ratio1;
                f = roundf(f);
                f = (f < min_f) ? min_f : ((f > max_f) ? max_f : f);
                out_ptr[hw] = (int8_t)f;
            }
        }
        
        dst += c1 * stride;
        for (int c = 0; c < c2; c++) {
            const int8_t *in_ptr = src2 + c * stride;
            int8_t *out_ptr = dst + c * stride;
            int hw = 0;
            for (; hw + 8 <= stride; hw += 8) {
                float f0 = (float)in_ptr[hw + 0] * scale_ratio2;
                float f1 = (float)in_ptr[hw + 1] * scale_ratio2;
                float f2 = (float)in_ptr[hw + 2] * scale_ratio2;
                float f3 = (float)in_ptr[hw + 3] * scale_ratio2;
                float f4 = (float)in_ptr[hw + 4] * scale_ratio2;
                float f5 = (float)in_ptr[hw + 5] * scale_ratio2;
                float f6 = (float)in_ptr[hw + 6] * scale_ratio2;
                float f7 = (float)in_ptr[hw + 7] * scale_ratio2;
                
                f0 = roundf(f0);
                f1 = roundf(f1);
                f2 = roundf(f2);
                f3 = roundf(f3);
                f4 = roundf(f4);
                f5 = roundf(f5);
                f6 = roundf(f6);
                f7 = roundf(f7);
                
                f0 = (f0 < min_f) ? min_f : ((f0 > max_f) ? max_f : f0);
                f1 = (f1 < min_f) ? min_f : ((f1 > max_f) ? max_f : f1);
                f2 = (f2 < min_f) ? min_f : ((f2 > max_f) ? max_f : f2);
                f3 = (f3 < min_f) ? min_f : ((f3 > max_f) ? max_f : f3);
                f4 = (f4 < min_f) ? min_f : ((f4 > max_f) ? max_f : f4);
                f5 = (f5 < min_f) ? min_f : ((f5 > max_f) ? max_f : f5);
                f6 = (f6 < min_f) ? min_f : ((f6 > max_f) ? max_f : f6);
                f7 = (f7 < min_f) ? min_f : ((f7 > max_f) ? max_f : f7);
                
                out_ptr[hw + 0] = (int8_t)f0;
                out_ptr[hw + 1] = (int8_t)f1;
                out_ptr[hw + 2] = (int8_t)f2;
                out_ptr[hw + 3] = (int8_t)f3;
                out_ptr[hw + 4] = (int8_t)f4;
                out_ptr[hw + 5] = (int8_t)f5;
                out_ptr[hw + 6] = (int8_t)f6;
                out_ptr[hw + 7] = (int8_t)f7;
            }
            for (; hw < stride; hw++) {
                float f = (float)in_ptr[hw] * scale_ratio2;
                f = roundf(f);
                f = (f < min_f) ? min_f : ((f > max_f) ? max_f : f);
                out_ptr[hw] = (int8_t)f;
            }
        }
    }
}