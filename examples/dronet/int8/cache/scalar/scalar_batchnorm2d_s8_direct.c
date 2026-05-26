void kernel_batchnorm2d_s8(const int8_t *input, const float *scale, const float *bias, int8_t *output, int N, int C, int H, int W, float scale_in, float scale_out, int activation_min, int activation_max) {
    float inv_scale_out = 1.0f / scale_out;
    int spatial_size = H * W;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c] * scale_in;
            float b = bias[c];
            const int8_t *in_ptr = input + (n * C + c) * spatial_size;
            int8_t *out_ptr = output + (n * C + c) * spatial_size;
            
            int hw = 0;
            for (; hw + 4 <= spatial_size; hw += 4) {
                float fv0 = (float)in_ptr[hw + 0];
                float fv1 = (float)in_ptr[hw + 1];
                float fv2 = (float)in_ptr[hw + 2];
                float fv3 = (float)in_ptr[hw + 3];
                
                float y0 = s * fv0 + b;
                float y1 = s * fv1 + b;
                float y2 = s * fv2 + b;
                float y3 = s * fv3 + b;
                
                int32_t v0 = (int32_t)roundf(y0 * inv_scale_out);
                int32_t v1 = (int32_t)roundf(y1 * inv_scale_out);
                int32_t v2 = (int32_t)roundf(y2 * inv_scale_out);
                int32_t v3 = (int32_t)roundf(y3 * inv_scale_out);
                
                if (v0 < activation_min) v0 = activation_min;
                if (v0 > activation_max) v0 = activation_max;
                if (v1 < activation_min) v1 = activation_min;
                if (v1 > activation_max) v1 = activation_max;
                if (v2 < activation_min) v2 = activation_min;
                if (v2 > activation_max) v2 = activation_max;
                if (v3 < activation_min) v3 = activation_min;
                if (v3 > activation_max) v3 = activation_max;
                
                out_ptr[hw + 0] = (int8_t)v0;
                out_ptr[hw + 1] = (int8_t)v1;
                out_ptr[hw + 2] = (int8_t)v2;
                out_ptr[hw + 3] = (int8_t)v3;
            }
            
            for (; hw < spatial_size; hw++) {
                float fv = (float)in_ptr[hw];
                float y = s * fv + b;
                int32_t v = (int32_t)roundf(y * inv_scale_out);
                if (v < activation_min) v = activation_min;
                if (v > activation_max) v = activation_max;
                out_ptr[hw] = (int8_t)v;
            }
        }
    }
}