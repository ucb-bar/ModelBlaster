#include <math.h>
#include <riscv_vector.h>

static inline int8_t silu_lut_scalar(int8_t bn_val, float silu_scale_in, float silu_scale_out, int silu_activation_min, int silu_activation_max) {
    float f = (float)bn_val * silu_scale_in;
    float y = f / (1.0f + expf(-f));
    int32_t q = (int32_t)roundf(y / silu_scale_out);
    if (q < silu_activation_min) q = silu_activation_min;
    if (q > silu_activation_max) q = silu_activation_max;
    return (int8_t)q;
}

void kernel_batchnorm2d_silu_s8(const int8_t *input, const float *scale, const float *bias, int8_t *output, int N, int C, int H, int W, float bn_scale_in, float bn_scale_out, int bn_activation_min, int bn_activation_max, float silu_scale_in, float silu_scale_out, int silu_activation_min, int silu_activation_max) {
    int8_t silu_lut[256];
    for (int v = -128; v < 128; v++) {
        int8_t iv = (int8_t)v;
        silu_lut[v + 128] = silu_lut_scalar(iv, silu_scale_in, silu_scale_out, silu_activation_min, silu_activation_max);
    }

    int spatial_size = H * W;
    float inv_bn_scale_out = 1.0f / bn_scale_out;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            int base_idx = (n * C + c) * spatial_size;

            for (int i = 0; i < spatial_size; i++) {
                int8_t in_val = input[base_idx + i];
                float f = (float)in_val * bn_scale_in;
                float y = f * s + b;
                int32_t q = (int32_t)roundf(y * inv_bn_scale_out);
                if (q < bn_activation_min) q = bn_activation_min;
                if (q > bn_activation_max) q = bn_activation_max;
                int8_t bn_val = (int8_t)q;
                
                uint8_t lut_idx = (uint8_t)((int)bn_val + 128);
                output[base_idx + i] = silu_lut[lut_idx];
            }
        }
    }
}