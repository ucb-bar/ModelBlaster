static inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    if (exponent == 0) return x;
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int32_t saturate_int8(int32_t x) {
    if (x < -128) return -128;
    if (x > 127) return 127;
    return x;
}

void kernel_conv2d_batchnorm2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, const float *bn_scale, const float *bn_bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int input_offset, int filter_offset, int conv_output_offset, int conv_output_multiplier, int conv_output_shift, int conv_activation_min, int conv_activation_max, float bn_scale_in, float bn_scale_out, int bn_activation_min, int bn_activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float bn_s = bn_scale[oc];
            float bn_b = bn_bias[oc];
            
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v = (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                    ? input_offset
                                    : (int32_t)input[((n*IC + ic)*IH + ih)*IW + iw] + input_offset;
                                int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw] + filter_offset;
                                acc += in_v * w_v;
                            }
                        }
                    }
                    
                    // Conv requantize via Q0.31 rounding multiply
                    int64_t prod = (int64_t)acc * (int64_t)conv_output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    int32_t scaled = (int32_t)prod;
                    
                    if (conv_output_shift > 0) {
                        scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (conv_output_shift - 1)))
                                            >> conv_output_shift);
                    } else if (conv_output_shift < 0) {
                        scaled = scaled << (-conv_output_shift);
                    }
                    
                    scaled += conv_output_offset;
                    if (scaled < conv_activation_min) scaled = conv_activation_min;
                    if (scaled > conv_activation_max) scaled = conv_activation_max;
                    int8_t conv_int8 = (int8_t)scaled;
                    
                    // BN affine in-register
                    float fv = (float)conv_int8 * bn_scale_in;
                    float y = bn_s * fv + bn_b;
                    int32_t v = (int32_t)roundf(y / bn_scale_out);
                    if (v < bn_activation_min) v = bn_activation_min;
                    if (v > bn_activation_max) v = bn_activation_max;
                    
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)v;
                }
            }
        }
    }
}