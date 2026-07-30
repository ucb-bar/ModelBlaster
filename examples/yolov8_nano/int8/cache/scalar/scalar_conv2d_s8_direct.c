void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    
    const int TILE_OC = 8;
    
    for (int n = 0; n < N; n++) {
        for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
            int oc_end = oc_outer + TILE_OC;
            if (oc_end > OC) oc_end = OC;
            int oc_count = oc_end - oc_outer;
            
            int32_t bias_vals[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (bias) {
                for (int i = 0; i < oc_count; i++) {
                    bias_vals[i] = bias[oc_outer + i];
                }
            }
            
            for (int oh = 0; oh < OH; oh++) {
                int ih_base = oh * SH - PH;
                
                for (int ow = 0; ow < OW; ow++) {
                    int iw_base = ow * SW - PW;
                    
                    int32_t acc0 = bias_vals[0];
                    int32_t acc1 = bias_vals[1];
                    int32_t acc2 = bias_vals[2];
                    int32_t acc3 = bias_vals[3];
                    int32_t acc4 = bias_vals[4];
                    int32_t acc5 = bias_vals[5];
                    int32_t acc6 = bias_vals[6];
                    int32_t acc7 = bias_vals[7];
                    
                    for (int ic = 0; ic < IC; ic++) {
                        const int8_t *in_base = input + ((size_t)n * IC + ic) * IH * IW;
                        const int8_t *w_base = weight + ((size_t)oc_outer * IC + ic) * KH * KW;
                        
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = ih_base + kh;
                            
                            if (ih >= 0 && ih < IH) {
                                const int8_t *in_row = in_base + ih * IW;
                                
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = iw_base + kw;
                                    
                                    int32_t in_v;
                                    if (iw >= 0 && iw < IW) {
                                        in_v = (int32_t)in_row[iw] + input_offset;
                                    } else {
                                        in_v = input_offset;
                                    }
                                    
                                    const int8_t *w_ptr = w_base + kh * KW + kw;
                                    size_t w_stride = IC * KH * KW;
                                    
                                    int32_t w0 = (int32_t)w_ptr[0] + filter_offset;
                                    acc0 += in_v * w0;
                                    
                                    if (oc_count > 1) {
                                        int32_t w1 = (int32_t)w_ptr[w_stride] + filter_offset;
                                        acc1 += in_v * w1;
                                    }
                                    if (oc_count > 2) {
                                        int32_t w2 = (int32_t)w_ptr[2*w_stride] + filter_offset;
                                        acc2 += in_v * w2;
                                    }
                                    if (oc_count > 3) {
                                        int32_t w3 = (int32_t)w_ptr[3*w_stride] + filter_offset;
                                        acc3 += in_v * w3;
                                    }
                                    if (oc_count > 4) {
                                        int32_t w4 = (int32_t)w_ptr[4*w_stride] + filter_offset;
                                        acc4 += in_v * w4;
                                    }
                                    if (oc_count > 5) {
                                        int32_t w5 = (int32_t)w_ptr[5*w_stride] + filter_offset;
                                        acc5 += in_v * w5;
                                    }
                                    if (oc_count > 6) {
                                        int32_t w6 = (int32_t)w_ptr[6*w_stride] + filter_offset;
                                        acc6 += in_v * w6;
                                    }
                                    if (oc_count > 7) {
                                        int32_t w7 = (int32_t)w_ptr[7*w_stride] + filter_offset;
                                        acc7 += in_v * w7;
                                    }
                                }
                            } else {
                                for (int kw = 0; kw < KW; kw++) {
                                    int32_t in_v = input_offset;
                                    
                                    const int8_t *w_ptr = w_base + kh * KW + kw;
                                    size_t w_stride = IC * KH * KW;
                                    
                                    int32_t w0 = (int32_t)w_ptr[0] + filter_offset;
                                    acc0 += in_v * w0;
                                    
                                    if (oc_count > 1) {
                                        int32_t w1 = (int32_t)w_ptr[w_stride] + filter_offset;
                                        acc1 += in_v * w1;
                                    }
                                    if (oc_count > 2) {
                                        int32_t w2 = (int32_t)w_ptr[2*w_stride] + filter_offset;
                                        acc2 += in_v * w2;
                                    }
                                    if (oc_count > 3) {
                                        int32_t w3 = (int32_t)w_ptr[3*w_stride] + filter_offset;
                                        acc3 += in_v * w3;
                                    }
                                    if (oc_count > 4) {
                                        int32_t w4 = (int32_t)w_ptr[4*w_stride] + filter_offset;
                                        acc4 += in_v * w4;
                                    }
                                    if (oc_count > 5) {
                                        int32_t w5 = (int32_t)w_ptr[5*w_stride] + filter_offset;
                                        acc5 += in_v * w5;
                                    }
                                    if (oc_count > 6) {
                                        int32_t w6 = (int32_t)w_ptr[6*w_stride] + filter_offset;
                                        acc6 += in_v * w6;
                                    }
                                    if (oc_count > 7) {
                                        int32_t w7 = (int32_t)w_ptr[7*w_stride] + filter_offset;
                                        acc7 += in_v * w7;
                                    }
                                }
                            }
                        }
                    }
                    
                    int8_t *out_ptr = output + ((size_t)n * OC + oc_outer) * OH * OW + oh * OW + ow;
                    size_t out_stride = OH * OW;
                    
                    int32_t accs[8] = {acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7};
                    for (int oc_inner = 0; oc_inner < oc_count; oc_inner++) {
                        int32_t acc = accs[oc_inner];
                        
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + (1LL << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled = scaled << (-output_shift);
                        }
                        
                        scaled += output_offset;
                        
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        
                        out_ptr[oc_inner * out_stride] = (int8_t)scaled;
                    }
                }
            }
        }
    }
}