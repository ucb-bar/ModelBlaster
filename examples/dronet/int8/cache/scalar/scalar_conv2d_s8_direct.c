void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    
    const int TILE_OC = 4;
    
    for (int n = 0; n < N; n++) {
        for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
            int oc_end = oc_outer + TILE_OC;
            if (oc_end > OC) oc_end = OC;
            
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc[4];
                    for (int t = 0; t < oc_end - oc_outer; t++) {
                        acc[t] = bias ? bias[oc_outer + t] : 0;
                    }
                    
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) {
                                for (int kw = 0; kw < KW; kw++) {
                                    int32_t in_v = input_offset;
                                    for (int t = 0; t < oc_end - oc_outer; t++) {
                                        int32_t w_v = (int32_t)weight[(((oc_outer + t)*IC + ic)*KH + kh)*KW + kw] + filter_offset;
                                        acc[t] += in_v * w_v;
                                    }
                                }
                            } else {
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (iw < 0 || iw >= IW) {
                                        in_v = input_offset;
                                    } else {
                                        in_v = (int32_t)input[((n*IC + ic)*IH + ih)*IW + iw] + input_offset;
                                    }
                                    for (int t = 0; t < oc_end - oc_outer; t++) {
                                        int32_t w_v = (int32_t)weight[(((oc_outer + t)*IC + ic)*KH + kh)*KW + kw] + filter_offset;
                                        acc[t] += in_v * w_v;
                                    }
                                }
                            }
                        }
                    }
                    
                    for (int t = 0; t < oc_end - oc_outer; t++) {
                        int64_t prod = (int64_t)acc[t] * (int64_t)output_multiplier;
                        prod = (prod + (1LL << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            int32_t round = (1 << (output_shift - 1));
                            scaled = (scaled + round) >> output_shift;
                        } else if (output_shift < 0) {
                            scaled = scaled << (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC + (oc_outer + t))*OH + oh)*OW + ow] = (int8_t)scaled;
                    }
                }
            }
        }
    }
}