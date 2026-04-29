void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v;
                                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                                    in_v = input_offset;
                                } else {
                                    in_v = (int32_t)input[((n*IC + ic)*IH + ih)*IW + iw]
                                         + input_offset;
                                }
                                int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw]
                                            + filter_offset;
                                acc += in_v * w_v;
                            }
                        }
                    }
                    int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
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
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)scaled;
                }
            }
        }
    }
}