void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            float b = bias ? bias[oc] : 0.0f;
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float acc = b;
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = weight[((oc*IC + ic)*KH + kh)*KW + kw];
                                acc += v * w;
                            }
                        }
                    }
                    output[((n*OC + oc)*OH + oh)*OW + ow] = acc;
                }
            }
        }
    }
}