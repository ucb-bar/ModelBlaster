void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    int8_t m = input[((n*C + c)*IH + ih0)*IW + iw0];
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int8_t v = input[((n*C + c)*IH + ih0+kh)*IW + iw0+kw];
                            if (v > m) m = v;
                        }
                    }
                    output[((n*C + c)*OH + oh)*OW + ow] = m;
                }
            }
        }
    }
}

