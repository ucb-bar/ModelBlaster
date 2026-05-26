void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    
    int has_padding = (PH != 0 || PW != 0);
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *input_nc = input + ((n*C + c)*IH)*IW;
            int8_t *output_nc = output + ((n*C + c)*OH)*OW;
            
            if (!has_padding) {
                for (int oh = 0; oh < OH; oh++) {
                    int ih_base = oh*SH;
                    for (int ow = 0; ow < OW; ow++) {
                        int iw_base = ow*SW;
                        
                        int8_t m0 = INT8_MIN;
                        int8_t m1 = INT8_MIN;
                        int8_t m2 = INT8_MIN;
                        int8_t m3 = INT8_MIN;
                        
                        int kh = 0;
                        for (; kh + 4 <= KH; kh += 4) {
                            int ih0 = ih_base + kh*DH;
                            int ih1 = ih_base + (kh+1)*DH;
                            int ih2 = ih_base + (kh+2)*DH;
                            int ih3 = ih_base + (kh+3)*DH;
                            const int8_t *row0 = input_nc + ih0*IW;
                            const int8_t *row1 = input_nc + ih1*IW;
                            const int8_t *row2 = input_nc + ih2*IW;
                            const int8_t *row3 = input_nc + ih3*IW;
                            
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = iw_base + kw*DW;
                                int8_t v0 = row0[iw];
                                int8_t v1 = row1[iw];
                                int8_t v2 = row2[iw];
                                int8_t v3 = row3[iw];
                                if (v0 > m0) m0 = v0;
                                if (v1 > m1) m1 = v1;
                                if (v2 > m2) m2 = v2;
                                if (v3 > m3) m3 = v3;
                            }
                        }
                        
                        for (; kh < KH; kh++) {
                            int ih = ih_base + kh*DH;
                            const int8_t *row = input_nc + ih*IW;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = iw_base + kw*DW;
                                int8_t v = row[iw];
                                if (v > m0) m0 = v;
                            }
                        }
                        
                        int8_t m01 = (m0 > m1) ? m0 : m1;
                        int8_t m23 = (m2 > m3) ? m2 : m3;
                        int8_t m = (m01 > m23) ? m01 : m23;
                        
                        output_nc[oh*OW + ow] = m;
                    }
                }
            } else {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int8_t m = INT8_MIN;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh*SH - PH + kh*DH;
                            if (ih < 0 || ih >= IH) continue;
                            const int8_t *row = input_nc + ih*IW;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow*SW - PW + kw*DW;
                                if (iw < 0 || iw >= IW) continue;
                                int8_t v = row[iw];
                                if (v > m) m = v;
                            }
                        }
                        output_nc[oh*OW + ow] = m;
                    }
                }
            }
        }
    }
}