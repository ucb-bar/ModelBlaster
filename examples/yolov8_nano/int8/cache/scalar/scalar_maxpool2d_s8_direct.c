void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *input_base = input + (n*C + c)*IH*IW;
            int8_t *output_base = output + (n*C + c)*OH*OW;
            
            if (PH == 0 && PW == 0 && DH == 1 && DW == 1) {
                if (KH == 2 && KW == 2 && SH == 2 && SW == 2) {
                    for (int oh = 0; oh < OH; oh++) {
                        int ih = oh * 2;
                        const int8_t *row0 = input_base + ih * IW;
                        const int8_t *row1 = row0 + IW;
                        int8_t *out_row = output_base + oh * OW;
                        int ow = 0;
                        for (; ow + 4 <= OW; ow += 4) {
                            int iw0 = ow * 2;
                            int iw1 = iw0 + 2;
                            int iw2 = iw1 + 2;
                            int iw3 = iw2 + 2;
                            
                            int8_t v00 = row0[iw0], v01 = row0[iw0 + 1];
                            int8_t v02 = row1[iw0], v03 = row1[iw0 + 1];
                            int8_t m0a = (v00 > v01) ? v00 : v01;
                            int8_t m0b = (v02 > v03) ? v02 : v03;
                            int8_t m0 = (m0a > m0b) ? m0a : m0b;
                            
                            int8_t v10 = row0[iw1], v11 = row0[iw1 + 1];
                            int8_t v12 = row1[iw1], v13 = row1[iw1 + 1];
                            int8_t m1a = (v10 > v11) ? v10 : v11;
                            int8_t m1b = (v12 > v13) ? v12 : v13;
                            int8_t m1 = (m1a > m1b) ? m1a : m1b;
                            
                            int8_t v20 = row0[iw2], v21 = row0[iw2 + 1];
                            int8_t v22 = row1[iw2], v23 = row1[iw2 + 1];
                            int8_t m2a = (v20 > v21) ? v20 : v21;
                            int8_t m2b = (v22 > v23) ? v22 : v23;
                            int8_t m2 = (m2a > m2b) ? m2a : m2b;
                            
                            int8_t v30 = row0[iw3], v31 = row0[iw3 + 1];
                            int8_t v32 = row1[iw3], v33 = row1[iw3 + 1];
                            int8_t m3a = (v30 > v31) ? v30 : v31;
                            int8_t m3b = (v32 > v33) ? v32 : v33;
                            int8_t m3 = (m3a > m3b) ? m3a : m3b;
                            
                            out_row[ow] = m0;
                            out_row[ow + 1] = m1;
                            out_row[ow + 2] = m2;
                            out_row[ow + 3] = m3;
                        }
                        for (; ow < OW; ow++) {
                            int iw = ow * 2;
                            int8_t v0 = row0[iw];
                            int8_t v1 = row0[iw + 1];
                            int8_t v2 = row1[iw];
                            int8_t v3 = row1[iw + 1];
                            int8_t m01 = (v0 > v1) ? v0 : v1;
                            int8_t m23 = (v2 > v3) ? v2 : v3;
                            out_row[ow] = (m01 > m23) ? m01 : m23;
                        }
                    }
                } else if (KH == 3 && KW == 3 && SH == 2 && SW == 2) {
                    for (int oh = 0; oh < OH; oh++) {
                        int ih = oh * 2;
                        const int8_t *row0 = input_base + ih * IW;
                        const int8_t *row1 = row0 + IW;
                        const int8_t *row2 = row1 + IW;
                        int8_t *out_row = output_base + oh * OW;
                        int ow = 0;
                        for (; ow + 2 <= OW; ow += 2) {
                            int iw0 = ow * 2;
                            int iw1 = iw0 + 2;
                            
                            int8_t m0 = row0[iw0];
                            int8_t v;
                            v = row0[iw0 + 1]; if (v > m0) m0 = v;
                            v = row0[iw0 + 2]; if (v > m0) m0 = v;
                            v = row1[iw0];     if (v > m0) m0 = v;
                            v = row1[iw0 + 1]; if (v > m0) m0 = v;
                            v = row1[iw0 + 2]; if (v > m0) m0 = v;
                            v = row2[iw0];     if (v > m0) m0 = v;
                            v = row2[iw0 + 1]; if (v > m0) m0 = v;
                            v = row2[iw0 + 2]; if (v > m0) m0 = v;
                            
                            int8_t m1 = row0[iw1];
                            v = row0[iw1 + 1]; if (v > m1) m1 = v;
                            v = row0[iw1 + 2]; if (v > m1) m1 = v;
                            v = row1[iw1];     if (v > m1) m1 = v;
                            v = row1[iw1 + 1]; if (v > m1) m1 = v;
                            v = row1[iw1 + 2]; if (v > m1) m1 = v;
                            v = row2[iw1];     if (v > m1) m1 = v;
                            v = row2[iw1 + 1]; if (v > m1) m1 = v;
                            v = row2[iw1 + 2]; if (v > m1) m1 = v;
                            
                            out_row[ow] = m0;
                            out_row[ow + 1] = m1;
                        }
                        for (; ow < OW; ow++) {
                            int iw = ow * 2;
                            int8_t m = row0[iw];
                            int8_t v;
                            v = row0[iw + 1]; if (v > m) m = v;
                            v = row0[iw + 2]; if (v > m) m = v;
                            v = row1[iw];     if (v > m) m = v;
                            v = row1[iw + 1]; if (v > m) m = v;
                            v = row1[iw + 2]; if (v > m) m = v;
                            v = row2[iw];     if (v > m) m = v;
                            v = row2[iw + 1]; if (v > m) m = v;
                            v = row2[iw + 2]; if (v > m) m = v;
                            out_row[ow] = m;
                        }
                    }
                } else {
                    for (int oh = 0; oh < OH; oh++) {
                        int ih_base = oh * SH;
                        int8_t *out_row = output_base + oh * OW;
                        for (int ow = 0; ow < OW; ow++) {
                            int iw_base = ow * SW;
                            int8_t m = INT8_MIN;
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = ih_base + kh;
                                const int8_t *row = input_base + ih * IW;
                                for (int kw = 0; kw < KW; kw++) {
                                    int8_t v = row[iw_base + kw];
                                    if (v > m) m = v;
                                }
                            }
                            out_row[ow] = m;
                        }
                    }
                }
            } else {
                for (int oh = 0; oh < OH; oh++) {
                    int ih_base = oh*SH - PH;
                    int kh_start = (ih_base < 0) ? (-ih_base + DH - 1) / DH : 0;
                    int kh_end = ((ih_base + (KH-1)*DH) >= IH) ? 
                        ((IH - 1 - ih_base) / DH + 1) : KH;
                    int8_t *out_row = output_base + oh * OW;
                    
                    for (int ow = 0; ow < OW; ow++) {
                        int iw_base = ow*SW - PW;
                        int kw_start = (iw_base < 0) ? (-iw_base + DW - 1) / DW : 0;
                        int kw_end = ((iw_base + (KW-1)*DW) >= IW) ? 
                            ((IW - 1 - iw_base) / DW + 1) : KW;
                        
                        int8_t m = INT8_MIN;
                        for (int kh = kh_start; kh < kh_end; kh++) {
                            int ih = ih_base + kh*DH;
                            const int8_t *row = input_base + ih * IW;
                            for (int kw = kw_start; kw < kw_end; kw++) {
                                int8_t v = row[iw_base + kw*DW];
                                if (v > m) m = v;
                            }
                        }
                        out_row[ow] = m;
                    }
                }
            }
        }
    }
}