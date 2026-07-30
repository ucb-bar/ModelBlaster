void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                 int N, int C, int IH, int IW, int scale) {
    int OH = IH * scale;
    int OW = IW * scale;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *input_channel = input + (n * C + c) * IH * IW;
            int8_t *output_channel = output + (n * C + c) * OH * OW;
            
            if (scale == 2) {
                for (int ih = 0; ih < IH; ih++) {
                    const int8_t *input_row = input_channel + ih * IW;
                    int8_t *out_row0 = output_channel + (ih * 2) * OW;
                    int8_t *out_row1 = out_row0 + OW;
                    
                    int iw = 0;
                    for (; iw + 4 <= IW; iw += 4) {
                        int8_t p0 = input_row[iw];
                        int8_t p1 = input_row[iw + 1];
                        int8_t p2 = input_row[iw + 2];
                        int8_t p3 = input_row[iw + 3];
                        
                        int ow = iw * 2;
                        out_row0[ow] = p0;
                        out_row0[ow + 1] = p0;
                        out_row1[ow] = p0;
                        out_row1[ow + 1] = p0;
                        
                        out_row0[ow + 2] = p1;
                        out_row0[ow + 3] = p1;
                        out_row1[ow + 2] = p1;
                        out_row1[ow + 3] = p1;
                        
                        out_row0[ow + 4] = p2;
                        out_row0[ow + 5] = p2;
                        out_row1[ow + 4] = p2;
                        out_row1[ow + 5] = p2;
                        
                        out_row0[ow + 6] = p3;
                        out_row0[ow + 7] = p3;
                        out_row1[ow + 6] = p3;
                        out_row1[ow + 7] = p3;
                    }
                    
                    for (; iw < IW; iw++) {
                        int8_t pixel = input_row[iw];
                        int ow = iw * 2;
                        out_row0[ow] = pixel;
                        out_row0[ow + 1] = pixel;
                        out_row1[ow] = pixel;
                        out_row1[ow + 1] = pixel;
                    }
                }
            } else {
                for (int ih = 0; ih < IH; ih++) {
                    const int8_t *input_row = input_channel + ih * IW;
                    int8_t *output_base = output_channel + ih * scale * OW;
                    
                    for (int iw = 0; iw < IW; iw++) {
                        int8_t pixel = input_row[iw];
                        int8_t *output_block = output_base + iw * scale;
                        
                        for (int s = 0; s < scale; s++) {
                            int8_t *output_row = output_block + s * OW;
                            for (int t = 0; t < scale; t++) {
                                output_row[t] = pixel;
                            }
                        }
                    }
                }
            }
        }
    }
}