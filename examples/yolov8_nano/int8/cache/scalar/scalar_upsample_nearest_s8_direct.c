void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                 int N, int C, int IH, int IW, int scale) {
    int OH = IH * scale;
    int OW = IW * scale;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *input_channel = input + (n * C + c) * IH * IW;
            int8_t *output_channel = output + (n * C + c) * OH * OW;
            
            for (int ih = 0; ih < IH; ih++) {
                const int8_t *input_row = input_channel + ih * IW;
                int8_t *output_row_base = output_channel + ih * scale * OW;
                
                // Build the first upsampled output row with 8-way unroll
                int8_t *first_output_row = output_row_base;
                int iw = 0;
                for (; iw + 8 <= IW; iw += 8) {
                    int8_t p0 = input_row[iw + 0];
                    int8_t p1 = input_row[iw + 1];
                    int8_t p2 = input_row[iw + 2];
                    int8_t p3 = input_row[iw + 3];
                    int8_t p4 = input_row[iw + 4];
                    int8_t p5 = input_row[iw + 5];
                    int8_t p6 = input_row[iw + 6];
                    int8_t p7 = input_row[iw + 7];
                    
                    int8_t *out_ptr = first_output_row + iw * scale;
                    for (int s = 0; s < scale; s++) {
                        out_ptr[s] = p0;
                        out_ptr[scale + s] = p1;
                        out_ptr[2*scale + s] = p2;
                        out_ptr[3*scale + s] = p3;
                        out_ptr[4*scale + s] = p4;
                        out_ptr[5*scale + s] = p5;
                        out_ptr[6*scale + s] = p6;
                        out_ptr[7*scale + s] = p7;
                    }
                }
                
                // Tail loop for remaining pixels
                for (; iw < IW; iw++) {
                    int8_t pixel = input_row[iw];
                    for (int s = 0; s < scale; s++) {
                        first_output_row[iw * scale + s] = pixel;
                    }
                }
                
                // Replicate the first output row vertically with 8-byte unroll
                for (int s = 1; s < scale; s++) {
                    int8_t *dest_row = output_row_base + s * OW;
                    int ow = 0;
                    for (; ow + 8 <= OW; ow += 8) {
                        dest_row[ow + 0] = first_output_row[ow + 0];
                        dest_row[ow + 1] = first_output_row[ow + 1];
                        dest_row[ow + 2] = first_output_row[ow + 2];
                        dest_row[ow + 3] = first_output_row[ow + 3];
                        dest_row[ow + 4] = first_output_row[ow + 4];
                        dest_row[ow + 5] = first_output_row[ow + 5];
                        dest_row[ow + 6] = first_output_row[ow + 6];
                        dest_row[ow + 7] = first_output_row[ow + 7];
                    }
                    for (; ow < OW; ow++) {
                        dest_row[ow] = first_output_row[ow];
                    }
                }
            }
        }
    }
}