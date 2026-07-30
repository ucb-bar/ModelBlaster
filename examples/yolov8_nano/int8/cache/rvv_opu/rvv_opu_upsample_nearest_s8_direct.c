void kernel_upsample_nearest_s8(const int8_t *input, int8_t *output,
                                 int N, int C, int IH, int IW, int scale) {
    int OH = IH * scale, OW = IW * scale;
    
    if (scale == 2) {
        /* For scale=2, vectorize over the output width dimension.
         * Each vector lane computes one output column, reading from
         * input[iw/2] where iw is the output column index.
         * Process both output rows (oh and oh+1) together. */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int ih = 0; ih < IH; ih++) {
                    const int8_t *input_row = input + ((n*C + c)*IH + ih)*IW;
                    int8_t *output_row0 = output + ((n*C + c)*OH + ih*2)*OW;
                    int8_t *output_row1 = output_row0 + OW;
                    
                    int ow = 0;
                    size_t vl;
                    
                    /* Process output columns in vector chunks. Each lane reads
                     * from input[ow/2], which is a strided load with stride=1
                     * but offset pattern [0,0,1,1,2,2,...]. Use indexed load. */
                    while (ow < OW) {
                        vl = __riscv_vsetvl_e8m4(OW - ow);
                        
                        /* Build index vector: [ow/2, (ow+1)/2, (ow+2)/2, ...].
                         * For scale=2, this is [ow>>1, (ow+1)>>1, ...]. */
                        vuint16m8_t vidx_base = __riscv_vid_v_u16m8(vl);
                        vuint16m8_t vow = __riscv_vadd_vx_u16m8(vidx_base, (uint16_t)ow, vl);
                        vuint16m8_t vidx = __riscv_vsrl_vx_u16m8(vow, 1, vl);
                        
                        /* Indexed load from input row. */
                        vint8m4_t vin = __riscv_vluxei16_v_i8m4(input_row, vidx, vl);
                        
                        /* Store to both output rows. */
                        __riscv_vse8_v_i8m4(output_row0 + ow, vin, vl);
                        __riscv_vse8_v_i8m4(output_row1 + ow, vin, vl);
                        
                        ow += (int)vl;
                    }
                }
            }
        }
    } else {
        /* General scale: vectorize over output width, indexed load from input.
         * Process all scale output rows for each input row together. */
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int ih = 0; ih < IH; ih++) {
                    const int8_t *input_row = input + ((n*C + c)*IH + ih)*IW;
                    
                    for (int s = 0; s < scale; s++) {
                        int8_t *output_row = output + ((n*C + c)*OH + ih*scale + s)*OW;
                        
                        int ow = 0;
                        size_t vl;
                        
                        while (ow < OW) {
                            vl = __riscv_vsetvl_e8m4(OW - ow);
                            
                            /* Index vector: [ow/scale, (ow+1)/scale, ...]. */
                            vuint16m8_t vidx_base = __riscv_vid_v_u16m8(vl);
                            vuint16m8_t vow = __riscv_vadd_vx_u16m8(vidx_base, (uint16_t)ow, vl);
                            vuint16m8_t vidx = __riscv_vdivu_vx_u16m8(vow, (uint16_t)scale, vl);
                            
                            vint8m4_t vin = __riscv_vluxei16_v_i8m4(input_row, vidx, vl);
                            
                            __riscv_vse8_v_i8m4(output_row + ow, vin, vl);
                            
                            ow += (int)vl;
                        }
                    }
                }
            }
        }
    }
}