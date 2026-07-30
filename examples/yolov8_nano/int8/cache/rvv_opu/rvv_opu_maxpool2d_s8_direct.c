void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    
    const ptrdiff_t c_stride_bytes = (ptrdiff_t)IH * (ptrdiff_t)IW;
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * (ptrdiff_t)OW;
    
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int c = 0;
                while (c < C) {
                    size_t vl = __riscv_vsetvl_e8m8((size_t)(C - c));
                    
                    vint8m8_t vmax = __riscv_vmv_v_x_i8m8(INT8_MIN, vl);
                    
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh*SH - PH + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow*SW - PW + kw*DW;
                            if (iw < 0 || iw >= IW) continue;
                            
                            const int8_t *input_ptr = input + ((n*C + c)*IH + ih)*IW + iw;
                            vint8m8_t v = __riscv_vlse8_v_i8m8(input_ptr, c_stride_bytes, vl);
                            vmax = __riscv_vmax_vv_i8m8(vmax, v, vl);
                        }
                    }
                    
                    int8_t *output_ptr = output + ((n*C + c)*OH + oh)*OW + ow;
                    __riscv_vsse8_v_i8m8(output_ptr, out_c_stride_bytes, vmax, vl);
                    c += (int)vl;
                }
            }
        }
    }
}