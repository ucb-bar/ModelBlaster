void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                int ih0 = oh * SH;
                for (int ow = 0; ow < OW; ) {
                    size_t vl = __riscv_vsetvl_e8m4(OW - ow);
                    vint8m4_t vmax = __riscv_vmv_v_x_i8m4(INT8_MIN, vl);
                    int iw0 = ow * SW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = ih0 + kh;
                        if (ih >= 0 && ih < IH) {
                            vint8m4_t vinput = __riscv_vlse8_v_i8m4(
                                input + ((n*C + c)*IH + ih)*IW + iw0,
                                (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t), vl);
                            vmax = __riscv_vmax_vv_i8m4(vmax, vinput, vl);
                            for (int kw = 1; kw < KW; kw++) {
                                vinput = __riscv_vlse8_v_i8m4(
                                    input + ((n*C + c)*IH + ih)*IW + iw0 + kw,
                                    (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t), vl);
                                vmax = __riscv_vmax_vv_i8m4(vmax, vinput, vl);
                            }
                        }
                    }
                    __riscv_vse8_v_i8m4(
                        output + ((n*C + c)*OH + oh)*OW + ow, vmax, vl);
                    ow += vl;
                }
            }
        }
    }
}