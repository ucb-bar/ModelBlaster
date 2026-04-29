void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow += __riscv_vsetvl_e8m1(OW - ow)) {
                    size_t vl = __riscv_vsetvl_e8m1(OW - ow);
                    vint8m1_t vmax = __riscv_vmv_v_x_i8m1(INT8_MIN, vl);
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = ih0 + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = iw0 + kw;
                            if (iw < 0 || iw >= IW) continue;
                            vint8m1_t vinput = __riscv_vlse8_v_i8m1(
                                input + ((n*C + c)*IH + ih)*IW + iw,
                                (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t), vl);
                            vmax = __riscv_vmax_vv_i8m1(vmax, vinput, vl);
                        }
                    }
                    __riscv_vse8_v_i8m1(
                        output + ((n*C + c)*OH + oh)*OW + ow, vmax, vl);
                }
            }
        }
    }
}