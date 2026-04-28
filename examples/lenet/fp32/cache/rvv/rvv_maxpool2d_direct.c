void kernel_maxpool2d(const float *input, float *output, int N, int C, int IH, int IW, int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            size_t vl;
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow += vl) {
                    vl = __riscv_vsetvl_e32m1(OW - ow);
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    vfloat32m1_t vinput = __riscv_vlse32_v_f32m1(
                        input + ((n*C + c)*IH + ih0)*IW + iw0,
                        (ptrdiff_t)SW * (ptrdiff_t)sizeof(float), vl);
                    vfloat32m1_t vmax = vinput;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = (kh == 0) ? 1 : 0; kw < KW; kw++) {
                            vinput = __riscv_vlse32_v_f32m1(
                                input + ((n*C + c)*IH + ih0 + kh)*IW + iw0 + kw,
                                (ptrdiff_t)SW * (ptrdiff_t)sizeof(float), vl);
                            vmax = __riscv_vfmax_vv_f32m1(vmax, vinput, vl);
                        }
                    }
                    __riscv_vse32_v_f32m1(output + ((n*C + c)*OH + oh)*OW + ow, vmax, vl);
                }
            }
        }
    }
}