void kernel_conv2d_dw(const float *input, const float *weight, const float *bias, float *output, int N, int C, int IH, int IW, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t out_c_stride_bytes = (ptrdiff_t)OH * (ptrdiff_t)OW * (ptrdiff_t)sizeof(float);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int c = 0;
                while (c < C) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(C - c));

                    vfloat32m4_t vacc = bias
                        ? __riscv_vle32_v_f32m4(bias + c, vl)
                        : __riscv_vfmv_v_f_f32m4(0.0f, vl);

                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;
                            float v = input[((n*C + c)*IH + ih)*IW + iw];
                            const float *w_ptr = weight + (c*KH + kh)*KW + kw;
                            vfloat32m4_t vw = __riscv_vlse32_v_f32m4(
                                w_ptr, (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)sizeof(float), vl);
                            vacc = __riscv_vfmacc_vf_f32m4(vacc, v, vw, vl);
                        }
                    }

                    __riscv_vsse32_v_f32m4(
                        output + ((n*C + c)*OH + oh)*OW + ow,
                        out_c_stride_bytes, vacc, vl);
                    c += (int)vl;
                }
            }
        }
    }
}