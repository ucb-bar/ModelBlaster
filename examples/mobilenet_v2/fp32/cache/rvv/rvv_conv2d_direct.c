void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    const ptrdiff_t oc_stride_bytes =
        (ptrdiff_t)IC * (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)sizeof(float);
    const ptrdiff_t out_oc_stride_bytes =
        (ptrdiff_t)OH * (ptrdiff_t)OW * (ptrdiff_t)sizeof(float);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            int ih = oh * SH - PH;
            for (int ow = 0; ow < OW; ow++) {
                int oc = 0;
                int iw = ow * SW - PW;
                while (oc < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc));
                    vfloat32m4_t vacc = bias
                        ? __riscv_vle32_v_f32m4(bias + oc, vl)
                        : __riscv_vfmv_v_f_f32m4(0.0f, vl);
                    for (int ic = 0; ic < IC; ic++) {
                        const float *input_ptr = input + ((n*IC + ic)*IH + ih)*IW + iw;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih_cur = kh;
                            if ((unsigned int)ih_cur < (unsigned int)IH) {
                                const float *input_row = input_ptr + ih_cur * IW;
                                const float *w_ptr =
                                    weight + ((oc*IC + ic)*KH + kh)*KW;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw_cur = kw;
                                    if ((unsigned int)(iw + iw_cur) < (unsigned int)IW) {
                                        float v = input_row[iw_cur];
                                        vfloat32m4_t vw = __riscv_vlse32_v_f32m4(
                                            w_ptr + kw, oc_stride_bytes, vl);
                                        vacc = __riscv_vfmacc_vf_f32m4(vacc, v, vw, vl);
                                    }
                                }
                            }
                        }
                    }
                    __riscv_vsse32_v_f32m4(
                        output + ((n*OC + oc)*OH + oh)*OW + ow,
                        out_oc_stride_bytes, vacc, vl);
                    oc += (int)vl;
                }
            }
        }
    }
}