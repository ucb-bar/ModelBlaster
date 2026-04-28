void kernel_conv2d(const float *input, const float *weight, const float *bias,
                   float *output,
                   int N, int IC, int IH, int IW, int OC,
                   int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                size_t vl;
                for (int ow = 0; ow < OW; ow += vl) {
                    vl = __riscv_vsetvl_e32m1(OW - ow);
                    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(
                        bias ? bias[oc] : 0.0f, vl);
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw0 = ow * SW - PW + kw;
                                vfloat32m1_t vinput = __riscv_vlse32_v_f32m1(
                                    input + ((n*IC + ic)*IH + ih)*IW + iw0,
                                    (ptrdiff_t)SW * (ptrdiff_t)sizeof(float), vl);
                                float w = weight[((oc*IC + ic)*KH + kh)*KW + kw];
                                vfloat32m1_t vweight = __riscv_vfmv_v_f_f32m1(w, vl);
                                vacc = __riscv_vfmacc_vv_f32m1(vacc, vinput, vweight, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m1(
                        output + ((n*OC + oc)*OH + oh)*OW + ow, vacc, vl);
                }
            }
        }
    }
}