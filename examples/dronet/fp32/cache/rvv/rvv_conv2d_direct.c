void kernel_conv2d(const float *input, const float *weight, const float *bias, float *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int oc0 = 0; oc0 < OC; oc0 += 1) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow += 1) {
                    size_t vl = __riscv_vsetvl_e32m1(1);
                    vfloat32m1_t vacc = bias ? __riscv_vle32_v_f32m1(bias + oc0, vl) : __riscv_vfmv_v_f_f32m1(0.0f, vl);
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                float v = input[((n*IC + ic)*IH + ih)*IW + iw];
                                float w = weight[((oc0*IC + ic)*KH + kh)*KW + kw];
                                vfloat32m1_t vinput = __riscv_vfmv_v_f_f32m1(v, vl);
                                vfloat32m1_t vweight = __riscv_vfmv_v_f_f32m1(w, vl);
                                vacc = __riscv_vfmacc_vv_f32m1(vacc, vinput, vweight, vl);
                            }
                        }
                    }
                    __riscv_vse32_v_f32m1(output + ((n*OC + oc0)*OH + oh)*OW + ow, vacc, vl);
                }
            }
        }
    }
}