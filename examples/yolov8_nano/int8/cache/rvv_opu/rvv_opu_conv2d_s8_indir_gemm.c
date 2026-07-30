void kernel_conv2d_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int N, int IC, int IH, int IW, int OC, int KH, int KW, int SH, int SW, int PH, int PW, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    
    const ptrdiff_t oc_stride_bytes = (ptrdiff_t)IC * (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)sizeof(int8_t);
    const ptrdiff_t out_oc_stride_bytes = (ptrdiff_t)OH * (ptrdiff_t)OW * (ptrdiff_t)sizeof(int8_t);
    
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc = 0;
                while (oc < OC) {
                    size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc));
                    
                    vint32m4_t vacc = bias
                        ? __riscv_vle32_v_i32m4(bias + oc, vl)
                        : __riscv_vmv_v_x_i32m4(0, vl);
                    
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            if (ih < 0 || ih >= IH) continue;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                if (iw < 0 || iw >= IW) continue;
                                
                                int8_t v = input[((n * IC + ic) * IH + ih) * IW + iw];
                                const int8_t *w_ptr = weight + ((oc * IC + ic) * KH + kh) * KW + kw;
                                vint8m1_t vw = __riscv_vlse8_v_i8m1(w_ptr, oc_stride_bytes, vl);
                                
                                vint16m2_t prod = __riscv_vwmul_vx_i16m2(vw, v, vl);
                                vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
                            }
                        }
                    }
                    
                    size_t vlmax = __riscv_vsetvlmax_e32m4();
                    for (size_t lane = 0; lane < vl; lane++) {
                        vint32m4_t shifted = __riscv_vslidedown_vx_i32m4(vacc, lane, vlmax);
                        int32_t acc = __riscv_vmv_x_s_i32m4_i32(shifted);
                        
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + (1LL << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (scaled + (1 << (output_shift - 1))) >> output_shift;
                        } else {
                            scaled = scaled << (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        
                        int oc_idx = oc + (int)lane;
                        if (oc_idx < OC) {
                            output[((n * OC + oc_idx) * OH + oh) * OW + ow] = (int8_t)scaled;
                        }
                    }
                    
                    oc += (int)vl;
                }
            }
        }
    }
}