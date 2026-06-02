void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;

    const int OC_BUDGET_BYTES = 24 * 1024;
    const int oc_slab_bytes = IC * KH * KW;
    int TILE_OC = OC_BUDGET_BYTES / (oc_slab_bytes > 0 ? oc_slab_bytes : 1);
    size_t vlmax = __riscv_vsetvlmax_e32m4();
    if (TILE_OC < (int)vlmax) TILE_OC = (int)vlmax;
    if (TILE_OC > OC) TILE_OC = OC;

    const ptrdiff_t out_oc_stride = (ptrdiff_t)OH * (ptrdiff_t)OW;
    const int symmetric = (input_offset == 0 && filter_offset == 0);

    for (int oc_outer = 0; oc_outer < OC; oc_outer += TILE_OC) {
        int oc_end = oc_outer + TILE_OC;
        if (oc_end > OC) oc_end = OC;

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int oc = oc_outer;
                    while (oc < oc_end) {
                        size_t vl = __riscv_vsetvl_e32m4((size_t)(oc_end - oc));

                        vint32m4_t vacc;
                        if (bias != NULL)
                            vacc = __riscv_vle32_v_i32m4(bias + oc, vl);
                        else
                            vacc = __riscv_vmv_v_x_i32m4(0, vl);

                        if (symmetric) {
                            for (int ic = 0; ic < IC; ic++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    int ih = oh * SH - PH + kh;
                                    if (ih < 0 || ih >= IH) continue;
                                    for (int kw = 0; kw < KW; kw++) {
                                        int iw = ow * SW - PW + kw;
                                        if (iw < 0 || iw >= IW) continue;

                                        int8_t in_byte = input[((n*IC + ic)*IH + ih)*IW + iw];

                                        const int8_t *wp = weight
                                            + ((size_t)ic * KH * KW + (size_t)kh * KW + kw) * OC + oc;
                                        vint8m1_t vw8 = __riscv_vle8_v_i8m1(wp, vl);

                                        vint16m2_t prod = __riscv_vwmul_vx_i16m2(vw8, in_byte, vl);
                                        vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
                                    }
                                }
                            }
                        } else {
                            for (int ic = 0; ic < IC; ic++) {
                                for (int kh = 0; kh < KH; kh++) {
                                    int ih = oh * SH - PH + kh;
                                    if (ih < 0 || ih >= IH) continue;
                                    for (int kw = 0; kw < KW; kw++) {
                                        int iw = ow * SW - PW + kw;
                                        if (iw < 0 || iw >= IW) continue;

                                        int8_t in_byte = input[((n*IC + ic)*IH + ih)*IW + iw];
                                        int32_t in_v = (int32_t)in_byte + input_offset;

                                        const int8_t *wp = weight
                                            + ((size_t)ic * KH * KW + (size_t)kh * KW + kw) * OC + oc;
                                        vint8m1_t vw8 = __riscv_vle8_v_i8m1(wp, vl);

                                        vint16m2_t vw16 = __riscv_vwadd_vx_i16m2(
                                            vw8, (int16_t)filter_offset, vl);

                                        vacc = __riscv_vwmacc_vx_i32m4(
                                            vacc, (int16_t)in_v, vw16, vl);
                                    }
                                }
                            }
                        }

                        vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
                            vacc, output_multiplier, __RISCV_VXRM_RNU, vl);

                        vint16m2_t vout16;
                        if (output_shift < 0) {
                            vint32m4_t vshifted = __riscv_vsll_vx_i32m4(
                                vscaled, (size_t)(-output_shift), vl);
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vshifted, 0, __RISCV_VXRM_RNU, vl);
                        } else {
                            vout16 = __riscv_vnclip_wx_i16m2(
                                vscaled, (size_t)(output_shift < 32 ? output_shift : 31), 
                                __RISCV_VXRM_RNU, vl);
                        }

                        vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
                        vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
                        vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);

                        vint8m1_t vout8 = __riscv_vnsra_wx_i8m1(vout16, 0, vl);

                        int8_t *op = output
                            + ((size_t)n * OC + oc) * OH * OW
                            + (size_t)oh * OW + ow;
                        __riscv_vsse8_v_i8m1(op, out_oc_stride, vout8, vl);

                        oc += (int)vl;
                    }
                }
            }
        }
    }
}