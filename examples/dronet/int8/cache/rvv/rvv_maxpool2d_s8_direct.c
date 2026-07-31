void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow += (int)({size_t vl = __riscv_vsetvl_e8m4(OW - ow); vl;})) {
                    /* vectorize over ow dimension */
                    size_t vl = __riscv_vsetvl_e8m4(OW - ow);
                    /* init accumulator to INT8_MIN */
                    vint8m4_t vacc = __riscv_vmv_v_x_i8m4((int8_t)(-128), vl);
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh * DH;
                        if (ih < 0 || ih >= IH) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            /* Each lane l corresponds to output column ow+l,
                             * so input column = (ow+l)*SW - PW + kw*DW.
                             * Base iw for lane 0: */
                            int iw0 = ow * SW - PW + kw * DW;
                            /* For SW==1: adjacent lanes are adjacent in input */
                            /* For SW>1: stride is SW floats apart */
                            /* We need to handle out-of-bounds lanes.
                             * Check if any lane could be in bounds first. */
                            /* iw for lane l = iw0 + l*SW */
                            /* min iw = iw0 (l=0), max iw = iw0 + (vl-1)*SW */
                            int iw_max = iw0 + (int)(vl - 1) * SW;
                            if (iw0 >= IW || iw_max < 0) continue;
                            /* Some lanes may be out of bounds. Use masked load. */
                            /* Compute per-lane iw validity:
                             * lane l is valid if iw0 + l*SW >= 0 && iw0 + l*SW < IW */
                            /* For simplicity, if all lanes are in bounds, use strided load.
                             * Otherwise fall back to scalar gather. */
                            int all_valid = (iw0 >= 0 && iw_max < IW);
                            if (all_valid) {
                                const int8_t *src = input + ((n * C + c) * IH + ih) * IW + iw0;
                                vint8m4_t vval;
                                if (SW == 1) {
                                    vval = __riscv_vle8_v_i8m4(src, vl);
                                } else {
                                    vval = __riscv_vlse8_v_i8m4(src, (ptrdiff_t)SW * sizeof(int8_t), vl);
                                }
                                vacc = __riscv_vmax_vv_i8m4(vacc, vval, vl);
                            } else {
                                /* scalar fallback for boundary lanes */
                                /* extract current acc, update per lane, put back */
                                /* Use a temporary array */
                                int8_t tmp[256];
                                __riscv_vse8_v_i8m4(tmp, vacc, vl);
                                for (size_t l = 0; l < vl; l++) {
                                    int iw = iw0 + (int)l * SW;
                                    if (iw < 0 || iw >= IW) continue;
                                    int8_t v = input[((n * C + c) * IH + ih) * IW + iw];
                                    if (v > tmp[l]) tmp[l] = v;
                                }
                                vacc = __riscv_vle8_v_i8m4(tmp, vl);
                            }
                        }
                    }
                    __riscv_vse8_v_i8m4(output + ((n * C + c) * OH + oh) * OW + ow, vacc, vl);
                }
            }
        }
    }
}