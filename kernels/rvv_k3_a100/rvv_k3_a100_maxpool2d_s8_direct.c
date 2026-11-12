/* source: curated */
/* algorithm: direct */
/* origin: A100 (VLEN 1024) maxpool2d_s8: no strided loads, no scalar boundary paths.
 * Per (n,c,oh): (1) vertical max over valid kh rows, unit-stride e8m8, into a
 * -128-padded scratch row; (2) horizontal window max from the scratch row using
 * unit-stride loads only (stride 2: load 2*vl bytes, de-interleave with vnsrl). */

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    int PWID = IW + 2*PW + 2*SW + 8;
    int8_t row[PWID];

    for (int i = 0; i < PW; i++) row[i] = -128;
    for (int i = PW + IW; i < PWID; i++) row[i] = -128;

    for (int nc = 0; nc < N*C; nc++) {
        const int8_t *in_nc = input + (size_t)nc*IH*IW;
        int8_t *out_nc = output + (size_t)nc*OH*OW;

        for (int oh = 0; oh < OH; oh++) {
            /* vertical */
            for (int x = 0; x < IW; ) {
                size_t vl = __riscv_vsetvl_e8m8(IW - x);
                vint8m8_t acc = __riscv_vmv_v_x_i8m8((int8_t)-128, vl);
                for (int kh = 0; kh < KH; kh++) {
                    int ih = oh*SH - PH + kh*DH;
                    if (ih < 0 || ih >= IH) continue;
                    acc = __riscv_vmax_vv_i8m8(acc,
                              __riscv_vle8_v_i8m8(in_nc + ih*IW + x, vl), vl);
                }
                __riscv_vse8_v_i8m8(row + PW + x, acc, vl);
                x += (int)vl;
            }
            /* horizontal */
            int8_t *out_row = out_nc + oh*OW;
            for (int ow = 0; ow < OW; ) {
                size_t vl = __riscv_vsetvl_e8m4(OW - ow);
                vint8m4_t acc = __riscv_vmv_v_x_i8m4((int8_t)-128, vl);
                if (SW == 1) {
                    for (int kw = 0; kw < KW; kw++)
                        acc = __riscv_vmax_vv_i8m4(acc,
                                  __riscv_vle8_v_i8m4(row + ow + kw*DW, vl), vl);
                } else if (SW == 2) {
                    for (int kw = 0; kw < KW; kw++) {
                        int pair = (DW == 1) && (kw + 1 < KW);
                        size_t vb = __riscv_vsetvl_e8m8(2*vl);
                        vint8m8_t raw = __riscv_vle8_v_i8m8(row + 2*ow + kw*DW, vb);
                        vuint16m8_t w = __riscv_vreinterpret_v_u8m8_u16m8(__riscv_vreinterpret_v_i8m8_u8m8(raw));
                        vuint8m4_t ev = __riscv_vnsrl_wx_u8m4(w, 0, vl);
                        acc = __riscv_vmax_vv_i8m4(acc,
                                  __riscv_vreinterpret_v_u8m4_i8m4(ev), vl);
                        if (pair) {
                            vuint8m4_t od = __riscv_vnsrl_wx_u8m4(w, 8, vl);
                            acc = __riscv_vmax_vv_i8m4(acc,
                                      __riscv_vreinterpret_v_u8m4_i8m4(od), vl);
                            kw++;
                        }
                    }
                } else {
                    for (int kw = 0; kw < KW; kw++)
                        acc = __riscv_vmax_vv_i8m4(acc,
                                  __riscv_vlse8_v_i8m4(row + ow*SW + kw*DW, SW, vl), vl);
                }
                __riscv_vse8_v_i8m4(out_row + ow, acc, vl);
                ow += (int)vl;
            }
        }
    }
}
