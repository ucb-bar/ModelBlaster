void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW) {
    int OH = (IH - KH) / SH + 1;
    int OW = (IW - KW) / SW + 1;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                size_t vl;
                int ow = 0;
                for (; ow + (int)__riscv_vsetvlmax_e8m1() <= OW;
                     ow += (int)__riscv_vsetvlmax_e8m1()) {
                    vl = __riscv_vsetvlmax_e8m1();
                    vint8m1_t vm = __riscv_vmv_v_x_i8m1(INT8_MIN, vl);
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            vint8m1_t vv = __riscv_vlse8_v_i8m1(
                                input + ((n*C + c)*IH + ih0 + kh)*IW + iw0 + kw,
                                (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t), vl);
                            vm = __riscv_vmax_vv_i8m1(vm, vv, vl);
                        }
                    }
                    __riscv_vse8_v_i8m1(
                        output + ((n*C + c)*OH + oh)*OW + ow, vm, vl);
                }
                for (; ow < OW; ow++) {
                    vl = __riscv_vsetvl_e8m1((size_t)(OW - ow));
                    vint8m1_t vm = __riscv_vmv_v_x_i8m1(INT8_MIN, vl);
                    int ih0 = oh * SH;
                    int iw0 = ow * SW;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            vint8m1_t vv = __riscv_vlse8_v_i8m1(
                                input + ((n*C + c)*IH + ih0 + kh)*IW + iw0 + kw,
                                (ptrdiff_t)SW * (ptrdiff_t)sizeof(int8_t), vl);
                            vm = __riscv_vmax_vv_i8m1(vm, vv, vl);
                        }
                    }
                    __riscv_vse8_v_i8m1(
                        output + ((n*C + c)*OH + oh)*OW + ow, vm, vl);
                }
            }
        }
    }
}