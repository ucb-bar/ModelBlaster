void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    /* OC-vectorized i8 conv, RVV V1.0.
     *
     * LMUL choice: i32 acc at LMUL=2 means we need i16 ops at LMUL=1
     * and i8 ops at LMUL=1/2 (fractional) to stay element-count-matched
     * across widening intrinsics. vsetvl_e32m2(remaining_OC) gives one
     * vl that's reused for the i8 strided load (vlse8_v_i8mf2), the
     * widening offset add (vwadd_vx_i16m1), and the widening MAC
     * (vwmacc_vx_i32m2). VLEN=128 -> vlmax for e32m2 == 8 OC elems per
     * chunk. */
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    const ptrdiff_t oc_stride_bytes =
        (ptrdiff_t)IC * (ptrdiff_t)KH * (ptrdiff_t)KW * (ptrdiff_t)sizeof(int8_t);

    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int oc_base = 0;
                while (oc_base < OC) {
                    size_t vl = __riscv_vsetvl_e32m2((size_t)(OC - oc_base));
                    vint32m2_t vacc;
                    if (bias != NULL) {
                        vacc = __riscv_vle32_v_i32m2(bias + oc_base, vl);
                    } else {
                        vacc = __riscv_vmv_v_x_i32m2(0, vl);
                    }
                    for (int ic = 0; ic < IC; ic++) {
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            int row_in_bounds = (ih >= 0 && ih < IH);
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int8_t in_byte = 0;
                                if (row_in_bounds && iw >= 0 && iw < IW) {
                                    in_byte = input[((n*IC + ic)*IH + ih)*IW + iw];
                                }
                                int32_t in_v = (int32_t)in_byte + input_offset;
                                const int8_t *wp = weight
                                    + (size_t)oc_base * (size_t)IC * KH * KW
                                    + ((size_t)ic * KH + kh) * KW + kw;
                                /* strided i8 load at LMUL=1/2 so the
                                 * subsequent widen produces i16 LMUL=1. */
                                vint8mf2_t vw8 = __riscv_vlse8_v_i8mf2(
                                    wp, oc_stride_bytes, vl);
                                /* i8mf2 -> i16m1, fold filter_offset in. */
                                vint16m1_t vw16 = __riscv_vwadd_vx_i16m1(
                                    vw8, (int16_t)filter_offset, vl);
                                /* i32m2 += i16m1 * i16-scalar-extended-from-int. */
                                vacc = __riscv_vwmacc_vx_i32m2(
                                    vacc, (int16_t)in_v, vw16, vl);
                            }
                        }
                    }
                    /* Pull lanes out and run the Q0.31 requantize
                     * scalar — bit-exact match to the reference. */
                    int32_t lane[64];   /* vlmax for e32m2 at VLEN=512 is 32; 64 is generous */
                    __riscv_vse32_v_i32m2(lane, vacc, vl);
                    for (size_t j = 0; j < vl; j++) {
                        int32_t acc = lane[j];
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + (1LL << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            int32_t round = (1 << (output_shift - 1));
                            scaled = (scaled + round) >> output_shift;
                        } else if (output_shift < 0) {
                            scaled = scaled << (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC + oc_base + (int)j)*OH + oh)*OW + ow] =
                            (int8_t)scaled;
                    }
                    oc_base += (int)vl;
                }
            }
        }
    }
}
