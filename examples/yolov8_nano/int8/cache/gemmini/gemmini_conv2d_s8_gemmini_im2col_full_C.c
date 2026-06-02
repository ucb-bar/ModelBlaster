void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    static elem_t ws_im2col [16 * 256 * 9] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[16 * 256]     __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    if (input_offset != 0 || filter_offset != 0
            || (size_t)(N * IH * IW * IC) > 512 * 1024
            || (size_t)(K_inner * OC)      > 512 * 1024
            || (size_t)(N * OH * OW * OC)  > 512 * 1024
            || K_inner * 16                > 16 * 256 * 9
            || OC * 16                     > 16 * 256) {
        for (int n = 0; n < N; n++) {
            for (int oc = 0; oc < OC; oc++) {
                for (int oh = 0; oh < OH; oh++) {
                    for (int ow = 0; ow < OW; ow++) {
                        int32_t acc = bias ? bias[oc] : 0;
                        for (int ic = 0; ic < IC; ic++) {
                            for (int kh = 0; kh < KH; kh++) {
                                int ih = oh * SH - PH + kh;
                                for (int kw = 0; kw < KW; kw++) {
                                    int iw = ow * SW - PW + kw;
                                    int32_t in_v;
                                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                                        in_v = input_offset;
                                    else
                                        in_v = (int32_t)input[((n*IC+ic)*IH+ih)*IW+iw]
                                             + input_offset;
                                    acc += in_v * ((int32_t)weight[((kh*KW+kw)*IC+ic)*OC+oc]
                                                   + filter_offset);
                                }
                            }
                        }
                        int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                        prod = (prod + ((int64_t)1 << 30)) >> 31;
                        int32_t scaled = (int32_t)prod;
                        if (output_shift > 0) {
                            scaled = (int32_t)(((int64_t)scaled
                                + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                        } else if (output_shift < 0) {
                            scaled <<= (-output_shift);
                        }
                        scaled += output_offset;
                        if (scaled < activation_min) scaled = activation_min;
                        if (scaled > activation_max) scaled = activation_max;
                        output[((n*OC+oc)*OH+oh)*OW+ow] = (int8_t)scaled;
                    }
                }
            }
        }
        return;
    }

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");
    gemmini_flush(0);

    for (int tile_i = 0; tile_i < total_out; tile_i += 16) {
        int tile_rows = total_out - tile_i < 16 ? total_out - tile_i : 16;

        for (int i = 0; i < 16; i++) {
            elem_t *row = &ws_im2col[i * K_inner];
            if (i >= tile_rows) {
                for (int k = 0; k < K_inner; k++) row[k] = 0;
                continue;
            }
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);

            int base_ih = oh_idx * SH - PH;
            int base_iw = ow_idx * SW - PW;

            int k_off = 0;
            for (int kh = 0; kh < KH; kh++) {
                int ih = base_ih + kh;
                int ih_valid = (ih >= 0 && ih < IH);
                for (int kw = 0; kw < KW; kw++) {
                    int iw = base_iw + kw;
                    if (ih_valid && iw >= 0 && iw < IW) {
                        const int8_t *src = &input[((n_idx*IC)*IH + ih)*IW + iw];
                        int ic = 0;
                        for (; ic + 4 <= IC; ic += 4) {
                            row[k_off + ic + 0] = src[(ic+0)*IH*IW];
                            row[k_off + ic + 1] = src[(ic+1)*IH*IW];
                            row[k_off + ic + 2] = src[(ic+2)*IH*IW];
                            row[k_off + ic + 3] = src[(ic+3)*IH*IW];
                        }
                        for (; ic < IC; ic++) {
                            row[k_off + ic] = src[ic*IH*IW];
                        }
                    } else {
                        for (int ic = 0; ic < IC; ic++) {
                            row[k_off + ic] = 0;
                        }
                    }
                    k_off += IC;
                }
            }
        }

        asm volatile("fence" ::: "memory");

        tiled_matmul_auto(
            16, OC, K_inner,
            ws_im2col, weight,
            (const void *)bias, (void *)ws_acc_out,
            K_inner, OC, OC, OC,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
            bias != NULL,
            false, false,
            true, false,
            0, WS
        );

        gemmini_fence();
        gemmini_flush(0);

        for (int i = 0; i < tile_rows; i++) {
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            
            int oc = 0;
            for (; oc + 4 <= OC; oc += 4) {
                int32_t acc0 = ws_acc_out[i * OC + oc + 0];
                int32_t acc1 = ws_acc_out[i * OC + oc + 1];
                int32_t acc2 = ws_acc_out[i * OC + oc + 2];
                int32_t acc3 = ws_acc_out[i * OC + oc + 3];
                
                int64_t prod0 = (int64_t)acc0 * (int64_t)output_multiplier;
                int64_t prod1 = (int64_t)acc1 * (int64_t)output_multiplier;
                int64_t prod2 = (int64_t)acc2 * (int64_t)output_multiplier;
                int64_t prod3 = (int64_t)acc3 * (int64_t)output_multiplier;
                
                prod0 = (prod0 + ((int64_t)1 << 30)) >> 31;
                prod1 = (prod1 + ((int64_t)1 << 30)) >> 31;
                prod2 = (prod2 + ((int64_t)1 << 30)) >> 31;
                prod3 = (prod3 + ((int64_t)1 << 30)) >> 31;
                
                int32_t s0 = (int32_t)prod0;
                int32_t s1 = (int32_t)prod1;
                int32_t s2 = (int32_t)prod2;
                int32_t s3 = (int32_t)prod3;
                
                if (output_shift > 0) {
                    s0 = (int32_t)(((int64_t)s0 + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                    s1 = (int32_t)(((int64_t)s1 + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                    s2 = (int32_t)(((int64_t)s2 + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                    s3 = (int32_t)(((int64_t)s3 + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                } else if (output_shift < 0) {
                    int neg_shift = -output_shift;
                    s0 <<= neg_shift;
                    s1 <<= neg_shift;
                    s2 <<= neg_shift;
                    s3 <<= neg_shift;
                }
                
                s0 += output_offset;
                s1 += output_offset;
                s2 += output_offset;
                s3 += output_offset;
                
                if (s0 < activation_min) s0 = activation_min;
                if (s0 > activation_max) s0 = activation_max;
                if (s1 < activation_min) s1 = activation_min;
                if (s1 > activation_max) s1 = activation_max;
                if (s2 < activation_min) s2 = activation_min;
                if (s2 > activation_max) s2 = activation_max;
                if (s3 < activation_min) s3 = activation_min;
                if (s3 > activation_max) s3 = activation_max;
                
                output[((n_idx*OC + (oc+0))*OH + oh_idx)*OW + ow_idx] = (int8_t)s0;
                output[((n_idx*OC + (oc+1))*OH + oh_idx)*OW + ow_idx] = (int8_t)s1;
                output[((n_idx*OC + (oc+2))*OH + oh_idx)*OW + ow_idx] = (int8_t)s2;
                output[((n_idx*OC + (oc+3))*OH + oh_idx)*OW + ow_idx] = (int8_t)s3;
            }
            for (; oc < OC; oc++) {
                int32_t acc = ws_acc_out[i * OC + oc];
                int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                prod = (prod + ((int64_t)1 << 30)) >> 31;
                int32_t scaled = (int32_t)prod;
                if (output_shift > 0) {
                    scaled = (int32_t)(((int64_t)scaled
                        + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                } else if (output_shift < 0) {
                    scaled <<= (-output_shift);
                }
                scaled += output_offset;
                if (scaled < activation_min) scaled = activation_min;
                if (scaled > activation_max) scaled = activation_max;
                output[((n_idx*OC + oc)*OH + oh_idx)*OW + ow_idx] = (int8_t)scaled;
            }
        }
    }
}