#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

enum {
    WS_BYTES       = 512 * 1024,
    IM2COL_ELEMS   = DIM * 256 * 9,
    ACC_ELEMS      = DIM * 256,
};

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    static elem_t ws_im2col [IM2COL_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[ACC_ELEMS]    __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    if (input_offset != 0 || filter_offset != 0
            || K_inner * DIM > IM2COL_ELEMS
            || OC * DIM      > ACC_ELEMS) {
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

    for (int tile_i = 0; tile_i < total_out; tile_i += DIM) {
        int tile_rows = total_out - tile_i < DIM ? total_out - tile_i : DIM;

        for (int i = 0; i < DIM; i++) {
            elem_t *row = &ws_im2col[i * K_inner];
            if (i >= tile_rows) {
                for (int k = 0; k < K_inner; k++) row[k] = 0;
                continue;
            }
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            
            const int8_t *input_base = input + n_idx * IC * IH * IW;
            int col_idx = 0;
            
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh_idx * SH - PH + kh;
                int ih_valid = (ih >= 0 && ih < IH);
                
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow_idx * SW - PW + kw;
                    int iw_valid = (iw >= 0 && iw < IW);
                    
                    if (ih_valid && iw_valid) {
                        const int8_t *src = input_base + ih * IW + iw;
                        for (int c = 0; c < IC; c++) {
                            row[col_idx++] = src[c * IH * IW];
                        }
                    } else {
                        for (int c = 0; c < IC; c++) {
                            row[col_idx++] = 0;
                        }
                    }
                }
            }
        }

        asm volatile("fence" ::: "memory");

        tiled_matmul_auto(
            DIM, OC, K_inner,
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
            
            int8_t *out_base = output + ((n_idx * OC * OH + oh_idx) * OW + ow_idx);
            const acc_t *acc_row = ws_acc_out + i * OC;
            
            int oc = 0;
            for (; oc + 4 <= OC; oc += 4) {
                int32_t acc0 = acc_row[oc + 0];
                int32_t acc1 = acc_row[oc + 1];
                int32_t acc2 = acc_row[oc + 2];
                int32_t acc3 = acc_row[oc + 3];
                
                int64_t prod0 = (int64_t)acc0 * (int64_t)output_multiplier;
                int64_t prod1 = (int64_t)acc1 * (int64_t)output_multiplier;
                int64_t prod2 = (int64_t)acc2 * (int64_t)output_multiplier;
                int64_t prod3 = (int64_t)acc3 * (int64_t)output_multiplier;
                
                prod0 = (prod0 + ((int64_t)1 << 30)) >> 31;
                prod1 = (prod1 + ((int64_t)1 << 30)) >> 31;
                prod2 = (prod2 + ((int64_t)1 << 30)) >> 31;
                prod3 = (prod3 + ((int64_t)1 << 30)) >> 31;
                
                int32_t scaled0 = (int32_t)prod0;
                int32_t scaled1 = (int32_t)prod1;
                int32_t scaled2 = (int32_t)prod2;
                int32_t scaled3 = (int32_t)prod3;
                
                if (output_shift > 0) {
                    int64_t round = (int64_t)1 << (output_shift - 1);
                    scaled0 = (int32_t)(((int64_t)scaled0 + round) >> output_shift);
                    scaled1 = (int32_t)(((int64_t)scaled1 + round) >> output_shift);
                    scaled2 = (int32_t)(((int64_t)scaled2 + round) >> output_shift);
                    scaled3 = (int32_t)(((int64_t)scaled3 + round) >> output_shift);
                } else if (output_shift < 0) {
                    int neg_shift = -output_shift;
                    scaled0 <<= neg_shift;
                    scaled1 <<= neg_shift;
                    scaled2 <<= neg_shift;
                    scaled3 <<= neg_shift;
                }
                
                scaled0 += output_offset;
                scaled1 += output_offset;
                scaled2 += output_offset;
                scaled3 += output_offset;
                
                if (scaled0 < activation_min) scaled0 = activation_min;
                if (scaled0 > activation_max) scaled0 = activation_max;
                if (scaled1 < activation_min) scaled1 = activation_min;
                if (scaled1 > activation_max) scaled1 = activation_max;
                if (scaled2 < activation_min) scaled2 = activation_min;
                if (scaled2 > activation_max) scaled2 = activation_max;
                if (scaled3 < activation_min) scaled3 = activation_min;
                if (scaled3 > activation_max) scaled3 = activation_max;
                
                out_base[(oc + 0) * OH * OW] = (int8_t)scaled0;
                out_base[(oc + 1) * OH * OW] = (int8_t)scaled1;
                out_base[(oc + 2) * OH * OW] = (int8_t)scaled2;
                out_base[(oc + 3) * OH * OW] = (int8_t)scaled3;
            }
            
            for (; oc < OC; oc++) {
                int32_t acc = acc_row[oc];
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
                out_base[oc * OH * OW] = (int8_t)scaled;
            }
        }
    }
}