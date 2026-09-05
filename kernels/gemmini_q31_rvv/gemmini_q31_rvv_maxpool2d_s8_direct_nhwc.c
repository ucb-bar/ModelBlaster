/* source: curated */
/* algorithm: direct_nhwc */
/* accuracy_class: bit_exact */
/* act_layouts: nhwc */
/* NHWC RVV maxpool2d_s8.
 *
 * WHY THIS EXISTS. The other NHWC maxpool beside it,
 * gemmini_q31_rvv_maxpool2d_s8_gemmini_tiled_conv_pool_nhwc.c, runs a STANDALONE
 * maxpool through tiled_conv_auto's pool tail -- i.e. it materialises a full
 * (identity-ish) gemmini conv just to reach the HW pooler. That is the right
 * kernel only when the pool is FUSED into a real conv's mvout; run on its own it
 * is enormous. Measured on the flashed fast-conv At35 core, dronet's single
 * maxpool (C=32, 56x56->27x27, K=3 S=2): gemmini_tiled_conv_pool_nhwc = 1.78 M
 * cycles, vs the NCHW RVV "direct" maxpool at 89 K -- a 20x regression that
 * single-handedly erases the NHWC conv win.
 *
 * WHAT THIS IS. The plain vectorised maxpool, laid out for NHWC. NHWC puts the
 * channel innermost, so a pooling window tap at (ih,iw) is a CONTIGUOUS run of C
 * bytes and the max reduction over the KH*KW window is a sequence of unit-stride
 * vle8 + vmax over channels -- no strided gather at all (the NCHW kernel needed a
 * vnsrl deinterleave trick precisely because NCHW makes the taps strided). Every
 * load and store here is unit stride.
 *
 * BIT-EXACT: max is associative/commutative, so the pooled values are identical
 * to the NCHW kernel's; only their layout differs. The whole NHWC island is a
 * permutation, so model output stays bit-identical.
 *
 * ACTIVATION LAYOUT: input/output are [N,H,W,C]. Enforced by act_layouts=("nhwc",)
 * + the deny-by-default gate in generate_kernels.
 */

#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    const int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    const int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;

    for (int n = 0; n < N; n++) {
        const int8_t *in_n  = input  + (size_t)n * IH * IW * C;
        int8_t       *out_n = output + (size_t)n * OH * OW * C;
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                int8_t *op = out_n + ((size_t)oh * OW + ow) * C;
                /* vectorize over the (contiguous) channel dimension */
                int c = 0;
                size_t vl;
                for (; c < C; c += vl) {
                    vl = __riscv_vsetvl_e8m4((size_t)(C - c));
                    vint8m4_t vacc = __riscv_vmv_v_x_i8m4((int8_t)(-128), vl);
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh * SH - PH + kh * DH;
                        if (ih < 0 || ih >= IH) continue;   /* pad: skip (-inf) */
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow * SW - PW + kw * DW;
                            if (iw < 0 || iw >= IW) continue;
                            const int8_t *ip = in_n
                                + (((size_t)ih * IW + iw) * C) + c;  /* contiguous in c */
                            vint8m4_t vd = __riscv_vle8_v_i8m4(ip, vl);
                            vacc = __riscv_vmax_vv_i8m4(vacc, vd, vl);
                        }
                    }
                    __riscv_vse8_v_i8m4(op + c, vacc, vl);
                }
            }
        }
    }
}
