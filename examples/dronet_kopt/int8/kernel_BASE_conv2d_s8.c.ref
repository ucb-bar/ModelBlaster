/* source: curated */
/* algorithm: gemmini_im2col_full_C */
/* accuracy_class: bit_exact */
/* WEIGHT LAYOUT CONTRACT: like the sibling tiled_conv variant, this
 * kernel expects `weight` already in flat HWIO layout
 * ([KH*KW*IC, OC]) — the form tiled_matmul_auto consumes directly.
 * The skeleton emitter (generate_skeleton.py::_backend_pack_weight)
 * permutes OIHW→HWIO at codegen time when --backend gemmini, so
 * we pass weight straight through without a workspace copy. */
/*
 * origin: im2col -> tiled_matmul_auto(full_C=true) -> RVV Q0.31 requantize.
 *
 * NOTE: this file used to be a symlink shared with
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c (the plain,
 * non-Q31 "gemmini" target, which has no `v` in kernel_cflags). It is now
 * a standalone gemmini_q31-only file so it can use RVV intrinsics
 * (kernel_cflags gained rv64imafdc*v* + <riscv_vector.h> for this target
 * specifically, kernel_opt_log.jsonl id 800) without breaking the plain
 * "gemmini" target's build. If you need to change the shared logic (the
 * im2col / tiled_matmul_auto(full_C=true) structure), check both files.
 *
 * WHY THIS KERNEL EXISTS (root-cause of the gemmini_q31 accuracy bug):
 * The sibling gemmini_tiled_conv algorithm asks gemmini's HW mvout unit
 * to fold (output_multiplier, output_shift) into ONE Q0.31 scale
 * (scale_q31 = round(mult / 2^shift)) and does a SINGLE hardware
 * round-shift-by-31. The reference/golden (the scalar fallback below,
 * kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c, and the PyTorch
 * quantized op it was generated from) all compute a TWO-STAGE rounding:
 *   stage1 = round_half_up(acc * mult,   2^31)
 *   stage2 = round_half_up(stage1,       2^shift)   (shift > 0)
 * Pre-folding mult/shift into one scale BEFORE multiplying by acc changes
 * which values land exactly on a rounding boundary (the pre-rounding of
 * the scale is amplified by acc, instead of a fixed +-0.5 in output
 * units), so a single hardware round cannot reproduce the two-stage
 * golden bit-for-bit -- this is a genuine, unavoidable structural
 * mismatch for that algorithm (see gemmini_q31_conv2d_s8_gemmini_tiled_conv.c
 * for the analysis and its own header comment). It is not a bug in that
 * kernel's code, just a HW datapath that cannot represent double
 * rounding.
 *
 * The fix is this algorithm: bypass gemmini's mvout requantize entirely
 * (tiled_matmul_auto(..., full_C=true) drains the RAW int32 accumulator,
 * no scale/round/saturate in hardware) and do the exact two-stage Q0.31
 * requantize on the CPU with vsmul_vx (stage1) + vnclip_wx (stage2), both
 * __RISCV_VXRM_RNU -- bit-identical to the scalar int64 formula and to
 * kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c's mb_requant_i32m4, which
 * is independently verified bit-exact end-to-end elsewhere in this repo.
 * Verified in ISOLATION on dronet (this kernel alone, every other op
 * forced to scalar reference_impl): max_abs_err=0
 * (experiments/kernel_opt_log.jsonl id 1101). The unvectorized ancestor
 * of this file (identical arithmetic, scalar requantize loop) was
 * likewise measured bit-exact on Saturn RTL FireSim, May 2026 -- see
 * kernels/gemmini/gemmini_conv2d_s8_gemmini_im2col_full_C.c.
 *
 * COST: draining a raw int32 accumulator moves 4 bytes/output instead of
 * gemmini_tiled_conv's 1, and needs a CPU-side requantize pass gemmini's
 * float/Q31-scale mvout does inside the accelerator for free
 * (kernel_opt_log id 302/303/304: the unvectorized version of this
 * kernel was ~4-5x slower than gemmini_tiled_conv on dronet, almost
 * entirely the scalar per-element requantize loop). The RVV
 * vectorization here (vsmul/vnclip over the OC dimension, same rounding
 * mode) plus writing straight into `output`'s NCHW layout via a strided
 * vector store (no NHWC ws_output staging buffer, no separate
 * NHWC->NCHW transpose pass -- same fix as id 304) claws back most of
 * that gap; see experiments/kernel_opt_log.jsonl id 1105+ for the
 * measured trade.
 *
 * Handles non-square kernels, any stride/padding, and large
 * output_shift values (int64/RVV requantize, no UB).
 */

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>
#include <riscv_vector.h>

/*
 * Static workspace limits.  512 KB covers all square conv layers in
 * dronet and yolov8_nano:
 *   WS_BYTES:     max input  = IC=3,IH=160,IW=160 →  75 KB (yolov8 l0)
 *                 max output = IC=16,OH=80,OW=80  → 100 KB (yolov8 l0)
 *                 (ws_weight is gone — weight is pre-packed HWIO at
 *                  codegen time and passed straight to tiled_matmul_auto)
 *   IM2COL_ELEMS: max K_inner = IC=256,K=3×3     → 2304 (yolov8 detect head)
 *   ACC_ELEMS:    max OC      = 256               (yolov8 l7/l8/l9)
 */
enum {
    WS_BYTES       = 512 * 1024,
    IM2COL_ELEMS   = DIM * 256 * 9,   /* DIM rows × max K_inner (IC=256, 3×3) */
    ACC_ELEMS      = DIM * 256,        /* DIM rows × max OC (256 in yolov8)    */
};

/* Two-stage Q0.31 requantize, vectorized over a run of OC channels.
 * Bit-identical to the scalar int64 reference (see file header):
 *   stage1 = round_half_up(acc * output_multiplier, 2^31)
 *   stage2 = round_half_up(stage1, 2^output_shift)          (shift > 0)
 *          = stage1 << (-output_shift)                      (shift < 0, exact)
 * vsmul_vx(..., RNU) performs stage1 (Q0.31 multiply-round, matches the
 * scalar '+ (1<<30) >> 31'); vnclip_wx(..., RNU) performs stage2 (matches
 * the scalar '+ (1<<(shift-1)) >> shift'). Structurally the same
 * construction as kernels/rvv/rvv_conv2d_s8_rvv_vsmul_vnclip.c's
 * mb_requant_i32m4. */
static inline vint8m1_t gq31_requant_i32m4(vint32m4_t vacc,
                                           int output_multiplier, int output_shift,
                                           int output_offset,
                                           int activation_min, int activation_max,
                                           size_t vl)
{
    vint32m4_t vscaled = __riscv_vsmul_vx_i32m4(
        vacc, output_multiplier, __RISCV_VXRM_RNU, vl);
    vint16m2_t vout16;
    if (output_shift < 0) {
        vint32m4_t vsh = __riscv_vsll_vx_i32m4(vscaled, (size_t)(-output_shift), vl);
        vout16 = __riscv_vnclip_wx_i16m2(vsh, 0, __RISCV_VXRM_RNU, vl);
    } else if (output_shift < 32) {
        vout16 = __riscv_vnclip_wx_i16m2(vscaled, (size_t)output_shift,
                                         __RISCV_VXRM_RNU, vl);
    } else {
        int sa2 = output_shift - 31;
        if (sa2 > 31) sa2 = 31;
        vint32m4_t v2 = __riscv_vsra_vx_i32m4(vscaled, 31, vl);
        vout16 = __riscv_vnclip_wx_i16m2(v2, (size_t)sa2, __RISCV_VXRM_RNU, vl);
    }
    vout16 = __riscv_vadd_vx_i16m2(vout16, (int16_t)output_offset, vl);
    vout16 = __riscv_vmax_vx_i16m2(vout16, (int16_t)activation_min, vl);
    vout16 = __riscv_vmin_vx_i16m2(vout16, (int16_t)activation_max, vl);
    return __riscv_vnsra_wx_i8m1(vout16, 0, vl);
}

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    /* ws_weight is gone — generate_skeleton.py::_backend_pack_weight
     * pre-packs weights to flat HWIO at codegen time, so we pass
     * `weight` straight into tiled_matmul_auto below. */
    static elem_t ws_input  [WS_BYTES]     __attribute__((aligned(64)));
    static elem_t ws_im2col [IM2COL_ELEMS] __attribute__((aligned(64)));
    static acc_t  ws_acc_out[ACC_ELEMS]    __attribute__((aligned(64)));

    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    int K_inner   = IC * KH * KW;
    int total_out = N * OH * OW;

    /* Fall back to scalar for configs exceeding workspace or using offsets.
     * offsets != 0 would require zero-point subtraction inside the GEMM, which
     * tiled_matmul_auto does not support; gemmini_im2col_full_C assumes
     * symmetric per-tensor int8 (offsets == 0 from extract_int8). */
    if (input_offset != 0 || filter_offset != 0
            || (size_t)(N * IH * IW * IC) > WS_BYTES
            || (size_t)(K_inner * OC)      > WS_BYTES
            || (size_t)(N * OH * OW * OC)  > WS_BYTES
            || K_inner * DIM               > IM2COL_ELEMS
            || OC * DIM                    > ACC_ELEMS) {
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
                                    /* weight is HWIO-packed:
                                     * idx = ((kh*KW + kw)*IC + ic)*OC + oc */
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

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap. */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    /* Reset gemmini controller and drain any prior DMA. */
    gemmini_flush(0);

    /* Transpose input NCHW → NHWC into ws_input. */
    for (int n = 0; n < N; n++)
        for (int h = 0; h < IH; h++)
            for (int w = 0; w < IW; w++)
                for (int c = 0; c < IC; c++)
                    ws_input[((n*IH + h)*IW + w)*IC + c] =
                        input[((n*IC + c)*IH + h)*IW + w];

    /* Weight is already HWIO-packed by the codegen
     * (generate_skeleton.py::_backend_pack_weight, --backend gemmini).
     * Layout = `[KH*KW*IC, OC]` flat — exactly the B-matrix layout
     * tiled_matmul_auto wants — so no ws_weight copy needed; we'll
     * pass `weight` directly to tiled_matmul_auto below. */

    /* Drain CPU store buffer (covers ws_input writes — the weight
     * was a const blob so already coherent, but keep the fence here
     * as gemmini mvin sets up A and B together). */
    asm volatile("fence" ::: "memory");

    /* Process output positions in tiles of DIM rows. */
    for (int tile_i = 0; tile_i < total_out; tile_i += DIM) {
        int tile_rows = total_out - tile_i < DIM ? total_out - tile_i : DIM;

        /* Build im2col A-matrix: DIM rows × K_inner cols.
         * Row i holds the flattened receptive field for output position
         * (tile_i + i).  Rows past tile_rows are zero-padded. */
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
            for (int kh = 0; kh < KH; kh++) {
                int ih = oh_idx * SH - PH + kh;
                for (int kw = 0; kw < KW; kw++) {
                    int iw = ow_idx * SW - PW + kw;
                    elem_t *cell = row + (kh * KW + kw) * IC;
                    if (ih >= 0 && ih < IH && iw >= 0 && iw < IW) {
                        const elem_t *src = &ws_input[((n_idx*IH + ih)*IW + iw)*IC];
                        for (int c = 0; c < IC; c++) cell[c] = src[c];
                    } else {
                        for (int c = 0; c < IC; c++) cell[c] = 0;
                    }
                }
            }
        }

        /* Drain CPU stores to ws_im2col before gemmini mvin. */
        asm volatile("fence" ::: "memory");

        /* GEMM: ws_im2col [DIM × K_inner] × weight [K_inner × OC] + bias[OC].
         * weight is pre-packed HWIO from codegen — flat [KH*KW*IC, OC] is
         * exactly the B-matrix layout tiled_matmul_auto wants.
         * full_C=true → raw int32 accumulator output (no float-scale mvout). */
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

        /* Wait for gemmini DMA writes to ws_acc_out to reach L2.
         * gemmini_fence drains the in-flight mvout DMAs; gemmini_flush
         * resets the controller. Both are needed — without the fence,
         * yolov8-scale tiles (DIM * OC * K_inner large) can race with
         * the CPU's subsequent ws_acc_out reads in the requantize loop,
         * which corrupts the stack and surfaces as mcause=1 mepc=0
         * (return-address-zeroed) several frames later. See
         * modelblaster/notes/gemmini_tiled_conv_fence_required.md. */
        gemmini_fence();
        gemmini_flush(0);

        /* RVV Q0.31 requantize: int32 accumulator -> int8, written
         * straight into `output`'s NCHW layout (stride OH*OW elements
         * between consecutive oc) via a strided vector store. No NHWC
         * staging buffer, no separate transpose pass (kernel_opt_log
         * id 304's fix, applied here too). */
        for (int i = 0; i < tile_rows; i++) {
            int out_idx = tile_i + i;
            int ow_idx  = out_idx % OW;
            int oh_idx  = (out_idx / OW) % OH;
            int n_idx   = out_idx / (OH * OW);
            int8_t *dst0 = output + (((size_t)n_idx * OC) * OH + oh_idx) * OW + ow_idx;
            ptrdiff_t oc_stride_bytes = (ptrdiff_t)OH * OW;  /* elem_t is 1 byte */
            int oc = 0;
            while (oc < OC) {
                size_t vl = __riscv_vsetvl_e32m4((size_t)(OC - oc));
                vint32m4_t vacc = __riscv_vle32_v_i32m4(&ws_acc_out[i * OC + oc], vl);
                vint8m1_t vout = gq31_requant_i32m4(
                    vacc, output_multiplier, output_shift, output_offset,
                    activation_min, activation_max, vl);
                __riscv_vsse8_v_i8m1(dst0 + (size_t)oc * OH * OW,
                                     oc_stride_bytes, vout, vl);
                oc += (int)vl;
            }
        }
    }
}
