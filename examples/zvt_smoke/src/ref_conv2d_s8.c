/* Verbatim LINEAR/CONV2D_S8 reference_impl from pipeline/reference_kernels.py
 * (OIHW branch), renamed. Regenerate if the spec changes. */
#include <stdint.h>
#include <stddef.h>
#define kernel_conv2d_s8 ref_conv2d_s8
void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    int OH = (IH + 2*PH - KH) / SH + 1;
    int OW = (IW + 2*PW - KW) / SW + 1;
    /* Weight layout depends on the active hardware target. The gemmini
     * codegen pre-packs to flat HWIO at compile time
     * (generate_skeleton.py::_backend_pack_weight) so the gemmini ROCC
     * kernel can pass the blob straight into tiled_conv_auto without
     * a per-call transpose. backends.py wires
     * MODELBLASTER_GEMMINI_HWIO_WEIGHTS=1 into kernel_cflags only for the
     * gemmini backend; all other backends keep PyTorch OIHW. */
    for (int n = 0; n < N; n++) {
        for (int oc = 0; oc < OC; oc++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    int32_t acc = bias ? bias[oc] : 0;
                    for (int ic = 0; ic < IC; ic++) {
                        /* Hoist (n,ic,*) row offsets to size_t so the
                         * multiply by IH*IW happens in 64-bit. Without
                         * this, GCC -O2 occasionally splits the
                         * pointer-+-int32-index addition into 32-bit
                         * `addw` on the low half + 64-bit `add` on the
                         * upper half — fine when (input_low32 + idx)
                         * stays positive, but wraps modulo 2^32 when
                         * it crosses the int32 sign boundary. Hit on
                         * V512D256 firesim with yolov8 because the
                         * wider thread-vreg state shifted BSS placement
                         * past the wrap boundary. */
                        const size_t in_row_base =
                            ((size_t)n * IC + ic) * IH;
                        for (int kh = 0; kh < KH; kh++) {
                            int ih = oh * SH - PH + kh;
                            for (int kw = 0; kw < KW; kw++) {
                                int iw = ow * SW - PW + kw;
                                int32_t in_v;
                                if (ih < 0 || ih >= IH || iw < 0 || iw >= IW) {
                                    in_v = input_offset;
                                } else {
                                    in_v = (int32_t)input[(in_row_base + ih) * IW + iw]
                                         + input_offset;
                                }
/* HWIO and IHWOC are NOT the same permutation of OIHW -- hwio is
 * (H,W,I,O) and ihwoc is (I,H,W,O) (see generate_skeleton
 * _LAYOUT_PERMUTATION). Sharing one indexing expression between the two
 * macros read 64/72 elements from the wrong place on an IHWOC buffer.
 * Latent because rvv models always select a curated conv kernel; it
 * would only surface as a *silently wrong* fallback. */
#if defined(MODELBLASTER_GEMMINI_HWIO_WEIGHTS)
                                int32_t w_v = (int32_t)weight[((kh*KW + kw)*IC + ic)*OC + oc]
                                            + filter_offset;
#elif defined(MODELBLASTER_RVV_IHWOC_WEIGHTS)
                                int32_t w_v = (int32_t)weight[((ic*KH + kh)*KW + kw)*OC + oc]
                                            + filter_offset;
#else
                                int32_t w_v = (int32_t)weight[((oc*IC + ic)*KH + kh)*KW + kw]
                                            + filter_offset;
#endif
                                acc += in_v * w_v;
                            }
                        }
                    }
                    int64_t prod = (int64_t)acc * (int64_t)output_multiplier;
                    prod = (prod + (1LL << 30)) >> 31;
                    int32_t scaled = (int32_t)prod;
                    if (output_shift > 0) {
                        scaled = (int32_t)(((int64_t)scaled + ((int64_t)1 << (output_shift - 1))) >> output_shift);
                    } else if (output_shift < 0) {
                        scaled = scaled << (-output_shift);
                    }
                    scaled += output_offset;
                    if (scaled < activation_min) scaled = activation_min;
                    if (scaled > activation_max) scaled = activation_max;
                    output[((n*OC + oc)*OH + oh)*OW + ow] = (int8_t)scaled;
                }
            }
        }
    }
}
