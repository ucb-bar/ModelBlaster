/* source: curated */
/* algorithm: per_input_lut */
/* accuracy_class: bit_exact */
/* origin: racing-nets (2026-11-11), same LUT idea as
 *   kernels/rvv_opu/rvv_opu_cat2_c1_s8_per_input_lut.c, for the pure-Gemmini
 *   target (rv64imafdc, no V). Before this file every cat4_c1_s8 on the
 *   gemmini target ran the scalar reference: a float multiply, a float divide
 *   and a roundf per output byte.
 *
 *   For input i the output byte is a pure function of the input byte:
 *     q = clamp(roundf(((float)v * scale_i) / scale_out))
 *   so each input gets a 256-entry table built with EXACTLY that expression
 *   (same operand order, float32 throughout), and the copy is a lookup. The
 *   NCHW layout makes every input one contiguous run of c_i*H*W bytes in both
 *   source and destination. When the table is the identity (scale_i ==
 *   scale_out with a range that does not clip) the run is a memcpy.
 *
 *   BIT-EXACT by construction; MB_DRIFT_ATOL must not be needed. */

#include <math.h>
#include <stdint.h>
#include <string.h>

static int _mb_cat_lut_build(int8_t lut[256], float scale_in, float scale_out,
                             int activation_min, int activation_max)
{
    int identity = 1;
    for (int v = -128; v < 128; v++) {
        float f = (float)v * scale_in;
        int32_t q = (int32_t)roundf(f / scale_out);
        if (q < activation_min) q = activation_min;
        if (q > activation_max) q = activation_max;
        lut[v + 128] = (int8_t)q;
        if (q != v) identity = 0;
    }
    return identity;
}

static void _mb_cat_lut_copy(int8_t *dst, const int8_t *src, size_t n,
                             const int8_t lut[256], int identity)
{
    if (identity) { memcpy(dst, src, n); return; }
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        int8_t a = lut[(int)src[i] + 128], b = lut[(int)src[i + 1] + 128];
        int8_t c = lut[(int)src[i + 2] + 128], d = lut[(int)src[i + 3] + 128];
        dst[i] = a; dst[i + 1] = b; dst[i + 2] = c; dst[i + 3] = d;
    }
    for (; i < n; i++) dst[i] = lut[(int)src[i] + 128];
}

void kernel_cat4_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2, const int8_t *in3, int c3, float scale3,
                       int8_t *output,
                       int N, int H, int W,
                       float scale_out,
                       int activation_min, int activation_max)
{
    const size_t stride = (size_t)H * (size_t)W;
    const int8_t *ins[4] = { in0, in1, in2, in3 };
    const int cs[4] = { c0, c1, c2, c3 };
    const float scales[4] = { scale0, scale1, scale2, scale3 };
    const int c_total = c0 + c1 + c2 + c3;
    int8_t lut[4][256];
    int ident[4];
    for (int i = 0; i < 4; i++)
        ident[i] = _mb_cat_lut_build(lut[i], scales[i], scale_out,
                                     activation_min, activation_max);
    for (int n_ = 0; n_ < N; n_++) {
        int out_c = 0;
        for (int i = 0; i < 4; i++) {
            const size_t run = (size_t)cs[i] * stride;
            _mb_cat_lut_copy(output + ((size_t)n_ * c_total + out_c) * stride,
                             ins[i] + (size_t)n_ * run, run, lut[i], ident[i]);
            out_c += cs[i];
        }
    }
}
