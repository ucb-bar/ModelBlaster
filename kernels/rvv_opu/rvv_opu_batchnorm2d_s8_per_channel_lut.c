/* source: curated */
/* algorithm: per_channel_lut */
/* accuracy_class: bit_exact */
/* origin: v11g+ — batchnorm2d_s8 was the second-largest rvv_opu kernel
 *   category after the conv2d reduce kernel landed (73.3 M rdcycle
 *   summed across 60 calls in v11g). The reference impl is a per-pixel
 *   sequence of cast / mul / FMA / div / round / clamp.
 *
 *   For fixed (scale[c], bias[c], scale_in, scale_out, clamp range), the
 *   quantized BN output is a deterministic function of the int8 input
 *   pixel — at most 256 distinct outputs per channel. Precompute a
 *   256-entry int8 LUT per channel and replace the inner H*W loop with
 *   `output[i] = lut[(int)input[i] + 128]`.
 *
 *   The silu_s8 lut_gather kernel proved that the Saturn-OPU bitstream
 *   does NOT implement `vluxei8` (indexed gather) — scalar table lookup
 *   is the only option that runs on FPGA. The win is purely from
 *   replacing ~8 scalar fp ops per pixel with one load + one indexed
 *   load + one store. Bit-exact by construction (the LUT computes the
 *   same math the reference does, the gather just replays it).
 */

#include <math.h>
#include <stdint.h>

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max) {
    int spatial = H * W;
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            int8_t lut[256];
            for (int v = 0; v < 256; v++) {
                int8_t iv = (int8_t)(v - 128);
                float fv = (float)iv * scale_in;
                float y = s * fv + b;
                int32_t q = (int32_t)roundf(y / scale_out);
                if (q < activation_min) q = activation_min;
                if (q > activation_max) q = activation_max;
                lut[v] = (int8_t)q;
            }
            const int8_t *ip = input + (n * C + c) * spatial;
            int8_t *op = output + (n * C + c) * spatial;
            for (int i = 0; i < spatial; i++) {
                op[i] = lut[(int)ip[i] + 128];
            }
        }
    }
}
