/* algorithm: rvv_lut_gather */
/* accuracy_class: bit_exact */
/* origin: Phase G5 v2 — scalar LUT lookup (NOT RVV-vectorized).
 *
 * SiLU on int8 has at most 256 distinct outputs given fixed
 * scale_in/scale_out. Precompute lut[input + 128] = scalar_silu(input)
 * using the SAME math as the reference impl. Bit-exact by
 * construction.
 *
 * v1 used __riscv_vluxei8_v_i8m1 (indexed gather) to vectorize the
 * lookup. That works on spike but the FireSim Saturn-OPU bitstream
 * does NOT implement vluxei8 — Illegal instruction trap at runtime.
 *
 * Stay scalar in the lookup loop. The win comes from avoiding
 * expf+roundf+clip per element (replaces ~80 cycles with ~3-5 cycles
 * per element); on spike this measured at ~50x speedup over the
 * reference scalar impl on yolov8. */
#include <math.h>

void kernel_silu_s8(const int8_t *input, int8_t *output, int n,
                    float scale_in, float scale_out,
                    int activation_min, int activation_max) {
    int8_t lut[256];
    for (int v = 0; v < 256; v++) {
        int8_t iv = (int8_t)(v - 128);
        float f = (float)iv * scale_in;
        float y = f / (1.0f + expf(-f));
        int32_t q = (int32_t)roundf(y / scale_out);
        if (q < activation_min) q = activation_min;
        if (q > activation_max) q = activation_max;
        lut[v] = (int8_t)q;
    }
    for (int i = 0; i < n; i++) {
        output[i] = lut[(int)input[i] + 128];
    }
}
