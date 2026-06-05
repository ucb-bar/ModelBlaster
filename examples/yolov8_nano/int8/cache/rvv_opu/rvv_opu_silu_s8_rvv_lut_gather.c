/* algorithm: rvv_lut_gather */
/* accuracy_class: bit_exact */
/* origin: hand-rolled Phase G5. SiLU on int8 has at most 256 distinct
 * outputs (the input is int8, so there are exactly 256 possible
 * results given a fixed scale_in/scale_out). Precompute that LUT
 * using the SAME scalar reference math, then vector-gather the
 * lookup. Bit-exact by construction vs the scalar reference. */
#include <math.h>
#include <riscv_vector.h>

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

    int i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m1((size_t)(n - i));
        vint8m1_t vx = __riscv_vle8_v_i8m1(&input[i], vl);
        /* Convert signed input to unsigned LUT index: idx = (int)x + 128.
         * Add 128 to each byte; reinterpret as unsigned for the gather. */
        vint8m1_t vshift = __riscv_vadd_vx_i8m1(vx, (int8_t)-128, vl);
        /* Subtracting -128 == adding 128 (mod 256) gives the unsigned index. */
        vuint8m1_t vidx = __riscv_vreinterpret_v_i8m1_u8m1(vshift);
        /* vluxei8: gather signed-int8 elements from `lut` indexed by vidx. */
        vint8m1_t vy = __riscv_vluxei8_v_i8m1(
            (const int8_t *)lut, vidx, vl);
        __riscv_vse8_v_i8m1(&output[i], vy, vl);
        i += (int)vl;
    }
}
