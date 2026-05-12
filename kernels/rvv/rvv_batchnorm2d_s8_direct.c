/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV batchnorm2d_s8 (LMUL-tiled / LUT-gather where applicable). */

void kernel_batchnorm2d_s8(const int8_t *input, const float *scale,
                           const float *bias, int8_t *output,
                           int N, int C, int H, int W,
                           float scale_in, float scale_out,
                           int activation_min, int activation_max) {
    float inv_scale_out = 1.0f / scale_out;
    int hw = H * W;
    int act_min = activation_min;
    int act_max = activation_max;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            /* Precompute combined per-channel scalars */
            float cs = scale[c] * scale_in * inv_scale_out;
            float cb = bias[c] * inv_scale_out;
            int base = (n * C + c) * hw;
            const int8_t *in_ptr = input + base;
            int8_t *out_ptr = output + base;
            int i = 0;
            size_t vl;
            for (; i < hw; i += vl) {
                vl = __riscv_vsetvl_e8m2(hw - i);
                /* Load int8 input */
                vint8m2_t vi8 = __riscv_vle8_v_i8m2(in_ptr + i, vl);
                /* Sign-extend i8 -> i32 directly (4x widen) */
                vint32m8_t vi32 = __riscv_vsext_vf4_i32m8(vi8, vl);
                /* Convert to float */
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vi32, vl);
                /* Apply combined scale and bias: out = vf * cs + cb */
                vf = __riscv_vfmacc_vf_f32m8(
                         __riscv_vfmv_v_f_f32m8(cb, vl),
                         cs, vf, vl);
                /* Convert float to int32 with rounding (round-to-nearest) */
                vint32m8_t vi_out = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                /* Clamp to [activation_min, activation_max] */
                vi_out = __riscv_vmax_vx_i32m8(vi_out, act_min, vl);
                vi_out = __riscv_vmin_vx_i32m8(vi_out, act_max, vl);
                /* Narrow i32 -> i16 -> i8 */
                vint16m4_t vi16_out = __riscv_vncvt_x_x_w_i16m4(vi_out, vl);
                vint8m2_t vi8_out = __riscv_vncvt_x_x_w_i8m2(vi16_out, vl);
                /* Store */
                __riscv_vse8_v_i8m2(out_ptr + i, vi8_out, vl);
            }
        }
    }
}