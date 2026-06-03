static inline int8_t compute_silu_lut_entry(int v, float silu_scale_in, float silu_scale_out, int silu_activation_min, int silu_activation_max) {
    int8_t iv = (int8_t)(uint8_t)v;
    float f = (float)iv * silu_scale_in;
    float y = f / (1.0f + expf(-f));
    int32_t q = (int32_t)roundf(y / silu_scale_out);
    if (q < silu_activation_min) q = silu_activation_min;
    if (q > silu_activation_max) q = silu_activation_max;
    return (int8_t)q;
}

void kernel_batchnorm2d_silu_s8(const int8_t *input, const float *scale, const float *bias, int8_t *output, int N, int C, int H, int W, float bn_scale_in, float bn_scale_out, int bn_activation_min, int bn_activation_max, float silu_scale_in, float silu_scale_out, int silu_activation_min, int silu_activation_max) {
    int8_t silu_lut[256];
    for (int v = 0; v < 256; v++) {
        silu_lut[v] = compute_silu_lut_entry(v, silu_scale_in, silu_scale_out, silu_activation_min, silu_activation_max);
    }

    int HW = H * W;
    float inv_bn_scale_out = 1.0f / bn_scale_out;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            int base_idx = (n * C + c) * HW;

            for (int hw = 0; hw < HW; ) {
                size_t vl = __riscv_vsetvl_e8m1(HW - hw);

                vint8m1_t v_in = __riscv_vle8_v_i8m1(input + base_idx + hw, vl);
                vint16m2_t v_in16 = __riscv_vsext_vf2_i16m2(v_in, vl);
                vfloat32m4_t v_f = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(v_in16, vl), vl);
                v_f = __riscv_vfmul_vf_f32m4(v_f, bn_scale_in, vl);
                vfloat32m4_t v_y = __riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(b, vl), s, v_f, vl);
                v_y = __riscv_vfmul_vf_f32m4(v_y, inv_bn_scale_out, vl);
                vint32m4_t v_q = __riscv_vfcvt_x_f_v_i32m4(v_y, vl);
                v_q = __riscv_vmax_vx_i32m4(v_q, bn_activation_min, vl);
                v_q = __riscv_vmin_vx_i32m4(v_q, bn_activation_max, vl);
                vint16m2_t v_q16 = __riscv_vnclip_wx_i16m2(v_q, 0, vl);
                vint8m1_t v_bn8 = __riscv_vnclip_wx_i8m1(v_q16, 0, vl);

                vuint8m1_t v_idx = __riscv_vreinterpret_v_i8m1_u8m1(v_bn8);
                vuint16m2_t v_idx16 = __riscv_vzext_vf2_u16m2(v_idx, vl);
                vint8m1_t v_lut = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vle8_v_u8m1((const uint8_t*)silu_lut, 256));
                vint8m1_t v_out = __riscv_vrgather_vv_i8m1(v_lut, v_idx16, vl);

                __riscv_vse8_v_i8m1(output + base_idx + hw, v_out, vl);
                hw += vl;
            }
        }
    }
}