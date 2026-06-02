void kernel_cat4_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1, const int8_t *in2, int c2, float scale2, const int8_t *in3, int c3, float scale3,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    int total_c = c0 + c1 + c2 + c3;

    const int8_t *ins[4] = { in0, in1, in2, in3 };
    int cs[4] = { c0, c1, c2, c3 };
    float scales[4] = { scale0, scale1, scale2, scale3 };

    for (int n = 0; n < N; n++) {
        int out_c = 0;
        for (int i = 0; i < 4; i++) {
            float ratio = scales[i] / scale_out;
            int ci = cs[i];
            const int8_t *in_base = ins[i] + n * ci * stride;
            int8_t *out_base = output + (n * total_c + out_c) * stride;

            int total_elems = ci * stride;
            int hw = 0;
            size_t vl;
            
            /* Process with LMUL=4 for better throughput */
            for (; hw < total_elems; hw += vl) {
                vl = __riscv_vsetvl_e8m4(total_elems - hw);
                vint8m4_t v8 = __riscv_vle8_v_i8m4(in_base + hw, vl);
                
                /* Widen i8 -> i16 first (2x) */
                vint16m8_t v16 = __riscv_vsext_vf2_i16m8(v8, vl);
                
                /* Convert i16 -> f32 with widening (2x more) */
                vfloat32m8_t vf_lo = __riscv_vfwcvt_f_x_v_f32m8(__riscv_vget_v_i16m8_i16m4(v16, 0), vl/2);
                vfloat32m8_t vf_hi = __riscv_vfwcvt_f_x_v_f32m8(__riscv_vget_v_i16m8_i16m4(v16, 1), vl - vl/2);
                
                /* Scale both halves */
                vf_lo = __riscv_vfmul_vf_f32m8(vf_lo, ratio, vl/2);
                vf_hi = __riscv_vfmul_vf_f32m8(vf_hi, ratio, vl - vl/2);
                
                /* Convert back to i32 with rounding */
                vint32m8_t vr_lo = __riscv_vfcvt_x_f_v_i32m8(vf_lo, vl/2);
                vint32m8_t vr_hi = __riscv_vfcvt_x_f_v_i32m8(vf_hi, vl - vl/2);
                
                /* Clamp both halves */
                vr_lo = __riscv_vmax_vx_i32m8(vr_lo, activation_min, vl/2);
                vr_lo = __riscv_vmin_vx_i32m8(vr_lo, activation_max, vl/2);
                vr_hi = __riscv_vmax_vx_i32m8(vr_hi, activation_min, vl - vl/2);
                vr_hi = __riscv_vmin_vx_i32m8(vr_hi, activation_max, vl - vl/2);
                
                /* Narrow i32 -> i16 */
                vint16m4_t vr16_lo = __riscv_vncvt_x_x_w_i16m4(vr_lo, vl/2);
                vint16m4_t vr16_hi = __riscv_vncvt_x_x_w_i16m4(vr_hi, vl - vl/2);
                
                /* Combine back into i16m8 */
                vint16m8_t vr16 = __riscv_vundefined_i16m8();
                vr16 = __riscv_vset_v_i16m4_i16m8(vr16, 0, vr16_lo);
                vr16 = __riscv_vset_v_i16m4_i16m8(vr16, 1, vr16_hi);
                
                /* Narrow i16 -> i8 */
                vint8m4_t vr8 = __riscv_vncvt_x_x_w_i8m4(vr16, vl);
                
                __riscv_vse8_v_i8m4(out_base + hw, vr8, vl);
            }

            out_c += ci;
        }
    }
}