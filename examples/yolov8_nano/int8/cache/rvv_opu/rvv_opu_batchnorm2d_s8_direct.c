void kernel_batchnorm2d_s8(const int8_t *input, const float *scale, const float *bias, int8_t *output, int N, int C, int H, int W, float scale_in, float scale_out, int activation_min, int activation_max) {
    int spatial_size = H * W;
    float inv_scale_out = 1.0f / scale_out;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float s = scale[c];
            float b = bias[c];
            const int8_t *in_ptr = input + (n * C + c) * spatial_size;
            int8_t *out_ptr = output + (n * C + c) * spatial_size;
            
            size_t vl;
            for (int i = 0; i < spatial_size; i += vl) {
                vl = __riscv_vsetvl_e8m2(spatial_size - i);
                
                vint8m2_t vin = __riscv_vle8_v_i8m2(in_ptr + i, vl);
                vint32m8_t vin32 = __riscv_vsext_vf4_i32m8(vin, vl);
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vin32, vl);
                
                vf = __riscv_vfmul_vf_f32m8(vf, scale_in, vl);
                vf = __riscv_vfmul_vf_f32m8(vf, s, vl);
                vf = __riscv_vfadd_vf_f32m8(vf, b, vl);
                vf = __riscv_vfmul_vf_f32m8(vf, inv_scale_out, vl);
                
                vint32m8_t vi32 = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                vi32 = __riscv_vmax_vx_i32m8(vi32, activation_min, vl);
                vi32 = __riscv_vmin_vx_i32m8(vi32, activation_max, vl);
                
                vint16m4_t vi16 = __riscv_vnclip_wx_i16m4(vi32, 0, __RISCV_VXRM_RNU, vl);
                vint8m2_t vout = __riscv_vnclip_wx_i8m2(vi16, 0, __RISCV_VXRM_RNU, vl);
                
                __riscv_vse8_v_i8m2(out_ptr + i, vout, vl);
            }
        }
    }
}