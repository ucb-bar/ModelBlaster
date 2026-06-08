void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max) {
    float inv_scale_out = 1.0f / scale_out;
    size_t vl;
    size_t vlmax = __riscv_vsetvlmax_e8m2();
    
    int i = 0;
    for (; i + 2 * vlmax <= n; i += 2 * vlmax) {
        vl = vlmax;
        
        vint8m2_t va0 = __riscv_vle8_v_i8m2(a + i, vl);
        vint8m2_t vb0 = __riscv_vle8_v_i8m2(b + i, vl);
        vint8m2_t va1 = __riscv_vle8_v_i8m2(a + i + vl, vl);
        vint8m2_t vb1 = __riscv_vle8_v_i8m2(b + i + vl, vl);
        
        vint32m8_t va32_0 = __riscv_vsext_vf4_i32m8(va0, vl);
        vint32m8_t vb32_0 = __riscv_vsext_vf4_i32m8(vb0, vl);
        vint32m8_t va32_1 = __riscv_vsext_vf4_i32m8(va1, vl);
        vint32m8_t vb32_1 = __riscv_vsext_vf4_i32m8(vb1, vl);
        
        vfloat32m8_t vfa0 = __riscv_vfcvt_f_x_v_f32m8(va32_0, vl);
        vfloat32m8_t vfb0 = __riscv_vfcvt_f_x_v_f32m8(vb32_0, vl);
        vfloat32m8_t vfa1 = __riscv_vfcvt_f_x_v_f32m8(va32_1, vl);
        vfloat32m8_t vfb1 = __riscv_vfcvt_f_x_v_f32m8(vb32_1, vl);
        
        vfa0 = __riscv_vfmul_vf_f32m8(vfa0, scale_a, vl);
        vfb0 = __riscv_vfmul_vf_f32m8(vfb0, scale_b, vl);
        vfa1 = __riscv_vfmul_vf_f32m8(vfa1, scale_a, vl);
        vfb1 = __riscv_vfmul_vf_f32m8(vfb1, scale_b, vl);
        
        vfloat32m8_t vfout0 = __riscv_vfadd_vv_f32m8(vfa0, vfb0, vl);
        vfout0 = __riscv_vfmul_vf_f32m8(vfout0, inv_scale_out, vl);
        vfloat32m8_t vfout1 = __riscv_vfadd_vv_f32m8(vfa1, vfb1, vl);
        vfout1 = __riscv_vfmul_vf_f32m8(vfout1, inv_scale_out, vl);
        
        vint32m8_t vout32_0 = __riscv_vfcvt_x_f_v_i32m8(vfout0, vl);
        vint32m8_t vout32_1 = __riscv_vfcvt_x_f_v_i32m8(vfout1, vl);
        
        vout32_0 = __riscv_vmax_vx_i32m8(vout32_0, activation_min, vl);
        vout32_0 = __riscv_vmin_vx_i32m8(vout32_0, activation_max, vl);
        vout32_1 = __riscv_vmax_vx_i32m8(vout32_1, activation_min, vl);
        vout32_1 = __riscv_vmin_vx_i32m8(vout32_1, activation_max, vl);
        
        vint16m4_t vout16_0 = __riscv_vnsra_wx_i16m4(vout32_0, 0, vl);
        vint8m2_t vout8_0 = __riscv_vnsra_wx_i8m2(vout16_0, 0, vl);
        vint16m4_t vout16_1 = __riscv_vnsra_wx_i16m4(vout32_1, 0, vl);
        vint8m2_t vout8_1 = __riscv_vnsra_wx_i8m2(vout16_1, 0, vl);
        
        __riscv_vse8_v_i8m2(output + i, vout8_0, vl);
        __riscv_vse8_v_i8m2(output + i + vl, vout8_1, vl);
    }
    
    for (; i < n; i += vl) {
        vl = __riscv_vsetvl_e8m2(n - i);
        
        vint8m2_t va = __riscv_vle8_v_i8m2(a + i, vl);
        vint8m2_t vb = __riscv_vle8_v_i8m2(b + i, vl);
        
        vint32m8_t va32 = __riscv_vsext_vf4_i32m8(va, vl);
        vint32m8_t vb32 = __riscv_vsext_vf4_i32m8(vb, vl);
        
        vfloat32m8_t vfa = __riscv_vfcvt_f_x_v_f32m8(va32, vl);
        vfloat32m8_t vfb = __riscv_vfcvt_f_x_v_f32m8(vb32, vl);
        
        vfa = __riscv_vfmul_vf_f32m8(vfa, scale_a, vl);
        vfb = __riscv_vfmul_vf_f32m8(vfb, scale_b, vl);
        
        vfloat32m8_t vfout = __riscv_vfadd_vv_f32m8(vfa, vfb, vl);
        vfout = __riscv_vfmul_vf_f32m8(vfout, inv_scale_out, vl);
        
        vint32m8_t vout32 = __riscv_vfcvt_x_f_v_i32m8(vfout, vl);
        
        vout32 = __riscv_vmax_vx_i32m8(vout32, activation_min, vl);
        vout32 = __riscv_vmin_vx_i32m8(vout32, activation_max, vl);
        
        vint16m4_t vout16 = __riscv_vnsra_wx_i16m4(vout32, 0, vl);
        vint8m2_t vout8 = __riscv_vnsra_wx_i8m2(vout16, 0, vl);
        
        __riscv_vse8_v_i8m2(output + i, vout8, vl);
    }
}