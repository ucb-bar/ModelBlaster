void kernel_cat2_c1_s8(const int8_t *in0, int c0, float scale0, const int8_t *in1, int c1, float scale1,
                    int8_t *output,
                    int N, int H, int W,
                    float scale_out,
                    int activation_min, int activation_max) {
    int stride = H * W;
    float ratio0 = scale0 / scale_out;
    float ratio1 = scale1 / scale_out;
    int8_t amin = (int8_t)activation_min;
    int8_t amax = (int8_t)activation_max;

    for (int n = 0; n < N; n++) {
        /* Process first input (in0) */
        {
            float ratio = ratio0;
            const int8_t *in = in0 + n * c0 * stride;
            int8_t *out = output + n * (c0 + c1) * stride;
            
            /* Process all c0 channels in one vectorized pass over the spatial dimension */
            for (int hw = 0; hw < stride; ) {
                size_t vl = __riscv_vsetvl_e8m2(stride - hw);
                
                for (int c = 0; c < c0; c++) {
                    const int8_t *src = in + c * stride + hw;
                    int8_t *dst = out + c * stride + hw;
                    
                    /* Load int8 */
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src, vl);
                    /* Sign-extend i8 -> i16 */
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    /* Sign-extend i16 -> i32 */
                    vint32m8_t v32 = __riscv_vsext_vf2_i32m8(v16, vl);
                    /* Convert int32 -> float32 */
                    vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl);
                    /* Multiply by ratio */
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl);
                    /* Round to nearest int32 */
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    /* Narrow i32 -> i16 with saturation */
                    vint16m4_t vi16 = __riscv_vnclip_wx_i16m4(vi, 0, __RISCV_VXRM_RDN, vl);
                    /* Narrow i16 -> i8 with saturation */
                    vint8m2_t vi8 = __riscv_vnclip_wx_i8m2(vi16, 0, __RISCV_VXRM_RDN, vl);
                    /* Clamp to activation range */
                    vi8 = __riscv_vmax_vx_i8m2(vi8, amin, vl);
                    vi8 = __riscv_vmin_vx_i8m2(vi8, amax, vl);
                    /* Store */
                    __riscv_vse8_v_i8m2(dst, vi8, vl);
                }
                
                hw += (int)vl;
            }
        }
        
        /* Process second input (in1) */
        {
            float ratio = ratio1;
            const int8_t *in = in1 + n * c1 * stride;
            int8_t *out = output + n * (c0 + c1) * stride + c0 * stride;
            
            /* Process all c1 channels in one vectorized pass over the spatial dimension */
            for (int hw = 0; hw < stride; ) {
                size_t vl = __riscv_vsetvl_e8m2(stride - hw);
                
                for (int c = 0; c < c1; c++) {
                    const int8_t *src = in + c * stride + hw;
                    int8_t *dst = out + c * stride + hw;
                    
                    /* Load int8 */
                    vint8m2_t v8 = __riscv_vle8_v_i8m2(src, vl);
                    /* Sign-extend i8 -> i16 */
                    vint16m4_t v16 = __riscv_vsext_vf2_i16m4(v8, vl);
                    /* Sign-extend i16 -> i32 */
                    vint32m8_t v32 = __riscv_vsext_vf2_i32m8(v16, vl);
                    /* Convert int32 -> float32 */
                    vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(v32, vl);
                    /* Multiply by ratio */
                    vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl);
                    /* Round to nearest int32 */
                    vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
                    /* Narrow i32 -> i16 with saturation */
                    vint16m4_t vi16 = __riscv_vnclip_wx_i16m4(vi, 0, __RISCV_VXRM_RDN, vl);
                    /* Narrow i16 -> i8 with saturation */
                    vint8m2_t vi8 = __riscv_vnclip_wx_i8m2(vi16, 0, __RISCV_VXRM_RDN, vl);
                    /* Clamp to activation range */
                    vi8 = __riscv_vmax_vx_i8m2(vi8, amin, vl);
                    vi8 = __riscv_vmin_vx_i8m2(vi8, amax, vl);
                    /* Store */
                    __riscv_vse8_v_i8m2(dst, vi8, vl);
                }
                
                hw += (int)vl;
            }
        }
    }
}