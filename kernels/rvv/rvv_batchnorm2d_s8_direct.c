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
            /* Broadcast the bias ONCE, outside the element loop.
             *
             * It is loop-invariant -- cb is per-channel -- so hoisting is a
             * plain win, but it is also a correctness fix on GCC 13.2. Inline
             * in the loop body, `__riscv_vfmv_v_f_f32m8(cb, vl)` sat between
             * an e8m2 load and the f32m8 arithmetic, and the compiler emitted
             * it WITHOUT the intervening vtype change: `vsetvli e8,m2` then
             * `vfmv.v.f`, which is an illegal instruction because SEW=8 has no
             * float format. It SIGILLs on the first batchnorm dispatch.
             *
             * Hoisting it behind an explicit vsetvlmax_e32m8 gives the
             * broadcast its own unambiguous SEW=32 context. Using VLMAX rather
             * than the loop's vl is safe: the tail elements of vbias are never
             * read, since the vfmacc below runs at the loop's own (smaller or
             * equal) vl. */
            size_t vlmax_f32 = __riscv_vsetvlmax_e32m8();
            vfloat32m8_t vbias = __riscv_vfmv_v_f_f32m8(cb, vlmax_f32);

            int i = 0;
            size_t vl;
            for (; i < hw; i += vl) {
                const size_t n_elem = (size_t)(hw - i);
                vl = __riscv_vsetvl_e8m2(n_elem);
                /* Load int8 input */
                vint8m2_t vi8 = __riscv_vle8_v_i8m2(in_ptr + i, vl);

                /* THE AVL IS THE ELEMENT COUNT, NEVER A PREVIOUS vsetvl'S
                 * RESULT, AND THAT IS NOT STYLE. Written as
                 * `__riscv_vsetvl_e32m8(vl)` -- chaining one vsetvl on
                 * another's return value -- GCC 14.3 substitutes an
                 * unrelated register for the AVL. Measured in the avgpool
                 * kernel, which had the same shape: the second vsetvl was
                 * issued with the OUTER LOOP BOUND as its AVL, vl came out
                 * 5 where the row is 11 wide, the `vsetvli zero,zero` forms
                 * carried that 5 down to the store, and six of every eleven
                 * outputs were never written. max_abs_err=68, and silent.
                 *
                 * It was correct under GCC 13.2 -- the compiler these
                 * kernels were verified on, and still the default `CROSS`.
                 * 13.2 has its own bug (the paragraph below), which is why
                 * 14.3 is mandatory: these kernels moved from a compiler
                 * that crashes loudly to one that answers wrongly. Passing
                 * the element count to every width is correct under both.
                 */
                /* Switch to the 32-bit domain EXPLICITLY before widening.
                 *
                 * GCC 13.2 does not track vtype across this kernel's mixed
                 * widths. It left `vsetvli e8,m2` standing and then emitted
                 * `vsext.vf4` and `vfmv.v.f` under it -- both illegal at SEW=8
                 * (a vf4 extend would imply a 2-bit source, and there is no
                 * 8-bit float), so the first batchnorm dispatch SIGILLs on the
                 * board. Neither is a kernel error: the intrinsics are used
                 * correctly and the compiler owes the vtype change.
                 *
                 * An explicit __riscv_vsetvl_* is the documented lever -- it
                 * both emits the instruction and updates the compiler's model,
                 * so the ops that follow are placed in the right domain. The
                 * element COUNT is unchanged (EMUL scales with SEW: e8m2 and
                 * e32m8 hold the same number of elements), so this costs one
                 * vsetvli per iteration and changes no arithmetic. */
                size_t vl32 = __riscv_vsetvl_e32m8(n_elem);
                /* Sign-extend i8 -> i32 directly (4x widen) */
                vint32m8_t vi32 = __riscv_vsext_vf4_i32m8(vi8, vl32);
                /* Convert to float */
                vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vi32, vl32);
                /* Apply combined scale and bias: out = vf * cs + cb */
                vf = __riscv_vfmacc_vf_f32m8(vbias, cs, vf, vl32);
                /* Convert float to int32 with rounding (round-to-nearest) */
                vint32m8_t vi_out = __riscv_vfcvt_x_f_v_i32m8(vf, vl32);
                /* Clamp to [activation_min, activation_max] */
                vi_out = __riscv_vmax_vx_i32m8(vi_out, act_min, vl32);
                vi_out = __riscv_vmin_vx_i32m8(vi_out, act_max, vl32);
                /* Narrow i32 -> i16 -> i8. Each narrowing step names the
                 * DESTINATION width, so step back down through the domains
                 * explicitly for the same reason as the widening above. */
                size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
                vint16m4_t vi16_out = __riscv_vncvt_x_x_w_i16m4(vi_out, vl16);
                size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
                vint8m2_t vi8_out = __riscv_vncvt_x_x_w_i8m2(vi16_out, vl8);
                /* Store */
                __riscv_vse8_v_i8m2(out_ptr + i, vi8_out, vl8);
            }
        }
    }
}