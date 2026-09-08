/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV int8 LayerNorm (int8 gamma/beta with their own
 * per-tensor scales). Only the third pass is rewritten, and the reason is
 * the whole design of this file.
 *
 * WHERE THE REFERENCE'S TIME ACTUALLY GOES, measured before writing any
 * vector code rather than assumed. The reference makes three passes per row
 * and the obvious reading is "three scalar passes, vectorize all three".
 * That reading is wrong on this toolchain: GCC 14.3 at -O2 -march=rv64gcv
 * ALREADY auto-vectorizes the two moment loops, into one `vfredosum.vs`
 * each (an ORDERED reduction, so it is a vector instruction that still adds
 * left to right). Disassembling the reference impl alone gives 2
 * vfredosum.vs, 1 vfmsub.vv, 2 vfwcvt.f.x.v -- and exactly one `jalr`,
 * which is the per-element `roundf` call in pass 3. Pass 3 is the cost: a
 * scalar body with an `fdiv.s` and a libm call for every output element,
 * 69.3 cycles per element measured over this model's 1,922,944 layer_norm
 * outputs.
 *
 * So passes 1 and 2 are left EXACTLY as the reference writes them, character
 * for character (only the second loop's index is renamed to avoid shadowing).
 * That is not laziness, it is the bit-exactness mechanism: the same C under
 * the same flags gives the same instructions, and the disassembly confirms
 * it -- this kernel's moment passes are the reference's 2 vfredosum.vs, 1
 * vfmsub.vv and 2 vfwcvt.f.x.v, instruction for instruction. Only the libm
 * call disappears (this file: 0 jalr).
 *
 * Pass 3 is vectorized: widen int8 -> i32 -> f32, the reference's own
 * operation order for the affine, then a rounding-exact convert, an int32
 * clamp, a narrow and a store.
 *
 *   THE ROUNDING. The reference tail is `roundf(y / scale_out)` --
 *   round-to-nearest, TIES AWAY FROM ZERO. vfcvt.x.f rounds by `frm`, and
 *   frm=RMM (encoding 4, ties to Max Magnitude) IS that mode, so the
 *   conversion is not an approximation of roundf, it is roundf. frm is
 *   bracketed around the conversion only and never held across the
 *   arithmetic -- it also governs vfmul/vfadd/vfdiv, where C rounds
 *   ties-to-even. Same instrument, and the same on-board tie sweep, as
 *   rvv_add_s8_rvv_frm_rmm.c.
 *
 *   THE OPERATION ORDER. `(x*scale_in - mu) * inv`, then `* g`, then `+ b`,
 *   then `/ scale_out`, each issued separately. The reference does not
 *   contract `n*g + b` into an fma on this toolchain (checked: no fmadd.s in
 *   its disassembly), so neither does this. The divide stays a divide.
 *
 * WHY THE MOMENTS ARE NOT VECTORIZED, WITH THE MEASUREMENT THAT DECIDED IT.
 * Vectorizing them is easy and fast and it changes the model's output. Three
 * variants were built and run end-to-end on spike, each as the ONLY curated
 * kernel in an otherwise all-reference build of octo-small
 * (LAYERS=1 WINDOW=1, 14 layer_norm dispatches, K from 256 to 16384):
 *
 *     moments                          layer_norm  speedup   model output
 *     exact-int mu + tree variance         5.75M    23.2x    max|d| = 9 LSB
 *     reference mu + tree variance        18.24M     7.3x    max|d| = 9 LSB
 *     reference mu + reference variance   32.29M    4.12x    IDENTICAL
 *
 * mu and inv feed EVERY element of the row, and this model puts 19
 * per-channel int8 linears downstream of them, so a perturbation far below
 * one LSB inside the norm is requantized into visible movement at the
 * output: the whole-model error against the fp32-quantized golden went from
 * max_abs_err=16 (cos 0.9574) to 18 (cos 0.9516), i.e. measurably WORSE, for
 * 0.4% of the model's cycles. Note the second row: even a bit-exact mu is
 * not enough -- the tree variance alone still moves the output by 9 LSB,
 * although a host emulation over 1.9 M outputs at this model's own shapes
 * and scales rates its own drift at 3.967e-04 LSB and 24 differing
 * elements. That gap between "sub-milli-LSB inside the op" and "9 LSB at
 * the output" is the fragility NOTES section 13.4 documents, and it is why
 * this op is not the place to spend accuracy.
 *
 * This also independently confirms the judgement in the older curated
 * `layernorm_s8` kernel (rvv_layernorm_s8_rvv_f64_tail.c), whose header
 * argues the same thing from first principles: "mu and inv then feed every
 * output in the row, so a few ULP there is not a rounding detail -- it
 * moves every element." It was right, and now there is a number for it. It
 * is also the opposite call from `layer_norm_f16`
 * (kernels/rvv_f16/rvv_f16_layer_norm_f16_direct.c), which vectorizes both
 * moments -- correctly, because fp16 has no per-op requantize to amplify
 * the difference.
 *
 * gamma and beta are each optional (the reference reads NULL as 1.0 and
 * 0.0), and the NULL cases SKIP the load and the operation rather than
 * materializing a vector of constants -- 8 of this model's 14 dispatches are
 * GroupNorms lowered as reshape + layer_norm + per-channel affine and pass
 * NULL for both.
 *
 * VTYPE. Each width transition names its own width, and every vsetvl is
 * handed the ELEMENT COUNT rather than a previous vsetvl's result (this GCC
 * miscompiles the chained form on partial tails --
 * scripts/check_rvv_avl.py; both that checker and check_rvv_vtype.py pass
 * on this file). vsext.vf4 is issued under SEW=32; under SEW=8 it implies a
 * 2-bit source and is a SIGILL on the board, not a build error.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

#ifndef MB_RVV_FRM_HELPERS_
#define MB_RVV_FRM_HELPERS_
#define MB_FRM_RNE 0
#define MB_FRM_RMM 4

static inline unsigned mb_frm_read(void)
{
    unsigned r;
    __asm__ volatile("frrm %0" : "=r"(r));
    return r;
}

static inline void mb_frm_write(unsigned mode)
{
    __asm__ volatile("fsrm %0" :: "r"(mode));
}
#endif /* MB_RVV_FRM_HELPERS_ */

void kernel_layer_norm_s8(const int8_t *input, const int8_t *gamma,
                          const int8_t *beta, int8_t *output,
                          int M, int K,
                          float scale_in, float scale_gamma, float scale_beta,
                          float scale_out, float eps,
                          int activation_min, int activation_max)
{
    if (M <= 0 || K <= 0) {
        return;
    }
    const unsigned frm_saved = mb_frm_read();

    for (int m = 0; m < M; m++) {
        const int8_t *row_in  = input  + (size_t)m * (size_t)K;
        int8_t       *row_out = output + (size_t)m * (size_t)K;

        /* ---- pass 1: mu, the reference's own sequential fp32 sum ---- */
        float mu = 0.0f;
        for (int k = 0; k < K; k++) mu += (float)row_in[k] * scale_in;
        mu /= (float)K;

        /* ---- pass 2: var, the reference's own sequential fp32 sum ---- */
        float var = 0.0f;
        for (int kk = 0; kk < K; kk++) {
            float d = (float)row_in[kk] * scale_in - mu;
            var += d * d;
        }
        var /= (float)K;
        const float inv = 1.0f / sqrtf(var + eps);

        /* ---- pass 3: affine + requantize ---- */
        int k = 0;
        while (k < K) {
            const size_t n_elem = (size_t)(K - k);

            size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
            vint8m1_t v8 = __riscv_vle8_v_i8m1(row_in + k, vl8);

            size_t vl = __riscv_vsetvl_e32m4(n_elem);
            vint32m4_t v32 = __riscv_vsext_vf4_i32m4(v8, vl);
            vfloat32m4_t vy = __riscv_vfcvt_f_x_v_f32m4(v32, vl);
            vy = __riscv_vfmul_vf_f32m4(vy, scale_in, vl);
            vy = __riscv_vfsub_vf_f32m4(vy, mu, vl);
            vy = __riscv_vfmul_vf_f32m4(vy, inv, vl);

            if (gamma) {
                size_t g8 = __riscv_vsetvl_e8m1(n_elem);
                vint8m1_t vg8 = __riscv_vle8_v_i8m1(gamma + k, g8);
                size_t g32 = __riscv_vsetvl_e32m4(n_elem);
                vint32m4_t vg32 = __riscv_vsext_vf4_i32m4(vg8, g32);
                vfloat32m4_t vg = __riscv_vfcvt_f_x_v_f32m4(vg32, g32);
                vg = __riscv_vfmul_vf_f32m4(vg, scale_gamma, g32);
                vy = __riscv_vfmul_vv_f32m4(vy, vg, g32);
            }
            if (beta) {
                size_t b8 = __riscv_vsetvl_e8m1(n_elem);
                vint8m1_t vb8 = __riscv_vle8_v_i8m1(beta + k, b8);
                size_t b32 = __riscv_vsetvl_e32m4(n_elem);
                vint32m4_t vb32 = __riscv_vsext_vf4_i32m4(vb8, b32);
                vfloat32m4_t vb = __riscv_vfcvt_f_x_v_f32m4(vb32, b32);
                vb = __riscv_vfmul_vf_f32m4(vb, scale_beta, b32);
                vy = __riscv_vfadd_vv_f32m4(vy, vb, b32);
            }

            size_t vlo = __riscv_vsetvl_e32m4(n_elem);
            vy = __riscv_vfdiv_vf_f32m4(vy, scale_out, vlo);

            mb_frm_write(MB_FRM_RMM);
            vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vy, vlo);
            mb_frm_write(frm_saved);

            vi = __riscv_vmax_vx_i32m4(vi, activation_min, vlo);
            vi = __riscv_vmin_vx_i32m4(vi, activation_max, vlo);

            size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
            vint16m2_t vi16 = __riscv_vncvt_x_x_w_i16m2(vi, vl16);
            size_t vlo8 = __riscv_vsetvl_e8m1(n_elem);
            vint8m1_t vi8 = __riscv_vncvt_x_x_w_i8m1(vi16, vlo8);
            __riscv_vse8_v_i8m1(row_out + k, vi8, vlo8);

            k += (int)vl8;
        }
    }
}
