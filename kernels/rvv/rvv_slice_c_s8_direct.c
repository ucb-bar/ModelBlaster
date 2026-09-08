/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV slice_c_s8 (contiguous channel-axis slice of an
 * NCHW int8 tensor).
 *
 * TWO PATHS, because the reference has two and they are not the same kind of
 * work:
 *
 *   scale_in == scale_out  the requantize tail is structurally a no-op
 *     (round((x*s)/s) == x), so the reference degenerates to a copy. And the
 *     copy is over ONE contiguous run per batch, not OC separate ones: the
 *     source channels [C_start, C_end) are adjacent in the input and the
 *     destination channels [0, OC) are adjacent in the output, so the
 *     reference's per-channel loop is walking a single OC*H*W-byte block at
 *     both ends. Collapsing it lets the whole slice move at e8m8 with no
 *     per-channel loop overhead and no partial tail except at the very end.
 *     Byte-identical to the reference by construction.
 *
 *   scale_in != scale_out  every element is rescaled and clipped. Still one
 *     contiguous run per batch, so the load and the store are both
 *     unit-stride; the body is the standard int8 -> i32 -> f32 widen, the
 *     reference's multiply-then-divide, and a rounding-mode-exact tail.
 *
 * BIT-EXACT in the requantize path too, on the usual two grounds:
 *
 *   1. THE ROUNDING MODE. The reference tail is `roundf(...)`, which is
 *      round-to-nearest TIES AWAY FROM ZERO. vfcvt.x.f rounds by `frm`, and
 *      frm=RMM (ties to Max Magnitude) is that mode, so the conversion IS
 *      roundf rather than an approximation of it. frm is bracketed around
 *      the conversion only -- it also governs vfmul/vfdiv, where C rounds
 *      ties-to-even. See rvv_add_s8_rvv_frm_rmm.c for the tie sweep.
 *   2. THE OPERATION ORDER. `(float)src[i] * scale_in / scale_out` is a
 *      multiply THEN a divide, and is issued as such; folding the pair into
 *      one ratio constant, or turning the divide into a reciprocal
 *      multiply, is a different number.
 *
 * VTYPE. Each width transition names its own width and every vsetvl gets the
 * ELEMENT COUNT, not a previous vsetvl's result (this GCC miscompiles the
 * chained form on partial tails -- scripts/check_rvv_avl.py). vsext.vf4 is
 * issued under SEW=32; under SEW=8 it implies a 2-bit source and SIGILLs on
 * the board instead of failing the build.
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

void kernel_slice_c_s8(const int8_t *input, int8_t *output,
                       int N, int IC, int C_start, int C_end, int H, int W,
                       float scale_in, float scale_out,
                       int activation_min, int activation_max)
{
    const int OC = C_end - C_start;
    if (OC <= 0 || H <= 0 || W <= 0) {
        return;
    }
    const size_t plane = (size_t)H * (size_t)W;
    const size_t run = (size_t)OC * plane;
    const int pure = (scale_in == scale_out);
    const unsigned frm_saved = mb_frm_read();

    for (int n = 0; n < N; n++) {
        const int8_t *src = input
                          + ((size_t)n * (size_t)IC + (size_t)C_start) * plane;
        int8_t *dst = output + (size_t)n * run;

        size_t i = 0;
        if (pure) {
            /* One contiguous OC*H*W block at both ends. */
            while (i < run) {
                size_t vl = __riscv_vsetvl_e8m8(run - i);
                __riscv_vse8_v_i8m8(dst + i,
                                    __riscv_vle8_v_i8m8(src + i, vl), vl);
                i += vl;
            }
            continue;
        }

        while (i < run) {
            const size_t n_elem = run - i;

            size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
            vint8m2_t vs8 = __riscv_vle8_v_i8m2(src + i, vl8);

            size_t vl = __riscv_vsetvl_e32m8(n_elem);
            vint32m8_t vs32 = __riscv_vsext_vf4_i32m8(vs8, vl);

            vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vs32, vl);
            vf = __riscv_vfmul_vf_f32m8(vf, scale_in, vl);
            vf = __riscv_vfdiv_vf_f32m8(vf, scale_out, vl);

            mb_frm_write(MB_FRM_RMM);
            vint32m8_t vi = __riscv_vfcvt_x_f_v_i32m8(vf, vl);
            mb_frm_write(frm_saved);

            vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
            vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);

            size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
            vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl16);
            size_t vlo8 = __riscv_vsetvl_e8m2(n_elem);
            vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vlo8);
            __riscv_vse8_v_i8m2(dst + i, vi8, vlo8);

            i += vl8;
        }
    }
}
