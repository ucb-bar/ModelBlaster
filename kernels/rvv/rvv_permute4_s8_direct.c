/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV rank-4 permute for int8. A permute MOVES data, and
 * when scale_in == scale_out it cannot change a value at all, so the only
 * question is how many elements move per instruction.
 *
 * FOUR PATHS, from two independent binary choices, and the reason both
 * choices exist is worth stating because one of them is a performance cliff:
 *
 *   the STORE is always unit-stride -- the output is written in index order,
 *   which is what `w++` in the reference means.
 *
 *   the LOAD depends on where the permuted innermost axis came from:
 *     os[3] == 1   the run is contiguous in the input too -> vle8
 *     otherwise    -> vlse8 at byte stride os[3]
 *
 *   the VALUE is either copied (scale_in == scale_out) or requantized.
 *
 * WHICH PATH THIS MODEL ACTUALLY TAKES, and why that matters here more than
 * in the fp16 sibling. spike charges unit-stride vector loads roughly per
 * BYTE but strided/indexed loads per ELEMENT (notes/curated_rvv_kernels.md,
 * variant #1 measured 1.00x -- exactly zero win -- for an otherwise
 * identical strided kernel). So the strided branch buys the STORE side only
 * and should not be expected to scale like the contiguous one. The ViT head
 * split this op exists for, (1,S,H,D)->(1,H,S,D), keeps D innermost at both
 * ends and therefore lands in the CONTIGUOUS branch; the strided branch
 * serves NCHW<->NHWC-shaped permutes, and is present for correctness rather
 * than for a speedup claim.
 *
 * The pure-copy path is BIT-EXACT by construction (it is a byte move) and
 * runs at e8m8 -- the widest group, since there is no arithmetic to leave
 * register room for. The requantize path is bit-exact for two reasons:
 *
 *   1. THE ARITHMETIC IS ALREADY ONE MULTIPLY IN THE REFERENCE. It hoists
 *      `ratio = scale_in / scale_out` out of the loop and computes
 *      `(float)v * ratio`, so folding here is not a deviation -- it is the
 *      reference's own expression, and there is no intermediate rounding to
 *      preserve.
 *   2. THE ROUNDING. The reference tail is
 *      `(int32_t)(f >= 0 ? f + 0.5f : f - 0.5f)`, and the obvious
 *      vectorization -- vfcvt.x.f under frm=RMM, i.e. exactly `roundf` --
 *      is NOT equal to it: adding 0.5f is itself a rounded operation, so
 *      for the largest float below a .5 tie the add carries into the next
 *      integer and the truncation keeps it. Measured over the 400 floats
 *      either side of every x.5 for x in [-300, 300] (481,401 values),
 *      frm=RMM/roundf disagrees with the reference form on 2 of them. So
 *      the tail is reproduced literally: |f| + 0.5f (vfabs, vfadd.vf), sign
 *      restored from f (vfsgnj.vv), round-toward-zero convert
 *      (vfcvt.rtz.x.f.v); identical to the reference's branch because float
 *      negation is exact, verified over 108,606,540 random float bit
 *      patterns with |f| < 4096 plus that tie sweep, ZERO mismatches. See
 *      rvv_add_c1_s8_direct.c for the long form.
 *
 * VTYPE. Each width transition names its own width, and every vsetvl is
 * handed the ELEMENT COUNT rather than a previous vsetvl's result (this GCC
 * miscompiles the chained form on partial tails --
 * scripts/check_rvv_avl.py). vsext.vf4 is issued under SEW=32; under SEW=8
 * it implies a 2-bit source and is a SIGILL on the board, not a build error.
 */

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>

void kernel_permute4_s8(const int8_t *input, int8_t *output,
                        int d0, int d1, int d2, int d3,
                        int p0, int p1, int p2, int p3,
                        float scale_in, float scale_out,
                        int activation_min, int activation_max)
{
    const int din[4] = { d0, d1, d2, d3 };
    const int sin[4] = { d1*d2*d3, d2*d3, d3, 1 };
    const int perm[4] = { p0, p1, p2, p3 };
    int od[4], os[4];
    for (int k = 0; k < 4; k++) { od[k] = din[perm[k]]; os[k] = sin[perm[k]]; }

    const int pure = (scale_in == scale_out);
    const int contig = (os[3] == 1);
    /* The reference's own hoisted constant, not a fold of ours. */
    const float ratio = scale_in / scale_out;
    const ptrdiff_t bstride = (ptrdiff_t)os[3];

    size_t w = 0;
    for (int o0 = 0; o0 < od[0]; o0++) {
      for (int o1 = 0; o1 < od[1]; o1++) {
        for (int o2 = 0; o2 < od[2]; o2++) {
          const size_t base = (size_t)o0*os[0] + (size_t)o1*os[1]
                            + (size_t)o2*os[2];
          const int8_t *src = input + base;
          int o3 = 0;

          if (pure && contig) {
              while (o3 < od[3]) {
                  size_t vl = __riscv_vsetvl_e8m8((size_t)(od[3] - o3));
                  __riscv_vse8_v_i8m8(
                      output + w, __riscv_vle8_v_i8m8(src + o3, vl), vl);
                  w += vl; o3 += (int)vl;
              }
          } else if (pure) {
              while (o3 < od[3]) {
                  size_t vl = __riscv_vsetvl_e8m8((size_t)(od[3] - o3));
                  __riscv_vse8_v_i8m8(
                      output + w,
                      __riscv_vlse8_v_i8m8(src + (size_t)o3*os[3],
                                           bstride, vl), vl);
                  w += vl; o3 += (int)vl;
              }
          } else {
              while (o3 < od[3]) {
                  const size_t n_elem = (size_t)(od[3] - o3);

                  size_t vl8 = __riscv_vsetvl_e8m2(n_elem);
                  vint8m2_t vs8 = contig
                      ? __riscv_vle8_v_i8m2(src + o3, vl8)
                      : __riscv_vlse8_v_i8m2(src + (size_t)o3*os[3],
                                             bstride, vl8);

                  size_t vl = __riscv_vsetvl_e32m8(n_elem);
                  vint32m8_t vs32 = __riscv_vsext_vf4_i32m8(vs8, vl);
                  vfloat32m8_t vf = __riscv_vfcvt_f_x_v_f32m8(vs32, vl);
                  vf = __riscv_vfmul_vf_f32m8(vf, ratio, vl);

                  vfloat32m8_t vm = __riscv_vfabs_v_f32m8(vf, vl);
                  vm = __riscv_vfadd_vf_f32m8(vm, 0.5f, vl);
                  vm = __riscv_vfsgnj_vv_f32m8(vm, vf, vl);
                  vint32m8_t vi = __riscv_vfcvt_rtz_x_f_v_i32m8(vm, vl);

                  vi = __riscv_vmax_vx_i32m8(vi, activation_min, vl);
                  vi = __riscv_vmin_vx_i32m8(vi, activation_max, vl);

                  size_t vl16 = __riscv_vsetvl_e16m4(n_elem);
                  vint16m4_t vi16 = __riscv_vncvt_x_x_w_i16m4(vi, vl16);
                  size_t vlo8 = __riscv_vsetvl_e8m2(n_elem);
                  vint8m2_t vi8 = __riscv_vncvt_x_x_w_i8m2(vi16, vlo8);
                  __riscv_vse8_v_i8m2(output + w, vi8, vlo8);

                  w += vl8; o3 += (int)vl8;
              }
          }
        }
      }
    }
}
