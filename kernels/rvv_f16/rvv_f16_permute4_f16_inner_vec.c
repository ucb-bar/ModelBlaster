/* source: curated */
/* algorithm: inner_vec */
/* origin: RVV+Zvfh rank-4 permute. A permute moves data and cannot change a
 * value, so this is BIT_EXACT by construction -- the only question is how
 * many elements move per instruction.
 *
 * The output is written in order, so the store is always unit-stride. The
 * load depends on where the permuted innermost axis came from:
 *
 *   os[3] == 1  the run is contiguous in the input too -> vle16 (a strided
 *               load with stride 2 would work but wastes the fast path)
 *   otherwise   -> vlse16 at stride os[3]*2 bytes
 *
 * The strided case is the one the head split needs, (1,S,H,D)->(1,H,S,D):
 * D is innermost in both, so it actually lands in the CONTIGUOUS case, and
 * the strided path serves NCHW->NHWC.
 */

#include <riscv_vector.h>

void kernel_permute4_f16(const _Float16 *input, _Float16 *output,
                         int d0, int d1, int d2, int d3,
                         int p0, int p1, int p2, int p3) {
    const int din[4] = { d0, d1, d2, d3 };
    const int sin[4] = { d1*d2*d3, d2*d3, d3, 1 };
    const int perm[4] = { p0, p1, p2, p3 };
    int od[4], os[4];
    for (int k = 0; k < 4; k++) { od[k] = din[perm[k]]; os[k] = sin[perm[k]]; }

    size_t w = 0;
    for (int o0 = 0; o0 < od[0]; o0++) {
      for (int o1 = 0; o1 < od[1]; o1++) {
        for (int o2 = 0; o2 < od[2]; o2++) {
          const size_t base = (size_t)o0*os[0] + (size_t)o1*os[1]
                            + (size_t)o2*os[2];
          const _Float16 *src = input + base;
          int o3 = 0;
          if (os[3] == 1) {
              while (o3 < od[3]) {
                  size_t vl = __riscv_vsetvl_e16m8((size_t)(od[3] - o3));
                  __riscv_vse16_v_f16m8(
                      output + w, __riscv_vle16_v_f16m8(src + o3, vl), vl);
                  w += vl; o3 += (int)vl;
              }
          } else {
              const ptrdiff_t bstride = (ptrdiff_t)os[3] * 2;
              while (o3 < od[3]) {
                  size_t vl = __riscv_vsetvl_e16m8((size_t)(od[3] - o3));
                  __riscv_vse16_v_f16m8(
                      output + w,
                      __riscv_vlse16_v_f16m8(src + (size_t)o3*os[3],
                                             bstride, vl), vl);
                  w += vl; o3 += (int)vl;
              }
          }
        }
      }
    }
}
