/* source: curated */
/* algorithm: direct */
/* accuracy_class: bit_exact */
/* origin: vectorized RVV batched int8 GEMM. This is
 * rvv_matmul_s8_rvv_k_reduce_n_lanes.c one batch level up -- same two loop
 * orders, same reasons, an outer loop over the B independent M x K x N
 * products and a `bi` slice offset on each of the three pointers. B == 1
 * reduces to exactly that kernel. It differs from it in the requantize
 * tail, for the measured reason in the last paragraph.
 *
 * WHY TWO PATHS. The two attention matmuls have opposite layouts and one
 * loop order is wrong for one of them:
 *
 *   transpose_b != 0  (Q @ K.T per head)  b is [B,N,K], so a[bi,i,:] and
 *     b[bi,j,:] are BOTH K-contiguous. Reduce along K: vle8 both operands,
 *     vwmul into an i32 accumulator, one vredsum per (i,j). Tail strips
 *     accumulate tail-UNDISTURBED so lanes above vl keep the previous
 *     strip's partial sums, which is what makes the single closing
 *     reduction over VLMAX correct when K is not a multiple of it.
 *
 *   transpose_b == 0  (@ V, FFN)  b is [B,K,N], so b[bi,k,j] at fixed j is
 *     strided by N. Reducing along K here would need a strided load per
 *     element, and on this unit a strided vlse8 over a reduction axis
 *     measured 1.00x -- exactly zero speedup -- because spike charges
 *     strided/indexed loads per ELEMENT while unit-stride loads are charged
 *     per byte (notes/curated_rvv_kernels.md, variant #1). So N goes in the
 *     lanes instead: hold an i32 accumulator over an N-strip and vwmul.vx
 *     it by the SCALAR a[i,k], with b[k,:] loaded contiguously. Every load
 *     in both paths is unit-stride.
 *
 * BIT-EXACT, and it costs nothing to be. The accumulation is int32 and
 * therefore exact whatever order it happens in, so unlike the fp16 sibling
 * (rvv_f16_bmm_tb_f16_k_reduce.c, which is only reduction-order equivalent)
 * there is no drift to declare here. int8 x int8 cannot overflow int32 at
 * any K this op will see: |acc| <= K * 127 * 127 = K * 16129, so K would
 * have to exceed 133k.
 *
 * THE REQUANTIZE TAIL, WHICH IS THE ONE PLACE THIS FILE IS NOT A COPY OF
 * ITS UNBATCHED SIBLING. That kernel's header explains that the tail stays
 * scalar because `roundf` is ties-away-from-zero and vfcvt is not -- true
 * of vfcvt's default behaviour, but not of the instruction, and the same
 * applies to the SCALAR conversion. Both are fixed here, and both were
 * checked in the disassembly rather than assumed:
 *
 *   * `(int32_t)roundf(x)` on this toolchain (Zephyr SDK GCC 14.3.0,
 *     rv64gcv) emits a CALL to libm's roundf followed by `fcvt.w.s rtz`.
 *     The call is the expensive part and not because of its own cost: the
 *     RVV calling convention makes every vector register caller-clobbered,
 *     so a call inside the (i,j) loop forces the accumulator group to be
 *     spilled and reloaded around it -- measured as 4 vs4r/vs1r/vl4re32/
 *     vl1re32 in the unbatched kernel's object code, i.e. 320 bytes of
 *     spill traffic per OUTPUT ELEMENT against 2*K bytes of useful operand
 *     loads. At octo's head_dim K = 64 that is more traffic than the dot
 *     product it guards. `fcvt.w.s rd, rs1, rmm` is one instruction and IS
 *     roundf-then-convert (RMM = round to nearest, ties to Max Magnitude),
 *     so the scalar path uses it directly and calls nothing.
 *   * the N-lanes path does not need a scalar tail at all. The accumulator
 *     strip is already in a vector register, so it is converted, scaled,
 *     rounded under frm=RMM (same mode, same exactness -- see
 *     rvv_add_s8_rvv_frm_rmm.c for the on-board tie sweep), clamped in the
 *     int32 domain and narrowed to int8 in place. That also removes the
 *     vse32 -> scalar-reload round trip through `lane_acc`.
 *
 * The obvious cheap alternative, `(int32_t)(x >= 0 ? x + 0.5f : x - 0.5f)`,
 * was rejected and it is worth saying why, because it appears in several
 * reference impls and looks equivalent: it is not roundf near a tie. For
 * x = 0x3EFFFFFF (the largest float below 0.5), x + 0.5f rounds to exactly
 * 1.0f under ties-to-even, so the expression yields 1 where roundf yields
 * 0. Over a 2.8M-element attention product that is a real 1-LSB mismatch
 * budget, not a theoretical one.
 *
 * VTYPE. Every vsetvl is handed the ELEMENT COUNT, never a previous
 * vsetvl's result (this GCC miscompiles the chained form on partial tails
 * -- scripts/check_rvv_avl.py), and the widening steps go i8 -> i16 -> i32
 * so no single instruction widens by more than 2x.
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

/* (int32_t)roundf(x), as the one instruction that computes it. Guarded for
 * the same reason as the frm helpers: kernel bodies are concatenated into a
 * single kernels.c. */
#ifndef MB_SCALAR_ROUND_RMM_
#define MB_SCALAR_ROUND_RMM_
static inline int32_t mb_roundf_i32(float x)
{
    int32_t r;
    __asm__("fcvt.w.s %0, %1, rmm" : "=r"(r) : "f"(x));
    return r;
}
#endif /* MB_SCALAR_ROUND_RMM_ */

void kernel_matmul_b_s8(const int8_t *a, const int8_t *b, int8_t *output,
                        int B, int M, int K, int N,
                        float scale_a, float scale_b, float scale_out,
                        int transpose_b, float scale_div,
                        int activation_min, int activation_max)
{
    const float total = (scale_a * scale_b) / (scale_out * scale_div);

    if (B <= 0 || M <= 0 || N <= 0) {
        return;
    }

    if (transpose_b) {
        const size_t vlmax = __riscv_vsetvlmax_e8m1();

        for (int bi = 0; bi < B; bi++) {
            const int8_t *ab = a + (size_t)bi * (size_t)M * (size_t)K;
            const int8_t *bb = b + (size_t)bi * (size_t)K * (size_t)N;
            int8_t *ob = output + (size_t)bi * (size_t)M * (size_t)N;

            for (int i = 0; i < M; i++) {
                const int8_t *a_row = ab + (size_t)i * (size_t)K;

                for (int j = 0; j < N; j++) {
                    const int8_t *b_row = bb + (size_t)j * (size_t)K;
                    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vlmax);
                    int k = 0;

                    while (k < K) {
                        size_t vl = __riscv_vsetvl_e8m1((size_t)(K - k));
                        vint8m1_t va = __riscv_vle8_v_i8m1(a_row + k, vl);
                        vint8m1_t vb = __riscv_vle8_v_i8m1(b_row + k, vl);
                        vint16m2_t prod = __riscv_vwmul_vv_i16m2(va, vb, vl);
                        vacc = __riscv_vwadd_wv_i32m4_tu(vacc, vacc, prod, vl);
                        k += (int)vl;
                    }

                    vint32m1_t vinit = __riscv_vmv_s_x_i32m1(0, 1);
                    int32_t acc = __riscv_vmv_x_s_i32m1_i32(
                        __riscv_vredsum_vs_i32m4_i32m1(vacc, vinit, vlmax));

                    int32_t v = mb_roundf_i32((float)acc * total);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    ob[(size_t)i * (size_t)N + (size_t)j] = (int8_t)v;
                }
            }
        }
        return;
    }

    {
        const unsigned frm_saved = mb_frm_read();

        for (int bi = 0; bi < B; bi++) {
            const int8_t *ab = a + (size_t)bi * (size_t)M * (size_t)K;
            const int8_t *bb = b + (size_t)bi * (size_t)K * (size_t)N;
            int8_t *ob = output + (size_t)bi * (size_t)M * (size_t)N;

            for (int i = 0; i < M; i++) {
                const int8_t *a_row = ab + (size_t)i * (size_t)K;
                int8_t *o_row = ob + (size_t)i * (size_t)N;
                int j0 = 0;

                while (j0 < N) {
                    const size_t n_elem = (size_t)(N - j0);
                    size_t vl = __riscv_vsetvl_e8m1(n_elem);
                    vint32m4_t vacc = __riscv_vmv_v_x_i32m4(0, vl);

                    for (int k = 0; k < K; k++) {
                        vint8m1_t vb = __riscv_vle8_v_i8m1(
                            bb + (size_t)k * (size_t)N + (size_t)j0, vl);
                        vint16m2_t prod =
                            __riscv_vwmul_vx_i16m2(vb, a_row[k], vl);
                        vacc = __riscv_vwadd_wv_i32m4(vacc, prod, vl);
                    }

                    /* roundf((float)acc * total), in the lanes. */
                    size_t vl32 = __riscv_vsetvl_e32m4(n_elem);
                    vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(vacc, vl32);
                    vf = __riscv_vfmul_vf_f32m4(vf, total, vl32);

                    mb_frm_write(MB_FRM_RMM);
                    vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vf, vl32);
                    mb_frm_write(frm_saved);

                    vi = __riscv_vmax_vx_i32m4(vi, activation_min, vl32);
                    vi = __riscv_vmin_vx_i32m4(vi, activation_max, vl32);

                    size_t vl16 = __riscv_vsetvl_e16m2(n_elem);
                    vint16m2_t vi16 = __riscv_vncvt_x_x_w_i16m2(vi, vl16);
                    size_t vl8 = __riscv_vsetvl_e8m1(n_elem);
                    vint8m1_t vi8 = __riscv_vncvt_x_x_w_i8m1(vi16, vl8);
                    __riscv_vse8_v_i8m1(o_row + j0, vi8, vl8);

                    j0 += (int)vl;
                }
            }
        }
    }
}
