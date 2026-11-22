/* source: curated */
/* algorithm: ime_vmadot_4x4x8 */
/* accuracy_class: bit_exact */
/* origin: conv2d_s8 on the SpaceMiT K1 IME (smt.vmadot) matrix engine, by
 *   im2col-into-GEMM lowering. The MAC core (4x4x8 micro-tile packing + the
 *   vmadot .insn accumulate) is copied VERBATIM from the board-proven
 *   kernels/ime/ime_matmul_s8_ime_vmadot_4x4x8.c -- read its header for the
 *   four hardware facts (4x4x8 hw-forced; vmadot ACCUMULATES vd += A.B^T; 16
 *   int32 results row-major across (vd,vd+1); CLUSTER 0 ONLY, harts 4-7 SIGILL).
 *
 * WHAT IS DIFFERENT from the matmul, and why it stays bit-exact:
 *   1. A is the im2col matrix A[M = OH*OW, K = IC*KH*KW], built ON THE FLY
 *      directly into each 4x8 A-tile: for packed row i (an output pixel
 *      m = oh*OW+ow) and packed column q (a filter tap k = (ic*KH+kh)*KW+kw)
 *      we gather input[(n,ic,ih,iw)] with ih=oh*SH-PH+kh, iw=ow*SW-PW+kw, and
 *      zero-pad out-of-bounds. No materialized im2col buffer.
 *   2. B is the weight in its NATIVE IHWOC layout weight[k*OC + oc]; that is
 *      exactly the transpose_b==0 branch of the matmul B-pack, so no copy/
 *      transpose is needed and the packed tiles are identical to a matmul with
 *      A=im2col, B=weight, N=OC.
 *   3. SYMMETRIC int8 only (input_offset==filter_offset==0 -- verified true for
 *      every conv in dronet and yolov8_nano). Then vmadot's raw int8 product IS
 *      the conv MAC; no row-sum/col-sum offset correction. Asymmetric convs are
 *      left on RVV (the picker chooses per dispatch).
 *   4. The requantize TAIL is the conv's scalar Q0.31 path (bias add in the
 *      int32 accumulator domain, then (acc*mult + (1<<30))>>31, rounding
 *      right-shift, +output_offset, activation clamp) copied from
 *      kernels/rvv/rvv_conv2d_s8_pc_direct.c (mb_q31_requant_pc) and the plain
 *      kernel_conv2d_s8 reference -- NOT the matmul kernel's float requantize
 *      (float roundf and Q0.31 round-half-up differ). The int32 MAC is
 *      identical to RVV; only the tail must match the conv oracle.
 *
 * Because M = OH*OW is large for early (big-spatial) convs and the matmul core
 * wins for large M (M=64 -> 1.43x, M=128 -> 2.30x vs RVV; it LOSES at small M,
 * M=7 -> 0.25x), this kernel is expected to WIN on early conv layers and lose
 * on late small-spatial layers -- so the scheduler keeps it as a per-dispatch
 * ALTERNATIVE and takes it only where it beats RVV.
 */
#include <stdint.h>
#include <stddef.h>

/* Scalar Q0.31 requantize, identical to rvv_conv2d_s8_pc_direct.c and the
 * plain kernel_conv2d_s8 reference. Per-tensor (scalar mult/shift). */
static inline int32_t mb_q31_requant(int32_t x, int32_t mult, int32_t shift) {
    int64_t prod = (int64_t)x * (int64_t)mult;
    prod = (prod + (1LL << 30)) >> 31;
    int32_t scaled = (int32_t)prod;
    if (shift > 0) {
        int32_t round = (1 << (shift - 1));
        return (scaled + round) >> shift;
    }
    return scaled << (-shift);
}

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max)
{
    if (OC <= 0 || IC <= 0) return;
    /* Symmetric fast path only. An asymmetric conv must not silently produce
     * wrong numbers -- leave it for the RVV kernel (the picker's job). */
    if (input_offset != 0 || filter_offset != 0) return;

    int OH = (IH + 2 * PH - KH) / SH + 1;
    int OW = (IW + 2 * PW - KW) / SW + 1;
    int M = OH * OW;                 /* im2col rows, per batch element   */
    int K = IC * KH * KW;            /* im2col inner (filter tap) length */
    int KHW = KH * KW;
    if (M <= 0 || K <= 0) return;

    int m_tiles = (M + 3) / 4;
    int k_slabs = (K + 7) / 8;
    size_t bytes_per_m_tile = (size_t)k_slabs * 32u;

    /* Block the m-tiles so the packed-A working set stays bounded (mirrors the
     * matmul's block_m_tiles budget). */
    size_t max_a_pack_bytes = 256u * 1024u;
    int block_m_tiles = (int)(max_a_pack_bytes / bytes_per_m_tile);
    if (block_m_tiles < 1) block_m_tiles = 1;
    if (block_m_tiles > m_tiles) block_m_tiles = m_tiles;

    int8_t packed_a[(size_t)block_m_tiles * bytes_per_m_tile];
    int8_t packed_b[(size_t)k_slabs * 32u];
    int32_t tile_output[16];

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;

        for (int mt_outer = 0; mt_outer < m_tiles; mt_outer += block_m_tiles) {
            int current_m_tiles = m_tiles - mt_outer;
            if (current_m_tiles > block_m_tiles) current_m_tiles = block_m_tiles;

            /* ---- pack A: im2col gather straight into 4x8 tiles ---- */
            for (int mt = 0; mt < current_m_tiles; mt++) {
                int m0 = (mt_outer + mt) * 4;
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    int8_t *dst = packed_a +
                        ((size_t)mt * (size_t)k_slabs + (size_t)ks) * 32u;
                    for (int i = 0; i < 4; i++) {
                        int row = m0 + i;
                        int oh = -1, ow = -1;
                        if (row < M) { oh = row / OW; ow = row - oh * OW; }
                        for (int q = 0; q < 8; q++) {
                            int k = k0 + q;
                            int8_t v = 0;
                            if (row < M && k < K) {
                                int ic = k / KHW;
                                int r  = k - ic * KHW;
                                int kh = r / KW;
                                int kw = r - kh * KW;
                                int ih = oh * SH - PH + kh;
                                int iw = ow * SW - PW + kw;
                                if (ih >= 0 && ih < IH && iw >= 0 && iw < IW)
                                    v = in_n[((size_t)ic * IH + ih) * IW + iw];
                            }
                            dst[i * 8 + q] = v;
                        }
                    }
                }
            }

            /* ---- for each 4-col panel of B (= 4 output channels) ---- */
            for (int n0 = 0; n0 < OC; n0 += 4) {
                for (int ks = 0; ks < k_slabs; ks++) {
                    int k0 = ks * 8;
                    int8_t *dst = packed_b + (size_t)ks * 32u;
                    for (int j = 0; j < 4; j++) {
                        int col = n0 + j;           /* output channel oc */
                        for (int q = 0; q < 8; q++) {
                            int k = k0 + q;
                            int8_t value = 0;
                            /* native IHWOC weight: weight[k*OC + oc]
                             * (== transpose_b==0 branch of the matmul). */
                            if (col < OC && k < K)
                                value = weight[(size_t)k * OC + col];
                            dst[j * 8 + q] = value;
                        }
                    }
                }

                for (int mt = 0; mt < current_m_tiles; mt++) {
                    int m0 = (mt_outer + mt) * 4;
                    const int8_t *a_tile = packed_a + (size_t)mt * bytes_per_m_tile;
                    const int8_t *b_tile = packed_b;
                    int slabs = k_slabs;
                    size_t n32 = 32, n8 = 8;

                    __asm__ volatile(
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        "vmv.v.i v8, 0\n\t"
                        "vmv.v.i v9, 0\n\t"
                        "1:\n\t"
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        "vle8.v v0, (%[pa])\n\t"
                        "vle8.v v4, (%[pb])\n\t"
                        "vsetvli t0, %[n32], e8, m1, ta, ma\n\t"
                        ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t"
                        "addi %[pa], %[pa], 32\n\t"
                        "addi %[pb], %[pb], 32\n\t"
                        "addi %[ks], %[ks], -1\n\t"
                        "bnez %[ks], 1b\n\t"
                        "vsetvli t0, %[n8], e32, m1, ta, ma\n\t"
                        "vse32.v v8, (%[o0])\n\t"
                        "vse32.v v9, (%[o1])\n\t"
                        : [pa] "+r"(a_tile), [pb] "+r"(b_tile), [ks] "+r"(slabs)
                        : [o0] "r"(tile_output), [o1] "r"(tile_output + 8),
                          [n32] "r"(n32), [n8] "r"(n8)
                        : "t0", "memory", "v0", "v4", "v8", "v9");

                    /* ---- conv Q0.31 tail: bias add, requant, clamp, store ---- */
                    for (int i = 0; i < 4 && m0 + i < M; i++) {
                        int row = m0 + i;
                        int oh = row / OW, ow = row - (row / OW) * OW;
                        for (int j = 0; j < 4 && n0 + j < OC; j++) {
                            int oc = n0 + j;
                            int32_t acc = tile_output[i * 4 + j];
                            if (bias) acc += bias[oc];
                            int32_t s = mb_q31_requant(acc, output_multiplier,
                                                       output_shift);
                            s += output_offset;
                            if (s < activation_min) s = activation_min;
                            if (s > activation_max) s = activation_max;
                            output[(((size_t)n * OC + oc) * OH + oh) * OW + ow]
                                = (int8_t)s;
                        }
                    }
                }
            }
        }
    }
}
