/* source: curated */
/* algorithm: zvt_tile_i8i32 */
/* accuracy_class: bit_exact */
/* origin: hand-written, first Zvt (draft RISC-V vector-matrix tile
 *   extension) kernel. Semantics checked against riscv-isa-sim 72958551
 *   riscv/insns/vtmms_tvv.h + riscv/vector_unit.h.
 *
 * ENGINE. vtmms.tvv with mtype widen=4 and vtype e8+altfmt accumulates
 *   acc[m][n] += sum_{ki<tk} A_ki[m] * B_ki[n]      (int8 x int8 -> int32)
 * for m < tm, n < tn (tn = vl), tk <= 4, where A_ki is vector register
 * group v(0+2*ki) and B_ki is v(8+2*ki) (spike: vreg_stride = 8/kmax = 2,
 * LMUL = 2 under msetmtype). So each operand is a K-COLUMN: tm int8
 * values for one k.
 *
 * LAYOUT IS FREE. input is [M,K] and weight is [N,K], both row-major, so
 * a K-column of either is one strided load (vlse8, stride K) straight
 * into the register the tile unit reads -- no packing buffers. A tile
 * step is 8 strided loads + 1 vtmms for a T x T x 4 MAC block.
 *
 * EDGES. msettm/msettn shrink the tile for the last row/column block;
 * msettn also sets vl, so A loads use vl = rows and B loads/vtmms use
 * vl = cols without a vsetvli (a vsetvli would clear mtype). The K
 * remainder (K % 4) is added in the scalar tail.
 *
 * SIZE-AGNOSTIC: the tile edge T is whatever the hart grants for
 * msettm/msettn (min(TE, 2*VLEN/8) for i8->i32), capped at MB_ZVT_TMAX,
 * so one binary adapts to any instance of the tile-unit generator.
 *
 * ASM BLOCKS: the tile config (mtype) only has to survive *within* one
 * asm statement -- any compiler-emitted vsetvl between statements clears
 * it, but never touches tile contents. So config + zero + K-loop is one
 * statement, and each read-out row is its own statement that starts by
 * re-establishing the config.
 *
 * SMALL N (< MB_ZVT_MIN_N, e.g. DroNet's 2048->1 heads) uses RVV dot
 * products instead: a near-empty tile wastes the engine. The requantize
 * tail is RVV too (zvt_requant_s8, bit-exact int64 lanes).
 *
 * OFFSETS. vtmms multiplies raw int8; nonzero input/filter offsets are
 * folded in with the usual sum correction (zero across this tree's
 * symmetric quantization, so the fast path is the common one).
 *
 * REQUIRES the calling thread to have Zvt enabled (mstatus.MS != Off):
 * the harness calls arch_riscv_zvt_enable() under CONFIG_RISCV_ISA_EXT_ZVT.
 */

#ifndef MB_ZVT_MIN_N
#define MB_ZVT_MIN_N 8 /* below this, a GEMV (RVV dot) beats the tile */
#endif

#ifndef MB_ZVT_TMAX
#define MB_ZVT_TMAX 128 /* cap on the granted tile edge (stack row buffer) */
#endif

/* Re-establish e8+altfmt / widen=4 / tk=4 / tm=mr / vl=tn=nc. */
#define ZVT_LIN_CFG \
    "vsetvl %[tmp], x0, %[vt]\n\t" \
    ZVT_S_MSETMTYPE("%[mt]", "%[vt]") \
    ZVT_S_MSETTM("%[tmp]", "%[mr]") \
    ZVT_S_MSETTN("%[tmp]", "%[nc]")

void kernel_linear_s8(const int8_t *input, const int8_t *weight, const int32_t *bias, int8_t *output, int M, int K, int N, int input_offset, int filter_offset, int output_offset, int output_multiplier, int output_shift, int activation_min, int activation_max) {
    if (M <= 0 || N <= 0) {
        return;
    }

    if (N < MB_ZVT_MIN_N) {
        /* Too few output columns to fill a tile (B loads would run at
         * vl = N and each vtmms would do 4*N MACs per row): contiguous-K
         * RVV dot products instead. */
        const int has_off_g = input_offset != 0 || filter_offset != 0;
        for (int m = 0; m < M; m++) {
            const int8_t *in_row = input + (size_t)m * (size_t)K;
            int32_t in_sum = 0;
            if (has_off_g) {
                for (int k = 0; k < K; k++) in_sum += in_row[k];
            }
            for (int n = 0; n < N; n++) {
                const int8_t *w_row = weight + (size_t)n * (size_t)K;
                int32_t acc = zvt_vdot8(in_row, w_row, K);
                if (has_off_g) {
                    int32_t w_sum = 0;
                    for (int k = 0; k < K; k++) w_sum += w_row[k];
                    acc += input_offset * w_sum + filter_offset * in_sum +
                           K * input_offset * filter_offset;
                }
                zvt_requant_s8(output + (size_t)m * (size_t)N + (size_t)n, &acc,
                               NULL, 0, NULL, 0, bias != NULL ? bias[n] : 0, 1,
                               output_multiplier, output_shift, output_offset,
                               activation_min, activation_max);
            }
        }
        return;
    }

    const unsigned long vtype = ZVT_VTYPE_E8 | ZVT_VTYPE_ALTFMT;
    const unsigned long mtype = ZVT_MTYPE(ZVT_WIDEN4, 4, 0);
    const long k_steps = K / 4;
    const int k_main = (int)k_steps * 4;
    const long stride = K;
    int32_t row_out[MB_ZVT_TMAX];

    /* Granted tile edge: msettm/msettn clamp to min(LMUL*VLEN/8, TE). */
    unsigned long T;
    {
        unsigned long want = MB_ZVT_TMAX, gm, gn, tmp;
        __asm__ volatile(
            "vsetvl %[tmp], x0, %[vt]\n\t"
            ZVT_S_MSETMTYPE("%[mt]", "%[vt]")
            ZVT_S_MSETTM("%[gm]", "%[w]")
            ZVT_S_MSETTN("%[gn]", "%[w]")
            : [gm] "=&r"(gm), [gn] "=&r"(gn), [tmp] "=&r"(tmp)
            : [mt] "r"(mtype), [vt] "r"(vtype), [w] "r"(want)
            : "memory");
        T = gm < gn ? gm : gn;
    }

    for (int m0 = 0; m0 < M; m0 += (int)T) {
        unsigned long mr = (unsigned long)((M - m0) < (int)T ? (M - m0) : (int)T);

        for (int n0 = 0; n0 < N; n0 += (int)T) {
            unsigned long nc = (unsigned long)((N - n0) < (int)T ? (N - n0) : (int)T);
            const int8_t *pa = input + (size_t)m0 * (size_t)K;
            const int8_t *pb = weight + (size_t)n0 * (size_t)K;
            long ks = k_steps;
            unsigned long tmp;

            __asm__ volatile(
                ZVT_LIN_CFG
                ZVT_S_VTZERO(0)
                "beqz %[ks], 2f\n"
                "1:\n\t"
                ZVT_S_MSETTN("%[tmp]", "%[mr]")
                "vlse8.v v0, (%[pa]), %[st]\n\t"
                "addi t1, %[pa], 1\n\t"
                "vlse8.v v2, (t1), %[st]\n\t"
                "addi t1, %[pa], 2\n\t"
                "vlse8.v v4, (t1), %[st]\n\t"
                "addi t1, %[pa], 3\n\t"
                "vlse8.v v6, (t1), %[st]\n\t"
                ZVT_S_MSETTN("%[tmp]", "%[nc]")
                "vlse8.v v8, (%[pb]), %[st]\n\t"
                "addi t1, %[pb], 1\n\t"
                "vlse8.v v10, (t1), %[st]\n\t"
                "addi t1, %[pb], 2\n\t"
                "vlse8.v v12, (t1), %[st]\n\t"
                "addi t1, %[pb], 3\n\t"
                "vlse8.v v14, (t1), %[st]\n\t"
                ZVT_S_VTMMS(0, 0, 8)
                "addi %[pa], %[pa], 4\n\t"
                "addi %[pb], %[pb], 4\n\t"
                "addi %[ks], %[ks], -1\n\t"
                "bnez %[ks], 1b\n"
                "2:\n\t"
                : [pa] "+r"(pa), [pb] "+r"(pb), [ks] "+r"(ks), [tmp] "=&r"(tmp)
                : [mt] "r"(mtype), [vt] "r"(vtype), [mr] "r"(mr), [nc] "r"(nc),
                  [st] "r"(stride)
                : "t1", "memory",
                  "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                  "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");

            /* Offset correction terms, hoisted per tile: O((mr+nc)*K)
             * instead of O(mr*nc*K). Only needed for asymmetric quant. */
            const int has_off = input_offset != 0 || filter_offset != 0;
            int32_t row_sum[MB_ZVT_TMAX], col_sum[MB_ZVT_TMAX];
            if (has_off) {
                for (int i = 0; i < (int)mr; i++) {
                    const int8_t *r = input + (size_t)(m0 + i) * (size_t)K;
                    int32_t acc_s = 0;
                    for (int k = 0; k < K; k++) acc_s += r[k];
                    row_sum[i] = acc_s;
                }
                for (int j = 0; j < (int)nc; j++) {
                    const int8_t *r = weight + (size_t)(n0 + j) * (size_t)K;
                    int32_t acc_s = 0;
                    for (int k = 0; k < K; k++) acc_s += r[k];
                    col_sum[j] = acc_s;
                }
            }

            for (int i = 0; i < (int)mr; i++) {
                int m = m0 + i;
                const int8_t *in_row = input + (size_t)m * (size_t)K;
                unsigned long tss = (unsigned long)i; /* tile 0, row i */

                __asm__ volatile(
                    ZVT_LIN_CFG
                    ZVT_S_VTSE32("%[po]", "%[tss]")
                    : [tmp] "=&r"(tmp)
                    : [mt] "r"(mtype), [vt] "r"(vtype), [mr] "r"(mr), [nc] "r"(nc),
                      [po] "r"(row_out), [tss] "r"(tss)
                    : "memory");

                if (k_main < K) { /* K % 4 remainder */
                    for (int jj = 0; jj < (int)nc; jj++) {
                        const int8_t *w_row = weight + (size_t)(n0 + jj) * (size_t)K;
                        for (int k = k_main; k < K; k++) {
                            row_out[jj] += (int32_t)in_row[k] * (int32_t)w_row[k];
                        }
                    }
                }
                zvt_requant_s8(output + (size_t)m * (size_t)N + (size_t)n0, row_out,
                               bias != NULL ? bias + n0 : NULL, 1,
                               has_off ? col_sum : NULL, input_offset,
                               has_off ? filter_offset * row_sum[i] +
                                             K * input_offset * filter_offset : 0,
                               (int)nc, output_multiplier, output_shift, output_offset,
                               activation_min, activation_max);
            }
        }
    }
}
