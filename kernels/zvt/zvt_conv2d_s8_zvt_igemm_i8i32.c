/* source: curated */
/* algorithm: zvt_igemm_i8i32 */
/* accuracy_class: bit_exact */
/* origin: hand-written Zvt (draft RISC-V vector-matrix tile extension)
 *   conv2d_s8, implicit GEMM. Same vtmms engine as
 *   kernels/zvt/zvt_linear_s8_zvt_tile_i8i32.c -- read that header first.
 *
 * MAPPING (per batch image):
 *   M = OC          A K-columns = weight rows, OIHW: weight[oc*K + k] is
 *                   K-contiguous, so a column over oc is one vlse8, stride K
 *   N = OH*OW       B K-columns = an im2col strip, packed 4 columns x nc
 *                   positions per K-step (zeros at padding positions)
 *   K = IC*KH*KW    k = (ic*KH + kh)*KW + kw, the OIHW inner order
 *   out[n][oc][p]  row m of the accumulator is contiguous in p (NCHW).
 *
 * REUSE: up to 4 accumulator tiles (int32 tiles 0/4/8/12) are live at once,
 * so each packed strip feeds up to 4*T output channels before the next
 * K-step.
 *
 * K TAIL: the last step's B columns k >= K are packed as zeros and the A
 * column addresses are clamped to k = K-1, so nothing is read out of
 * bounds and the extra products are 0.
 *
 * VECTORIZED TAILS: im2col packing (zvt_pack_col: one strided load per
 * output-row segment) and requantization (zvt_requant_s8, int64 lanes,
 * bit-exact) are RVV; scalar versions dominated the instruction count.
 *
 * OFFSETS: padding is raw 0 in the strip, exactly the reference's
 * "in_v = input_offset" after the usual sum correction:
 *   acc += io*sum_k w[oc][k] + fo*sum_k strip[k][p] + K*io*fo.
 *
 * Weights must be OIHW: the zvt backend does not set
 * MODELBLASTER_RVV_IHWOC_WEIGHTS (see pipeline/backends.py ZVT).
 *
 * Tile config only needs to hold within one asm statement; every
 * statement re-establishes it (cheap: vsetvl + 3 tile CSR ops).
 */

#ifndef MB_ZVT_TMAX
#define MB_ZVT_TMAX 128
#endif

#define ZVT_CONV_BASE_CFG \
    "vsetvl %[tmp], x0, %[vt]\n\t" \
    ZVT_S_MSETMTYPE("%[mt]", "%[vt]")

/* One accumulator group j (tile 4j): tm = vl = mr for the A loads, then
 * vl = nc for the MAC. A columns at a0..a3 (stride K). */
#define ZVT_CONV_GROUP(j) \
    "ld t1, " #j "*8(%[mrs])\n\t" \
    "beqz t1, 9" #j "f\n\t" \
    ZVT_S_MSETTM("%[tmp]", "t1") \
    ZVT_S_MSETTN("%[tmp]", "t1") \
    "ld t2, " #j "*32+0(%[pas])\n\t" \
    "vlse8.v v0, (t2), %[st]\n\t" \
    "ld t2, " #j "*32+8(%[pas])\n\t" \
    "vlse8.v v2, (t2), %[st]\n\t" \
    "ld t2, " #j "*32+16(%[pas])\n\t" \
    "vlse8.v v4, (t2), %[st]\n\t" \
    "ld t2, " #j "*32+24(%[pas])\n\t" \
    "vlse8.v v6, (t2), %[st]\n\t" \
    ZVT_S_MSETTN("%[tmp]", "%[nc]") \
    ZVT_S_VTMMS(4 * j, 0, 8) \
    "9" #j ":\n\t"

/* Pack one im2col column: dst[c] = input pixel feeding output position
 * p0+c through tap (kh,kw) of one input-channel plane, 0 at padding.
 * Vectorized per output-row segment: in-bounds pixels of a segment are one
 * strided (stride SW) load; the padded ends are zero-filled. */
static inline void zvt_pack_col(int8_t *dst, const int8_t *plane, int kh, int kw,
                                int p0, int nc, int IH, int IW, int OW,
                                int SH, int SW, int PH, int PW)
{
    int c = 0;
    while (c < nc) {
        int p = p0 + c, oh = p / OW, ow = p - oh * OW;
        int seg = OW - ow < nc - c ? OW - ow : nc - c;
        int ih = oh * SH - PH + kh;
        int8_t *d = dst + c;
        if (ih < 0 || ih >= IH) {
            zvt_vzero8(d, seg);
        } else {
            /* output columns ow' in [lo, hi) map to 0 <= iw < IW */
            int lo = ow, hi = ow + seg;
            int t = PW - kw;                      /* need ow'*SW >= t */
            if (t > 0) {
                int m = (t + SW - 1) / SW;
                if (lo < m) lo = m;
            }
            int u = IW - 1 + PW - kw;             /* need ow'*SW <= u */
            int mh = u < 0 ? -1 : u / SW;
            if (hi > mh + 1) hi = mh + 1;
            if (lo >= hi) {
                zvt_vzero8(d, seg);
            } else {
                zvt_vzero8(d, lo - ow);
                zvt_vzero8(d + (hi - ow), ow + seg - hi);
                zvt_vcopy8(d + (lo - ow),
                           plane + (size_t)ih * IW + (lo * SW - PW + kw), SW, hi - lo);
            }
        }
        c += seg;
    }
}

void kernel_conv2d_s8(const int8_t *input, const int8_t *weight,
                      const int32_t *bias, int8_t *output,
                      int N, int IC, int IH, int IW, int OC,
                      int KH, int KW, int SH, int SW, int PH, int PW,
                      int input_offset, int filter_offset, int output_offset,
                      int output_multiplier, int output_shift,
                      int activation_min, int activation_max) {
    const int OH = (IH + 2 * PH - KH) / SH + 1;
    const int OW = (IW + 2 * PW - KW) / SW + 1;
    const int P = OH * OW;
    const int K = IC * KH * KW;
    if (N <= 0 || OC <= 0 || P <= 0 || K <= 0) {
        return;
    }

    const unsigned long vtype = ZVT_VTYPE_E8 | ZVT_VTYPE_ALTFMT;
    const unsigned long mtype = ZVT_MTYPE(ZVT_WIDEN4, 4, 0);
    const long stride = K;
    const int has_off = input_offset != 0 || filter_offset != 0;

    unsigned long T;
    {
        unsigned long want = MB_ZVT_TMAX, gm, gn, tmp;
        __asm__ volatile(
            ZVT_CONV_BASE_CFG
            ZVT_S_MSETTM("%[gm]", "%[w]")
            ZVT_S_MSETTN("%[gn]", "%[w]")
            : [gm] "=&r"(gm), [gn] "=&r"(gn), [tmp] "=&r"(tmp)
            : [mt] "r"(mtype), [vt] "r"(vtype), [w] "r"(want)
            : "memory");
        T = gm < gn ? gm : gn;
    }

    int8_t strip[4 * MB_ZVT_TMAX] __attribute__((aligned(16)));
    int32_t row_out[MB_ZVT_TMAX];
    int32_t col_sum[MB_ZVT_TMAX];

    for (int n = 0; n < N; n++) {
        const int8_t *in_n = input + (size_t)n * IC * IH * IW;
        int8_t *out_n = output + (size_t)n * OC * P;

        for (int p0 = 0; p0 < P; p0 += (int)T) {
            unsigned long nc = (unsigned long)((P - p0) < (int)T ? (P - p0) : (int)T);

            if (has_off) {
                for (int c = 0; c < (int)nc; c++) col_sum[c] = 0;
            }

            for (int g0 = 0; g0 < OC; g0 += 4 * (int)T) {
                unsigned long mrs[4];
                for (int j = 0; j < 4; j++) {
                    int oc = g0 + j * (int)T;
                    int r = OC - oc;
                    mrs[j] = (unsigned long)(r <= 0 ? 0 : (r < (int)T ? r : (int)T));
                }

                /* zero the live accumulators */
                {
                    unsigned long tmp;
                    __asm__ volatile(
                        ZVT_CONV_BASE_CFG
                        ZVT_S_MSETTN("%[tmp]", "%[nc]")
                        "ld t1, 0(%[mrs])\n\t" ZVT_S_MSETTM("%[tmp]", "t1") ZVT_S_VTZERO(0)
                        "ld t1, 8(%[mrs])\n\t" "beqz t1, 1f\n\t" ZVT_S_MSETTM("%[tmp]", "t1") ZVT_S_VTZERO(4)
                        "ld t1, 16(%[mrs])\n\t" "beqz t1, 1f\n\t" ZVT_S_MSETTM("%[tmp]", "t1") ZVT_S_VTZERO(8)
                        "ld t1, 24(%[mrs])\n\t" "beqz t1, 1f\n\t" ZVT_S_MSETTM("%[tmp]", "t1") ZVT_S_VTZERO(12)
                        "1:\n\t"
                        : [tmp] "=&r"(tmp)
                        : [mt] "r"(mtype), [vt] "r"(vtype), [nc] "r"(nc), [mrs] "r"(mrs)
                        : "t1", "memory");
                }

                for (int k0 = 0; k0 < K; k0 += 4) {
                    /* pack 4 strip columns (B) */
                    for (int ki = 0; ki < 4; ki++) {
                        int k = k0 + ki;
                        int8_t *col = strip + ki * MB_ZVT_TMAX;
                        if (k >= K) {
                            zvt_vzero8(col, (int)nc);
                            continue;
                        }
                        int ic = k / (KH * KW), rem = k % (KH * KW);
                        zvt_pack_col(col, in_n + (size_t)ic * IH * IW, rem / KW, rem % KW,
                                     p0, (int)nc, IH, IW, OW, SH, SW, PH, PW);
                        if (has_off && g0 == 0) {
                            for (int c = 0; c < (int)nc; c++) col_sum[c] += col[c];
                        }
                    }
                    /* A column addresses per group, clamped to k <= K-1 */
                    const int8_t *pas[4][4];
                    for (int j = 0; j < 4; j++) {
                        const int8_t *wrow = weight + (size_t)(g0 + j * (int)T) * K;
                        for (int ki = 0; ki < 4; ki++) {
                            int k = k0 + ki < K ? k0 + ki : K - 1;
                            pas[j][ki] = mrs[j] ? wrow + k : weight;
                        }
                    }
                    unsigned long tmp;
                    __asm__ volatile(
                        ZVT_CONV_BASE_CFG
                        ZVT_S_MSETTN("%[tmp]", "%[nc]")
                        "vle8.v v8, (%[b0])\n\t"
                        "vle8.v v10, (%[b1])\n\t"
                        "vle8.v v12, (%[b2])\n\t"
                        "vle8.v v14, (%[b3])\n\t"
                        ZVT_CONV_GROUP(0)
                        ZVT_CONV_GROUP(1)
                        ZVT_CONV_GROUP(2)
                        ZVT_CONV_GROUP(3)
                        : [tmp] "=&r"(tmp)
                        : [mt] "r"(mtype), [vt] "r"(vtype), [nc] "r"(nc),
                          [mrs] "r"(mrs), [pas] "r"(pas), [st] "r"(stride),
                          [b0] "r"(strip), [b1] "r"(strip + MB_ZVT_TMAX),
                          [b2] "r"(strip + 2 * MB_ZVT_TMAX),
                          [b3] "r"(strip + 3 * MB_ZVT_TMAX)
                        : "t1", "t2", "memory",
                          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
                          "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
                }

                /* read out + requantize, row by row */
                for (int j = 0; j < 4; j++) {
                    for (int i = 0; i < (int)mrs[j]; i++) {
                        int oc = g0 + j * (int)T + i;
                        unsigned long tss = ZVT_TSS(4 * j, 0, i), tmp;
                        unsigned long mr = mrs[j];

                        __asm__ volatile(
                            ZVT_CONV_BASE_CFG
                            ZVT_S_MSETTM("%[tmp]", "%[mr]")
                            ZVT_S_MSETTN("%[tmp]", "%[nc]")
                            ZVT_S_VTSE32("%[po]", "%[tss]")
                            : [tmp] "=&r"(tmp)
                            : [mt] "r"(mtype), [vt] "r"(vtype), [nc] "r"(nc),
                              [mr] "r"(mr), [po] "r"(row_out), [tss] "r"(tss)
                            : "memory");

                        int32_t wsum = 0;
                        if (has_off) {
                            const int8_t *wrow = weight + (size_t)oc * K;
                            for (int k = 0; k < K; k++) wsum += wrow[k];
                        }
                        zvt_requant_s8(out_n + (size_t)oc * P + p0, row_out,
                                       has_off ? col_sum : NULL, filter_offset, NULL, 0,
                                       (bias != NULL ? bias[oc] : 0) +
                                           (has_off ? input_offset * wsum +
                                                      K * input_offset * filter_offset : 0),
                                       (int)nc, output_multiplier, output_shift,
                                       output_offset, activation_min, activation_max);
                    }
                }
            }
        }
    }
}
