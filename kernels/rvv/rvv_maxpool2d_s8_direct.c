/* source: curated */
/* algorithm: direct */
/* origin: vectorized RVV maxpool2d_s8. Rewritten 2026-11-04 (k3-work TASK-013):
 *   each output row is split into border columns (window partly outside the
 *   image) and interior columns. Border outputs take the max of the clipped
 *   window in scalar code; interior columns take unit-stride vector loads when
 *   SW == 1 (or SW == 2: contiguous byte loads, even bytes kept by a narrowing
 *   shift) and the run is wide enough to fill a vector, else scalar code too.
 *   The previous version handled every border window by storing the vector
 *   accumulator to the stack, updating it in scalar code and reloading it, and
 *   used strided loads (vlse8) for SW > 1. Both are slow on K3 A100 cores:
 *   DroNet's 3x3 stride-2 maxpool took 160k ticks there vs 10.5k on an X100,
 *   and YOLOv8-nano's SPPF 5x5 pools (all border at 5x5 and 2x3) 30k / 5k.
 *   Same result: max over the in-bounds window taps, INT8_MIN for an
 *   all-padding window. */

/* Max of the window taps at rows ih0 + kh*DH, cols iw0 + kw*DW that lie inside.
 * Kept scalar: GCC's auto-vectorized form of these short loops (vl of a few bytes)
 * is slower than scalar code on both K3 core types. */
static inline __attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize"))) int8_t mp2_window(const int8_t *in_nc, int IH, int IW, int ih0, int iw0,
                                int KH, int KW, int DH, int DW)
{
    int kh0 = ih0 < 0 ? (-ih0 + DH - 1) / DH : 0;
    int kh1 = IH - ih0 > 0 ? (IH - ih0 + DH - 1) / DH : 0;
    int kw0 = iw0 < 0 ? (-iw0 + DW - 1) / DW : 0;
    int kw1 = IW - iw0 > 0 ? (IW - iw0 + DW - 1) / DW : 0;
    if (kh1 > KH) kh1 = KH;
    if (kw1 > KW) kw1 = KW;
    int m = -128;
    for (int kh = kh0; kh < kh1; kh++) {
        const int8_t *row = in_nc + (size_t)(ih0 + kh * DH) * IW + iw0;
        for (int kw = kw0; kw < kw1; kw++) {
            int v = row[kw * DW];
            m = v > m ? v : m;
        }
    }
    return (int8_t)m;
}

__attribute__((optimize("no-tree-vectorize", "no-tree-slp-vectorize")))
void kernel_maxpool2d_s8(const int8_t *input, int8_t *output,
                         int N, int C, int IH, int IW,
                         int KH, int KW, int SH, int SW,
                         int PH, int PW, int DH, int DW) {
    int OH = (IH + 2*PH - DH*(KH-1) - 1) / SH + 1;
    int OW = (IW + 2*PW - DW*(KW-1) - 1) / SW + 1;
    if (OH <= 0 || OW <= 0) return;

    /* Interior columns [ow_lo, ow_hi): every kw tap of the window is inside the row. */
    int ow_lo = (PW + SW - 1) / SW;
    /* largest ow*SW with the last tap inside; for SW == 2 the vector path also reads
     * the byte after each tap, so that one must be inside too */
    int last = IW - (SW == 2 ? 2 : 1) + PW - DW * (KW - 1);
    int ow_hi = last >= 0 ? last / SW + 1 : 0;
    if (ow_lo > OW) ow_lo = OW;
    if (ow_hi > OW) ow_hi = OW;
    if (ow_hi < ow_lo) ow_hi = ow_lo;
    /* SW == 2: each tap loads 2*vl contiguous bytes, viewed as 16-bit lanes, and keeps
     * the low (even) byte of each by a narrowing shift, instead of a strided load. */
    int vec = (SW == 1 || SW == 2) && ow_hi - ow_lo >= 16;

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            const int8_t *in_nc = input + ((size_t)n*C + c)*IH*IW;
            int8_t *out_nc = output + ((size_t)n*C + c)*OH*OW;

            for (int oh = 0; oh < OH; oh++) {
                int ih0 = oh*SH - PH;
                int8_t *orow = out_nc + (size_t)oh*OW;
                if (!vec) {
                    for (int ow = 0; ow < OW; ow++)
                        orow[ow] = mp2_window(in_nc, IH, IW, ih0, ow*SW - PW, KH, KW, DH, DW);
                    continue;
                }
                for (int ow = 0; ow < ow_lo; ow++)
                    orow[ow] = mp2_window(in_nc, IH, IW, ih0, ow*SW - PW, KH, KW, DH, DW);
                size_t vl;
                for (int ow = ow_lo; ow < ow_hi; ow += (int)vl) {
                    vl = __riscv_vsetvl_e8m4((size_t)(ow_hi - ow));   /* 2*vl bytes fit an e8m8 load */
                    vint8m4_t vacc = __riscv_vmv_v_x_i8m4((int8_t)(-128), vl);
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = ih0 + kh*DH;
                        if (ih < 0 || ih >= IH) continue;
                        const int8_t *p = in_nc + (size_t)ih*IW + (size_t)ow*SW - PW;
                        for (int kw = 0; kw < KW; kw++) {
                            vint8m4_t v;
                            if (SW == 1) v = __riscv_vle8_v_i8m4(p + kw*DW, vl);
                            else v = __riscv_vncvt_x_x_w_i8m4(__riscv_vreinterpret_v_i8m8_i16m8(
                                         __riscv_vle8_v_i8m8(p + kw*DW, 2 * vl)), vl);
                            vacc = __riscv_vmax_vv_i8m4(vacc, v, vl);
                        }
                    }
                    __riscv_vse8_v_i8m4(orow + ow, vacc, vl);
                }
                for (int ow = ow_hi; ow < OW; ow++)
                    orow[ow] = mp2_window(in_nc, IH, IW, ih0, ow*SW - PW, KH, KW, DH, DW);
            }
        }
    }
}
