/* Load-once Gemmini conv2d+maxpool kernel (bit-exact reorder of tiled_conv_auto).
 *
 * WHY: conv0 (grayscale IC=1, 112^2 -> 56^2, OC=32, 3x3 s2, fused 3x3-s2 maxpool)
 * is LOAD-DMA-bound: loop_conv_ws / tiled_conv re-mvins each output tile's input
 * band, so the 12.5 KB input is re-fetched ~13x (overlapping 3x3-s2 halos). The
 * array GEMM (K=IC=1) is trivial (EX ~= 2 cyc); ~79% of the ~997K conv0 cost is
 * the input mvin.
 *
 * FIX: the whole conv0 input (padded 114x114x1 = ~13K scratchpad rows) fits the
 * 256 KB scratchpad, and the 3x3x1x32 weights are tiny. So mvin the ENTIRE padded
 * input ONCE and the weights ONCE, leave both resident, then loop over
 * accumulator-bounded output bands doing bias-mvin -> preload/compute (reading A
 * straight out of the RESIDENT global input at the right offset, never re-mvin) ->
 * HW pooled mvout (config_st pool tail, exactly as LoopConvSt does it). The input
 * DMA collapses from ~13x reload to a single load.
 *
 * BIT-EXACT: this only reorders DMA, not arithmetic. Each output pixel gets the
 * identical int32 accumulate (bias + full 3x3xIC sum), and the requant + max-pool
 * are applied by the SAME HW config_st(scale,act,pool)+mvout the fused path uses.
 * So the result equals tiled_conv_auto+pool (max_abs_err 0, within the numeric_drift
 * fast-conv envelope of <=3 vs any alternate reference).
 *
 * CONSTRAINTS (the caller's fast-path gate must already guarantee these):
 *   - N == 1, square conv kernel/stride, symmetric conv pad, IC <= DIM.
 *   - zero input/filter/output offsets, foldable Q0.31 shift (Q31 acc-scale build).
 *   - square pool, zero pool pad, unit pool dilation (HW pool fill is 0).
 *   - NHWC activations: input [IH,IW,IC], output [OHp,OWp,OC]; weights HWIO.
 *
 * Layout of the resident input (A scratchpad, base 0):
 *   one padded input pixel per spad row, IC channels in the columns.
 *   a_addr(padded_row, padded_col) = padded_row * (IW+2P) + padded_col.
 *   This is exactly tiled_conv/sp_tiled_conv's A layout with icols = full padded
 *   width, so the compute A_stride = conv stride reads consecutive output columns
 *   S apart, and each 3x3 tap indexes a_addr = gpir*(IW+2P) + gpic directly.
 */
#ifndef MB_CONV2D_POOL_LOADONCE_H
#define MB_CONV2D_POOL_LOADONCE_H

#include <stdint.h>
#include <stddef.h>
#include <gemmini.h>
#include <gemmini_params.h>

/* acc address bases (match sp_tiled_conv / sp_tiled_matmul):
 *   accumulate read/write  -> is_acc(bit31) | accumulate(bit30)
 *   overwrite write (bias) -> is_acc(bit31) only                         */
#define MB_LO_ACC_ACCUM      ((uint32_t)(3u << (ADDR_LEN - 2)))   /* 0xC0000000 */
#define MB_LO_ACC_OVERWRITE  ((uint32_t)(1u << (ADDR_LEN - 1)))   /* 0x80000000 */

/* Returns 0 on success (kernel ran), nonzero if the shape is outside the
 * fast-path this kernel supports (caller should fall back). */
static int mb_conv2d_pool_loadonce_s8(
        const int8_t *input, const int8_t *weight, const acc_t *bias,
        int8_t *output,
        int IC, int IH, int IW, int OC,
        int K, int S, int P,
        int pool_size, int pool_stride,
        int act, acc_scale_t scale)
{
    if (IC > DIM) return 1;                 /* one kch block only */

    const int OH  = (IH + 2 * P - K) / S + 1;
    const int OW  = (IW + 2 * P - K) / S + 1;
    const int OHp = (OH - pool_size) / pool_stride + 1;   /* pool pad 0 */
    const int OWp = (OW - pool_size) / pool_stride + 1;

    const int full_ih = IH + 2 * P;         /* padded input rows */
    const int full_iw = IW + 2 * P;         /* padded input cols == A row stride */

    /* full band conv-col extent the pool needs (full pooled width) */
    const int ocols_full = (OWp - 1) * pool_stride + pool_size;

    /* weight scratchpad region (top of spad), HWIO layout, one kch block */
    const int out_channels_per_bank = OC / DIM + (OC % DIM != 0);
    const int B_rows = out_channels_per_bank * K * K * IC;
    const uint32_t B_sp_start = (uint32_t)(BANK_NUM * BANK_ROWS) - (uint32_t)B_rows;

    /* capacity guard: resident input + weights must fit the scratchpad */
    if ((size_t)full_ih * full_iw + B_rows > (size_t)(BANK_NUM * BANK_ROWS))
        return 2;
    if ((size_t)ocols_full * pool_size > (size_t)ACC_ROWS)   /* need >=1 pooled row/band */
        return 3;

    /* First-layer pixel packing: fold up to (DIM/IC, clamped to K) kernel columns
     * into one compute's K dim. The mvin PixelRepeater must be configured with the
     * SAME pixel_repeats for the packed read to line up, so decide it up front. */
    int mpr = DIM / IC;
    if (mpr > K) mpr = K;
    if (mpr < 1) mpr = 1;

    /* ---------- 1. Load the whole padded input ONCE (A base 0) ----------
     * Port of sp_tiled_conv's input mvin with the "tile" == the entire image
     * (upad=dpad=lpad=rpad = P). is_zeros -> dram addr 0 => HW zero-fills the
     * padding border in the scratchpad, exactly as LoopConvLdA does.          */
    gemmini_extended5_config_ld((uint64_t)IC * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, full_ih * full_iw, mpr, 0);
    for (int irow = -P; irow < IH + P; irow++) {
        const int irow_padded = irow + P;
        for (int icol = -P; icol < IW + P; ) {
            int I;
            if (icol < 0)            I = (-icol > DIM ? DIM : -icol);
            else if (icol >= IW)     I = (IW + P - icol > DIM ? DIM : IW + P - icol);
            else                     I = (IW - icol > DIM ? DIM : IW - icol);
            const int icol_padded = icol + P;
            const uint32_t A_sp = (uint32_t)irow_padded * full_iw + icol_padded;
            const int is_zeros = (irow < 0 || irow >= IH || icol < 0 || icol >= IW);
            const elem_t *in = is_zeros ? NULL
                             : input + ((size_t)irow * IW + icol) * IC;
            gemmini_extended_mvin(in, A_sp, IC, I);
            icol += I;
        }
    }

    /* ---------- 2. Load the weights ONCE (B region) ----------
     * HWIO [K*K*IC, OC]; spad layout matches the compute's B_sp_addr formula. */
    gemmini_extended4_config_ld((uint64_t)OC * sizeof(elem_t), MVIN_SCALE_IDENTITY,
                                false, K * K * IC, 1);
    for (int och = 0; och < OC; och += DIM) {
        const int J = OC - och > DIM ? DIM : OC - och;
        for (int krow = 0; krow < K; krow++)
            for (int kcol = 0; kcol < K; kcol++) {
                const uint32_t B_sp = B_sp_start
                    + (uint32_t)(och / DIM) * K * K * IC + krow * K * IC + kcol * IC;
                const elem_t *w = weight + ((size_t)krow * K * IC + kcol * IC) * OC + och;
                gemmini_extended_mvin2(w, B_sp, J, IC);
            }
    }

    /* ---------- 3. Compute config: WS, A_stride = conv stride ----------
     * consecutive output columns are S input columns apart in the resident A.
     * First-layer packing: fold up to (DIM/IC, clamped to K) kernel columns into
     * one compute's K dimension (they are contiguous input columns in A), cutting
     * the array-dispatch count ~K-fold.  EX only, no effect on the result.       */
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, S, 0, 0, false);

    /* pooled rows per band, bounded by the accumulator */
    int Pp_max = 1;
    while (1) {
        const int orows_try = ((Pp_max + 1) - 1) * pool_stride + pool_size;
        if ((size_t)orows_try * ocols_full > (size_t)ACC_ROWS) break;
        Pp_max++;
    }

    const int no_bias = (bias == NULL);

    /* ---------- 4. Loop output bands x och-blocks; no input reload ---------- */
    for (int P0 = 0; P0 < OHp; P0 += Pp_max) {
        const int Pp     = OHp - P0 > Pp_max ? Pp_max : OHp - P0;
        const int orows_ = (Pp - 1) * pool_stride + pool_size;   /* conv rows in band */
        const int ocols_ = ocols_full;                            /* conv cols (full width) */
        const int gr_lo  = P0 * pool_stride;                      /* first global conv row */

        for (int och = 0; och < OC; och += DIM) {
            const int J = OC - och > DIM ? DIM : OC - och;

            /* 4a. bias -> accumulator (overwrite), replicated over all spatial.
             *     stride 0 => same bias[och..] vector for every acc row.        */
            if (!no_bias) {
                gemmini_extended4_config_ld(0, MVIN_SCALE_IDENTITY, false,
                                            orows_ * ocols_, 2);
                for (int orow = 0; orow < orows_; orow++)
                    for (int ocol = 0; ocol < ocols_; ocol += DIM) {
                        const int I = ocols_ - ocol > DIM ? DIM : ocols_ - ocol;
                        const uint32_t D_sp = MB_LO_ACC_OVERWRITE
                            + (uint32_t)orow * ocols_ + ocol;
                        gemmini_extended_mvin3(bias + och, D_sp, J, I);
                    }
            }

            /* 4b. matmul: accumulate all K*K taps onto the (biased) acc.
             *     A is read from the RESIDENT global input, never re-mvin'd.
             *     kcol is packed mpr-at-a-time into the K contraction dim.        */
            for (int krow = 0; krow < K; krow++)
                for (int kcol = 0; kcol < K; kcol += mpr) {
                    const int pixels = K - kcol > mpr ? mpr : K - kcol;
                    const int Kc = pixels * IC;                 /* contraction width */
                    int new_weights = 1;
                    for (int orow = 0; orow < orows_; orow++) {
                        const int gr = gr_lo + orow;               /* global conv row */
                        for (int ocol = 0; ocol < ocols_; ) {
                            const int I = ocols_ - ocol > DIM ? DIM : ocols_ - ocol;
                            const int gpir = gr * S + krow;        /* padded input row */
                            const int gpic = ocol * S + kcol;      /* padded input col */
                            const uint32_t A_sp = (uint32_t)gpir * full_iw + gpic;
                            const uint32_t B_sp = B_sp_start
                                + (uint32_t)(och / DIM) * K * K * IC
                                + krow * K * IC + kcol * IC;
                            uint32_t C_sp = MB_LO_ACC_ACCUM
                                + (uint32_t)orow * ocols_ + ocol;
                            /* First tap when there is no bias must OVERWRITE the
                             * accumulator instead of adding to stale contents. */
                            if (no_bias && krow == 0 && kcol == 0)
                                C_sp &= ~(uint32_t)(1u << (ADDR_LEN - 2));

                            const uint32_t pre = new_weights ? B_sp : GARBAGE_ADDR;
                            gemmini_extended_preload(pre, C_sp, J, Kc, J, I);
                            if (new_weights) {
                                gemmini_extended_compute_preloaded(A_sp, GARBAGE_ADDR, Kc, I, J, I);
                            } else {
                                gemmini_extended_compute_accumulated(A_sp, GARBAGE_ADDR, Kc, I, J, I);
                            }
                            new_weights = 0;
                            ocol += I;
                        }
                    }
                }

            /* 4c. HW pooled mvout of this band/och-block (config_st pool tail).
             *     Reads the resident acc, applies max-pool + Q0.31 requant+act,
             *     writes the pooled [Pp x OWp x J] block into the NHWC output.   */
            gemmini_extended2_config_st(
                (uint64_t)OC * sizeof(elem_t), act, scale,
                pool_stride, pool_size, OWp,
                Pp, OWp, orows_, ocols_,
                0, 0);
            int8_t *pool_dram = output + ((size_t)P0 * OWp) * OC + och;
            gemmini_extended_mvout(pool_dram, MB_LO_ACC_ACCUM, J, 0);

            gemmini_fence();   /* serialize acc reuse across tiles */
        }
    }

    return 0;
}

#endif /* MB_CONV2D_POOL_LOADONCE_H */
