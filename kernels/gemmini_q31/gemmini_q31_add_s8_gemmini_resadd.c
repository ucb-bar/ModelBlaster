/* source: curated */
/* algorithm: gemmini_resadd */
/* accuracy_class: numeric_drift */
/* origin: tiled_resadd_auto — gemmini's residual-add path. Computes
 *         C[i] = sat_int8(round(A_scale * A[i] + B_scale * B[i]) * C_scale)
 *         with optional ReLU fused into the requantize tail.
 *
 *         Maps the agents add_s8 contract:
 *           output[i] = round((a[i]*scale_a + b[i]*scale_b) / scale_out)
 *           clamp to [activation_min, activation_max]
 *         to gemmini parameters:
 *           A_scale = scale_a / scale_out  (mvin float scale on Q31 build)
 *           B_scale = scale_b / scale_out
 *           C_scale = ACC_SCALE_IDENTITY   (no further mvout scaling)
 *           relu    = (activation_min == 0)
 *
 *         The fused activation path covers the common (activation_min=0,
 *         activation_max=127) and (None, None) cases. For asymmetric
 *         clamps (e.g. activation_min < 0 and != INT8_MIN, or
 *         activation_max < 127), we post-clamp on the CPU side.
 *
 *         Falls back to scalar for very small n where the per-call
 *         gemmini setup (mstatus, gemmini_flush) exceeds a scalar
 *         elementwise pass.
 */
void kernel_add_s8(const int8_t *a, const int8_t *b, int8_t *output, int n,
                   float scale_a, float scale_b, float scale_out,
                   int activation_min, int activation_max)
{
    /* Per-call gemmini setup is ~50–100 cycles. Below ~256 elements the
     * scalar pass beats us. Also fall back when the scales are too
     * asymmetric for gemmini's float-scale mvin path (it rounds the
     * scaled int8 input to int8 before accumulation, so when
     * scale_a/scale_out or scale_b/scale_out is ≪ 1 we lose all
     * precision and the output diverges by tens of LSB — observed on
     * yolov8 residual blocks where the two add branches have very
     * different per-tensor scales). */
    float a_ratio = scale_a / scale_out;
    float b_ratio = scale_b / scale_out;
    float a_abs   = a_ratio < 0 ? -a_ratio : a_ratio;
    float b_abs   = b_ratio < 0 ? -b_ratio : b_ratio;
    /* Keep the gemmini path when both ratios are within ~[0.5, 2.0] of
     * 1.0 — that's the well-conditioned residual-add regime where the
     * mvin int rounding error stays under a single LSB. Outside this
     * range, fall back to scalar (still bit-exact vs the reference). */
    bool scales_ok = (a_abs >= 0.5f && a_abs <= 2.0f
                      && b_abs >= 0.5f && b_abs <= 2.0f);

    if (n <= 0 || n < 256 || !scales_ok) {
        for (int i = 0; i < n; i++) {
            float fa = (float)a[i] * scale_a;
            float fb = (float)b[i] * scale_b;
            float fout = (fa + fb) / scale_out;
            int32_t v = (int32_t)roundf(fout);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i] = (int8_t)v;
        }
        return;
    }

    /* gemmini's fused-relu only applies activation_min=0; any clamp tighter
     * than the int8 range needs a post-pass. We keep relu fusion for the
     * common (0, 127) case and post-clamp the rest. */
    bool fused_relu = (activation_min == 0 && activation_max == 127);
    bool need_post_clamp = !(activation_min == -128 && activation_max == 127)
                            && !fused_relu;

    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    scale_t a_scale = (scale_t)(scale_a / scale_out);
    scale_t b_scale = (scale_t)(scale_b / scale_out);

    /* tiled_resadd_auto's internal tiler shrinks tile_J in DIM-multiples
     * until acc_rows ≤ ACC_ROWS/2 = 512. The autotune is correct for
     * I=1 only up to a regime where the J tile-merge bookkeeping stays
     * coherent — empirically I=1 with very large J (yolov8: n=25600,
     * 12800, 6400, 3200) crashes with mcause=1 mepc=0 (stack corruption
     * from a stray DMA write past the output buffer when tile_J merges
     * across many J-tiles).
     *
     * Workaround: chunk the call into ≤6272-element pieces (the
     * largest size we've validated cleanly on dronet) and issue
     * back-to-back `tiled_resadd_auto`s with a fence between them.
     * Each chunk is independent — the math (per-element) doesn't
     * depend on neighbors. Adds at most ~10 µs per chunk for the
     * gemmini handshake; way under what the equivalent scalar pass
     * would cost. */
    enum { ADD_CHUNK_MAX = 6272 };
    int remaining = n;
    int offset = 0;
    while (remaining > 0) {
        int chunk = remaining > ADD_CHUNK_MAX ? ADD_CHUNK_MAX : remaining;

        gemmini_flush(0);
        asm volatile("fence" ::: "memory");

        tiled_resadd_auto(
            /* I = */ 1, /* J = */ (size_t)chunk,
            a_scale, b_scale, ACC_SCALE_IDENTITY,
            a + offset, b + offset, output + offset,
            /* relu = */ fused_relu,
            WS
        );

        gemmini_fence();
        gemmini_flush(0);

        offset    += chunk;
        remaining -= chunk;
    }

    /* Post-clamp for asymmetric activation ranges that gemmini's fused
     * path can't express. */
    if (need_post_clamp) {
        for (int i = 0; i < n; i++) {
            int v = output[i];
            if (v < activation_min) output[i] = (int8_t)activation_min;
            else if (v > activation_max) output[i] = (int8_t)activation_max;
        }
    }
}
