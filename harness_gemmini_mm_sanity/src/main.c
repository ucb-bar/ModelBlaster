/*
 * Minimal standalone gemmini int32-accumulate sanity test for hart1 of the
 * real riskybird v3 KU040 board (RocketKU040DroneDualConfig,
 * Q31Ws32x32AccGemminiConfig, DIM=32). NOT DroNet -- isolates whether the
 * gemmini systolic array's raw int32 MAC accumulate is correct on this
 * silicon at all, independent of any Q0.31 requant/mvout-scale path
 * (variant A's requant is deterministic and was already spike-bit-exact,
 * so a real-board mismatch there means the ACCUMULATOR itself is wrong).
 *
 * Uses tiled_matmul_auto(..., full_C=true, ...) -- the same high-level
 * gemmini entry point modelblaster's linear_s8/conv2d_s8 bit-exact
 * kernels already call (kernels/gemmini_q31/gemmini_q31_linear_s8_
 * gemmini_tiled_matmul.c) -- with full_C=true so gemmini drains the RAW
 * int32 accumulator with NO scale/round/clamp anywhere in the path. That
 * raw int32 C is compared element-by-element against a CPU int32
 * reference of the SAME known int8 A/B, computed with the identical
 * transpose_B=true convention (weight physically [N,K], gemmini reads it
 * as logical [K,N]).
 *
 * Two sizes, one ELF:
 *   (a) single-tile 32x32x32  (== DIM) -- no gemmini-side tiling/
 *       double-buffering at all; a fail here means basic compute/config
 *       is wrong (DIM/sp/acc/dataflow/mvin-scale mismatch vs the real
 *       bitstream).
 *   (b) multi-tile 128x128x128 (4x4x4 tiles of 32) -- forces the tiled_
 *       matmul_auto scratchpad + double-buffering path (the same class
 *       of path spike's "LOOP_WS bounds too large" aborted on post-BN-
 *       bake). A fail here with (a) passing points at tiling/scratchpad
 *       overflow specifically, not basic array compute.
 *
 * Console line format (one per test):
 *   GEMMINI_MM_INT32 size=<M>x<K>x<N> max_abs_err=<n> PASS|FAIL
 * followed by a handful of "  sample[i]: gemmini=<v> cpu=<v>" lines.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "gemmini.h"

/* ---- test A: 32x32x32, single tile (== DIM) ---------------------------- */
#define MA 32
#define KA 32
#define NA 32
static int8_t  A_a[MA][KA]  __attribute__((aligned(64)));
static int8_t  Bt_a[NA][KA] __attribute__((aligned(64))); /* physical [N,K] */
static int32_t C_a[MA][NA]  __attribute__((aligned(64)));
static int32_t ref_a[MA][NA];

/* ---- test B: 128x128x128, multi-tile (4x4x4 tiles of 32) --------------- */
#define MB 128
#define KB 128
#define NB 128
static int8_t  A_b[MB][KB]  __attribute__((aligned(64)));
static int8_t  Bt_b[NB][KB] __attribute__((aligned(64)));
static int32_t C_b[MB][NB]  __attribute__((aligned(64)));
static int32_t ref_b[MB][NB];

static void fill(int8_t *A, int M, int K, int8_t *Bt, int N) {
    for (int m = 0; m < M; m++)
        for (int k = 0; k < K; k++)
            A[m * K + k] = (int8_t)(((m * 3 + k * 5) % 15) - 7);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
            Bt[n * K + k] = (int8_t)(((n * 7 + k * 2) % 11) - 5);
}

static void cpu_ref(const int8_t *A, const int8_t *Bt, int32_t *ref,
                     int M, int K, int N) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)A[m * K + k] * (int32_t)Bt[n * K + k];
            }
            ref[m * N + n] = acc;
        }
    }
}

static void run_test(const char *label, int8_t *A, int8_t *Bt, int32_t *C,
                      int32_t *ref, int M, int K, int N) {
    fill(A, M, K, Bt, N);
    cpu_ref(A, Bt, ref, M, K, N);

    /* Enable mstatus.XS=Dirty so RoCC custom-3 instructions don't trap
     * (same as the modelblaster gemmini kernels do before their first
     * gemmini call). */
    asm volatile("csrs mstatus, %0" : : "r"(0x18000) : "memory");

    gemmini_flush(0);
    asm volatile("fence" ::: "memory");

    /* full_C=true: raw int32 accumulator out, no scale/round/clamp.
     * D=NULL (no bias). transpose_A=false, transpose_B=true (Bt is
     * physically [N,K], gemmini reads it as logical [K,N]) -- same
     * convention as gemmini_q31_linear_s8_gemmini_tiled_matmul.c. */
    tiled_matmul_auto(
        (size_t)M, (size_t)N, (size_t)K,
        A, Bt,
        NULL, (void *)C,
        (size_t)K, (size_t)K, (size_t)N, (size_t)N,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, (scale_acc_t)1,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, (acc_scale_t)0,
        false,
        false, true,
        true, false,
        0, WS
    );

    gemmini_fence();
    gemmini_flush(0);

    int32_t max_abs_err = 0;
    int first_bad = -1;
    for (int i = 0; i < M * N; i++) {
        int32_t d = C[i] - ref[i];
        if (d < 0) d = -d;
        if (d > max_abs_err) max_abs_err = d;
        if (d != 0 && first_bad < 0) first_bad = i;
    }

    printf("GEMMINI_MM_INT32 size=%dx%dx%d max_abs_err=%d %s\n",
           M, K, N, max_abs_err, max_abs_err == 0 ? "PASS" : "FAIL");
    for (int i = 0; i < 4 && i < M * N; i++) {
        printf("  sample[%d]: gemmini=%d cpu=%d\n", i, C[i], ref[i]);
    }
    if (first_bad >= 0) {
        printf("  first_mismatch[%d]: gemmini=%d cpu=%d\n",
               first_bad, C[first_bad], ref[first_bad]);
    }
    (void)label;
}

int main(void) {
    printf("gemmini_mm_sanity: DIM=%d BANK_ROWS=%d\n", DIM, BANK_ROWS);

    run_test("single-tile", (int8_t *)A_a, (int8_t *)Bt_a, (int32_t *)C_a,
              (int32_t *)ref_a, MA, KA, NA);
    run_test("multi-tile", (int8_t *)A_b, (int8_t *)Bt_b, (int32_t *)C_b,
              (int32_t *)ref_b, MB, KB, NB);

    printf("gemmini_mm_sanity: done\n");
#ifndef CONFIG_ARCH_POSIX
    sys_reboot(SYS_REBOOT_COLD);
#endif
    return 0;
}
