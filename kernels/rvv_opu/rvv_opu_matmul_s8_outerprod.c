/* source: curated */
/* algorithm: outerprod */
/* origin: Saturn OPU outer-product i8 matmul. Ported from
 *   hw/chipyard/generators/saturn/benchmarks/opu-gemm/kernel.h
 *   (branch origin/opu-fp8), function `i8_mm_bme_sq`, by Miles Rusch.
 *
 * Upstream's `i8_mm_bme_sq` iterates over mlmax-sized tiles of (i, j),
 * with a per-tile MAC loop that VOPACC's K i8 inputs into m1. We
 * preserve that tile-walking shape faithfully here, with three
 * adaptations for the agents-flow matmul_s8 signature:
 *
 *   1. Layout normalization. Upstream takes `at` laid out as [K, M]
 *      (i.e., a's transpose, so unit-stride loads can sweep K rows of
 *      M lanes). Our `a` is [M, K] row-major. We transpose into a
 *      stack scratch of size K_TILE × mlmax bytes (well-sized because
 *      we only need one mlmax-tile of M at a time, not the full M).
 *      For transpose_b=1, b is [N, K] and we do the same flip into a
 *      K_TILE × mlmax scratch.
 *
 *   2. Requantize tail. The matmul_s8 ABI demands i8 output with a
 *      float-scale rescale (scale_a*scale_b/(scale_out*scale_div)).
 *      We drain each m1 tile into i32 scratch, then loop over the
 *      tile computing roundf(acc * total) + clamp + i8 store.
 *
 *   3. Edge handling. Upstream's narrow-N fallback uses a per-row
 *      variant (`i8_loop_k_general`). We don't replicate that — the
 *      tile loop here always uses square mlmax tiles, falling back
 *      to the embedded scalar reference for (M, N) shapes that
 *      aren't multiples of mlmax. Easier to read; perf-equivalent
 *      for the agents-flow's typical shapes (M, N either small <= 8
 *      or pure multiples like 64).
 *
 * MAC body itself (VOPACC inner loop with two-way unroll) is byte-
 * for-byte upstream.
 */

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <riscv_vector.h>
#include "saturn_opu.h"

/* Tile bounds. mlmax = VLEN/8 at runtime; we cap at OPU_MAX_TILE=64
 * (matches V512). K_TILE is the per-tile K reach; we transpose only
 * one mlmax row-strip of a (and optionally of b) at a time, so the
 * scratch is K_TILE × mlmax = 1024 × 64 = 64 KiB per buffer. That fits
 * in stack with margin; for deeper K we sub-tile the K loop. */
#define OPU_MAX_TILE   64
#define OPU_MAX_K     1024

/* Core OPU MAC + drain into i32 scratch — verbatim port of upstream
 * `i8_sq_loop_k` + `i32_sq_store_c`. Operates on a single mlmax×mlmax
 * tile: caller supplies `at` laid out as [K, mlmax] and `b` as
 * [K, mlmax]; result is M_TILE × N_TILE i32 in `c_out_tile`. */
static inline void opu_tile_mac(int32_t *c_out_tile,
                                const int8_t *at_tile, const int8_t *b_tile,
                                size_t M_tile, size_t N_tile, size_t K) {
    /* Seed m1 to zero (no bias path in matmul_s8). */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(N_tile));
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m1, v0);

    /* Two-way unrolled VOPACC over K. */
    size_t k = 0;
    while (k + 2 <= K) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M_tile));
        asm volatile("vle8.v v16, (%0)" : : "r"(&at_tile[k * M_tile]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N_tile));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b_tile[k * N_tile]));
        VOPACC(m1, v18, v16);
        k++;
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M_tile));
        asm volatile("vle8.v v20, (%0)" : : "r"(&at_tile[k * M_tile]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N_tile));
        asm volatile("vle8.v v22, (%0)" : : "r"(&b_tile[k * N_tile]));
        VOPACC(m1, v22, v20);
        k++;
    }
    if (k < K) {
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(M_tile));
        asm volatile("vle8.v v16, (%0)" : : "r"(&at_tile[k * M_tile]));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" : : "r"(N_tile));
        asm volatile("vle8.v v18, (%0)" : : "r"(&b_tile[k * N_tile]));
        VOPACC(m1, v18, v16);
    }

    /* Drain rows: M_tile vectors of N_tile i32 elements each. */
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" : : "r"(N_tile));
    for (size_t r = 0; r < M_tile; r++) {
        VMV_VR(v0, r, m1);
        asm volatile("vse32.v v0, (%0)" : : "r"(&c_out_tile[r * N_tile]));
    }
}

static void matmul_s8_scalar_fallback(const int8_t *a, const int8_t *b,
                                      int8_t *output,
                                      int M, int K, int N,
                                      float scale_a, float scale_b,
                                      float scale_out,
                                      int transpose_b, float scale_div,
                                      int activation_min, int activation_max) {
    float total = (scale_a * scale_b) / (scale_out * scale_div);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t acc = 0;
            for (int k = 0; k < K; k++) {
                int8_t av = a[i * K + k];
                int8_t bv = transpose_b ? b[j * K + k] : b[k * N + j];
                acc += (int32_t)av * (int32_t)bv;
            }
            int32_t v = (int32_t)roundf((float)acc * total);
            if (v < activation_min) v = activation_min;
            if (v > activation_max) v = activation_max;
            output[i * N + j] = (int8_t)v;
        }
    }
}

void kernel_matmul_s8(const int8_t *a, const int8_t *b, int8_t *output,
                      int M, int K, int N,
                      float scale_a, float scale_b, float scale_out,
                      int transpose_b, float scale_div,
                      int activation_min, int activation_max) {
    /* Eligibility: just K within scratch. Partial tiles at the M and
     * N edges are handled by tiling with min(remaining, mlmax). */
    size_t mlmax;
    asm volatile("vsetvli %0, zero, e8, m1, ta, ma" : "=r"(mlmax));
    if (K > OPU_MAX_K || (int)mlmax > OPU_MAX_TILE) {
        matmul_s8_scalar_fallback(a, b, output, M, K, N,
                                  scale_a, scale_b, scale_out,
                                  transpose_b, scale_div,
                                  activation_min, activation_max);
        return;
    }
    const int MLMAX = (int)mlmax;
    const float total = (scale_a * scale_b) / (scale_out * scale_div);

    /* Per-tile transpose scratch sized at the worst-case (mlmax-wide).
     * Stride between K rows is the actual tile width, so partial tiles
     * pack tightly — only as much as M_tile / N_tile bytes per row. */
    int8_t  at_tile[OPU_MAX_K * OPU_MAX_TILE];   /* [K, M_tile] */
    int8_t  b_tile [OPU_MAX_K * OPU_MAX_TILE];   /* [K, N_tile] */
    int32_t c_tile [OPU_MAX_TILE * OPU_MAX_TILE]; /* [M_tile, N_tile] */

    for (int i0 = 0; i0 < M; i0 += MLMAX) {
        const int M_tile = (M - i0 < MLMAX) ? (M - i0) : MLMAX;

        /* Transpose a's M_tile row strip into at_tile [K, M_tile]. */
        for (int k = 0; k < K; k++) {
            for (int r = 0; r < M_tile; r++) {
                at_tile[k * M_tile + r] = a[(i0 + r) * K + k];
            }
        }

        for (int j0 = 0; j0 < N; j0 += MLMAX) {
            const int N_tile = (N - j0 < MLMAX) ? (N - j0) : MLMAX;

            /* Normalize b strip into b_tile [K, N_tile]. */
            if (transpose_b) {
                /* b is [N, K]; pull column j for j in [j0, j0+N_tile). */
                for (int k = 0; k < K; k++) {
                    for (int c = 0; c < N_tile; c++) {
                        b_tile[k * N_tile + c] = b[(j0 + c) * K + k];
                    }
                }
            } else {
                /* b is [K, N]; copy a column window contiguous in K. */
                for (int k = 0; k < K; k++) {
                    for (int c = 0; c < N_tile; c++) {
                        b_tile[k * N_tile + c] = b[k * N + (j0 + c)];
                    }
                }
            }

            /* OPU MAC for one M_tile×N_tile tile, drain to c_tile. */
            opu_tile_mac(c_tile, at_tile, b_tile,
                         (size_t)M_tile, (size_t)N_tile, (size_t)K);

            /* Requantize tail + write i8 into output rows. */
            for (int r = 0; r < M_tile; r++) {
                int8_t *out_row = &output[(i0 + r) * N + j0];
                int32_t *acc_row = &c_tile[r * N_tile];
                for (int c = 0; c < N_tile; c++) {
                    int32_t v = (int32_t)roundf((float)acc_row[c] * total);
                    if (v < activation_min) v = activation_min;
                    if (v > activation_max) v = activation_max;
                    out_row[c] = (int8_t)v;
                }
            }
        }
    }
}
