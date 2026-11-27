/* Optimised K3 kernels of the IR runtime.
 *
 * int8 GEMM (linear_dq_s8 / linear3_dq_s8 / linear_dq_add_s8): SpacemiT IME vmadot, C += A . B^T,
 *   tile TM x TK x TN = 8x16x8 on the A100 (vlenb 128), 4x8x4 on the X100 (vlenb 32); the tile shape
 *   follows vlenb at run time, so one code path serves both clusters. 2x4 register blocking (8
 *   accumulator pairs, 6 operand loads per 8 vmadots; the A100 is load-bound, see
 *   artifacts/k3_probes/ime/summary.md). Weights are packed once at load into
 *   [n-block of 4 tiles][k-slab][4][TN x TK]; activations per call into [m-block][k-slab][2][TM x TK].
 *   The int32 accumulators are exact, and the epilogue performs the reference's arithmetic in the
 *   same order and precision: s = (double)row_scale * (double)w_scale (exact), y = (float)(acc * s)
 *   (one RNE rounding), y + bias (fp32), addend + y (fp32). So the result is byte-identical to the
 *   KernelSpec reference C, which the runtime's --check mode verifies op by op.
 * quantize_rows_s8: vector x * col_inv, vfredmax, the scalar scale and reciprocal, vfcvt with
 *   round-to-nearest-max-magnitude (= roundf), clamp; byte-identical to the reference.
 * fp32 kinds: RVV (vector-length agnostic) GEMM for linear / bmm / conv (im2col), row kernels for
 *   softmax / layer_norm / rms_norm, elementwise exp-based kernels (silu_mul, gelu_tanh). These are
 *   held to the fp32 per-slot bound, not to bit-exactness.
 * Every kernel splits its work over the pool (k3_parallel). */
#include <math.h>
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3ir.h"

#define IN(i) k3_ptr(g, op->in[i])
#define OUT(i) k3_ptr(g, op->out[i])
#define P(i) (op->ip[i])

#ifndef K3_NO_IME
__asm__(".macro k3_vmadot_s8 vd, vs1, vs2\n"
        " .word (0x2b | (\\vd<<7) | (3<<12) | (\\vs1<<15) | (\\vs2<<20) | (0x71<<25))\n"
        ".endm\n"
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
        " .set v\\n, \\n\n"
        ".endr\n");
#define VMADOT(c, a, b) __asm__ volatile("k3_vmadot_s8 %0, %1, %2" : "+vr"(c) : "vr"(a), "vr"(b))
#endif

/* tile geometry follows the executing thread's cluster (vlenb): thread-local, so two pools on the two
 * clusters can run in one process */
static __thread size_t g_vlenb;
static __thread int TM, TN, TK;
extern int k3_vstore, k3_bpf, k3_mtail;               /* IME tile */

#define NBT 4  /* n-tiles per block */
#define MBT 2  /* m-tiles per block */
static __thread uint16_t g_perm[256];
int k3_vperm = -1;   /* -1: auto (X100 only; vrgather is slow on the A100) */

static void init_tiles(void)
{
    if (g_vlenb) return;
    g_vlenb = __riscv_vlenb();
    TM = TN = (int)(g_vlenb / 16);   /* 8 (A100) / 2... see below */
    if (g_vlenb == 128) { TM = 8; TN = 8; TK = 16; }
    else if (g_vlenb == 32) { TM = 4; TN = 4; TK = 8; }
    else { TM = TN = TK = 0; }
    if (TM) {                    /* [row][tile][pair] <- [tile][row][pair] over one m-tile's NBT tiles */
        const int PR = TN / 2, np = NBT * TM * PR;
        for (int t = 0; t < np; t++) {
            const int r = t / (NBT * PR), q = t % (NBT * PR), j = q / PR, pp = q % PR;
            g_perm[t] = (uint16_t)(j * TM * PR + r * PR + pp);
        }
    }
}

static void *xalloc(size_t n)
{
    void *p = aligned_alloc(64, (n + 127) & ~(size_t)63);
    memset(p, 0, (n + 127) & ~(size_t)63);
    return p;
}

/* per-thread grow-only scratch */
static __thread void *scr_p[4];
static __thread size_t scr_n[4];
static void *scratch(int i, size_t n)
{
    if (scr_n[i] < n) {
        free(scr_p[i]);
        scr_n[i] = (n + 4095) & ~(size_t)4095;
        scr_p[i] = aligned_alloc(64, scr_n[i]);
    }
    return scr_p[i];
}

/* shared (per-op-call) scratch: packed A, row-scale patterns */
static void *sh_p[4];
static size_t sh_n[4];
static void *shared_scratch(int i, size_t n)
{
    if (sh_n[i] < n) {
        free(sh_p[i]);
        sh_n[i] = (n + 4095) & ~(size_t)4095;
        sh_p[i] = aligned_alloc(64, sh_n[i]);
    }
    return sh_p[i];
}

/* Run small ops on the calling thread: a pool dispatch + barrier costs ~10 us on the A100s, more
 * than a 50 x 720 row op takes. `work` ~ elements touched. */
int k3_par_min = 32768;
static void par(int64_t n, int64_t work, k3_range_fn fn, void *ctx)
{
    const int nw = k3_pool_size();
    if (nw == 1 || work < k3_par_min || n < 2) {
        fn(ctx, 0, n, 0);
        return;
    }
    int64_t grain = (n + 4 * nw - 1) / (4 * nw);
    k3_parallel(n, grain < 1 ? 1 : grain, fn, ctx);
}

/* ======================================================================== int8 IME GEMM */

typedef struct {
    int8_t *bp;            /* packed weights */
    double *wsp;           /* [nblocks*NBT][TM*TN] w_scale pattern (col = idx % TN) */
    float *wsp32;          /* the same pattern in fp32 (epilogue mode 3) */
    float *bip;            /* [nblocks*NBT][TM*TN] bias pattern, NULL if no bias */
    int N, K, Np, Kp, nblk;
    int N0, N1, N2;        /* output split (linear3) */
} gemm_w;

static int gemm_prepare(k3_op *op, int N, int K)
{
    init_tiles();
    if (!TM) return 0;
    if (k3_vstore < 0) k3_vstore = TM == 8;
    gemm_w *w = calloc(1, sizeof(gemm_w));
    const int nbw = NBT * TN;
    w->N = N;
    w->K = K;
    w->Np = (N + nbw - 1) / nbw * nbw;
    w->Kp = (K + TK - 1) / TK * TK;
    w->nblk = w->Np / nbw;
    const int ks = w->Kp / TK, tt = TN * TK;
    const int8_t *src = op->wp[0];
    w->bp = xalloc((size_t)w->Np * w->Kp);
    for (int nb = 0; nb < w->nblk; nb++)
        for (int s = 0; s < ks; s++)
            for (int j = 0; j < NBT; j++) {
                int8_t *d = w->bp + (((size_t)nb * ks + s) * NBT + j) * tt;
                for (int c = 0; c < TN; c++) {
                    const int n = (nb * NBT + j) * TN + c;
                    for (int kk = 0; kk < TK; kk++) {
                        const int k = s * TK + kk;
                        d[c * TK + kk] = (n < N && k < K) ? src[(size_t)n * K + k] : 0;
                    }
                }
            }
    const float *ws = op->wp[2], *bias = op->wp[1];
    const int ntl = w->nblk * NBT, tsz = TM * TN;
    w->wsp = xalloc(sizeof(double) * (size_t)ntl * tsz);
    w->wsp32 = xalloc(sizeof(float) * (size_t)ntl * tsz);
    if (bias) w->bip = xalloc(sizeof(float) * (size_t)ntl * tsz);
    for (int t = 0; t < ntl; t++)
        for (int e = 0; e < tsz; e++) {
            const int n = t * TN + e % TN;
            w->wsp[(size_t)t * tsz + e] = n < N ? (double)ws[n] : 0.0;
            w->wsp32[(size_t)t * tsz + e] = n < N ? ws[n] : 0.0f;
            if (bias) w->bip[(size_t)t * tsz + e] = n < N ? bias[n] : 0.0f;
        }
    op->prep.wpack = w;
    return 1;
}

typedef struct {
    k3_graph *g;
    k3_op *op;
    gemm_w *w;
    const int8_t *x;
    const float *rs;
    int8_t *ap;            /* packed activations */
    double *rsp;           /* [mtiles][TM*TN] row-scale pattern (row = idx / TN) */
    float *rsp32;          /* [mtiles][TM*TN] the same in fp32, one copy per m-tile (epilogue mode 3) */
    int M, Mp, mblk, ks;
    float *out[3];
    const float *addend;
    int mgroups, mper;
    const k3_epi16 *e16;   /* binary16 output(s) or addend: the block store of kern_f16.c (NULL: all fp32) */
} gemm_ctx;

static void pack_a_range(void *vc, int64_t b0, int64_t b1, int wid)
{
    init_tiles();
    (void)wid;
    gemm_ctx *c = vc;
    const int K = c->w->K, tt = TM * TK;
    for (int64_t mb = b0; mb < b1; mb++) {
        for (int i = 0; i < MBT; i++) {
            const int mt = (int)mb * MBT + i;
            for (int r = 0; r < TM; r++) {
                const int m = mt * TM + r;
                const int8_t *row = c->x + (size_t)m * K;
                for (int s = 0; s < c->ks; s++) {
                    int8_t *d = c->ap + (((size_t)mb * c->ks + s) * MBT + i) * tt + r * TK;
                    const int k0 = s * TK;
                    if (m < c->M && k0 + TK <= K) {
                        if (TK == 16) {
                            ((uint64_t *)d)[0] = *(const uint64_t *)(row + k0);
                            ((uint64_t *)d)[1] = *(const uint64_t *)(row + k0 + 8);
                        } else {
                            ((uint64_t *)d)[0] = *(const uint64_t *)(row + k0);
                        }
                    } else {
                        for (int kk = 0; kk < TK; kk++)
                            d[kk] = (m < c->M && k0 + kk < K) ? row[k0 + kk] : 0;
                    }
                }
            }
            const int tsz = TM * TN;
            double *rp = c->rsp + (size_t)mt * tsz * NBT;      /* NBT copies: one per n-tile of a block */
            for (int e = 0; e < tsz; e++) {
                const int m = mt * TM + e / TN;
                rp[e] = m < c->M ? (double)c->rs[m] : 0.0;
            }
            for (int j = 1; j < NBT; j++) memcpy(rp + j * tsz, rp, sizeof(double) * tsz);
            float *rq = c->rsp32 + (size_t)mt * tsz * 2;      /* two copies: one m4 chunk = 2 tiles */
            for (int e = 0; e < tsz; e++) rq[e] = rq[e + tsz] = (float)rp[e];
        }
    }
}

/* write one fp32 tile (row-major TM x TN in `y`) to the outputs. Rows go out as scalar 64-bit copies:
 * on the A100 a short (vl = 8) vle32 / vse32 pair costs ~37 ns per row, a 64-bit ld / sd ~1 ns. */
static inline void copy_row(float *d, const float *s, int n, int even)
{
    if (even) {
        uint64_t *dd = (uint64_t *)d;
        const uint64_t *ss = (const uint64_t *)s;
        for (int i = 0; i < n / 2; i++) dd[i] = ss[i];
    } else {
        uint32_t *dd = (uint32_t *)d;
        const uint32_t *ss = (const uint32_t *)s;
        for (int i = 0; i < n; i++) dd[i] = ss[i];
    }
}

extern int k3_vstore;
static inline void store_tile(gemm_ctx *c, float *y, int mt, int nt)
{
    gemm_w *w = c->w;
    const int n0 = nt * TN;
    if (n0 >= w->N) return;
    float *dst;
    int ld, col;
    if (n0 + TN <= w->N0) { dst = c->out[0]; ld = w->N0; col = n0; }
    else if (n0 >= w->N0 && n0 + TN <= w->N0 + w->N1) { dst = c->out[1]; ld = w->N1; col = n0 - w->N0; }
    else if (n0 >= w->N0 + w->N1 && n0 + TN <= w->N) { dst = c->out[2]; ld = w->N2; col = n0 - w->N0 - w->N1; }
    else dst = NULL;                                       /* tile straddles an output split or N */
    const int rows = c->M - mt * TM < TM ? c->M - mt * TM : TM;
    if (dst && k3_vstore) {        /* vector row copies: no scalar reads of vector-written data (slow on the A100) */
        const size_t vl = __riscv_vsetvl_e32m1(TN);
        for (int r = 0; r < rows; r++) {
            vfloat32m1_t v = __riscv_vle32_v_f32m1(y + r * TN, vl);
            if (c->addend) v = __riscv_vfadd_vv_f32m1(__riscv_vle32_v_f32m1(c->addend + (size_t)(mt * TM + r) * w->N + n0, vl), v, vl);
            __riscv_vse32_v_f32m1(dst + (size_t)(mt * TM + r) * ld + col, v, vl);
        }
        return;
    }
    if (dst) {
        const int even = !(ld & 1) && !(col & 1);
        if (c->addend) {                                   /* gather the addend tile, one vector add */
            float *ab = scratch(2, sizeof(float) * TM * TN);
            const int aeven = !(w->N & 1);
            for (int r = 0; r < rows; r++) copy_row(ab + r * TN, c->addend + (size_t)(mt * TM + r) * w->N + n0, TN, aeven);
            size_t vl = __riscv_vsetvl_e32m2(TM * TN);
            __riscv_vse32_v_f32m2(y, __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(ab, vl), __riscv_vle32_v_f32m2(y, vl), vl), vl);
        }
        for (int r = 0; r < rows; r++) copy_row(dst + (size_t)(mt * TM + r) * ld + col, y + r * TN, TN, even);
        return;
    }
    for (int r = 0; r < rows; r++) {
        const int m = mt * TM + r;
        const float *yr = y + r * TN;
        for (int e = 0; e < TN; e++) {
            const int n = n0 + e;
            if (n >= w->N) break;
            float v = yr[e];
            if (c->addend) v = c->addend[(size_t)m * w->N + n] + v;
            if (n < w->N0) c->out[0][(size_t)m * w->N0 + n] = v;
            else if (n < w->N0 + w->N1) c->out[1][(size_t)m * w->N1 + n - w->N0] = v;
            else c->out[2][(size_t)m * w->N2 + n - w->N0 - w->N1] = v;
        }
    }
}

/* prefetch.w the output (and prefetch.r the addend) lines this block will write: the A100 stalls ~90 ns
 * on every write-allocate miss of a 32-byte tile row otherwise. Issued before the K loop, so the
 * misses overlap the vmadots. */
int k3_prefetch = 0;   /* measured: no gain on the A100 */
int k3_vstore = -1;
int k3_bpf = 0;        /* --bpf N: prefetch.r the weight stream N k-slabs ahead (first m-block of a unit) */
int k3_mtail = 0;      /* --mtail: one-m-tile K loop when the block's second m-tile is all padding */    /* -1 auto: on for the A100 (TM == 8); --vstore 0/1 */
int k3_cbozero = 1;
int k3_skip_epilogue;
static inline void prefetch_block(gemm_ctx *c, int mt0, int nt0)
{
    gemm_w *w = c->w;
    for (int j = 0; j < NBT; j++) {
        const int n0 = (nt0 + j) * TN;
        if (n0 >= w->N) break;
        float *dst;
        int ld, col;
        if (n0 < w->N0) { dst = c->out[0]; ld = w->N0; col = n0; }
        else if (n0 < w->N0 + w->N1) { dst = c->out[1]; ld = w->N1; col = n0 - w->N0; }
        else { dst = c->out[2]; ld = w->N2; col = n0 - w->N0 - w->N1; }
        for (int r = 0; r < MBT * TM; r++) {
            const int m = mt0 * TM + r;
            if (m >= c->M) break;
            __asm__ volatile("prefetch.w 0(%0)" ::"r"(dst + (size_t)m * ld + col));
            if (c->addend) __asm__ volatile("prefetch.r 0(%0)" ::"r"(c->addend + (size_t)m * w->N + n0));
        }
    }
}

#ifndef K3_NO_IME

/* Epilogue of one 2x4 block: the 8 int32 tiles go to an L1 buffer [i][j][TM*TN] (8 stores), the
 * dequantisation runs over the whole block with long vectors (e64 m8), and the fp32 tile rows are
 * copied to the row-major outputs with scalar 64-bit moves: one vector->scalar turnaround per block
 * instead of per tile (the A100 pays ~100+ ns per turnaround and ~10 ns per short vector op). */
static inline void block_epilogue(gemm_ctx *c, int32_t *cb, int mt0, int nt0, float *yb)
{
    gemm_w *w = c->w;
    const int tsz = TM * TN, n4 = NBT * tsz;
    const int ni = (mt0 + 1) * TM < c->M ? 2 : 1;
    if (k3_skip_epilogue == 1) return;
    if (k3_skip_epilogue == 3 || k3_skip_epilogue == 4) {
        /* exact as above: (double)rs * (double)ws is formed by a widening fp32 multiply (exact: 24 + 24 bits
         * < 53), from fp32 patterns (half the bytes); the column-scale and bias chunks are loaded once for
         * both m-tiles. Byte-identical to the mode-0 epilogue. */
        const float *bp = w->bip ? w->bip + (size_t)nt0 * tsz : NULL, *wq = w->wsp32 + (size_t)nt0 * tsz;
        for (int e = 0; e < n4;) {
            size_t vl = __riscv_vsetvl_e32m4(n4 - e);
            vfloat32m4_t wv = __riscv_vle32_v_f32m4(wq + e, vl), bv;
            if (bp) bv = __riscv_vle32_v_f32m4(bp + e, vl);
            for (int i = 0; i < ni; i++) {
                const float *rq = c->rsp32 + (size_t)(mt0 + i) * tsz * 2;
                vfloat64m8_t sc = __riscv_vfwmul_vv_f64m8(__riscv_vle32_v_f32m4(rq + (e % (2 * tsz)), vl), wv, vl);
                vfloat64m8_t a = __riscv_vfwcvt_f_x_v_f64m8(__riscv_vle32_v_i32m4(cb + i * n4 + e, vl), vl);
                vfloat32m4_t y = __riscv_vfncvt_f_f_w_f32m4(__riscv_vfmul_vv_f64m8(a, sc, vl), vl);
                if (bp) y = __riscv_vfadd_vv_f32m4(y, bv, vl);
                __riscv_vse32_v_f32m4(yb + i * n4 + e, y, vl);
            }
            e += (int)vl;
        }
    } else
    for (int i = 0; i < ni; i++) {
        const int32_t *ci = cb + i * n4;
        const double *rp = c->rsp + (size_t)(mt0 + i) * n4, *wp = w->wsp + (size_t)nt0 * tsz;
        const float *bp = w->bip ? w->bip + (size_t)nt0 * tsz : NULL;
        float *yi = yb + i * n4;
        for (int e = 0; e < n4;) {
            size_t vl = __riscv_vsetvl_e64m8(n4 - e);
            vfloat64m8_t a = __riscv_vfwcvt_f_x_v_f64m8(__riscv_vle32_v_i32m4(ci + e, vl), vl);
            vfloat64m8_t sc = __riscv_vfmul_vv_f64m8(__riscv_vle64_v_f64m8(rp + e, vl), __riscv_vle64_v_f64m8(wp + e, vl), vl);
            vfloat32m4_t y = __riscv_vfncvt_f_f_w_f32m4(__riscv_vfmul_vv_f64m8(a, sc, vl), vl);
            if (bp) y = __riscv_vfadd_vv_f32m4(y, __riscv_vle32_v_f32m4(bp + e, vl), vl);
            __riscv_vse32_v_f32m4(yi + e, y, vl);
            e += (int)vl;
        }
    }
    if (k3_skip_epilogue == 2) return;
    if (k3_skip_epilogue == 4) return;             /* mode 3 math without the stores (timing only) */
    if (c->e16) { k3_f16_block_store(c->e16, yb, ni, mt0, nt0); return; }
    /* Fast store: the block's 4 tiles of one m-tile land in one output as rows of NBT*TN floats (128 B on
     * the A100, 64 B on the X100). One vrgather (e64 pairs) turns [tile][row][col] into [row][tile][col],
     * and each row leaves with one vector store: no per-tile-row short stores, no scalar turnaround. */
    if (k3_vperm > 0 || (k3_vperm < 0 && TM == 4)) {
        const int n0 = nt0 * TN, span = NBT * TN;
        float *dst = NULL;
        int ld = 0, col = 0;
        if (n0 + span <= w->N0) { dst = c->out[0]; ld = w->N0; col = n0; }
        else if (n0 >= w->N0 && n0 + span <= w->N0 + w->N1) { dst = c->out[1]; ld = w->N1; col = n0 - w->N0; }
        else if (n0 >= w->N0 + w->N1 && n0 + span <= w->N) { dst = c->out[2]; ld = w->N2; col = n0 - w->N0 - w->N1; }
        if (dst && !(ld & 1) && !(col & 1) && !(((uintptr_t)dst) & 7)) {
            const int np = n4 / 2;                          /* pairs per m-tile block */
            for (int i = 0; i < ni; i++) {
                size_t vl = __riscv_vsetvl_e64m8(np);
                vuint64m8_t v = __riscv_vle64_v_u64m8((const uint64_t *)(yb + i * n4), vl);
                v = __riscv_vrgatherei16_vv_u64m8(v, __riscv_vle16_v_u16m2(g_perm, vl), vl);
                const int rows = c->M - (mt0 + i) * TM < TM ? c->M - (mt0 + i) * TM : TM;
                float *d0 = dst + (size_t)((mt0 + i) * TM) * ld + col;
                if (c->addend) {
                    __riscv_vse64_v_u64m8((uint64_t *)(yb + i * n4), v, vl);   /* rows now contiguous */
                    for (int r = 0; r < rows; r++) {
                        size_t v2 = __riscv_vsetvl_e32m2(span);
                        vfloat32m2_t y = __riscv_vle32_v_f32m2(yb + i * n4 + r * span, v2);
                        y = __riscv_vfadd_vv_f32m2(__riscv_vle32_v_f32m2(c->addend + (size_t)((mt0 + i) * TM + r) * w->N + n0, v2), y, v2);
                        __riscv_vse32_v_f32m2(d0 + (size_t)r * ld, y, v2);
                    }
                } else if (TM == 8) {                       /* A100: a row is one m1 register (16 pairs) */
                    if (rows > 0) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 0 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 0), 16);
                    if (rows > 1) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 1 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 1), 16);
                    if (rows > 2) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 2 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 2), 16);
                    if (rows > 3) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 3 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 3), 16);
                    if (rows > 4) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 4 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 4), 16);
                    if (rows > 5) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 5 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 5), 16);
                    if (rows > 6) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 6 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 6), 16);
                    if (rows > 7) __riscv_vse64_v_u64m1((uint64_t *)(d0 + 7 * (size_t)ld), __riscv_vget_v_u64m8_u64m1(v, 7), 16);
                } else {                                    /* X100: a row is one m2 register pair (8 pairs) */
                    if (rows > 0) __riscv_vse64_v_u64m2((uint64_t *)(d0 + 0 * (size_t)ld), __riscv_vget_v_u64m8_u64m2(v, 0), 8);
                    if (rows > 1) __riscv_vse64_v_u64m2((uint64_t *)(d0 + 1 * (size_t)ld), __riscv_vget_v_u64m8_u64m2(v, 1), 8);
                    if (rows > 2) __riscv_vse64_v_u64m2((uint64_t *)(d0 + 2 * (size_t)ld), __riscv_vget_v_u64m8_u64m2(v, 2), 8);
                    if (rows > 3) __riscv_vse64_v_u64m2((uint64_t *)(d0 + 3 * (size_t)ld), __riscv_vget_v_u64m8_u64m2(v, 3), 8);
                }
            }
            return;
        }
    }
    for (int i = 0; i < ni; i++)
        for (int j = 0; j < NBT; j++) store_tile(c, yb + (i * NBT + j) * tsz, mt0 + i, nt0 + j);
}

static void gemm_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    init_tiles();
    (void)wid;
    gemm_ctx *c = vc;
    gemm_w *w = c->w;
    const int tt = TM * TK, ks = c->ks;
    float *ybuf = scratch(0, sizeof(float) * (MBT * NBT + 1) * TM * TN);
    int32_t *cbuf = scratch(3, sizeof(int32_t) * MBT * NBT * TM * TN);
    for (int64_t u = u0; u < u1; u++) {
        const int nb = (int)(u / c->mgroups), mg = (int)(u % c->mgroups);
        const int mb0 = mg * c->mper, mb1 = mb0 + c->mper < c->mblk ? mb0 + c->mper : c->mblk;
        const int8_t *bbase = w->bp + (size_t)nb * ks * NBT * tt;
        for (int mb = mb0; mb < mb1; mb++) {
            const int8_t *abase = c->ap + (size_t)mb * ks * MBT * tt;
            if (k3_prefetch) prefetch_block(c, mb * MBT, nb * NBT);
            size_t vl32 = __riscv_vsetvl_e32m2(TM * TN);
            vint32m2_t c00 = __riscv_vmv_v_x_i32m2(0, vl32), c01 = c00, c02 = c00, c03 = c00;
            vint32m2_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            const int8_t *ap = abase, *bp = bbase;
            const int pf = k3_bpf && mb == mb0, half = k3_mtail && (mb * MBT + 1) * TM >= c->M;
            if (half) {                 /* second m-tile all padding (e.g. M = 50 on the A100): 4 vmadots per slab */
                for (int s = 0; s < ks; s++, ap += MBT * tt, bp += NBT * tt) {
                    size_t vl8 = __riscv_vsetvl_e8m1(tt);
                    if (pf) for (int o = 0; o < NBT * tt; o += 64) __asm__ volatile("prefetch.r 0(%0)" ::"r"(bp + k3_bpf * NBT * tt + o));
                    vint8m1_t a0 = __riscv_vle8_v_i8m1(ap, vl8);
                    vint8m1_t b0 = __riscv_vle8_v_i8m1(bp, vl8), b1 = __riscv_vle8_v_i8m1(bp + tt, vl8);
                    vint8m1_t b2 = __riscv_vle8_v_i8m1(bp + 2 * tt, vl8), b3 = __riscv_vle8_v_i8m1(bp + 3 * tt, vl8);
                    VMADOT(c00, a0, b0);
                    VMADOT(c01, a0, b1);
                    VMADOT(c02, a0, b2);
                    VMADOT(c03, a0, b3);
                }
            }
            for (int s = half ? ks : 0; s < ks; s++, ap += MBT * tt, bp += NBT * tt) {
                size_t vl8 = __riscv_vsetvl_e8m1(tt);
                if (pf) for (int o = 0; o < NBT * tt; o += 64) __asm__ volatile("prefetch.r 0(%0)" ::"r"(bp + k3_bpf * NBT * tt + o));
                vint8m1_t a0 = __riscv_vle8_v_i8m1(ap, vl8), a1 = __riscv_vle8_v_i8m1(ap + tt, vl8);
                vint8m1_t b0 = __riscv_vle8_v_i8m1(bp, vl8), b1 = __riscv_vle8_v_i8m1(bp + tt, vl8);
                VMADOT(c00, a0, b0);
                VMADOT(c10, a1, b0);
                VMADOT(c01, a0, b1);
                VMADOT(c11, a1, b1);
                vint8m1_t b2 = __riscv_vle8_v_i8m1(bp + 2 * tt, vl8), b3 = __riscv_vle8_v_i8m1(bp + 3 * tt, vl8);
                VMADOT(c02, a0, b2);
                VMADOT(c12, a1, b2);
                VMADOT(c03, a0, b3);
                VMADOT(c13, a1, b3);
            }
            const int mt = mb * MBT, nt = nb * NBT, tsz = TM * TN;
            {
                size_t vl = __riscv_vsetvl_e32m2(tsz);
                __riscv_vse32_v_i32m2(cbuf + 0 * tsz, c00, vl);
                __riscv_vse32_v_i32m2(cbuf + 1 * tsz, c01, vl);
                __riscv_vse32_v_i32m2(cbuf + 2 * tsz, c02, vl);
                __riscv_vse32_v_i32m2(cbuf + 3 * tsz, c03, vl);
                __riscv_vse32_v_i32m2(cbuf + 4 * tsz, c10, vl);
                __riscv_vse32_v_i32m2(cbuf + 5 * tsz, c11, vl);
                __riscv_vse32_v_i32m2(cbuf + 6 * tsz, c12, vl);
                __riscv_vse32_v_i32m2(cbuf + 7 * tsz, c13, vl);
            }
            block_epilogue(c, cbuf, mt, nt, ybuf);
        }
    }
}
#endif

static void gemm_run(k3_graph *g, k3_op *op)
{
    init_tiles();
#ifdef K3_NO_IME
    k3_ref_run(g, op);
#else
    gemm_w *w = op->prep.wpack;
    gemm_ctx c = {0};
    c.g = g;
    c.op = op;
    c.w = w;
    c.x = IN(0);
    c.rs = IN(1);
    c.M = (int)P(0);
    c.ks = w->Kp / TK;
    const int mbr = MBT * TM;
    c.mblk = (c.M + mbr - 1) / mbr;
    c.Mp = c.mblk * mbr;
    c.ap = shared_scratch(0, (size_t)c.Mp * w->Kp);
    c.rsp = shared_scratch(1, sizeof(double) * (size_t)c.mblk * MBT * TM * TN * NBT);
    c.rsp32 = shared_scratch(3, sizeof(float) * (size_t)c.mblk * MBT * TM * TN * 2);
    if (op->kind == K_LINEAR3_DQ_S8) {
        w->N0 = (int)P(2); w->N1 = (int)P(3); w->N2 = (int)P(4);
        c.out[0] = OUT(0); c.out[1] = OUT(1); c.out[2] = op->nout > 2 ? OUT(2) : NULL;
    } else {
        w->N0 = w->N; w->N1 = w->N2 = 0;
        c.out[0] = OUT(0);
        c.addend = op->kind == K_LINEAR_DQ_ADD_S8 ? IN(2) : NULL;
    }
    k3_epi16 e16;
    {
        int any = 0;
        memset(&e16, 0, sizeof e16);
        for (int j = 0; j < op->nout && j < 3; j++) {
            e16.out[j] = OUT(j);
            any |= e16.is16[j] = g->t[op->out[j]].dtype == DT_F16;
        }
        if (op->kind == K_LINEAR_DQ_ADD_S8) {
            e16.addend = IN(2);
            any |= e16.add16 = g->t[op->in[2]].dtype == DT_F16;
        }
        if (any) {
            e16.M = c.M; e16.N = w->N; e16.N0 = w->N0; e16.N1 = w->N1; e16.N2 = w->N2;
            e16.tm = TM; e16.tn = TN; e16.nbt = NBT;
            c.e16 = &e16;
        }
    }
    k3_parallel(c.mblk, 1, pack_a_range, &c);
    /* units: n-blocks x m-groups; enough units for every worker, n-block weights reused over the group */
    const int nw = k3_pool_size();
    c.mgroups = 1;
    while (c.mgroups < c.mblk && w->nblk * c.mgroups < 4 * nw) c.mgroups++;
    c.mper = (c.mblk + c.mgroups - 1) / c.mgroups;
    c.mgroups = (c.mblk + c.mper - 1) / c.mper;
    k3_parallel((int64_t)w->nblk * c.mgroups, 1, gemm_range, &c);
#endif
}

/* ======================================================================== quantize_rows_s8 */
typedef struct { k3_graph *g; k3_op *op; } opctx;

static void qrows_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const float *in = IN(0), *ci = IN(1);
    int8_t *out = OUT(0);
    float *rs = OUT(1);
    for (int64_t m = m0; m < m1; m++) {
        const float *x = in + (size_t)m * K;
        size_t vlm = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vfloat32m4_t v = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(x + k, vl), __riscv_vle32_v_f32m4(ci + k, vl), vl);
            vmax = __riscv_vfmax_vv_f32m4_tu(vmax, vmax, __riscv_vfabs_v_f32m4(v, vl), vl);
            k += (int)vl;
        }
        vfloat32m1_t r = __riscv_vfredmax_vs_f32m4_f32m1(vmax, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm);
        const float amax = __riscv_vfmv_f_s_f32m1_f32(r);
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        rs[m] = s;
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vfloat32m4_t v = __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(x + k, vl), __riscv_vle32_v_f32m4(ci + k, vl), vl);
            v = __riscv_vfmul_vf_f32m4(v, inv, vl);
            vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4_rm(v, __RISCV_FRM_RMM, vl);
            q = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(q, 127, vl), -127, vl);
            vint16m2_t q16 = __riscv_vnsra_wx_i16m2(q, 0, vl);
            __riscv_vse8_v_i8m1(out + (size_t)m * K + k, __riscv_vnsra_wx_i8m1(q16, 0, vl), vl);
            k += (int)vl;
        }
    }
}

/* ======================================================================== fp32 GEMM */
/* C[m, n] (+)= sum_k A[m*lda + k] * B[k*ldb + n], row-major, rows m in [m0, m1), all N */
#define sgemm_rows k3_sgemm_rows
void k3_sgemm_rows(const float *A, int64_t lda, const float *B, int64_t ldb, float *C, int64_t ldc, int m0,
                   int m1, int K, int N, const float *bias, int accumulate)
{
    for (int n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m2(N - n0);
        int m = m0;
        for (; m + 4 <= m1; m += 4) {
            vfloat32m2_t c0, c1, c2, c3;
            if (accumulate) {
                c0 = __riscv_vle32_v_f32m2(C + (size_t)m * ldc + n0, vl);
                c1 = __riscv_vle32_v_f32m2(C + (size_t)(m + 1) * ldc + n0, vl);
                c2 = __riscv_vle32_v_f32m2(C + (size_t)(m + 2) * ldc + n0, vl);
                c3 = __riscv_vle32_v_f32m2(C + (size_t)(m + 3) * ldc + n0, vl);
            } else if (bias) {
                c0 = __riscv_vle32_v_f32m2(bias + n0, vl);
                c1 = c0; c2 = c0; c3 = c0;
            } else {
                c0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
                c1 = c0; c2 = c0; c3 = c0;
            }
            const float *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
            const float *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) {
                vfloat32m2_t bv = __riscv_vle32_v_f32m2(b, vl);
                c0 = __riscv_vfmacc_vf_f32m2(c0, a0[k], bv, vl);
                c1 = __riscv_vfmacc_vf_f32m2(c1, a1[k], bv, vl);
                c2 = __riscv_vfmacc_vf_f32m2(c2, a2[k], bv, vl);
                c3 = __riscv_vfmacc_vf_f32m2(c3, a3[k], bv, vl);
            }
            __riscv_vse32_v_f32m2(C + (size_t)m * ldc + n0, c0, vl);
            __riscv_vse32_v_f32m2(C + (size_t)(m + 1) * ldc + n0, c1, vl);
            __riscv_vse32_v_f32m2(C + (size_t)(m + 2) * ldc + n0, c2, vl);
            __riscv_vse32_v_f32m2(C + (size_t)(m + 3) * ldc + n0, c3, vl);
        }
        for (; m < m1; m++) {
            vfloat32m2_t c0 = accumulate ? __riscv_vle32_v_f32m2(C + (size_t)m * ldc + n0, vl)
                              : bias ? __riscv_vle32_v_f32m2(bias + n0, vl) : __riscv_vfmv_v_f_f32m2(0.0f, vl);
            const float *a0 = A + (size_t)m * lda, *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) c0 = __riscv_vfmacc_vf_f32m2(c0, a0[k], __riscv_vle32_v_f32m2(b, vl), vl);
            __riscv_vse32_v_f32m2(C + (size_t)m * ldc + n0, c0, vl);
        }
        n0 += (int)vl;
    }
}

/* linear fp32: W [N, K] is transposed at load into [n-block][K][NB] (NB = VLMAX e32 m2) */
typedef struct { float *wt; int NB, nblk, N, K; } lin_w;

static int linear_prepare(k3_op *op)
{
    const int K = (int)P(1), N = (int)P(2);
    lin_w *w = calloc(1, sizeof(lin_w));
    w->NB = (int)__riscv_vsetvlmax_e32m2();
    w->N = N;
    w->K = K;
    w->nblk = (N + w->NB - 1) / w->NB;
    w->wt = xalloc(sizeof(float) * (size_t)w->nblk * w->NB * K);
    const float *src = op->wp[0];
    for (int nb = 0; nb < w->nblk; nb++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < w->NB; j++) {
                const int n = nb * w->NB + j;
                w->wt[((size_t)nb * K + k) * w->NB + j] = n < N ? src[(size_t)n * K + k] : 0.0f;
            }
    op->prep.wpack = w;
    return 1;
}

typedef struct { k3_graph *g; k3_op *op; int mper, mgroups; } lin_ctx;

static void linear_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    lin_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    lin_w *w = op->prep.wpack;
    const int M = (int)P(0), K = w->K, N = w->N;
    const float *x = IN(0), *bias = op->wp[1];
    float *y = OUT(0);
    float *tmp = scratch(1, sizeof(float) * (size_t)M * w->NB);
    for (int64_t u = u0; u < u1; u++) {
        const int nb = (int)(u / c->mgroups), mg = (int)(u % c->mgroups);
        const int m0 = mg * c->mper, m1 = m0 + c->mper < M ? m0 + c->mper : M;
        const int n0 = nb * w->NB, nn = N - n0 < w->NB ? N - n0 : w->NB;
        const float *bt = w->wt + (size_t)nb * K * w->NB;
        if (nn == w->NB) {
            sgemm_rows(x, K, bt, w->NB, y + n0, N, m0, m1, K, w->NB, bias ? bias + n0 : NULL, 0);
        } else {
            sgemm_rows(x, K, bt, w->NB, tmp, w->NB, m0, m1, K, w->NB, NULL, 0);
            for (int m = m0; m < m1; m++)
                for (int j = 0; j < nn; j++) y[(size_t)m * N + n0 + j] = tmp[(size_t)m * w->NB + j] + (bias ? bias[n0 + j] : 0.0f);
        }
    }
}

static void linear_run(k3_graph *g, k3_op *op)
{
    lin_w *w = op->prep.wpack;
    const int M = (int)P(0), nw = k3_pool_size();
    lin_ctx c = {g, op, M, 1};
    while (c.mgroups < (M + 3) / 4 && w->nblk * c.mgroups < 3 * nw) c.mgroups++;
    c.mper = ((M + c.mgroups - 1) / c.mgroups + 3) / 4 * 4;
    c.mgroups = (M + c.mper - 1) / c.mper;
    k3_parallel((int64_t)w->nblk * c.mgroups, 1, linear_range, &c);
}

/* bmm_strided with unit-stride rows: C = A B per (b0, b1) batch. B with unit k-stride (Q K^T) is
 * transposed into scratch first. */
typedef struct { k3_graph *g; k3_op *op; int mper, mgroups; float *bt; } bmm_ctx;

/* B with unit k-stride (Q K^T): transposed once per batch into shared scratch [batch][K][N] */
static void bmm_tr_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    bmm_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *p = op->ip;
    const int64_t B1 = p[1], K = p[3], N = p[4];
    const float *b = IN(1);
    for (int64_t u = u0; u < u1; u++) {
        const int64_t x = u / B1, yy = u % B1;
        const float *B = b + p[10] + x * p[11] + yy * p[12];
        float *d = c->bt + u * K * N;
        for (int64_t n = 0; n < N; n++) {
            const float *src = B + n * p[14];
            for (int64_t k = 0; k < K; k++) d[k * N + n] = src[k];
        }
    }
}

static void bmm_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    bmm_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *p = op->ip;
    const int64_t B1 = p[1], M = p[2], K = p[3], N = p[4];
    const float *a = IN(0), *b = IN(1);
    float *cc = OUT(0);
    for (int64_t u = u0; u < u1; u++) {
        const int64_t bt = u / c->mgroups, mg = u % c->mgroups;
        const int64_t x = bt / B1, yy = bt % B1;
        const float *A = a + p[5] + x * p[6] + yy * p[7];
        float *C = cc + p[15] + x * p[16] + yy * p[17];
        const int m0 = (int)(mg * c->mper), m1 = (int)(m0 + c->mper < M ? m0 + c->mper : M);
        if (c->bt) sgemm_rows(A, p[8], c->bt + bt * K * N, N, C, p[18], m0, m1, (int)K, (int)N, NULL, 0);
        else sgemm_rows(A, p[8], b + p[10] + x * p[11] + yy * p[12], p[13], C, p[18], m0, m1, (int)K, (int)N, NULL, 0);
    }
}

static int bmm_ok(k3_op *op)
{
    const int64_t *p = op->ip;
    /* unit k-stride in A, unit n-stride in C, B either n- or k-unit-stride; output fully covered */
    return p[9] == 1 && p[19] == 1 && (p[14] == 1 || p[13] == 1) && p[0] * p[1] * p[2] * p[4] == p[20];
}

static void bmm_run(k3_graph *g, k3_op *op)
{
    const int64_t *p = op->ip;
    const int64_t nb = p[0] * p[1], M = p[2], K = p[3], N = p[4];
    const int nw = k3_pool_size();
    bmm_ctx c = {g, op, (int)M, 1, NULL};
    if (p[14] != 1) {
        c.bt = shared_scratch(2, sizeof(float) * (size_t)(nb * K * N));
        par(nb, nb * K * N, bmm_tr_range, &c);
    }
    const int64_t work = nb * M * N * K;
    if (nw > 1 && work >= 4 * k3_par_min)
        while (c.mgroups < (M + 3) / 4 && nb * c.mgroups < 3 * nw) c.mgroups++;
    c.mper = (int)(((M + c.mgroups - 1) / c.mgroups + 3) / 4 * 4);
    c.mgroups = (int)((M + c.mper - 1) / c.mper);
    if (nw == 1 || work < 4 * k3_par_min) bmm_range(&c, 0, nb * c.mgroups, 0);
    else k3_parallel(nb * c.mgroups, 1, bmm_range, &c);
}

/* conv2d: im2col [OH*OW][IC*KH*KW] then out^T = W cols; output NCHW written per (oc) */
typedef struct { float *wt; int KK; } conv_w;

static int conv_prepare(k3_op *op)
{
    const int IC = (int)P(1), OC = (int)P(4), KH = (int)P(7), KW = (int)P(8);
    conv_w *w = calloc(1, sizeof(conv_w));
    w->KK = IC * KH * KW;
    w->wt = xalloc(sizeof(float) * (size_t)w->KK * OC);   /* [KK][OC] */
    const float *src = op->wp[0];
    for (int oc = 0; oc < OC; oc++)
        for (int k = 0; k < w->KK; k++) w->wt[(size_t)k * OC + oc] = src[(size_t)oc * w->KK + k];
    op->prep.wpack = w;
    return P(0) == 1;
}

typedef struct { k3_graph *g; k3_op *op; } conv_ctx;

/* units of CR output pixels: im2col chunk + GEMM + NCHW store, all in per-thread scratch */
#define CR 64
static void conv_unit_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    conv_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    conv_w *w = op->prep.wpack;
    const int IC = (int)P(1), IH = (int)P(2), IW = (int)P(3), OC = (int)P(4), OW = (int)P(6), KH = (int)P(7),
              KW = (int)P(8), SH = (int)P(9), SW = (int)P(10), PH = (int)P(11), PW = (int)P(12), DH = (int)P(13),
              DW = (int)P(14);
    const int OHW = (int)(P(5) * P(6)), KK = w->KK;
    const float *x = IN(0);
    float *y = OUT(0);
    float *col = scratch(1, sizeof(float) * (size_t)CR * KK), *tmp = scratch(2, sizeof(float) * (size_t)CR * OC);
    for (int64_t u = u0; u < u1; u++) {
        const int r0 = (int)u * CR, r1 = r0 + CR < OHW ? r0 + CR : OHW, nr = r1 - r0;
        for (int r = r0; r < r1; r++) {
            const int oh = r / OW, ow = r % OW;
            float *d = col + (size_t)(r - r0) * KK;
            for (int ic = 0; ic < IC; ic++)
                for (int kh = 0; kh < KH; kh++) {
                    const int ih = oh * SH - PH + kh * DH;
                    const float *xr = x + ((size_t)ic * IH + ih) * IW;
                    for (int kw = 0; kw < KW; kw++) {
                        const int iw = ow * SW - PW + kw * DW;
                        *d++ = (ih >= 0 && ih < IH && iw >= 0 && iw < IW) ? xr[iw] : 0.0f;
                    }
                }
        }
        sgemm_rows(col, KK, w->wt, OC, tmp, OC, 0, nr, KK, OC, op->wp[1], 0);
        for (int oc = 0; oc < OC; oc++) {
            float *yo = y + (size_t)oc * OHW + r0;
            for (int r = 0; r < nr; r++) yo[r] = tmp[(size_t)r * OC + oc];
        }
    }
}

static void conv_run(k3_graph *g, k3_op *op)
{
    conv_ctx c = {g, op};
    const int OHW = (int)(P(5) * P(6));
    const int64_t nu = (OHW + CR - 1) / CR;
    if (k3_pool_size() == 1 || nu < 2) conv_unit_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, conv_unit_range, &c);
}

/* conv_transpose2d with kernel == stride, no padding / dilation / groups: one GEMM [IH*IW, IC] x
 * [IC, OC*KH*KW] (the weight as stored), then a scatter of each pixel's OC x KH x KW block */
typedef struct { k3_graph *g; k3_op *op; float *xt; } ct_ctx;

static int convt_ok(k3_op *op)
{
    return P(0) == 1 && P(7) == P(9) && P(8) == P(10) && P(11) == 0 && P(12) == 0 && P(13) == 1 && P(14) == 1 &&
           P(15) == 1 && P(5) == P(2) * P(9) && P(6) == P(3) * P(10);
}

static void convt_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    ct_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int IC = (int)P(1), IH = (int)P(2), IW = (int)P(3), OC = (int)P(4), OW = (int)P(6), KH = (int)P(7),
              KW = (int)P(8);
    const int NO = OC * KH * KW, OHW = (int)(P(5) * P(6));
    const float *b = op->wp[1];
    float *y = OUT(0), *tmp = scratch(2, sizeof(float) * 16 * (size_t)NO);
    for (int64_t u = u0; u < u1; u++) {
        const int p0 = (int)u * 16, p1 = p0 + 16 < IH * IW ? p0 + 16 : IH * IW;
        sgemm_rows(c->xt + (size_t)p0 * IC, IC, op->wp[0], NO, tmp, NO, 0, p1 - p0, IC, NO, NULL, 0);
        for (int p = p0; p < p1; p++) {
            const int ih = p / IW, iw = p % IW;
            const float *t = tmp + (size_t)(p - p0) * NO;
            for (int oc = 0; oc < OC; oc++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        y[(size_t)oc * OHW + (size_t)(ih * KH + kh) * OW + iw * KW + kw] =
                            t[(oc * KH + kh) * KW + kw] + (b ? b[oc] : 0.0f);
        }
    }
}

static void convt_run(k3_graph *g, k3_op *op)
{
    const int IC = (int)P(1), HW = (int)(P(2) * P(3));
    const float *x = IN(0);
    ct_ctx c = {g, op, shared_scratch(2, sizeof(float) * (size_t)HW * IC)};
    for (int ic = 0; ic < IC; ic++)
        for (int p = 0; p < HW; p++) c.xt[(size_t)p * IC + ic] = x[(size_t)ic * HW + p];
    k3_parallel((HW + 15) / 16, 1, convt_range, &c);
}

static void im2col_range(void *vc, int64_t R0, int64_t R1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    {                 /* [IC, IH, IW] -> [OH*OW, IC*KH*KW] */
        const int64_t IC = P(0), IH = P(1), IW = P(2), OW = P(4), KH = P(5), KW = P(6), SH = P(7), SW = P(8),
                      PH = P(9), PW = P(10), DH = P(11), DW = P(12), KK = IC * KH * KW;
        const float *x = IN(0);
        float *y = OUT(0);
        for (int64_t r = R0; r < R1; r++) {
            const int64_t oh = r / OW, ow = r % OW;
            float *d = y + r * KK;
            for (int64_t ic = 0; ic < IC; ic++)
                for (int64_t kh = 0; kh < KH; kh++) {
                    const int64_t ih = oh * SH - PH + kh * DH;
                    for (int64_t kw = 0; kw < KW; kw++) {
                        const int64_t iw = ow * SW - PW + kw * DW;
                        *d++ = (ih >= 0 && ih < IH && iw >= 0 && iw < IW) ? x[(ic * IH + ih) * IW + iw] : 0.0f;
                    }
                }
        }
}
}

/* matmul A [M, K] x B [K, N], row-major */
static void matmul_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int M = (int)P(0), K = (int)P(1), N = (int)P(2);
    const int m0 = (int)u0 * 16, m1 = (int)u1 * 16 < M ? (int)u1 * 16 : M;
    sgemm_rows(IN(0), K, IN(1), N, OUT(0), N, m0, m1, K, N, NULL, 0);
}

static void vsoftmax_row(float *x, int n, float scale);

/* sdpa [BH, S, D]: K transposed once (shared), then per (head, 32-row block): scores, softmax, P V */
typedef struct { k3_graph *g; k3_op *op; float *kt; } sdpa_ctx;

static void sdpa_tr(void *vc, int64_t b0, int64_t b1, int wid)
{
    (void)wid;
    sdpa_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int S = (int)P(1), D = (int)P(2);
    const float *k = IN(1);
    for (int64_t b = b0; b < b1; b++)
        for (int j = 0; j < S; j++)
            for (int d = 0; d < D; d++) c->kt[((size_t)b * D + d) * S + j] = k[((size_t)b * S + j) * D + d];
}

static void sdpa_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    sdpa_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int S = (int)P(1), D = (int)P(2), nb = (S + 31) / 32;
    const float *q = IN(0), *v = IN(2);
    float *y = OUT(0), *sc = scratch(1, sizeof(float) * 32 * (size_t)S);
    const float scale = 1.0f / sqrtf((float)D);
    for (int64_t u = u0; u < u1; u++) {
        const int b = (int)(u / nb), i0 = (int)(u % nb) * 32, i1 = i0 + 32 < S ? i0 + 32 : S;
        sgemm_rows(q + (size_t)b * S * D, D, c->kt + (size_t)b * D * S, S, sc - (size_t)i0 * S, S, i0, i1, D, S, NULL, 0);
        for (int i = i0; i < i1; i++) vsoftmax_row(sc + (size_t)(i - i0) * S, S, scale);
        sgemm_rows(sc - (size_t)i0 * S, S, v + (size_t)b * S * D, D, y + (size_t)b * S * D, D, i0, i1, S, D, NULL, 0);
    }
}

static void sdpa_run(k3_graph *g, k3_op *op)
{
    const int BH = (int)P(0), S = (int)P(1), D = (int)P(2);
    sdpa_ctx c = {g, op, shared_scratch(2, sizeof(float) * (size_t)BH * S * D)};
    par(BH, (int64_t)BH * S * D, sdpa_tr, &c);
    const int64_t nu = (int64_t)BH * ((S + 31) / 32);
    if (k3_pool_size() == 1) sdpa_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, sdpa_range, &c);
}

/* ======================================================================== vector math */
static inline vfloat32m4_t vexp(vfloat32m4_t x, size_t vl)
{
    /* exp(x) = 2^n * p(r), r = x - n ln2, |r| <= ln2/2; degree-6 polynomial (rel. err ~2e-7) */
    x = __riscv_vfmin_vf_f32m4(__riscv_vfmax_vf_f32m4(x, -87.3f, vl), 88.3f, vl);
    vfloat32m4_t t = __riscv_vfmul_vf_f32m4(x, 1.44269504088896341f, vl);
    vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4_rm(t, __RISCV_FRM_RNE, vl);
    vfloat32m4_t nf = __riscv_vfcvt_f_x_v_f32m4(n, vl);
    vfloat32m4_t r = __riscv_vfnmsac_vf_f32m4(x, 0.693359375f, nf, vl);
    r = __riscv_vfnmsac_vf_f32m4(r, -2.12194440e-4f, nf, vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.9875691500e-4f, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.3981999507e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(8.3334519073e-3f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(4.1665795894e-2f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(1.6666665459e-1f, vl), vl);
    p = __riscv_vfmadd_vv_f32m4(p, r, __riscv_vfmv_v_f_f32m4(5.0000001201e-1f, vl), vl);
    vfloat32m4_t r2 = __riscv_vfmul_vv_f32m4(r, r, vl);
    p = __riscv_vfmadd_vv_f32m4(p, r2, __riscv_vfadd_vf_f32m4(r, 1.0f, vl), vl);
    vint32m4_t e = __riscv_vsll_vx_i32m4(__riscv_vadd_vx_i32m4(n, 127, vl), 23, vl);
    return __riscv_vfmul_vv_f32m4(p, __riscv_vreinterpret_v_i32m4_f32m4(e), vl);
}

/* log(x), x > 0 (Cephes logf: x = m 2^e, m in [sqrt(1/2), sqrt(2)), degree-8 polynomial) */
static inline vfloat32m4_t vlog(vfloat32m4_t x, size_t vl)
{
    vint32m4_t xi = __riscv_vreinterpret_v_f32m4_i32m4(x);
    vint32m4_t e = __riscv_vsub_vx_i32m4(__riscv_vsra_vx_i32m4(xi, 23, vl), 127, vl);
    vfloat32m4_t m = __riscv_vreinterpret_v_i32m4_f32m4(__riscv_vor_vx_i32m4(__riscv_vand_vx_i32m4(xi, 0x007fffff, vl), 0x3f800000, vl));
    vbool8_t big = __riscv_vmfgt_vf_f32m4_b8(m, 1.41421356f, vl);
    m = __riscv_vfmul_vf_f32m4_mu(big, m, m, 0.5f, vl);
    e = __riscv_vadd_vx_i32m4_mu(big, e, e, 1, vl);
    vfloat32m4_t f = __riscv_vfsub_vf_f32m4(m, 1.0f, vl), z = __riscv_vfmul_vv_f32m4(f, f, vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(7.0376836292e-2f, vl);
    const float c[8] = {-1.1514610310e-1f, 1.1676998740e-1f, -1.2420140846e-1f, 1.4249322787e-1f,
                        -1.6668057665e-1f, 2.0000714765e-1f, -2.4999993993e-1f, 3.3333331174e-1f};
    for (int i = 0; i < 8; i++) p = __riscv_vfmadd_vv_f32m4(p, f, __riscv_vfmv_v_f_f32m4(c[i], vl), vl);
    p = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), z, vl);
    vfloat32m4_t ef = __riscv_vfcvt_f_x_v_f32m4(e, vl);
    p = __riscv_vfmacc_vf_f32m4(p, -2.12194440e-4f, ef, vl);
    p = __riscv_vfnmsac_vf_f32m4(p, 0.5f, z, vl);
    vfloat32m4_t r = __riscv_vfadd_vv_f32m4(f, p, vl);
    return __riscv_vfmacc_vf_f32m4(r, 0.693359375f, ef, vl);
}

/* tanh(x) = sign(x) (1 - 2 / (exp(2|x|) + 1)) */
static inline vfloat32m4_t vtanh(vfloat32m4_t x, size_t vl)
{
    vfloat32m4_t a = __riscv_vfabs_v_f32m4(x, vl);
    vfloat32m4_t e = vexp(__riscv_vfmul_vf_f32m4(a, 2.0f, vl), vl);
    vfloat32m4_t t = __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(e, 1.0f, vl), 2.0f, vl), 1.0f, vl);
    return __riscv_vfsgnj_vv_f32m4(t, x, vl);
}

/* erf(x): Abramowitz-Stegun 7.1.26 (|error| < 1.5e-7) */
static inline vfloat32m4_t verf(vfloat32m4_t x, size_t vl)
{
    vfloat32m4_t a = __riscv_vfabs_v_f32m4(x, vl);
    vfloat32m4_t t = __riscv_vfrdiv_vf_f32m4(__riscv_vfmacc_vf_f32m4(__riscv_vfmv_v_f_f32m4(1.0f, vl), 0.3275911f, a, vl), 1.0f, vl);
    vfloat32m4_t p = __riscv_vfmv_v_f_f32m4(1.061405429f, vl);
    const float c[4] = {-1.453152027f, 1.421413741f, -0.284496736f, 0.254829592f};
    for (int i = 0; i < 4; i++) p = __riscv_vfmadd_vv_f32m4(p, t, __riscv_vfmv_v_f_f32m4(c[i], vl), vl);
    p = __riscv_vfmul_vv_f32m4(p, t, vl);
    vfloat32m4_t e = vexp(__riscv_vfneg_v_f32m4(__riscv_vfmul_vv_f32m4(a, a, vl), vl), vl);
    vfloat32m4_t y = __riscv_vfrsub_vf_f32m4(__riscv_vfmul_vv_f32m4(p, e, vl), 1.0f, vl);
    return __riscv_vfsgnj_vv_f32m4(y, x, vl);
}

static void vsoftmax_row(float *y, int K, float scale)
{
    size_t vlm = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vm = __riscv_vfmv_v_f_f32m4(-INFINITY, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t t = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), scale, vl);
        __riscv_vse32_v_f32m4(y + k, t, vl);
        vm = __riscv_vfmax_vv_f32m4_tu(vm, vm, t, vl);
        k += (int)vl;
    }
    const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(vm, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), vlm));
    vfloat32m4_t vs = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t e = vexp(__riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), mx, vl), vl);
        __riscv_vse32_v_f32m4(y + k, e, vl);
        vs = __riscv_vfadd_vv_f32m4_tu(vs, vs, e, vl);
        k += (int)vl;
    }
    const float inv = 1.0f / __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vs, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        __riscv_vse32_v_f32m4(y + k, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), inv, vl), vl);
        k += (int)vl;
    }
}

static void softmax_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const float *in = IN(0);
    float *out = OUT(0);
    for (int64_t m = m0; m < m1; m++) {
        const float *x = in + (size_t)m * K;
        float *y = out + (size_t)m * K;
        size_t vlm = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t vm = __riscv_vfmv_v_f_f32m4(-INFINITY, vlm);
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vm = __riscv_vfmax_vv_f32m4_tu(vm, vm, __riscv_vle32_v_f32m4(x + k, vl), vl);
            k += (int)vl;
        }
        const float mx = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredmax_vs_f32m4_f32m1(vm, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), vlm));
        vfloat32m4_t vs = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vfloat32m4_t e = vexp(__riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), mx, vl), vl);
            if (mx == -INFINITY) e = __riscv_vfmv_v_f_f32m4(NAN, vl);
            __riscv_vse32_v_f32m4(y + k, e, vl);
            vs = __riscv_vfadd_vv_f32m4_tu(vs, vs, e, vl);
            k += (int)vl;
        }
        const float s = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(vs, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
        const float inv = 1.0f / s;
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            __riscv_vse32_v_f32m4(y + k, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), inv, vl), vl);
            k += (int)vl;
        }
    }
}

static float vsum(const float *x, int K, float sub, int sq)
{
    size_t vlm = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t v = __riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), sub, vl);
        if (sq) v = __riscv_vfmul_vv_f32m4(v, v, vl);
        acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, v, vl);
        k += (int)vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
}

static void norm_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const float eps = (float)op->fp[0];
    const float *in = IN(0), *ga = op->wp[0], *be = op->wp[1];
    float *out = OUT(0);
    for (int64_t m = m0; m < m1; m++) {
        const float *x = in + (size_t)m * K;
        float *y = out + (size_t)m * K;
        if (op->kind == K_RMS_NORM) {
            const float d = sqrtf(vsum(x, K, 0.0f, 1) / (float)K + eps);
            const float inv = 1.0f / d;
            for (int k = 0; k < K;) {
                size_t vl = __riscv_vsetvl_e32m4(K - k);
                __riscv_vse32_v_f32m4(y + k, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), inv, vl), vl);
                k += (int)vl;
            }
        } else {
            const float mean = vsum(x, K, 0.0f, 0) / (float)K;
            const float var = vsum(x, K, mean, 1) / (float)K;
            const float rstd = 1.0f / sqrtf(var + eps);
            for (int k = 0; k < K;) {
                size_t vl = __riscv_vsetvl_e32m4(K - k);
                vfloat32m4_t v = __riscv_vfmul_vf_f32m4(__riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), mean, vl), rstd, vl);
                if (ga) v = __riscv_vfmul_vv_f32m4(v, __riscv_vle32_v_f32m4(ga + k, vl), vl);
                if (be) v = __riscv_vfadd_vv_f32m4(v, __riscv_vle32_v_f32m4(be + k, vl), vl);
                __riscv_vse32_v_f32m4(y + k, v, vl);
                k += (int)vl;
            }
        }
    }
}

/* elementwise over [i0, i1) */
static void ewise_range(void *vc, int64_t i0, int64_t i1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const float *a = IN(0), *b = op->nin > 1 ? IN(1) : NULL;
    float *y = OUT(0);
    const float sc = (float)op->fp[0];
    for (int64_t i = i0; i < i1;) {
        size_t vl = __riscv_vsetvl_e32m4(i1 - i);
        vfloat32m4_t x = __riscv_vle32_v_f32m4(a + i, vl), r;
        switch (op->kind) {
        case K_ADD: r = __riscv_vfadd_vv_f32m4(x, __riscv_vle32_v_f32m4(b + i, vl), vl); break;
        case K_MUL_SCALAR: r = __riscv_vfmul_vf_f32m4(x, sc, vl); break;
        case K_SILU_MUL: case K_SILU: case K_EWISE1: {      /* x / (1 + exp(-x)) [* b] */
            vfloat32m4_t e = vexp(__riscv_vfneg_v_f32m4(x, vl), vl);
            r = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(e, 1.0f, vl), vl);
            if (op->kind == K_SILU_MUL) r = __riscv_vfmul_vv_f32m4(r, __riscv_vle32_v_f32m4(b + i, vl), vl);
            break;
        }
        case K_GELU_EXACT: {                                 /* 0.5 x (1 + tanh(u)), tanh(u) = 1 - 2 / (exp(2u) + 1) */
            vfloat32m4_t x3 = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vv_f32m4(x, x, vl), x, vl);
            vfloat32m4_t u = __riscv_vfmul_vf_f32m4(__riscv_vfmacc_vf_f32m4(x, 0.044715f, x3, vl), 0.79788456080286536f, vl);
            vfloat32m4_t e = vexp(__riscv_vfmul_vf_f32m4(u, 2.0f, vl), vl);
            vfloat32m4_t th = __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(e, 1.0f, vl), 2.0f, vl), 1.0f, vl);
            r = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(x, 0.5f, vl), __riscv_vfadd_vf_f32m4(th, 1.0f, vl), vl);
            break;
        }
        case K_GELU:
            r = __riscv_vfmul_vv_f32m4(__riscv_vfmul_vf_f32m4(x, 0.5f, vl),
                                      __riscv_vfadd_vf_f32m4(verf(__riscv_vfmul_vf_f32m4(x, 0.70710678118654752f, vl), vl), 1.0f, vl), vl);
            break;
        case K_TANH: r = vtanh(x, vl); break;
        case K_RELU: r = __riscv_vfmax_vf_f32m4(x, 0.0f, vl); break;
        case K_SIGMOID: r = __riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(vexp(__riscv_vfneg_v_f32m4(x, vl), vl), 1.0f, vl), 1.0f, vl); break;
        case K_SOFTPLUS: {                                  /* log1p(exp(x)), x for x > 20 (torch threshold) */
            vfloat32m4_t l = vlog(__riscv_vfadd_vf_f32m4(vexp(x, vl), 1.0f, vl), vl);
            r = __riscv_vmerge_vvm_f32m4(l, x, __riscv_vmfgt_vf_f32m4_b8(x, 20.0f, vl), vl);
            break;
        }
        case K_MUL: r = __riscv_vfmul_vv_f32m4(x, __riscv_vle32_v_f32m4(b + i, vl), vl); break;
        default: r = x;
        }
        __riscv_vse32_v_f32m4(y + i, r, vl);
        i += (int64_t)vl;
    }
}

/* ewise2 add / mul with innermost unit or zero strides (the broadcasts these graphs use) */
static void ewise2_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *d = op->ip + 1, *sa = op->ip + 7, *sb = op->ip + 13;
    const float *a = IN(0), *b = IN(1);
    float *y = OUT(0);
    const int opc = (int)P(0);
    const int64_t n5 = d[5];
    for (int64_t r = r0; r < r1; r++) {               /* r enumerates (i0..i4) */
        int64_t t = r, idx[5];
        for (int k = 4; k >= 0; k--) { idx[k] = t % d[k]; t /= d[k]; }
        int64_t pa = 0, pb = 0;
        for (int k = 0; k < 5; k++) { pa += idx[k] * sa[k]; pb += idx[k] * sb[k]; }
        float *yo = y + r * n5;
        for (int64_t i = 0; i < n5;) {
            size_t vl = __riscv_vsetvl_e32m4(n5 - i);
            vfloat32m4_t va = sa[5] ? __riscv_vle32_v_f32m4(a + pa + i, vl) : __riscv_vfmv_v_f_f32m4(a[pa], vl);
            vfloat32m4_t vb = sb[5] ? __riscv_vle32_v_f32m4(b + pb + i, vl) : __riscv_vfmv_v_f_f32m4(b[pb], vl);
            vfloat32m4_t vr = opc == 0 ? __riscv_vfadd_vv_f32m4(va, vb, vl) : __riscv_vfmul_vv_f32m4(va, vb, vl);
            __riscv_vse32_v_f32m4(yo + i, vr, vl);
            i += (int64_t)vl;
        }
    }
}

static void rope_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int H = (int)P(1), half = (int)P(2);
    const float *in = IN(0), *ct = IN(1), *st = IN(2);
    float *out = OUT(0);
    for (int64_t r = r0; r < r1; r++)
        for (int h = 0; h < H; h++) {
            const float *x = in + ((size_t)r * H + h) * 2 * half;
            float *y = out + ((size_t)r * H + h) * 2 * half;
            for (int j = 0; j < half;) {
                size_t vl = __riscv_vsetvl_e32m4(half - j);
                vfloat32m4_t x1 = __riscv_vle32_v_f32m4(x + j, vl), x2 = __riscv_vle32_v_f32m4(x + half + j, vl);
                vfloat32m4_t cc = __riscv_vle32_v_f32m4(ct + (size_t)r * half + j, vl), ss = __riscv_vle32_v_f32m4(st + (size_t)r * half + j, vl);
                __riscv_vse32_v_f32m4(y + j, __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x1, cc, vl), __riscv_vfmul_vv_f32m4(x2, ss, vl), vl), vl);
                __riscv_vse32_v_f32m4(y + half + j, __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x2, cc, vl), __riscv_vfmul_vv_f32m4(x1, ss, vl), vl), vl);
                j += (int)vl;
            }
        }
}

/* strided_copy: rows of the innermost dim; unit stride -> memcpy */
static void scopy_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t off = P(0), *d = op->ip + 1, *s = op->ip + 7;
    const size_t es = k3_esz(g->t[op->in[0]].dtype);
    const uint8_t *in = IN(0);
    uint8_t *out = OUT(0);
    for (int64_t r = r0; r < r1; r++) {
        int64_t t = r, b = off;
        for (int k = 4; k >= 0; k--) { b += (t % d[k]) * s[k]; t /= d[k]; }
        uint8_t *o = out + (size_t)r * d[5] * es;
        if (s[5] == 1) memcpy(o, in + b * es, (size_t)d[5] * es);
        else if (es == 4) {
            const float *src = (const float *)in + b;
            float *dst = (float *)o;
            for (int64_t i = 0; i < d[5]; i++) dst[i] = src[i * s[5]];
        } else for (int64_t i = 0; i < d[5]; i++) memcpy(o + i * es, in + (b + i * s[5]) * es, es);
    }
}


/* ======================================================================== row op + quantize_rows_s8 */
/* A norm or elementwise op whose output is read by the next op, a quantize_rows_s8 over the same rows:
 * one pass per chunk of rows (producer rows, then their quantisation while they are in L1), one pool
 * dispatch instead of two. The producer output is still written, so later readers are unaffected, and both
 * halves run the same code as the separate kernels: byte-identical results. */
int k3_rowq_mode = 0;

static int rowq_producer(k3_op *op)
{
    switch (op->kind) {
    case K_LAYER_NORM: case K_RMS_NORM: case K_SILU_MUL: case K_GELU_EXACT: case K_GELU: case K_SILU: case K_MUL:
    case K_ADD: case K_RELU: case K_SIGMOID: case K_TANH: return 1;
    case K_EWISE1: return P(0) == 14;
    default: return 0;
    }
}

int k3_rowq_fuse(k3_graph *g)
{
    int n = 0;
    if (!k3_rowq_mode) return 0;
    for (int i = 0; i + 1 < g->n_ops; i++) {
        k3_op *op = &g->ops[i], *q = &g->ops[i + 1];
        if (op->impl != 1 || q->impl != 1 || q->kind != K_QUANTIZE_ROWS_S8 || !rowq_producer(op)) continue;
        if (g->t[q->in[0]].data != g->t[op->out[0]].data || g->t[q->in[0]].nelem != g->t[op->out[0]].nelem) continue;
        const int64_t M = q->ip[0], K = q->ip[1];
        if (op->kind == K_LAYER_NORM || op->kind == K_RMS_NORM) { if (op->ip[0] != M || op->ip[1] != K) continue; }
        else if ((op->kind == K_EWISE1 ? op->ip[1] : op->ip[0]) != M * K) continue;
        op->impl = 4;
        op->prep.npad = i + 1;          /* index of the fused quantize op */
        q->impl = 2;
        n++, i++;
    }
    return n;
}

typedef struct { k3_graph *g; k3_op *op, *q; int64_t K, rows; } rowq_ctx;
static void rowq_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    rowq_ctx *c = vc;
    opctx pc = {c->g, c->op}, qc = {c->g, c->q};
    const int norm = c->op->kind == K_LAYER_NORM || c->op->kind == K_RMS_NORM;
    for (int64_t m = m0; m < m1; m += c->rows) {
        const int64_t e = m + c->rows < m1 ? m + c->rows : m1;
        if (norm) norm_range(&pc, m, e, wid);
        else ewise_range(&pc, m * c->K, e * c->K, wid);
        qrows_range(&qc, m, e, wid);
    }
}

void k3_rowq_run(k3_graph *g, k3_op *op)
{
    k3_op *q = &g->ops[op->prep.npad];
    rowq_ctx c = {g, op, q, q->ip[1], 0};
    c.rows = 16384 / (4 * c.K) > 1 ? 16384 / (4 * c.K) : 1;
    par(q->ip[0], q->ip[0] * q->ip[1], rowq_range, &c);
}

/* ======================================================================== dispatch */
int k3_opt_prepare(k3_graph *g, k3_op *op)
{
    (void)g;
    switch (op->kind) {
    case K_LINEAR_DQ_S8: return gemm_prepare(op, (int)P(2), (int)P(1));
    case K_LINEAR_DQ_ADD_S8: return gemm_prepare(op, (int)P(2), (int)P(1));
    case K_LINEAR3_DQ_S8: return gemm_prepare(op, (int)(P(2) + P(3) + P(4)), (int)P(1));
    case K_QUANTIZE_ROWS_S8: return 1;
    case K_LINEAR: return linear_prepare(op);
    case K_BMM_STRIDED: return bmm_ok(op);
    case K_CONV2D: return conv_prepare(op);
    case K_SOFTMAX: case K_LAYER_NORM: case K_RMS_NORM: case K_ROPE_HALF: case K_SILU_MUL: case K_GELU_EXACT:
    case K_ADD: case K_MUL_SCALAR: case K_STRIDED_COPY: case K_SILU: case K_GELU: case K_TANH: case K_SOFTPLUS:
    case K_MUL: case K_RELU: case K_SIGMOID: case K_MATMUL: case K_SDPA: return 1;
    case K_CONV_T2D: return convt_ok(op);
    case K_IM2COL: return 1;
    case K_EWISE1: return P(0) == 14;
    case K_EWISE2: return (P(0) == 0 || P(0) == 2) && op->fp[0] == 1.0 &&
                          (op->ip[7 + 5] == 0 || op->ip[7 + 5] == 1) && (op->ip[13 + 5] == 0 || op->ip[13 + 5] == 1);
    default: return 0;
    }
}

void k3_opt_run(k3_graph *g, k3_op *op)
{
    opctx c = {g, op};
    switch (op->kind) {
    case K_LINEAR_DQ_S8: case K_LINEAR_DQ_ADD_S8: case K_LINEAR3_DQ_S8: gemm_run(g, op); break;
    case K_QUANTIZE_ROWS_S8: par(P(0), P(0) * P(1), qrows_range, &c); break;
    case K_LINEAR: linear_run(g, op); break;
    case K_BMM_STRIDED: bmm_run(g, op); break;
    case K_CONV2D: conv_run(g, op); break;
    case K_CONV_T2D: convt_run(g, op); break;
    case K_SDPA: sdpa_run(g, op); break;
    case K_IM2COL: par(P(3) * P(4), P(3) * P(4) * P(0) * P(5) * P(6), im2col_range, &c); break;
    case K_MATMUL: par((P(0) + 15) / 16, P(0) * P(1) * P(2) / 8, matmul_range, &c); break;
    case K_SOFTMAX: par(P(0), P(0) * P(1), softmax_range, &c); break;
    case K_LAYER_NORM: case K_RMS_NORM: par(P(0), P(0) * P(1), norm_range, &c); break;
    case K_ROPE_HALF: par(P(0), P(0) * P(1) * P(2) * 2, rope_range, &c); break;
    case K_EWISE1: if (P(1) < k3_par_min) ewise_range(&c, 0, P(1), 0); else k3_parallel(P(1), 8192, ewise_range, &c); break;
    case K_SILU_MUL: case K_GELU_EXACT: case K_ADD: case K_MUL_SCALAR: case K_SILU: case K_GELU: case K_TANH:
    case K_SOFTPLUS: case K_MUL: case K_RELU: case K_SIGMOID:
        if (P(0) < k3_par_min) ewise_range(&c, 0, P(0), 0); else k3_parallel(P(0), 8192, ewise_range, &c);
        break;
    case K_EWISE2: {
        const int64_t *d = op->ip + 1;
        par(d[0] * d[1] * d[2] * d[3] * d[4], d[0] * d[1] * d[2] * d[3] * d[4] * d[5], ewise2_range, &c);
        break;
    }
    case K_STRIDED_COPY: {
        const int64_t *d = op->ip + 1;
        par(d[0] * d[1] * d[2] * d[3] * d[4], d[0] * d[1] * d[2] * d[3] * d[4] * d[5], scopy_range, &c);
        break;
    }
    default: k3_ref_run(g, op);
    }
}
