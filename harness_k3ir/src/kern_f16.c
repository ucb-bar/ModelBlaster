/* fp16-storage kernels of the K3 IR runtime (RVV + Zvfh; ModelBlaster's _f16 contract, f16_ir.py).
 *
 * Storage binary16, math fp32, one round-to-nearest-even rounding on store. Precision per op class:
 *   GEMM-class (linear, matmul, bmm_strided, conv2d via im2col, conv_transpose2d, the two products of sdpa):
 *     fp16 operands, vfwmacc.vf (fp16 x fp16 -> fp32, the product exact), fp32 accumulators in the
 *     reference's k order, bias in fp32 at the reference's position (after the sum; before it for
 *     conv_transpose2d), one rounding to fp16. Byte-identical to the reference C (--check).
 *   add / mul / ewise2 add|mul / relu: native e16 arithmetic. binary32 has p = 24 >= 2*11 + 2 bits, so
 *     fp16(fp32(a op b)) == fp16(a op b) for + and x (double rounding is innocuous): byte-identical.
 *   mul_scalar, rope_half, quantize_rows_s8: widened to fp32, the reference's fp32 operations in order:
 *     byte-identical.
 *   softmax, layer_norm, rms_norm: fp32 statistics; partial sums in vector lanes, final reduction ordered
 *     (vfredosum: deterministic for a given VLEN, so Spike and the board agree), not the reference's
 *     scalar order: held to the fp16 slot bound.
 *   silu / silu_mul / gelu / gelu_tanh / sigmoid / tanh / softplus: fp32 polynomials (kern_opt.c's),
 *     slot bound.
 *   linear_dq*_s8 with fp16 activations: kern_opt.c's IME kernel on an fp32 view of the operands
 *     (the addend widened exactly), result rounded once to fp16: byte-identical.
 *   copies (strided_copy, catn, index_select, upsample, maxpool) move binary16 elements unchanged. */
#include <math.h>
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3ir.h"

typedef _Float16 h16;
#define IN(i) k3_ptr(g, op->in[i])
#define OUT(i) k3_ptr(g, op->out[i])
#define P(i) (op->ip[i])

extern int k3_par_min;
int k3_hg_cfg = 0;     /* GEMM register blocking: 0 auto (= 2), 1 = 8 rows x e32m2, 2 = 4 rows x e32m4. Both run at
                        * the vector fp32 FMA rate on the board (bench: A100 ~10.6, X100 13-16 MAC/ns per core);
                        * 2 is 5-12 % faster on the X100 and not slower on the A100 */

static void *xalloc(size_t n)
{
    void *p = aligned_alloc(64, (n + 127) & ~(size_t)63);
    memset(p, 0, (n + 127) & ~(size_t)63);
    return p;
}

static __thread void *scr_p[6];
static __thread size_t scr_n[6];
static void *scratch(int i, size_t n)
{
    if (scr_n[i] < n) {
        free(scr_p[i]);
        scr_n[i] = (n + 4095) & ~(size_t)4095;
        scr_p[i] = aligned_alloc(64, scr_n[i]);
    }
    return scr_p[i];
}

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

/* register blocking for a GEMM with M rows: forced by --hg-cfg, else 4 rows x e32m4 except on the X100 (vlenb 32)
 * for M <= 64, where the 4-row panel [K][32] fp16 outgrows L1 for the few row groups that reuse it (SmolVLA
 * step, M = 50: 57 ms with 8 x m2 vs 71 ms with 4 x m4 on 8 X100) */
static int cfg_m(int64_t M)
{
    if (k3_hg_cfg) return k3_hg_cfg;
    return (__riscv_vlenb() == 32 && M <= 64) ? 1 : 2;
}

/* ======================================================================== GEMM
 * C[m, n] = sum_k A[m*lda + k] * B[k*ldb + n] (+ bias[n]) for rows [m0, m1), columns [0, N).
 * out16: C is binary16 (rounded once), else fp32. bias_first: the accumulators start at the bias. */
#define ST2(C, off, c, vl)                                                                                        \
    do {                                                                                                          \
        if (out16) __riscv_vse16_v_f16m1((h16 *)(C) + (off), __riscv_vfncvt_f_f_w_f16m1((c), (vl)), (vl));      \
        else __riscv_vse32_v_f32m2((float *)(C) + (off), (c), (vl));                                             \
    } while (0)
#define ST4(C, off, c, vl)                                                                                        \
    do {                                                                                                          \
        if (out16) __riscv_vse16_v_f16m2((h16 *)(C) + (off), __riscv_vfncvt_f_f_w_f16m2((c), (vl)), (vl));      \
        else __riscv_vse32_v_f32m4((float *)(C) + (off), (c), (vl));                                             \
    } while (0)

static void hgemm_m2(const h16 *A, int64_t lda, const h16 *B, int64_t ldb, void *C, int64_t ldc, int m0, int m1,
                     int K, int N, const h16 *bias, int bias_first, int out16)
{
    for (int n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m2(N - n0);
        vfloat32m2_t z = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        vfloat32m2_t bv = bias ? __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1(bias + n0, vl), vl) : z;
        vfloat32m2_t init = bias_first ? bv : z;
        const int addb = bias && !bias_first;
        int m = m0;
        for (; m + 8 <= m1; m += 8) {
            vfloat32m2_t c0 = init, c1 = init, c2 = init, c3 = init, c4 = init, c5 = init, c6 = init, c7 = init;
            const h16 *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda, *a4 = a3 + lda,
                      *a5 = a4 + lda, *a6 = a5 + lda, *a7 = a6 + lda;
            const h16 *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) {
                vfloat16m1_t v = __riscv_vle16_v_f16m1(b, vl);
                c0 = __riscv_vfwmacc_vf_f32m2(c0, a0[k], v, vl);
                c1 = __riscv_vfwmacc_vf_f32m2(c1, a1[k], v, vl);
                c2 = __riscv_vfwmacc_vf_f32m2(c2, a2[k], v, vl);
                c3 = __riscv_vfwmacc_vf_f32m2(c3, a3[k], v, vl);
                c4 = __riscv_vfwmacc_vf_f32m2(c4, a4[k], v, vl);
                c5 = __riscv_vfwmacc_vf_f32m2(c5, a5[k], v, vl);
                c6 = __riscv_vfwmacc_vf_f32m2(c6, a6[k], v, vl);
                c7 = __riscv_vfwmacc_vf_f32m2(c7, a7[k], v, vl);
            }
            if (addb) {
                c0 = __riscv_vfadd_vv_f32m2(c0, bv, vl); c1 = __riscv_vfadd_vv_f32m2(c1, bv, vl);
                c2 = __riscv_vfadd_vv_f32m2(c2, bv, vl); c3 = __riscv_vfadd_vv_f32m2(c3, bv, vl);
                c4 = __riscv_vfadd_vv_f32m2(c4, bv, vl); c5 = __riscv_vfadd_vv_f32m2(c5, bv, vl);
                c6 = __riscv_vfadd_vv_f32m2(c6, bv, vl); c7 = __riscv_vfadd_vv_f32m2(c7, bv, vl);
            }
            ST2(C, (size_t)(m + 0) * ldc + n0, c0, vl); ST2(C, (size_t)(m + 1) * ldc + n0, c1, vl);
            ST2(C, (size_t)(m + 2) * ldc + n0, c2, vl); ST2(C, (size_t)(m + 3) * ldc + n0, c3, vl);
            ST2(C, (size_t)(m + 4) * ldc + n0, c4, vl); ST2(C, (size_t)(m + 5) * ldc + n0, c5, vl);
            ST2(C, (size_t)(m + 6) * ldc + n0, c6, vl); ST2(C, (size_t)(m + 7) * ldc + n0, c7, vl);
        }
        for (; m + 2 <= m1; m += 2) {
            vfloat32m2_t c0 = init, c1 = init;
            const h16 *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) {
                vfloat16m1_t v = __riscv_vle16_v_f16m1(b, vl);
                c0 = __riscv_vfwmacc_vf_f32m2(c0, a0[k], v, vl);
                c1 = __riscv_vfwmacc_vf_f32m2(c1, a1[k], v, vl);
            }
            if (addb) { c0 = __riscv_vfadd_vv_f32m2(c0, bv, vl); c1 = __riscv_vfadd_vv_f32m2(c1, bv, vl); }
            ST2(C, (size_t)m * ldc + n0, c0, vl);
            ST2(C, (size_t)(m + 1) * ldc + n0, c1, vl);
        }
        for (; m < m1; m++) {
            vfloat32m2_t c0 = init;
            const h16 *a0 = A + (size_t)m * lda, *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) c0 = __riscv_vfwmacc_vf_f32m2(c0, a0[k], __riscv_vle16_v_f16m1(b, vl), vl);
            if (addb) c0 = __riscv_vfadd_vv_f32m2(c0, bv, vl);
            ST2(C, (size_t)m * ldc + n0, c0, vl);
        }
        n0 += (int)vl;
    }
}

static void hgemm_m4(const h16 *A, int64_t lda, const h16 *B, int64_t ldb, void *C, int64_t ldc, int m0, int m1,
                     int K, int N, const h16 *bias, int bias_first, int out16)
{
    for (int n0 = 0; n0 < N;) {
        size_t vl = __riscv_vsetvl_e32m4(N - n0);
        vfloat32m4_t z = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t bv = bias ? __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(bias + n0, vl), vl) : z;
        vfloat32m4_t init = bias_first ? bv : z;
        const int addb = bias && !bias_first;
        int m = m0;
        for (; m + 4 <= m1; m += 4) {
            vfloat32m4_t c0 = init, c1 = init, c2 = init, c3 = init;
            const h16 *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *a2 = a1 + lda, *a3 = a2 + lda;
            const h16 *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) {
                vfloat16m2_t v = __riscv_vle16_v_f16m2(b, vl);
                c0 = __riscv_vfwmacc_vf_f32m4(c0, a0[k], v, vl);
                c1 = __riscv_vfwmacc_vf_f32m4(c1, a1[k], v, vl);
                c2 = __riscv_vfwmacc_vf_f32m4(c2, a2[k], v, vl);
                c3 = __riscv_vfwmacc_vf_f32m4(c3, a3[k], v, vl);
            }
            if (addb) {
                c0 = __riscv_vfadd_vv_f32m4(c0, bv, vl); c1 = __riscv_vfadd_vv_f32m4(c1, bv, vl);
                c2 = __riscv_vfadd_vv_f32m4(c2, bv, vl); c3 = __riscv_vfadd_vv_f32m4(c3, bv, vl);
            }
            ST4(C, (size_t)(m + 0) * ldc + n0, c0, vl); ST4(C, (size_t)(m + 1) * ldc + n0, c1, vl);
            ST4(C, (size_t)(m + 2) * ldc + n0, c2, vl); ST4(C, (size_t)(m + 3) * ldc + n0, c3, vl);
        }
        for (; m + 2 <= m1; m += 2) {
            vfloat32m4_t c0 = init, c1 = init;
            const h16 *a0 = A + (size_t)m * lda, *a1 = a0 + lda, *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) {
                vfloat16m2_t v = __riscv_vle16_v_f16m2(b, vl);
                c0 = __riscv_vfwmacc_vf_f32m4(c0, a0[k], v, vl);
                c1 = __riscv_vfwmacc_vf_f32m4(c1, a1[k], v, vl);
            }
            if (addb) { c0 = __riscv_vfadd_vv_f32m4(c0, bv, vl); c1 = __riscv_vfadd_vv_f32m4(c1, bv, vl); }
            ST4(C, (size_t)m * ldc + n0, c0, vl);
            ST4(C, (size_t)(m + 1) * ldc + n0, c1, vl);
        }
        for (; m < m1; m++) {
            vfloat32m4_t c0 = init;
            const h16 *a0 = A + (size_t)m * lda, *b = B + n0;
            for (int k = 0; k < K; k++, b += ldb) c0 = __riscv_vfwmacc_vf_f32m4(c0, a0[k], __riscv_vle16_v_f16m2(b, vl), vl);
            if (addb) c0 = __riscv_vfadd_vv_f32m4(c0, bv, vl);
            ST4(C, (size_t)m * ldc + n0, c0, vl);
        }
        n0 += (int)vl;
    }
}

static inline int hg_rows(int c) { return c == 2 ? 4 : 8; }
static inline int hg_nb(int c) { return c == 2 ? (int)__riscv_vsetvlmax_e32m4() : (int)__riscv_vsetvlmax_e32m2(); }

static void hgemm_c(int c, const h16 *A, int64_t lda, const h16 *B, int64_t ldb, void *C, int64_t ldc, int m0, int m1,
                    int K, int N, const h16 *bias, int bias_first, int out16)
{
    if (c == 2) hgemm_m4(A, lda, B, ldb, C, ldc, m0, m1, K, N, bias, bias_first, out16);
    else hgemm_m2(A, lda, B, ldb, C, ldc, m0, m1, K, N, bias, bias_first, out16);
}
/* both blockings give the same bytes (same k order): the choice is speed only */
#define hgemm(A, lda, B, ldb, C, ldc, m0, m1, K, N, bias, bf, o16) \
    hgemm_c(cfg_m((m1) - (m0)), A, lda, B, ldb, C, ldc, m0, m1, K, N, bias, bf, o16)

/* ---- linear: W [N, K] packed at load into [n-block][K][NB] (NB = the GEMM's vector length) */
typedef struct { h16 *wt; h16 *bias; int NB, nblk, N, K, cfg; } hlin_w;

static int hlinear_prepare(k3_op *op)
{
    const int K = (int)P(1), N = (int)P(2);
    if (!((P(K3_IP_WF16) & 1)) || (op->wp[1] && !((P(K3_IP_WF16) >> 1) & 1))) return 0;
    hlin_w *w = calloc(1, sizeof(hlin_w));
    w->cfg = cfg_m(P(0));
    w->NB = hg_nb(w->cfg);
    w->N = N;
    w->K = K;
    w->nblk = (N + w->NB - 1) / w->NB;
    w->wt = xalloc(sizeof(h16) * (size_t)w->nblk * w->NB * K);
    const h16 *src = op->wp[0];
    for (int nb = 0; nb < w->nblk; nb++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < w->NB; j++) {
                const int n = nb * w->NB + j;
                w->wt[((size_t)nb * K + k) * w->NB + j] = n < N ? src[(size_t)n * K + k] : (h16)0;
            }
    if (op->wp[1]) {
        w->bias = xalloc(sizeof(h16) * (size_t)w->nblk * w->NB);
        memcpy(w->bias, op->wp[1], sizeof(h16) * N);
    }
    op->prep.wpack = w;
    return 1;
}

typedef struct { k3_graph *g; k3_op *op; int mper, mgroups; } hlin_ctx;

static void hlinear_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hlin_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    hlin_w *w = op->prep.wpack;
    const int M = (int)P(0), K = w->K, N = w->N;
    const h16 *x = IN(0);
    h16 *y = OUT(0);
    for (int64_t u = u0; u < u1; u++) {
        const int nb = (int)(u / c->mgroups), mg = (int)(u % c->mgroups);
        const int m0 = mg * c->mper, m1 = m0 + c->mper < M ? m0 + c->mper : M;
        const int n0 = nb * w->NB, nn = N - n0 < w->NB ? N - n0 : w->NB;
        const h16 *bt = w->wt + (size_t)nb * K * w->NB;
        if (m0 >= m1) continue;
        hgemm_c(w->cfg, x, K, bt, w->NB, y + n0, N, m0, m1, K, nn, w->bias ? w->bias + n0 : NULL, 0, 1);
    }
}

static void hlinear_run(k3_graph *g, k3_op *op)
{
    hlin_w *w = op->prep.wpack;
    const int M = (int)P(0), nw = k3_pool_size(), R = hg_rows(w->cfg);
    hlin_ctx c = {g, op, M, 1};
    while (c.mgroups < (M + R - 1) / R && w->nblk * c.mgroups < 3 * nw) c.mgroups++;
    c.mper = ((M + c.mgroups - 1) / c.mgroups + R - 1) / R * R;
    c.mgroups = (M + c.mper - 1) / c.mper;
    const int64_t nu = (int64_t)w->nblk * c.mgroups;
    if (nw == 1 || nu < 2) hlinear_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, hlinear_range, &c);
}

/* ---- matmul A [M, K] x B [K, N] */
typedef struct { k3_graph *g; k3_op *op; } opctx;

static void hmatmul_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int M = (int)P(0), K = (int)P(1), N = (int)P(2);
    const int m0 = (int)u0 * 16, m1 = (int)u1 * 16 < M ? (int)u1 * 16 : M;
    hgemm(IN(0), K, IN(1), N, OUT(0), N, m0, m1, K, N, NULL, 0, 1);
}

/* ---- bmm_strided: unit k-stride A, unit n-stride C, B with unit n- or k-stride (the latter transposed) */
typedef struct { k3_graph *g; k3_op *op; int mper, mgroups; h16 *bt; } hbmm_ctx;

static void hbmm_tr_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hbmm_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *p = op->ip;
    const int64_t B1 = p[1], K = p[3], N = p[4];
    const h16 *b = IN(1);
    for (int64_t u = u0; u < u1; u++) {
        const int64_t x = u / B1, yy = u % B1;
        const h16 *Bm = b + p[10] + x * p[11] + yy * p[12];
        h16 *d = c->bt + u * K * N;
        for (int64_t n = 0; n < N; n++) {
            const h16 *src = Bm + n * p[14];
            for (int64_t k = 0; k < K; k++) d[k * N + n] = src[k];
        }
    }
}

static void hbmm_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hbmm_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *p = op->ip;
    const int64_t B1 = p[1], M = p[2], K = p[3], N = p[4];
    const h16 *a = IN(0), *b = IN(1);
    h16 *cc = OUT(0);
    for (int64_t u = u0; u < u1; u++) {
        const int64_t bt = u / c->mgroups, mg = u % c->mgroups;
        const int64_t x = bt / B1, yy = bt % B1;
        const h16 *A = a + p[5] + x * p[6] + yy * p[7];
        h16 *C = cc + p[15] + x * p[16] + yy * p[17];
        const int m0 = (int)(mg * c->mper), m1 = (int)(m0 + c->mper < M ? m0 + c->mper : M);
        if (c->bt) hgemm(A, p[8], c->bt + bt * K * N, N, C, p[18], m0, m1, (int)K, (int)N, NULL, 0, 1);
        else hgemm(A, p[8], b + p[10] + x * p[11] + yy * p[12], p[13], C, p[18], m0, m1, (int)K, (int)N, NULL, 0, 1);
    }
}

static int hbmm_ok(k3_op *op)
{
    const int64_t *p = op->ip;
    return p[9] == 1 && p[19] == 1 && (p[14] == 1 || p[13] == 1) && p[0] * p[1] * p[2] * p[4] == p[20];
}

static void hbmm_run(k3_graph *g, k3_op *op)
{
    const int64_t *p = op->ip;
    const int64_t nb = p[0] * p[1], M = p[2], K = p[3], N = p[4];
    const int nw = k3_pool_size(), R = hg_rows(cfg_m(M));
    hbmm_ctx c = {g, op, (int)M, 1, NULL};
    if (p[14] != 1) {
        c.bt = shared_scratch(2, sizeof(h16) * (size_t)(nb * K * N));
        par(nb, nb * K * N, hbmm_tr_range, &c);
    }
    const int64_t work = nb * M * N * K;
    if (nw > 1 && work >= 4 * k3_par_min)
        while (c.mgroups < (M + R - 1) / R && nb * c.mgroups < 3 * nw) c.mgroups++;
    c.mper = (int)(((M + c.mgroups - 1) / c.mgroups + R - 1) / R * R);
    c.mgroups = (int)((M + c.mper - 1) / c.mper);
    if (nw == 1 || work < 4 * k3_par_min) hbmm_range(&c, 0, nb * c.mgroups, 0);
    else k3_parallel(nb * c.mgroups, 1, hbmm_range, &c);
}

/* ---- conv2d (N = 1): im2col chunk [CR][KK] (fp16, exact) -> GEMM with W^T [KK][OC] -> fp32 tile -> NCHW */
typedef struct { h16 *wt; int KK; } hconv_w;
#define CR 32

static int hconv_prepare(k3_op *op)
{
    if (P(0) != 1 || !(P(K3_IP_WF16) & 1) || (op->wp[1] && !((P(K3_IP_WF16) >> 1) & 1))) return 0;
    const int IC = (int)P(1), OC = (int)P(4), KH = (int)P(7), KW = (int)P(8);
    hconv_w *w = calloc(1, sizeof(hconv_w));
    w->KK = IC * KH * KW;
    w->wt = xalloc(sizeof(h16) * (size_t)w->KK * OC);
    const h16 *src = op->wp[0];
    for (int oc = 0; oc < OC; oc++)
        for (int k = 0; k < w->KK; k++) w->wt[(size_t)k * OC + oc] = src[(size_t)oc * w->KK + k];
    op->prep.wpack = w;
    return 1;
}

/* units: (pixel chunk of CR, output-channel block of OCb): late layers (10 x 10 pixels) have few pixel chunks,
 * so the output channels are split too (each unit re-does its chunk's im2col, which is ~1/OCb of its GEMM) */
typedef struct { k3_graph *g; k3_op *op; int noc, ocb; } hconv_ctx;

static void hconv_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hconv_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    hconv_w *w = op->prep.wpack;
    const int IC = (int)P(1), IH = (int)P(2), IW = (int)P(3), OC = (int)P(4), OW = (int)P(6), KH = (int)P(7),
              KW = (int)P(8), SH = (int)P(9), SW = (int)P(10), PH = (int)P(11), PW = (int)P(12), DH = (int)P(13),
              DW = (int)P(14);
    const int OHW = (int)(P(5) * P(6)), KK = w->KK;
    const h16 *x = IN(0), *bias = op->wp[1];
    h16 *y = OUT(0);
    h16 *col = scratch(1, sizeof(h16) * (size_t)CR * KK);
    float *tmp = scratch(2, sizeof(float) * (size_t)CR * c->ocb);
    int last = -1;
    for (int64_t u = u0; u < u1; u++) {
        const int pc = (int)(u / c->noc), q = (int)(u % c->noc);
        const int r0 = pc * CR, r1 = r0 + CR < OHW ? r0 + CR : OHW, nr = r1 - r0;
        const int oc0 = q * c->ocb, oc1 = oc0 + c->ocb < OC ? oc0 + c->ocb : OC, noc = oc1 - oc0;
        if (pc != last) {
            for (int r = r0; r < r1; r++) {
                const int oh = r / OW, ow = r % OW;
                h16 *d = col + (size_t)(r - r0) * KK;
                for (int ic = 0; ic < IC; ic++)
                    for (int kh = 0; kh < KH; kh++) {
                        const int ih = oh * SH - PH + kh * DH;
                        const h16 *xr = x + ((size_t)ic * IH + ih) * IW;
                        if (ih < 0 || ih >= IH) {
                            for (int kw = 0; kw < KW; kw++) *d++ = (h16)0;
                            continue;
                        }
                        for (int kw = 0; kw < KW; kw++) {
                            const int iw = ow * SW - PW + kw * DW;
                            *d++ = (iw >= 0 && iw < IW) ? xr[iw] : (h16)0;
                        }
                    }
            }
            last = pc;
        }
        hgemm(col, KK, w->wt + oc0, OC, tmp, noc, 0, nr, KK, noc, bias ? bias + oc0 : NULL, 0, 0);
        for (int oc = 0; oc < noc; oc++) {
            h16 *yo = y + (size_t)(oc0 + oc) * OHW + r0;
            const float *t = tmp + oc;
            for (int r = 0; r < nr; r++) yo[r] = (h16)t[(size_t)r * noc];
        }
    }
}

static void hconv_run(k3_graph *g, k3_op *op)
{
    const int OHW = (int)(P(5) * P(6)), OC = (int)P(4), nw = k3_pool_size();
    const int npc = (OHW + CR - 1) / CR;
    int noc = 1;
    if (nw > 1 && npc < 4 * nw) {
        noc = (4 * nw + npc - 1) / npc;
        const int mx = OC / 64 > 1 ? OC / 64 : 1;          /* blocks of >= 64 output channels */
        if (noc > mx) noc = mx;
    }
    int ocb = (OC + noc - 1) / noc;
    ocb = (ocb + 31) / 32 * 32;
    noc = (OC + ocb - 1) / ocb;
    hconv_ctx c = {g, op, noc, ocb};
    const int64_t nu = (int64_t)npc * noc;
    if (nw == 1 || nu < 2) hconv_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, hconv_range, &c);
}

/* ---- conv_transpose2d, kernel == stride, no padding / dilation / groups: [HW, IC] x [IC, OC*KH*KW] with the
 * accumulators starting at the bias (the reference adds the products to the bias in ic order) */
typedef struct { h16 *bexp; } hct_w;
typedef struct { k3_graph *g; k3_op *op; h16 *xt; } hct_ctx;

static int hconvt_prepare(k3_op *op)
{
    if (!(P(0) == 1 && P(7) == P(9) && P(8) == P(10) && P(11) == 0 && P(12) == 0 && P(13) == 1 && P(14) == 1 &&
          P(15) == 1 && P(5) == P(2) * P(9) && P(6) == P(3) * P(10)))
        return 0;
    if (!(P(K3_IP_WF16) & 1) || (op->wp[1] && !((P(K3_IP_WF16) >> 1) & 1))) return 0;
    const int OC = (int)P(4), KH = (int)P(7), KW = (int)P(8);
    hct_w *w = calloc(1, sizeof(hct_w));
    w->bexp = xalloc(sizeof(h16) * (size_t)OC * KH * KW);
    const h16 *b = op->wp[1];
    for (int i = 0; i < OC * KH * KW; i++) w->bexp[i] = b ? b[i / (KH * KW)] : (h16)0;
    op->prep.wpack = w;
    return 1;
}

static void hconvt_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hct_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    hct_w *w = op->prep.wpack;
    const int IC = (int)P(1), IH = (int)P(2), IW = (int)P(3), OC = (int)P(4), OW = (int)P(6), KH = (int)P(7),
              KW = (int)P(8);
    const int NO = OC * KH * KW, OHW = (int)(P(5) * P(6));
    h16 *y = OUT(0);
    float *tmp = scratch(2, sizeof(float) * 16 * (size_t)NO);
    for (int64_t u = u0; u < u1; u++) {
        const int p0 = (int)u * 16, p1 = p0 + 16 < IH * IW ? p0 + 16 : IH * IW;
        hgemm(c->xt + (size_t)p0 * IC, IC, op->wp[0], NO, tmp, NO, 0, p1 - p0, IC, NO, w->bexp, 1, 0);
        for (int p = p0; p < p1; p++) {
            const int ih = p / IW, iw = p % IW;
            const float *t = tmp + (size_t)(p - p0) * NO;
            for (int oc = 0; oc < OC; oc++)
                for (int kh = 0; kh < KH; kh++)
                    for (int kw = 0; kw < KW; kw++)
                        y[(size_t)oc * OHW + (size_t)(ih * KH + kh) * OW + iw * KW + kw] = (h16)t[(oc * KH + kh) * KW + kw];
        }
    }
}

static void hconvt_run(k3_graph *g, k3_op *op)
{
    const int IC = (int)P(1), HW = (int)(P(2) * P(3));
    const h16 *x = IN(0);
    hct_ctx c = {g, op, shared_scratch(2, sizeof(h16) * (size_t)HW * IC)};
    for (int ic = 0; ic < IC; ic++)
        for (int p = 0; p < HW; p++) c.xt[(size_t)p * IC + ic] = x[(size_t)ic * HW + p];
    const int64_t nu = (HW + 15) / 16;
    if (k3_pool_size() == 1 || nu < 2) hconvt_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, hconvt_range, &c);
}

/* ======================================================================== fp32 vector math (kern_opt.c's) */
static inline vfloat32m4_t vexp(vfloat32m4_t x, size_t vl)
{
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

static inline vfloat32m4_t vtanh(vfloat32m4_t x, size_t vl)
{
    vfloat32m4_t a = __riscv_vfabs_v_f32m4(x, vl);
    vfloat32m4_t e = vexp(__riscv_vfmul_vf_f32m4(a, 2.0f, vl), vl);
    vfloat32m4_t t = __riscv_vfrsub_vf_f32m4(__riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(e, 1.0f, vl), 2.0f, vl), 1.0f, vl);
    return __riscv_vfsgnj_vv_f32m4(t, x, vl);
}

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

static inline vfloat32m4_t ld16(const h16 *p, size_t vl) { return __riscv_vfwcvt_f_f_v_f32m4(__riscv_vle16_v_f16m2(p, vl), vl); }
static inline void st16(h16 *p, vfloat32m4_t v, size_t vl) { __riscv_vse16_v_f16m2(p, __riscv_vfncvt_f_f_w_f16m2(v, vl), vl); }

/* ======================================================================== elementwise */
static void hew_range(void *vc, int64_t i0, int64_t i1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const h16 *a = IN(0), *b = op->nin > 1 ? IN(1) : NULL;
    h16 *y = OUT(0);
    const float sc = (float)op->fp[0];
    switch (op->kind) {
    case K_ADD: case K_MUL: case K_RELU:                 /* native binary16: equal to fp32 math + one rounding */
        for (int64_t i = i0; i < i1;) {
            size_t vl = __riscv_vsetvl_e16m8(i1 - i);
            vfloat16m8_t x = __riscv_vle16_v_f16m8(a + i, vl), r;
            if (op->kind == K_ADD) r = __riscv_vfadd_vv_f16m8(x, __riscv_vle16_v_f16m8(b + i, vl), vl);
            else if (op->kind == K_MUL) r = __riscv_vfmul_vv_f16m8(x, __riscv_vle16_v_f16m8(b + i, vl), vl);
            else r = __riscv_vfmax_vf_f16m8(x, (h16)0.0f, vl);
            __riscv_vse16_v_f16m8(y + i, r, vl);
            i += (int64_t)vl;
        }
        return;
    default: break;
    }
    for (int64_t i = i0; i < i1;) {
        size_t vl = __riscv_vsetvl_e32m4(i1 - i);
        vfloat32m4_t x = ld16(a + i, vl), r;
        switch (op->kind) {
        case K_MUL_SCALAR: r = __riscv_vfmul_vf_f32m4(x, sc, vl); break;
        case K_SILU_MUL: case K_SILU: case K_EWISE1: {
            vfloat32m4_t e = vexp(__riscv_vfneg_v_f32m4(x, vl), vl);
            r = __riscv_vfdiv_vv_f32m4(x, __riscv_vfadd_vf_f32m4(e, 1.0f, vl), vl);
            if (op->kind == K_SILU_MUL) r = __riscv_vfmul_vv_f32m4(r, ld16(b + i, vl), vl);
            break;
        }
        case K_GELU_EXACT: {
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
        case K_SIGMOID: r = __riscv_vfrdiv_vf_f32m4(__riscv_vfadd_vf_f32m4(vexp(__riscv_vfneg_v_f32m4(x, vl), vl), 1.0f, vl), 1.0f, vl); break;
        case K_SOFTPLUS: {
            vfloat32m4_t l = vlog(__riscv_vfadd_vf_f32m4(vexp(x, vl), 1.0f, vl), vl);
            r = __riscv_vmerge_vvm_f32m4(l, x, __riscv_vmfgt_vf_f32m4_b8(x, 20.0f, vl), vl);
            break;
        }
        default: r = x;
        }
        st16(y + i, r, vl);
        i += (int64_t)vl;
    }
}

/* ewise2 add / mul with innermost unit or zero strides: native binary16 */
static void hew2_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t *d = op->ip + 1, *sa = op->ip + 7, *sb = op->ip + 13;
    const h16 *a = IN(0), *b = IN(1);
    h16 *y = OUT(0);
    const int opc = (int)P(0);
    const int64_t n5 = d[5];
    for (int64_t r = r0; r < r1; r++) {
        int64_t t = r, idx[5];
        for (int k = 4; k >= 0; k--) { idx[k] = t % d[k]; t /= d[k]; }
        int64_t pa = 0, pb = 0;
        for (int k = 0; k < 5; k++) { pa += idx[k] * sa[k]; pb += idx[k] * sb[k]; }
        h16 *yo = y + r * n5;
        for (int64_t i = 0; i < n5;) {
            size_t vl = __riscv_vsetvl_e16m8(n5 - i);
            vfloat16m8_t va = sa[5] ? __riscv_vle16_v_f16m8(a + pa + i, vl) : __riscv_vfmv_v_f_f16m8(a[pa], vl);
            vfloat16m8_t vb = sb[5] ? __riscv_vle16_v_f16m8(b + pb + i, vl) : __riscv_vfmv_v_f_f16m8(b[pb], vl);
            vfloat16m8_t vr = opc == 0 ? __riscv_vfadd_vv_f16m8(va, vb, vl) : __riscv_vfmul_vv_f16m8(va, vb, vl);
            __riscv_vse16_v_f16m8(yo + i, vr, vl);
            i += (int64_t)vl;
        }
    }
}

/* ======================================================================== rows: softmax, norms */
static float osum(vfloat32m4_t acc, size_t vlm)
{
    return __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredosum_vs_f32m4_f32m1(acc, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
}

/* softmax of one fp32 row x (scaled by `scale`) into fp32 e (may alias x): max, exp, ordered sum, x 1/sum */
static void soft_row32(const float *x, float *e, int K, float scale)
{
    size_t vlm = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t vm = __riscv_vfmv_v_f_f32m4(-INFINITY, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vm = __riscv_vfmax_vv_f32m4_tu(vm, vm, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), scale, vl), vl);
        k += (int)vl;
    }
    const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(vm, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), vlm));
    vfloat32m4_t vs = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t t = vexp(__riscv_vfsub_vf_f32m4(__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), scale, vl), mx, vl), vl);
        if (mx == -INFINITY) t = __riscv_vfmv_v_f_f32m4(NAN, vl);
        __riscv_vse32_v_f32m4(e + k, t, vl);
        vs = __riscv_vfadd_vv_f32m4_tu(vs, vs, t, vl);
        k += (int)vl;
    }
    const float inv = 1.0f / osum(vs, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        __riscv_vse32_v_f32m4(e + k, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(e + k, vl), inv, vl), vl);
        k += (int)vl;
    }
}

static void hsoftmax_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const h16 *in = IN(0);
    h16 *out = OUT(0);
    float *row = scratch(3, sizeof(float) * (size_t)K);
    for (int64_t m = m0; m < m1; m++) {
        const h16 *x = in + (size_t)m * K;
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            __riscv_vse32_v_f32m4(row + k, ld16(x + k, vl), vl);
            k += (int)vl;
        }
        soft_row32(row, row, K, 1.0f);
        h16 *y = out + (size_t)m * K;
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            st16(y + k, __riscv_vle32_v_f32m4(row + k, vl), vl);
            k += (int)vl;
        }
    }
}

static float hsum(const h16 *x, int K, float sub, int sq)
{
    size_t vlm = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
    for (int k = 0; k < K;) {
        size_t vl = __riscv_vsetvl_e32m4(K - k);
        vfloat32m4_t v = __riscv_vfsub_vf_f32m4(ld16(x + k, vl), sub, vl);
        if (sq) v = __riscv_vfmul_vv_f32m4(v, v, vl);
        acc = __riscv_vfadd_vv_f32m4_tu(acc, acc, v, vl);
        k += (int)vl;
    }
    return osum(acc, vlm);
}

static void hnorm_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const float eps = (float)op->fp[0];
    const h16 *in = IN(0), *ga = op->wp[0], *be = op->wp[1];
    h16 *out = OUT(0);
    for (int64_t m = m0; m < m1; m++) {
        const h16 *x = in + (size_t)m * K;
        h16 *y = out + (size_t)m * K;
        if (op->kind == K_RMS_NORM) {
            const float inv = 1.0f / sqrtf(hsum(x, K, 0.0f, 1) / (float)K + eps);
            for (int k = 0; k < K;) {
                size_t vl = __riscv_vsetvl_e32m4(K - k);
                st16(y + k, __riscv_vfmul_vf_f32m4(ld16(x + k, vl), inv, vl), vl);
                k += (int)vl;
            }
        } else {
            const float mean = hsum(x, K, 0.0f, 0) / (float)K;
            const float var = hsum(x, K, mean, 1) / (float)K;
            const float rstd = 1.0f / sqrtf(var + eps);
            for (int k = 0; k < K;) {
                size_t vl = __riscv_vsetvl_e32m4(K - k);
                vfloat32m4_t v = __riscv_vfmul_vf_f32m4(__riscv_vfsub_vf_f32m4(ld16(x + k, vl), mean, vl), rstd, vl);
                if (ga) v = __riscv_vfmul_vv_f32m4(v, ld16(ga + k, vl), vl);
                if (be) v = __riscv_vfadd_vv_f32m4(v, ld16(be + k, vl), vl);
                st16(y + k, v, vl);
                k += (int)vl;
            }
        }
    }
}

/* ---- sdpa [BH, S, D]: q K^T (fp16 GEMM, fp32 scores), fp32 softmax, P V with V widened, one rounding */
typedef struct { k3_graph *g; k3_op *op; h16 *kt; } hsdpa_ctx;

static void hsdpa_tr(void *vc, int64_t b0, int64_t b1, int wid)
{
    (void)wid;
    hsdpa_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int S = (int)P(1), D = (int)P(2);
    const h16 *k = IN(1);
    for (int64_t b = b0; b < b1; b++)
        for (int j = 0; j < S; j++)
            for (int d = 0; d < D; d++) c->kt[((size_t)b * D + d) * S + j] = k[((size_t)b * S + j) * D + d];
}

static void hsdpa_range(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    hsdpa_ctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int S = (int)P(1), D = (int)P(2), nb = (S + 31) / 32;
    const h16 *q = IN(0), *v = IN(2);
    h16 *y = OUT(0);
    float *sc = scratch(1, sizeof(float) * 32 * (size_t)S);
    const float scale = 1.0f / sqrtf((float)D);
    for (int64_t u = u0; u < u1; u++) {
        const int b = (int)(u / nb), i0 = (int)(u % nb) * 32, i1 = i0 + 32 < S ? i0 + 32 : S;
        hgemm(q + (size_t)b * S * D + (size_t)i0 * D, D, c->kt + (size_t)b * D * S, S, sc, S, 0, i1 - i0, D, S, NULL, 0, 0);
        for (int i = 0; i < i1 - i0; i++) soft_row32(sc + (size_t)i * S, sc + (size_t)i * S, S, scale);
        const h16 *vb = v + (size_t)b * S * D;
        for (int i = 0; i < i1 - i0; i++) {
            const float *pr = sc + (size_t)i * S;
            h16 *yo = y + ((size_t)b * S + i0 + i) * D;
            for (int d0 = 0; d0 < D;) {
                size_t vl = __riscv_vsetvl_e32m4(D - d0);
                vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                for (int j = 0; j < S; j++) acc = __riscv_vfmacc_vf_f32m4(acc, pr[j], ld16(vb + (size_t)j * D + d0, vl), vl);
                st16(yo + d0, acc, vl);
                d0 += (int)vl;
            }
        }
    }
}

static void hsdpa_run(k3_graph *g, k3_op *op)
{
    const int BH = (int)P(0), S = (int)P(1), D = (int)P(2);
    hsdpa_ctx c = {g, op, shared_scratch(2, sizeof(h16) * (size_t)BH * S * D)};
    par(BH, (int64_t)BH * S * D, hsdpa_tr, &c);
    const int64_t nu = (int64_t)BH * ((S + 31) / 32);
    if (k3_pool_size() == 1) hsdpa_range(&c, 0, nu, 0);
    else k3_parallel(nu, 1, hsdpa_range, &c);
}

/* ---- rope_half: widened, the reference's fp32 products and sums in order, one rounding */
static void hrope_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int H = (int)P(1), half = (int)P(2);
    const h16 *in = IN(0), *ct = IN(1), *st = IN(2);
    h16 *out = OUT(0);
    for (int64_t r = r0; r < r1; r++)
        for (int h = 0; h < H; h++) {
            const h16 *x = in + ((size_t)r * H + h) * 2 * half;
            h16 *y = out + ((size_t)r * H + h) * 2 * half;
            for (int j = 0; j < half;) {
                size_t vl = __riscv_vsetvl_e32m4(half - j);
                vfloat32m4_t x1 = ld16(x + j, vl), x2 = ld16(x + half + j, vl);
                vfloat32m4_t cc = ld16(ct + (size_t)r * half + j, vl), ss = ld16(st + (size_t)r * half + j, vl);
                st16(y + j, __riscv_vfsub_vv_f32m4(__riscv_vfmul_vv_f32m4(x1, cc, vl), __riscv_vfmul_vv_f32m4(x2, ss, vl), vl), vl);
                st16(y + half + j, __riscv_vfadd_vv_f32m4(__riscv_vfmul_vv_f32m4(x2, cc, vl), __riscv_vfmul_vv_f32m4(x1, ss, vl), vl), vl);
                j += (int)vl;
            }
        }
}

/* ---- quantize_rows_s8 from a binary16 input: the fp32 kernel's arithmetic on the exact widened values */
static void hqrows_range(void *vc, int64_t m0, int64_t m1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int K = (int)P(1);
    const h16 *in = IN(0);
    const float *ci = IN(1);
    int8_t *out = OUT(0);
    float *rs = OUT(1);
    for (int64_t m = m0; m < m1; m++) {
        const h16 *x = in + (size_t)m * K;
        size_t vlm = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t vmax = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vfloat32m4_t v = __riscv_vfmul_vv_f32m4(ld16(x + k, vl), __riscv_vle32_v_f32m4(ci + k, vl), vl);
            vmax = __riscv_vfmax_vv_f32m4_tu(vmax, vmax, __riscv_vfabs_v_f32m4(v, vl), vl);
            k += (int)vl;
        }
        const float amax = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(vmax, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = 1.0f / s;
        rs[m] = s;
        for (int k = 0; k < K;) {
            size_t vl = __riscv_vsetvl_e32m4(K - k);
            vfloat32m4_t v = __riscv_vfmul_vv_f32m4(ld16(x + k, vl), __riscv_vle32_v_f32m4(ci + k, vl), vl);
            v = __riscv_vfmul_vf_f32m4(v, inv, vl);
            vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4_rm(v, __RISCV_FRM_RMM, vl);
            q = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(q, 127, vl), -127, vl);
            vint16m2_t q16 = __riscv_vnsra_wx_i16m2(q, 0, vl);
            __riscv_vse8_v_i8m1(out + (size_t)m * K + k, __riscv_vnsra_wx_i8m1(q16, 0, vl), vl);
            k += (int)vl;
        }
    }
}

/* ---- casts and conversions */
typedef struct { const void *src; void *dst; int dir; } cvt_ctx;      /* dir 0: f16 -> f32, 1: f32 -> f16 */

static void cvt_range(void *vc, int64_t i0, int64_t i1, int wid)
{
    (void)wid;
    cvt_ctx *c = vc;
    for (int64_t i = i0; i < i1;) {
        size_t vl = __riscv_vsetvl_e32m8(i1 - i);
        if (c->dir == 0)
            __riscv_vse32_v_f32m8((float *)c->dst + i, __riscv_vfwcvt_f_f_v_f32m8(__riscv_vle16_v_f16m4((const h16 *)c->src + i, vl), vl), vl);
        else
            __riscv_vse16_v_f16m4((h16 *)c->dst + i, __riscv_vfncvt_f_f_w_f16m4(__riscv_vle32_v_f32m8((const float *)c->src + i, vl), vl), vl);
        i += (int64_t)vl;
    }
}

static void convert(const void *src, void *dst, int64_t n, int dir)
{
    cvt_ctx c = {src, dst, dir};
    if (n < k3_par_min || k3_pool_size() == 1) cvt_range(&c, 0, n, 0);
    else k3_parallel(n, 16384, cvt_range, &c);
}

void k3_f16_convert(const void *src, void *dst, int64_t n, int dir) { convert(src, dst, n, dir); }

/* ---- int8 IME GEMM epilogue with binary16 outputs (called by kern_opt.c's block_epilogue, k3_f16_epi = 1).
 * yb: [ni][NBT][TM x TN] fp32 values y = dequant(acc) (+ bias), exactly the values kern_opt.c would store in
 * fp32. Each element leaves as fp16(addend + y) (addend widened exactly) or as addend + y in fp32 for an fp32
 * output: the arithmetic of dq_run's fp32 view and its conversion pass, without the pass. */
int k3_f16_epi = 1;
static inline int epi_out(const k3_epi16 *e, int n, int *ld, int *col)
{
    if (n < e->N0) { *ld = e->N0; *col = n; return 0; }
    if (n < e->N0 + e->N1) { *ld = e->N1; *col = n - e->N0; return 1; }
    *ld = e->N2; *col = n - e->N0 - e->N1; return 2;
}
void k3_f16_block_store(const k3_epi16 *e, const float *yb, int ni, int mt0, int nt0)
{
    const int TM = e->tm, TN = e->tn, tsz = TM * TN;
    for (int i = 0; i < ni; i++) {
        const int mt = mt0 + i, rows = e->M - mt * TM < TM ? e->M - mt * TM : TM;
        for (int j = 0; j < e->nbt; j++) {
            const int n0 = (nt0 + j) * TN;
            if (n0 >= e->N) break;
            const float *y = yb + (size_t)(i * e->nbt + j) * tsz;
            int ld, col, ld1, col1;
            const int o = epi_out(e, n0, &ld, &col), o1 = epi_out(e, n0 + TN - 1, &ld1, &col1);
            if (o == o1 && n0 + TN <= e->N) {          /* whole tile in one output: vector rows */
                const size_t vl = __riscv_vsetvl_e32m1(TN);
                for (int r = 0; r < rows; r++) {
                    const size_t m = (size_t)(mt * TM + r);
                    vfloat32m1_t v = __riscv_vle32_v_f32m1(y + r * TN, vl);
                    if (e->addend) {
                        vfloat32m1_t a = e->add16
                            ? __riscv_vfwcvt_f_f_v_f32m1(__riscv_vle16_v_f16mf2((const h16 *)e->addend + m * e->N + n0, vl), vl)
                            : __riscv_vle32_v_f32m1((const float *)e->addend + m * e->N + n0, vl);
                        v = __riscv_vfadd_vv_f32m1(a, v, vl);
                    }
                    if (e->is16[o]) __riscv_vse16_v_f16mf2((h16 *)e->out[o] + m * ld + col, __riscv_vfncvt_f_f_w_f16mf2(v, vl), vl);
                    else __riscv_vse32_v_f32m1((float *)e->out[o] + m * ld + col, v, vl);
                }
                continue;
            }
            for (int r = 0; r < rows; r++) {             /* tile straddles an output split or N */
                const size_t m = (size_t)(mt * TM + r);
                for (int q = 0; q < TN && n0 + q < e->N; q++) {
                    const int n = n0 + q, oo = epi_out(e, n, &ld, &col);
                    float v = y[r * TN + q];
                    if (e->addend) v = (e->add16 ? (float)((const h16 *)e->addend)[m * e->N + n] : ((const float *)e->addend)[m * e->N + n]) + v;
                    if (e->is16[oo]) ((h16 *)e->out[oo])[m * ld + col] = (h16)v;
                    else ((float *)e->out[oo])[m * ld + col] = v;
                }
            }
        }
    }
}

/* ---- int8 IME GEMMs with binary16 activations: kern_opt.c's kernel on fp32 views (k3_f16_epi = 0, and the
 * no-IME build), or with the fp16 block store above */
static void dq_run(k3_graph *g, k3_op *op)
{
#ifndef K3_NO_IME
    if (k3_f16_epi) { k3_opt_run(g, op); return; }
#endif
    k3_tensor *t[K3_NIN + K3_NOUT];
    void *orig[K3_NIN + K3_NOUT];
    int nt = 0;
    if (op->kind == K_LINEAR_DQ_ADD_S8 && g->t[op->in[2]].dtype == DT_F16) {
        k3_tensor *a = &g->t[op->in[2]];
        float *b = scratch(4, (size_t)a->nelem * 4 + 64);
        convert(a->data, b, a->nelem, 0);
        t[nt] = a; orig[nt] = a->data; nt++;
        a->data = b;
        a->dtype = DT_F32;
    }
    const int first_out = nt;
    for (int j = 0; j < op->nout; j++) {
        k3_tensor *o = &g->t[op->out[j]];
        if (o->dtype != DT_F16) continue;
        t[nt] = o; orig[nt] = o->data;
        o->data = shared_scratch(j < 3 ? j : 3, (size_t)o->nelem * 4 + 64);
        o->dtype = DT_F32;
        nt++;
    }
    k3_opt_run(g, op);
    for (int q = 0; q < nt; q++) {
        if (q >= first_out) convert(t[q]->data, orig[q], t[q]->nelem, 1);
        t[q]->data = orig[q];
        t[q]->dtype = DT_F16;
    }
}

/* ---- strided_copy of binary16 rows */
static void hscopy_range(void *vc, int64_t r0, int64_t r1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t off = P(0), *d = op->ip + 1, *s = op->ip + 7;
    const h16 *in = IN(0);
    h16 *out = OUT(0);
    for (int64_t r = r0; r < r1; r++) {
        int64_t t = r, b = off;
        for (int k = 4; k >= 0; k--) { b += (t % d[k]) * s[k]; t /= d[k]; }
        h16 *o = out + (size_t)r * d[5];
        if (s[5] == 1) memcpy(o, in + b, (size_t)d[5] * 2);
        else {
            const uint16_t *src = (const uint16_t *)in + b;
            uint16_t *dst = (uint16_t *)o;
            for (int64_t i = 0; i < d[5]; i++) dst[i] = src[i * s[5]];
        }
    }
}

/* ---- maxpool2d (binary16, exact) */
static void hmaxpool_range(void *vc, int64_t c0, int64_t c1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t IH = P(2), IW = P(3), OH = P(4), OW = P(5), KH = P(6), KW = P(7), SH = P(8), SW = P(9),
                  PH = P(10), PW = P(11);
    const h16 *x = IN(0);
    h16 *y = OUT(0);
    for (int64_t ch = c0; ch < c1; ch++) {
        const h16 *xc = x + ch * IH * IW;
        h16 *yc = y + ch * OH * OW;
        for (int64_t oh = 0; oh < OH; oh++) {
            if (SW == 1) {        /* rows of the window, vector over ow */
                for (int64_t ow0 = 0; ow0 < OW;) {
                    size_t vl = __riscv_vsetvl_e16m4(OW - ow0);
                    vfloat16m4_t m = __riscv_vfmv_v_f_f16m4((h16)-INFINITY, vl);
                    for (int64_t kh = 0; kh < KH; kh++) {
                        const int64_t ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int64_t kw = 0; kw < KW; kw++) {
                            const int64_t iw0 = ow0 - PW + kw;
                            /* lanes with iw out of range: masked load (padding never wins) */
                            int64_t lo = iw0 < 0 ? -iw0 : 0, hi = (int64_t)vl;
                            if (iw0 + hi > IW) hi = IW - iw0;
                            if (hi <= lo) continue;
                            if (lo == 0 && hi == (int64_t)vl) {
                                m = __riscv_vfmax_vv_f16m4(m, __riscv_vle16_v_f16m4(xc + ih * IW + iw0, vl), vl);
                            } else {
                                vbool4_t msk = __riscv_vmand_mm_b4(
                                    __riscv_vmsgeu_vx_u16m4_b4(__riscv_vid_v_u16m4(vl), (uint16_t)lo, vl),
                                    __riscv_vmsltu_vx_u16m4_b4(__riscv_vid_v_u16m4(vl), (uint16_t)hi, vl), vl);
                                vfloat16m4_t ld = __riscv_vle16_v_f16m4_mu(msk, __riscv_vfmv_v_f_f16m4((h16)-INFINITY, vl),
                                                                          xc + ih * IW + iw0, vl);
                                m = __riscv_vfmax_vv_f16m4(m, ld, vl);
                            }
                        }
                    }
                    __riscv_vse16_v_f16m4(yc + oh * OW + ow0, m, vl);
                    ow0 += (int64_t)vl;
                }
            } else {
                for (int64_t ow = 0; ow < OW; ow++) {
                    float m = -INFINITY;
                    for (int64_t kh = 0; kh < KH; kh++) {
                        const int64_t ih = oh * SH - PH + kh;
                        if (ih < 0 || ih >= IH) continue;
                        for (int64_t kw = 0; kw < KW; kw++) {
                            const int64_t iw = ow * SW - PW + kw;
                            if (iw < 0 || iw >= IW) continue;
                            const float v = (float)xc[ih * IW + iw];
                            m = v > m ? v : m;
                        }
                    }
                    yc[oh * OW + ow] = (h16)m;
                }
            }
        }
    }
}

/* ---- depthwise conv2d (G = IC = OC, N = 1, no padding: ViNT pads with explicit gathers): vector over output
 * columns, vfwmacc in the reference's kh, kw order, bias after, one rounding: byte-exact */
static int hdw_ok(k3_op *op)
{
    return P(0) == 1 && P(15) == P(1) && P(4) == P(1) && P(11) == 0 && P(12) == 0 && (P(K3_IP_WF16) & 1) &&
           (!op->wp[1] || ((P(K3_IP_WF16) >> 1) & 1));
}

static void hdw_range(void *vc, int64_t c0, int64_t c1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t IH = P(2), IW = P(3), OH = P(5), OW = P(6), KH = P(7), KW = P(8), SH = P(9), SW = P(10),
                  DH = P(13), DW = P(14);
    const h16 *x = IN(0), *W = op->wp[0], *bias = op->wp[1];
    h16 *y = OUT(0);
    for (int64_t ch = c0; ch < c1; ch++) {
        const h16 *wc = W + ch * KH * KW, *xc = x + ch * IH * IW;
        const float bb = bias ? (float)bias[ch] : 0.0f;
        for (int64_t oh = 0; oh < OH; oh++)
            for (int64_t ow0 = 0; ow0 < OW;) {
                size_t vl = __riscv_vsetvl_e32m4(OW - ow0);
                vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vl);
                for (int64_t kh = 0; kh < KH; kh++) {
                    const h16 *xr = xc + (oh * SH + kh * DH) * IW + ow0 * SW;
                    for (int64_t kw = 0; kw < KW; kw++) {
                        vfloat16m2_t v = SW == 1 ? __riscv_vle16_v_f16m2(xr + kw * DW, vl)
                                                 : __riscv_vlse16_v_f16m2(xr + kw * DW, SW * 2, vl);
                        acc = __riscv_vfwmacc_vf_f32m4(acc, wc[kh * KW + kw], v, vl);
                    }
                }
                if (bias) acc = __riscv_vfadd_vf_f32m4(acc, bb, vl);
                st16(y + (ch * OH + oh) * OW + ow0, acc, vl);
                ow0 += (int64_t)vl;
            }
    }
}

/* ---- im2col [IC, IH, IW] -> [OH*OW, IC*KH*KW] of binary16 elements */
static void him2col_range(void *vc, int64_t R0, int64_t R1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t IH = P(1), IW = P(2), OW = P(4), KH = P(5), KW = P(6), SH = P(7), SW = P(8),
                  PH = P(9), PW = P(10), DH = P(11), DW = P(12), IC = P(0), KK = IC * KH * KW;
    const uint16_t *x = IN(0);
    uint16_t *y = OUT(0);
    for (int64_t r = R0; r < R1; r++) {
        const int64_t oh = r / OW, ow = r % OW;
        uint16_t *d = y + r * KK;
        for (int64_t ic = 0; ic < IC; ic++)
            for (int64_t kh = 0; kh < KH; kh++) {
                const int64_t ih = oh * SH - PH + kh * DH;
                const uint16_t *xr = x + (ic * IH + ih) * IW;
                if (ih < 0 || ih >= IH) {
                    for (int64_t kw = 0; kw < KW; kw++) *d++ = 0;
                    continue;
                }
                for (int64_t kw = 0; kw < KW; kw++) {
                    const int64_t iw = ow * SW - PW + kw * DW;
                    *d++ = (iw >= 0 && iw < IW) ? xr[iw] : 0;
                }
            }
    }
}

/* ---- upsample nearest, integer factors (binary16 copies) */
static void hups_range(void *vc, int64_t c0, int64_t c1, int wid)
{
    (void)wid;
    opctx *c = vc;
    k3_graph *g = c->g;
    k3_op *op = c->op;
    const int64_t IH = P(1), IW = P(2), OH = P(3), OW = P(4);
    const uint16_t *x = IN(0);
    uint16_t *y = OUT(0);
    for (int64_t ch = c0; ch < c1; ch++)
        for (int64_t oh = 0; oh < OH; oh++) {
            const uint16_t *xr = x + (ch * IH + oh * IH / OH) * IW;
            uint16_t *yr = y + (ch * OH + oh) * OW;
            for (int64_t ow = 0; ow < OW; ow++) yr[ow] = xr[ow * IW / OW];
        }
}

/* ======================================================================== dispatch */
int k3_f16_exact_kind(int kind)
{
    switch (kind) {
    case K_LINEAR: case K_MATMUL: case K_BMM_STRIDED: case K_CONV2D: case K_CONV_T2D: case K_ADD: case K_MUL:
    case K_RELU: case K_EWISE2: case K_MUL_SCALAR: case K_ROPE_HALF: case K_QUANTIZE_ROWS_S8: case K_LINEAR_DQ_S8:
    case K_LINEAR3_DQ_S8: case K_LINEAR_DQ_ADD_S8: case K_CAST: case K_STRIDED_COPY: case K_MAXPOOL2D:
    case K_UPSAMPLE_NEAREST: case K_IM2COL: case K_CONV2D_GROUPED:
        return 1;
    default: return 0;
    }
}

static int all_f16(k3_graph *g, k3_op *op)
{
    for (int j = 0; j < op->nin; j++)
        if (g->t[op->in[j]].dtype != DT_F16) return 0;
    for (int j = 0; j < op->nout; j++)
        if (g->t[op->out[j]].dtype != DT_F16) return 0;
    return 1;
}

int k3_f16_prepare(k3_graph *g, k3_op *op)
{
    switch (op->kind) {
    case K_CAST: return 1;
    case K_LINEAR: return all_f16(g, op) && hlinear_prepare(op);
    case K_MATMUL: return all_f16(g, op);
    case K_BMM_STRIDED: return all_f16(g, op) && hbmm_ok(op);
    case K_CONV2D: return all_f16(g, op) && hconv_prepare(op);
    case K_CONV_T2D: return all_f16(g, op) && hconvt_prepare(op);
    case K_SDPA: return all_f16(g, op);
    case K_CONV2D_GROUPED: return all_f16(g, op) && hdw_ok(op);
    case K_SOFTMAX: case K_ROPE_HALF: case K_SILU_MUL: case K_GELU_EXACT: case K_GELU: case K_ADD: case K_MUL:
    case K_MUL_SCALAR: case K_SILU: case K_TANH: case K_SOFTPLUS: case K_RELU: case K_SIGMOID: case K_STRIDED_COPY:
    case K_MAXPOOL2D: case K_UPSAMPLE_NEAREST: case K_IM2COL:
        return all_f16(g, op);
    case K_LAYER_NORM: case K_RMS_NORM:
        return all_f16(g, op) && (!op->wp[0] || (P(K3_IP_WF16) & 1)) && (!op->wp[1] || ((P(K3_IP_WF16) >> 1) & 1));
    case K_EWISE1: return all_f16(g, op) && P(0) == 14;
    case K_EWISE2: return all_f16(g, op) && (P(0) == 0 || P(0) == 2) && op->fp[0] == 1.0 &&
                          (op->ip[7 + 5] == 0 || op->ip[7 + 5] == 1) && (op->ip[13 + 5] == 0 || op->ip[13 + 5] == 1);
    case K_QUANTIZE_ROWS_S8: return g->t[op->in[0]].dtype == DT_F16;
    case K_LINEAR_DQ_S8: case K_LINEAR3_DQ_S8: case K_LINEAR_DQ_ADD_S8: return k3_opt_prepare(g, op);
    default: return 0;
    }
}

void k3_f16_run(k3_graph *g, k3_op *op)
{
    opctx c = {g, op};
    switch (op->kind) {
    case K_CAST: {
        const k3_tensor *a = &g->t[op->in[0]], *b = &g->t[op->out[0]];
        if (a->dtype == b->dtype) memcpy(b->data, a->data, (size_t)P(0) * k3_esz(a->dtype));
        else convert(a->data, b->data, P(0), a->dtype == DT_F32);
        break;
    }
    case K_LINEAR: hlinear_run(g, op); break;
    case K_MATMUL: par((P(0) + 15) / 16, P(0) * P(1) * P(2) / 8, hmatmul_range, &c); break;
    case K_BMM_STRIDED: hbmm_run(g, op); break;
    case K_CONV2D: hconv_run(g, op); break;
    case K_CONV_T2D: hconvt_run(g, op); break;
    case K_SDPA: hsdpa_run(g, op); break;
    case K_SOFTMAX: par(P(0), P(0) * P(1), hsoftmax_range, &c); break;
    case K_LAYER_NORM: case K_RMS_NORM: par(P(0), P(0) * P(1), hnorm_range, &c); break;
    case K_ROPE_HALF: par(P(0), P(0) * P(1) * P(2) * 2, hrope_range, &c); break;
    case K_EWISE1: if (P(1) < k3_par_min) hew_range(&c, 0, P(1), 0); else k3_parallel(P(1), 8192, hew_range, &c); break;
    case K_SILU_MUL: case K_GELU_EXACT: case K_ADD: case K_MUL_SCALAR: case K_SILU: case K_GELU: case K_TANH:
    case K_SOFTPLUS: case K_MUL: case K_RELU: case K_SIGMOID:
        if (P(0) < k3_par_min) hew_range(&c, 0, P(0), 0); else k3_parallel(P(0), 8192, hew_range, &c);
        break;
    case K_EWISE2: {
        const int64_t *d = op->ip + 1;
        par(d[0] * d[1] * d[2] * d[3] * d[4], d[0] * d[1] * d[2] * d[3] * d[4] * d[5], hew2_range, &c);
        break;
    }
    case K_STRIDED_COPY: {
        const int64_t *d = op->ip + 1;
        par(d[0] * d[1] * d[2] * d[3] * d[4], d[0] * d[1] * d[2] * d[3] * d[4] * d[5], hscopy_range, &c);
        break;
    }
    case K_QUANTIZE_ROWS_S8: par(P(0), P(0) * P(1), hqrows_range, &c); break;
    case K_LINEAR_DQ_S8: case K_LINEAR3_DQ_S8: case K_LINEAR_DQ_ADD_S8: dq_run(g, op); break;
    case K_MAXPOOL2D: par(P(0) * P(1), P(0) * P(1) * P(4) * P(5) * P(6) * P(7), hmaxpool_range, &c); break;
    case K_UPSAMPLE_NEAREST: par(P(0), P(0) * P(3) * P(4), hups_range, &c); break;
    case K_IM2COL: par(P(3) * P(4), P(3) * P(4) * P(0) * P(5) * P(6), him2col_range, &c); break;
    case K_CONV2D_GROUPED: par(P(1), P(1) * P(5) * P(6) * P(7) * P(8), hdw_range, &c); break;
    default: k3_ref_run(g, op);
    }
}
