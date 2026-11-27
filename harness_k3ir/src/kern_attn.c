/* Fused attention for the K3 IR runtime.
 *
 * Pattern (found at load time, k3_attn_fuse): S = bmm_strided(Q, K^T) [-> mul_scalar c] [-> ewise2 add mask]
 * -> softmax (rows) -> O = bmm_strided(P, V), every intermediate read only inside the chain; and every sdpa
 * op. The chain runs as one op at the position of the final bmm (all inputs exist by then); the other chain
 * ops are skipped (impl 2). No score matrix is materialised: each unit is one (batch, row block) and keeps
 * its scores in L1/L2.
 *
 * mode 1 (fp32): Q K^T and P V with the RVV fp32 GEMM, accurate exp; held to the fp32 slot bound.
 * mode 3 (int8): Q rows quantised per row (absmax / 127); K smoothed (the per-channel mean over keys is
 *   subtracted: softmax is invariant to the per-query constant q . kmean this removes) and quantised per
 *   key; Q K^T on the IME (s8 x s8). Scores t = acc sq sk c log2e + mask log2e, softmax in base 2 with a
 *   degree-4 exp2; the unnormalised probabilities e in (0, 1] quantised as u8 = round(255 e); V quantised
 *   per channel over keys; P V on the IME (u8 x s8); O = acc sv / (255 sum e). An fp32-island numerics
 *   change, checked by the per-model accuracy metrics, not by bit-exactness.
 */
#include <math.h>
#include <riscv_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3ir.h"

int k3_attn_mode = 0;
int k3_attn_i8_min = 128;     /* --attn 5: int8 path when N >= this, fp32 fused below */
static int g_dbg = -1;          /* K3_ATTN_DBG bits: 1 no K smoothing, 2 fp32 P V (int8 Q K^T only) */
static uint64_t g_tprep, g_tmask, g_tunit;
static uint64_t g_ph[16][6];    /* per worker: q, qk, softmax, ppack, pv */
static long g_ncall;

#ifndef K3_NO_IME
__asm__(".macro k3a_vmadot vd, vs1, vs2\n"
        " .word (0x2b | (\\vd<<7) | (3<<12) | (\\vs1<<15) | (\\vs2<<20) | (0x71<<25))\n"
        ".endm\n"
        ".macro k3a_vmadotus vd, vs1, vs2\n"            /* funct3 1: A unsigned x B signed */
        " .word (0x2b | (\\vd<<7) | (1<<12) | (\\vs1<<15) | (\\vs2<<20) | (0x71<<25))\n"
        ".endm\n"
        ".irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"
        " .set v\\n, \\n\n"
        ".endr\n");
#define VMADOT(c, a, b) __asm__ volatile("k3a_vmadot %0, %1, %2" : "+vr"(c) : "vr"(a), "vr"(b))
#define VMADOTUS(c, a, b) __asm__ volatile("k3a_vmadotus %0, %1, %2" : "+vr"(c) : "vr"(a), "vr"(b))
#endif

#define NBT 4
#define MBT 2
static __thread int TM, TN, TK;

static void tiles(void)
{
    if (TM) return;
    size_t vb = __riscv_vlenb();
    if (vb == 128) { TM = 8; TN = 8; TK = 16; }
    else if (vb == 32) { TM = 4; TN = 4; TK = 8; }
}

static void *xalloc(size_t n)
{
    void *p = aligned_alloc(64, (n + 127) & ~(size_t)63);
    memset(p, 0, (n + 127) & ~(size_t)63);
    return p;
}

static __thread void *scr_p[8];
static __thread size_t scr_n[8];
static void *scratch(int i, size_t n)
{
    if (scr_n[i] < n) {
        free(scr_p[i]);
        scr_n[i] = (n + 4095) & ~(size_t)4095;
        scr_p[i] = aligned_alloc(64, scr_n[i]);
    }
    return scr_p[i];
}

typedef struct {
    int B0, B1, M, N, D, DV, NB;
    int qt, kt, vt, ot, mt;                 /* tensor ids; mt -1: no mask */
    int64_t qo, qs0, qs1, qsm, ko, ks0, ks1, ksn, vo, vs0, vs1, vsn, oo, os0, os1, osm;
    float scale;
    int64_t *mrow, msn;                     /* mask: element (row r, key n) at mrow[r] + n msn */
    int mask_shared;                        /* mask rows identical for every batch */
    int ops[5], oimpl[5], nops, mode;
    /* int8 geometry and per-call buffers */
    int RB, nrb, Npad, NKT, Dp, ksq, ksp, DVp, ndvb;
    int8_t *k8;                             /* [NB][Npad*Dp] IME B layout */
    float *skp;                             /* [NB][NKT][T] */
    int8_t *v8;                             /* [NB][DVp*Npad] IME B layout (n = dv, k = key) */
    float *svp;                             /* [NB][DVp/TN][T] */
    float *kmean, *vsc, *vinv;              /* [NB][Dp] K channel means; [NB][DVp] V scales and inverses */
    float *mp;                              /* [nmask][MBT][NKT][T] additive pattern (log2 domain), or [NKT][T] */
    int nmask;
    float *kt32;                            /* fp32 mode: [NB][D][N] */
    /* binary16 operands (fp16 / mixed builds): Q, K, V, mask, O widened to fp32 views per call (exact), the
     * fused fp32 or int8 kernel runs on the views, O rounded once to fp16. The chain's fp16 intermediates
     * (scores, probabilities) stay fp32 inside the fused op: a numerics change held to the --attn-check
     * bound, like the int8 modes. */
    int f16, nh, hid[5];
    float *hbuf[5];
} attn_d;

/* ======================================================================== matching */
static int64_t tbytes(k3_graph *g, int id) { return g->t[id].nelem * (int64_t)k3_esz(g->t[id].dtype); }
static int overlap(k3_graph *g, int a, int b)
{
    const char *pa = g->t[a].data, *pb = g->t[b].data;
    return pa < pb + tbytes(g, b) && pb < pa + tbytes(g, a);
}

/* the next op after `from` that reads tensor `t` (same data pointer and size); -1 if none */
static int next_reader(k3_graph *g, int from, int t, int *slot)
{
    for (int j = from + 1; j < g->n_ops; j++)
        for (int s = 0; s < g->ops[j].nin; s++) {
            int u = g->ops[j].in[s];
            if (g->t[u].data == g->t[t].data && g->t[u].nelem == g->t[t].nelem) { *slot = s; return j; }
        }
    return -1;
}

static int bmm_contig_out(const int64_t *p)        /* C = [B0][B1][M][N] dense */
{
    const int64_t B1 = p[1], M = p[2], N = p[4];
    return p[15] == 0 && p[19] == 1 && p[18] == N && (B1 == 1 || p[17] == M * N) &&
           (p[0] == 1 || p[16] == B1 * M * N) && p[20] == p[0] * B1 * M * N;
}

static attn_d *match_chain(k3_graph *g, int i0)
{
    k3_op *b1 = &g->ops[i0];
    const int64_t *p = b1->ip;
    if (b1->kind != K_BMM_STRIDED || p[9] != 1 || p[13] != 1 || !bmm_contig_out(p)) return NULL;
    attn_d a = {0};
    a.B0 = (int)p[0]; a.B1 = (int)p[1]; a.M = (int)p[2]; a.D = (int)p[3]; a.N = (int)p[4];
    a.qt = b1->in[0]; a.kt = b1->in[1];
    a.qo = p[5]; a.qs0 = p[6]; a.qs1 = p[7]; a.qsm = p[8];
    a.ko = p[10]; a.ks0 = p[11]; a.ks1 = p[12]; a.ksn = p[14];
    a.scale = 1.0f;
    a.mt = -1;
    a.ops[a.nops++] = i0;
    int cur = b1->out[0], j, slot;
    int64_t rows = (int64_t)a.B0 * a.B1 * a.M;
    for (;;) {
        j = next_reader(g, a.ops[a.nops - 1], cur, &slot);
        if (j < 0 || a.nops >= 4) return NULL;
        k3_op *o = &g->ops[j];
        if (o->kind == K_MUL_SCALAR && o->ip[0] == rows * a.N && a.nops == 1) {
            a.scale = (float)o->fp[0];
        } else if (o->kind == K_EWISE2 && o->ip[0] == 0 && o->fp[0] == 1.0 && a.mt < 0) {
            const int64_t *d = o->ip + 1, *sa = o->ip + 7 + 6 * slot, *sb = o->ip + 7 + 6 * (1 - slot);
            int64_t n = 1, st = 1;
            for (int k = 5; k >= 0; k--) {                   /* chain operand dense in the 6-d space */
                if (d[k] > 1 && sa[k] != st) return NULL;
                st *= d[k];
                n *= d[k];
            }
            if (n != rows * a.N || d[5] % a.N || (sb[5] != 0 && sb[5] != 1)) return NULL;
            a.mt = o->in[1 - slot];
            a.msn = sb[5];
            a.mrow = malloc(sizeof(int64_t) * rows);
            for (int64_t r = 0; r < rows; r++) {
                int64_t f = r * a.N, i5 = f % d[5], t = f / d[5], off = i5 * sb[5];
                for (int k = 4; k >= 0; k--) { off += (t % d[k]) * sb[k]; t /= d[k]; }
                a.mrow[r] = off;
            }
            a.mask_shared = 1;
            for (int64_t r = 0; r < rows; r++)
                if (a.mrow[r] != a.mrow[r % a.M]) { a.mask_shared = 0; break; }
        } else if (o->kind == K_SOFTMAX && o->ip[0] == rows && o->ip[1] == a.N) {
            a.ops[a.nops++] = j;
            cur = o->out[0];
            break;
        } else return NULL;
        a.ops[a.nops++] = j;
        cur = o->out[0];
    }
    /* final bmm: A = the softmax output, dense [B0][B1][M][N]; V with unit dv stride; O unit dv stride */
    int jb = -1;
    for (int k = a.ops[a.nops - 1] + 1; k < g->n_ops && jb < 0; k++) {
        k3_op *o = &g->ops[k];
        for (int s = 0; s < o->nin; s++)
            if (overlap(g, o->in[s], cur)) {
                if (o->kind != K_BMM_STRIDED || s != 0) return NULL;
                jb = k;
                break;
            }
    }
    if (jb < 0) return NULL;
    k3_op *b2 = &g->ops[jb];
    const int64_t *q = b2->ip;
    const char *pa = (const char *)g->t[b2->in[0]].data + q[5] * (int64_t)k3_esz(g->t[b2->in[0]].dtype);
    if (pa != (const char *)g->t[cur].data || q[0] != a.B0 || q[1] != a.B1 || q[2] != a.M || q[3] != a.N ||
        q[9] != 1 || q[8] != a.N || (a.B1 > 1 && q[7] != (int64_t)a.M * a.N) ||
        (a.B0 > 1 && q[6] != (int64_t)a.B1 * a.M * a.N) || q[14] != 1 || q[19] != 1)
        return NULL;
    a.DV = (int)q[4];
    a.vt = b2->in[1]; a.vo = q[10]; a.vs0 = q[11]; a.vs1 = q[12]; a.vsn = q[13];
    a.ot = b2->out[0]; a.oo = q[15]; a.os0 = q[16]; a.os1 = q[17]; a.osm = q[18];
    a.ops[a.nops++] = jb;
    /* intermediates (every chain op's output but the last) read by nothing outside the chain, not outputs;
     * Q, K, V, mask not written between the first and the last chain op */
    for (int c = 0; c < a.nops - 1; c++) {
        int t = g->ops[a.ops[c]].out[0];
        for (int k = 0; k < g->n_ops; k++) {
            int inchain = 0;
            for (int e = 0; e < a.nops; e++) inchain |= a.ops[e] == k;
            if (inchain) continue;
            for (int s = 0; s < g->ops[k].nin; s++)
                if (overlap(g, g->ops[k].in[s], t)) return NULL;
        }
        for (int s = 0; s < g->n_out; s++)
            if (overlap(g, g->outputs[s], t)) return NULL;
    }
    for (int k = a.ops[0] + 1; k < jb; k++) {
        int inchain = 0;
        for (int e = 0; e < a.nops; e++) inchain |= a.ops[e] == k;
        if (inchain) continue;
        for (int s = 0; s < g->ops[k].nout; s++) {
            int w = g->ops[k].out[s];
            if (overlap(g, w, a.qt) || overlap(g, w, a.kt) || overlap(g, w, a.vt) || (a.mt >= 0 && overlap(g, w, a.mt)))
                return NULL;
        }
    }
    attn_d *r = malloc(sizeof a);
    *r = a;
    return r;
}

static attn_d *match_sdpa(k3_graph *g, int i)
{
    k3_op *o = &g->ops[i];
    attn_d a = {0};
    const int BH = (int)o->ip[0], S = (int)o->ip[1], D = (int)o->ip[2];
    a.B0 = 1; a.B1 = BH; a.M = S; a.N = S; a.D = D; a.DV = D;
    a.qt = o->in[0]; a.kt = o->in[1]; a.vt = o->in[2]; a.ot = o->out[0]; a.mt = -1;
    a.qs1 = a.ks1 = a.vs1 = a.os1 = (int64_t)S * D;
    a.qsm = a.ksn = a.vsn = a.osm = D;
    a.scale = 1.0f / sqrtf((float)D);
    a.ops[a.nops++] = i;
    attn_d *r = malloc(sizeof a);
    *r = a;
    return r;
}

static void geometry(attn_d *a)
{
    tiles();
    a->NB = a->B0 * a->B1;
    if (a->mode >= 3 && TM) {
        const int T = TM * TN;
        a->RB = MBT * TM;
        a->nrb = (a->M + a->RB - 1) / a->RB;
        a->Npad = (a->N + NBT * TN - 1) / (NBT * TN) * (NBT * TN);
        a->NKT = a->Npad / TN;
        a->Dp = (a->D + TK - 1) / TK * TK;
        a->ksq = a->Dp / TK;
        a->ksp = a->Npad / TK;
        a->DVp = (a->DV + NBT * TN - 1) / (NBT * TN) * (NBT * TN);
        a->ndvb = a->DVp / (NBT * TN);
        a->k8 = xalloc((size_t)a->NB * a->Npad * a->Dp);
        a->skp = xalloc(sizeof(float) * a->NB * a->NKT * T);
        a->v8 = xalloc((size_t)a->NB * a->DVp * a->Npad);
        a->svp = xalloc(sizeof(float) * a->NB * (a->DVp / TN) * T);
        a->kmean = xalloc(sizeof(float) * (size_t)a->NB * a->Dp);
        a->vsc = xalloc(sizeof(float) * (size_t)a->NB * a->DVp);
        a->vinv = xalloc(sizeof(float) * (size_t)a->NB * a->DVp);
        a->nmask = a->mt < 0 ? 0 : (a->mask_shared ? 1 : a->NB) * a->nrb;
        if (a->mt < 0) {                            /* padding keys only */
            a->mp = xalloc(sizeof(float) * a->NKT * T);
            for (int kt = 0; kt < a->NKT; kt++)
                for (int e = 0; e < T; e++) a->mp[kt * T + e] = kt * TN + e % TN < a->N ? 0.0f : -INFINITY;
        } else {
            a->mp = xalloc(sizeof(float) * (size_t)a->nmask * MBT * a->NKT * T);
        }
    } else {
        a->mode = 1;
        a->RB = 32;
        a->nrb = (a->M + 31) / 32;
        a->kt32 = xalloc(sizeof(float) * (size_t)a->NB * a->D * a->N);
    }
}

int k3_attn_fuse(k3_graph *g)
{
    if (!k3_attn_mode) return 0;
    int n = 0;
    for (int i = 0; i < g->n_ops; i++) {
        attn_d *a = NULL;
        if (g->ops[i].kind == K_SDPA) a = match_sdpa(g, i);
        else if (g->ops[i].kind == K_BMM_STRIDED && g->ops[i].impl != 2) a = match_chain(g, i);
        if (!a) continue;
        a->mode = k3_attn_mode;
        if (k3_attn_mode == 5) a->mode = (int64_t)a->N >= k3_attn_i8_min ? 3 : 1;   /* auto: int8 for long key sets */
        if (k3_attn_mode == 7) a->mode = (int64_t)a->N >= k3_attn_i8_min ? 6 : 1;   /* auto with 16-bit Q */
#ifdef K3_NO_IME
        a->mode = 1;
#endif
        geometry(a);
        {
            const int ids[5] = {a->qt, a->kt, a->vt, a->mt, a->ot};
            for (int k = 0; k < 5; k++) {
                if (ids[k] < 0 || g->t[ids[k]].dtype != DT_F16) continue;
                int dup = 0;
                for (int h = 0; h < a->nh; h++) dup |= a->hid[h] == ids[k];
                if (dup) continue;
                a->hid[a->nh] = ids[k];
                a->hbuf[a->nh++] = xalloc(sizeof(float) * (size_t)g->t[ids[k]].nelem);
            }
            a->f16 = a->nh > 0;
        }
        for (int e = 0; e < a->nops; e++) a->oimpl[e] = g->ops[a->ops[e]].impl;
        for (int e = 0; e < a->nops - 1; e++) g->ops[a->ops[e]].impl = 2;
        k3_op *h = &g->ops[a->ops[a->nops - 1]];
        h->impl = 3;
        h->prep.wpack = a;
        n++;
    }
    return n;
}

void k3_attn_report(k3_graph *g, const char *tag)
{
    int n = 0, first = 1;
    for (int i = 0; i < g->n_ops; i++)
        if (g->ops[i].impl == 3) {
            attn_d *a = g->ops[i].prep.wpack;
            if (first)
                printf("ATTN %s mode=%d B=%dx%d M=%d N=%d D=%d DV=%d scale=%g mask=%d shared=%d f16=%d ops=", tag, a->mode,
                       a->B0, a->B1, a->M, a->N, a->D, a->DV, a->scale, a->mt >= 0, a->mask_shared, a->f16);
            if (first) for (int e = 0; e < a->nops; e++) printf("%s%s", e ? "," : "", k3_kind_names[g->ops[a->ops[e]].kind]);
            if (first) printf("\n");
            first = 0;
            n++;
        }
    printf("ATTN %s fused_groups=%d\n", tag, n);
}

/* ======================================================================== shared math */
static inline vfloat32m4_t vexp_acc(vfloat32m4_t x, size_t vl)     /* as kern_opt.c's vexp */
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

/* 2^t for t <= 0 (t clamped at -126), degree-4 Taylor on [-1/2, 1/2]: relative error < 5e-5 */
static inline vfloat32m4_t vexp2_fast(vfloat32m4_t t, size_t vl)
{
    t = __riscv_vfmax_vf_f32m4(t, -125.0f, vl);
    vint32m4_t n = __riscv_vfcvt_x_f_v_i32m4_rm(t, __RISCV_FRM_RNE, vl);
    vfloat32m4_t f = __riscv_vfsub_vv_f32m4(t, __riscv_vfcvt_f_x_v_f32m4(n, vl), vl);
    vfloat32m4_t p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vf_f32m4(f, 9.6181291e-3f, vl), 5.5504109e-2f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 2.4022651e-1f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 6.9314718e-1f, vl);
    p = __riscv_vfadd_vf_f32m4(__riscv_vfmul_vv_f32m4(p, f, vl), 1.0f, vl);
    vint32m4_t bits = __riscv_vadd_vv_i32m4(__riscv_vreinterpret_v_f32m4_i32m4(p), __riscv_vsll_vx_i32m4(n, 23, vl), vl);
    return __riscv_vreinterpret_v_i32m4_f32m4(bits);
}

static inline int64_t qbase(attn_d *a, int bt) { return a->qo + (int64_t)(bt / a->B1) * a->qs0 + (int64_t)(bt % a->B1) * a->qs1; }
static inline int64_t kbase(attn_d *a, int bt) { return a->ko + (int64_t)(bt / a->B1) * a->ks0 + (int64_t)(bt % a->B1) * a->ks1; }
static inline int64_t vbase(attn_d *a, int bt) { return a->vo + (int64_t)(bt / a->B1) * a->vs0 + (int64_t)(bt % a->B1) * a->vs1; }
static inline int64_t obase(attn_d *a, int bt) { return a->oo + (int64_t)(bt / a->B1) * a->os0 + (int64_t)(bt % a->B1) * a->os1; }

typedef struct { k3_graph *g; attn_d *a; } actx;

/* ======================================================================== fp32 fused */
static void f32_tr(void *vc, int64_t b0, int64_t b1, int wid)
{
    (void)wid;
    actx *c = vc;
    attn_d *a = c->a;
    const float *k = k3_ptr(c->g, a->kt);
    for (int64_t bt = b0; bt < b1; bt++) {
        const float *K = k + kbase(a, (int)bt);
        float *d = a->kt32 + bt * a->D * a->N;
        for (int n = 0; n < a->N; n++)
            for (int j = 0; j < a->D; j++) d[(size_t)j * a->N + n] = K[n * a->ksn + j];
    }
}

static void f32_unit(void *vc, int64_t u0, int64_t u1, int wid)
{
    (void)wid;
    actx *c = vc;
    attn_d *a = c->a;
    k3_graph *g = c->g;
    const float *q = k3_ptr(g, a->qt), *v = k3_ptr(g, a->vt), *mk = a->mt >= 0 ? k3_ptr(g, a->mt) : NULL;
    float *o = k3_ptr(g, a->ot);
    const int N = a->N;
    float *S = scratch(0, sizeof(float) * 32 * (size_t)N);
    for (int64_t u = u0; u < u1; u++) {
        const int bt = (int)(u / a->nrb), i0 = (int)(u % a->nrb) * 32, i1 = i0 + 32 < a->M ? i0 + 32 : a->M;
        const int nr = i1 - i0;
        k3_sgemm_rows(q + qbase(a, bt) + (int64_t)i0 * a->qsm, a->qsm, a->kt32 + (size_t)bt * a->D * N, N, S, N, 0, nr,
                      a->D, N, NULL, 0);
        for (int r = 0; r < nr; r++) {
            float *y = S + (size_t)r * N;
            const float *mr = mk ? mk + a->mrow[(int64_t)bt * a->M + i0 + r] : NULL;
            size_t vlm = __riscv_vsetvlmax_e32m4();
            vfloat32m4_t vm = __riscv_vfmv_v_f_f32m4(-INFINITY, vlm);
            for (int k = 0; k < N;) {
                size_t vl = __riscv_vsetvl_e32m4(N - k);
                vfloat32m4_t t = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), a->scale, vl);
                if (mr) t = __riscv_vfadd_vv_f32m4(t, a->msn ? __riscv_vle32_v_f32m4(mr + k, vl) : __riscv_vfmv_v_f_f32m4(mr[0], vl), vl);
                __riscv_vse32_v_f32m4(y + k, t, vl);
                vm = __riscv_vfmax_vv_f32m4_tu(vm, vm, t, vl);
                k += (int)vl;
            }
            const float mx = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m4_f32m1(vm, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), vlm));
            vfloat32m4_t vs = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
            for (int k = 0; k < N;) {
                size_t vl = __riscv_vsetvl_e32m4(N - k);
                vfloat32m4_t e = vexp_acc(__riscv_vfsub_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), mx, vl), vl);
                __riscv_vse32_v_f32m4(y + k, e, vl);
                vs = __riscv_vfadd_vv_f32m4_tu(vs, vs, e, vl);
                k += (int)vl;
            }
            const float inv = 1.0f / __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vs, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm));
            for (int k = 0; k < N;) {
                size_t vl = __riscv_vsetvl_e32m4(N - k);
                __riscv_vse32_v_f32m4(y + k, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(y + k, vl), inv, vl), vl);
                k += (int)vl;
            }
        }
        k3_sgemm_rows(S, N, v + vbase(a, bt), a->vsn, o + obase(a, bt) + (int64_t)i0 * a->osm, a->osm, 0, nr, N, a->DV,
                      NULL, 0);
    }
}

/* ======================================================================== int8 fused */
#ifndef K3_NO_IME
static inline void tr8(uint64_t x[8])      /* 8x8 byte transpose: byte c of x[r] -> byte r of x[c] */
{
    for (int i = 0; i < 4; i++) {
        const uint64_t a = x[i], b = x[i + 4];
        x[i] = (a & 0x00000000FFFFFFFFull) | (b << 32);
        x[i + 4] = (a >> 32) | (b & 0xFFFFFFFF00000000ull);
    }
    static const int ix[4] = {0, 1, 4, 5};
    for (int t = 0; t < 4; t++) {
        const int i = ix[t];
        const uint64_t a = x[i], b = x[i + 2];
        x[i] = (a & 0x0000FFFF0000FFFFull) | ((b & 0x0000FFFF0000FFFFull) << 16);
        x[i + 2] = ((a >> 16) & 0x0000FFFF0000FFFFull) | (b & 0xFFFF0000FFFF0000ull);
    }
    for (int i = 0; i < 8; i += 2) {
        const uint64_t a = x[i], b = x[i + 1];
        x[i] = (a & 0x00FF00FF00FF00FFull) | ((b & 0x00FF00FF00FF00FFull) << 8);
        x[i + 1] = ((a >> 8) & 0x00FF00FF00FF00FFull) | (b & 0xFF00FF00FF00FF00ull);
    }
}

/* absmax of each of n rows (x + r*ld - sub) -> amax[r], without a vector->scalar move per row */
static void rows_absmax(const float *x, int64_t ld, int n, int D, const float *sub, float *amax)
{
    for (int r = 0; r < n; r++) {
        const float *xr = x + (int64_t)r * ld;
        size_t vlm = __riscv_vsetvlmax_e32m4();
        vfloat32m4_t m = __riscv_vfmv_v_f_f32m4(0.0f, vlm);
        for (int k = 0; k < D;) {
            size_t vl = __riscv_vsetvl_e32m4(D - k);
            vfloat32m4_t v = __riscv_vle32_v_f32m4(xr + k, vl);
            if (sub) v = __riscv_vfsub_vv_f32m4(v, __riscv_vle32_v_f32m4(sub + k, vl), vl);
            m = __riscv_vfmax_vv_f32m4_tu(m, m, __riscv_vfabs_v_f32m4(v, vl), vl);
            k += (int)vl;
        }
        vfloat32m1_t red = __riscv_vfredmax_vs_f32m4_f32m1(m, __riscv_vfmv_s_f_f32m1(0.0f, 1), vlm);
        __riscv_vse32_v_f32m1(amax + r, red, 1);
    }
}

/* scale s = amax / 127 (1 for a zero row), inv = 1 / s, vectorised over n rows */
static void scales(const float *amax, int n, float *s, float *inv)
{
    for (int k = 0; k < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - k);
        vfloat32m4_t a = __riscv_vle32_v_f32m4(amax + k, vl);
        vfloat32m4_t sc = __riscv_vfmerge_vfm_f32m4(__riscv_vfdiv_vf_f32m4(a, 127.0f, vl), 1.0f,
                                                    __riscv_vmfle_vf_f32m4_b8(a, 0.0f, vl), vl);
        __riscv_vse32_v_f32m4(s + k, sc, vl);
        __riscv_vse32_v_f32m4(inv + k, __riscv_vfrdiv_vf_f32m4(sc, 1.0f, vl), vl);
        k += (int)vl;
    }
}

/* q8[0..n) = round((x - sub) * inv) clamped to +-127 (inv a vector when vinv) */
static inline void quant_row(const float *x, int n, const float *sub, float inv, const float *vinv, int8_t *q8)
{
    for (int k = 0; k < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - k);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(x + k, vl);
        if (sub) v = __riscv_vfsub_vv_f32m4(v, __riscv_vle32_v_f32m4(sub + k, vl), vl);
        v = vinv ? __riscv_vfmul_vv_f32m4(v, __riscv_vle32_v_f32m4(vinv + k, vl), vl) : __riscv_vfmul_vf_f32m4(v, inv, vl);
        vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4_rm(v, __RISCV_FRM_RNE, vl);
        q = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(q, 127, vl), -127, vl);
        __riscv_vse8_v_i8m1(q8 + k, __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(q, 0, vl), 0, vl), vl);
        k += (int)vl;
    }
}

/* 16-bit row quantisation split in bytes: q16 = round(x * inv) in +-32767, hi = q16 >> 8 (s8), lo = q16 & 255 (u8) */
static inline void quant_row16(const float *x, int n, float inv, int8_t *hi, uint8_t *lo)
{
    for (int k = 0; k < n;) {
        size_t vl = __riscv_vsetvl_e32m4(n - k);
        vint32m4_t q = __riscv_vfcvt_x_f_v_i32m4_rm(__riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(x + k, vl), inv, vl), __RISCV_FRM_RNE, vl);
        q = __riscv_vmax_vx_i32m4(__riscv_vmin_vx_i32m4(q, 32767, vl), -32767, vl);
        __riscv_vse8_v_i8m1(hi + k, __riscv_vnsra_wx_i8m1(__riscv_vnsra_wx_i16m2(q, 8, vl), 0, vl), vl);
        vuint32m4_t l = __riscv_vand_vx_u32m4(__riscv_vreinterpret_v_i32m4_u32m4(q), 255, vl);
        __riscv_vse8_v_u8m1(lo + k, __riscv_vnsrl_wx_u8m1(__riscv_vnsrl_wx_u16m2(l, 0, vl), 0, vl), vl);
        k += (int)vl;
    }
}

static inline void cp8(void *d, const void *s) { *(uint64_t *)d = *(const uint64_t *)s; }
static inline void cp4(void *d, const void *s) { *(uint32_t *)d = *(const uint32_t *)s; }

/* phase A, per batch: K channel mean over keys (smoothing), V per-channel absmax -> scales, patterns */
static void i8_prepA(void *vc, int64_t b0, int64_t b1, int wid)
{
    tiles();
    (void)wid;
    actx *c = vc;
    attn_d *a = c->a;
    k3_graph *g = c->g;
    const int T = TM * TN, N = a->N, D = a->D, DV = a->DV;
    const float *k = k3_ptr(g, a->kt), *v = k3_ptr(g, a->vt);
    float *vmax = scratch(1, sizeof(float) * (a->DVp + 64));
    for (int64_t bt = b0; bt < b1; bt++) {
        const float *K = k + kbase(a, (int)bt), *V = v + vbase(a, (int)bt);
        float *mean = a->kmean + bt * a->Dp;
        for (int j = 0; j < D;) {
            size_t vl = __riscv_vsetvl_e32m4(D - j);
            vfloat32m4_t s0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), s1 = s0;
            int n = 0;
            for (; n + 1 < N; n += 2) {
                s0 = __riscv_vfadd_vv_f32m4(s0, __riscv_vle32_v_f32m4(K + (int64_t)n * a->ksn + j, vl), vl);
                s1 = __riscv_vfadd_vv_f32m4(s1, __riscv_vle32_v_f32m4(K + (int64_t)(n + 1) * a->ksn + j, vl), vl);
            }
            if (n < N) s0 = __riscv_vfadd_vv_f32m4(s0, __riscv_vle32_v_f32m4(K + (int64_t)n * a->ksn + j, vl), vl);
            __riscv_vse32_v_f32m4(mean + j, __riscv_vfmul_vf_f32m4(__riscv_vfadd_vv_f32m4(s0, s1, vl), (g_dbg & 1) ? 0.0f : 1.0f / N, vl), vl);
            j += (int)vl;
        }
        for (int j = 0; j < DV;) {
            size_t vl = __riscv_vsetvl_e32m4(DV - j);
            vfloat32m4_t m0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), m1 = m0;
            int n = 0;
            for (; n + 1 < N; n += 2) {
                m0 = __riscv_vfmax_vv_f32m4(m0, __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(V + (int64_t)n * a->vsn + j, vl), vl), vl);
                m1 = __riscv_vfmax_vv_f32m4(m1, __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(V + (int64_t)(n + 1) * a->vsn + j, vl), vl), vl);
            }
            if (n < N) m0 = __riscv_vfmax_vv_f32m4(m0, __riscv_vfabs_v_f32m4(__riscv_vle32_v_f32m4(V + (int64_t)n * a->vsn + j, vl), vl), vl);
            __riscv_vse32_v_f32m4(vmax + j, __riscv_vfmax_vv_f32m4(m0, m1, vl), vl);
            j += (int)vl;
        }
        float *vs = a->vsc + bt * a->DVp, *vi = a->vinv + bt * a->DVp;
        scales(vmax, DV, vs, vi);
        float *svp = a->svp + (size_t)bt * (a->DVp / TN) * T;
        for (int t = 0; t < a->DVp / TN; t++)
            for (int e = 0; e < T; e++) {
                const int dv = t * TN + e % TN;
                svp[t * T + e] = dv < DV ? vs[dv] : 0.0f;
            }
    }
}

/* phase B, per (batch, key block of NBT*TN keys): K quantised per key -> IME B layout (n = key, k = d);
 * V quantised per channel -> IME B layout (n = dv, k = key) through 8x8 byte transposes */
static void i8_prepB(void *vc, int64_t u0, int64_t u1, int wid)
{
    tiles();
    (void)wid;
    actx *c = vc;
    attn_d *a = c->a;
    k3_graph *g = c->g;
    const int T = TM * TN, N = a->N, D = a->D, DV = a->DV, ksq = a->ksq, ksp = a->ksp, KB = NBT * TN;
    const int nkb = a->Npad / KB;
    const float *k = k3_ptr(g, a->kt), *v = k3_ptr(g, a->vt);
    float *amax = scratch(1, sizeof(float) * 3 * (KB + 16)), *sk = amax + KB + 16, *ik = sk + KB + 16;
    int8_t *rows = scratch(2, (size_t)KB * (a->Dp > a->DVp ? a->Dp : a->DVp) + 64);
    for (int64_t u = u0; u < u1; u++) {
        const int bt = (int)(u / nkb), kb0 = (int)(u % nkb) * KB, nk = N - kb0 < KB ? (N - kb0 > 0 ? N - kb0 : 0) : KB;
        const float *K = k + kbase(a, bt) + (int64_t)kb0 * a->ksn, *V = v + vbase(a, bt) + (int64_t)kb0 * a->vsn;
        const float *mean = a->kmean + (size_t)bt * a->Dp;
        /* K */
        rows_absmax(K, a->ksn, nk, D, mean, amax);
        scales(amax, nk, sk, ik);
        memset(rows, 0, (size_t)KB * a->Dp);
        for (int r = 0; r < nk; r++) quant_row(K + (int64_t)r * a->ksn, D, mean, ik[r], NULL, rows + (size_t)r * a->Dp);
        int8_t *kb = a->k8 + (size_t)bt * a->Npad * a->Dp + (size_t)(kb0 / KB) * ksq * NBT * TN * TK;
        for (int r = 0; r < KB; r++) {
            const int jj = r / TN, cc = r % TN;
            for (int sl = 0; sl < ksq; sl++) {
                int8_t *d = kb + ((size_t)sl * NBT + jj) * TN * TK + cc * TK;
                const int8_t *s = rows + (size_t)r * a->Dp + sl * TK;
                cp8(d, s);
                if (TK == 16) cp8(d + 8, s + 8);
            }
        }
        float *skp = a->skp + ((size_t)bt * a->NKT + kb0 / TN) * T;
        const float cs = a->scale * 1.44269504088896341f;
        for (int t = 0; t < NBT; t++)
            for (int e = 0; e < T; e++) {
                const int r = t * TN + e % TN;
                skp[t * T + e] = r < nk ? sk[r] * cs : 0.0f;
            }
        /* V: rows [KB][DVp] int8, then [dv][key] 8x8 blocks into the B layout */
        memset(rows, 0, (size_t)KB * a->DVp);
        const float *vi = a->vinv + (size_t)bt * a->DVp;
        for (int r = 0; r < nk; r++) quant_row(V + (int64_t)r * a->vsn, DV, NULL, 0.0f, vi, rows + (size_t)r * a->DVp);
        int8_t *vb = a->v8 + (size_t)bt * a->DVp * a->Npad;
        for (int kg = 0; kg < KB; kg += 8)
            for (int dg = 0; dg < a->DVp; dg += 8) {
                uint64_t x[8];
                for (int r = 0; r < 8; r++) x[r] = *(const uint64_t *)(rows + (size_t)(kg + r) * a->DVp + dg);
                tr8(x);
                const int key = kb0 + kg, sl = key / TK, kk0 = key % TK;
                for (int cdv = 0; cdv < 8; cdv++) {
                    const int dv = dg + cdv, dvb = dv / KB, jj = (dv / TN) % NBT, cc = dv % TN;
                    *(uint64_t *)(vb + ((((size_t)dvb * ksp + sl) * NBT + jj) * TN + cc) * TK + kk0) = x[cdv];
                }
            }
    }
}

/* additive mask in the tile layout [MBT][NKT][T], log2 domain, -inf on padding keys; one per (mask set, rb) */
static void i8_mask(void *vc, int64_t u0, int64_t u1, int wid)
{
    tiles();
    (void)wid;
    actx *c = vc;
    attn_d *a = c->a;
    const float *mk = k3_ptr(c->g, a->mt);
    const int T = TM * TN;
    for (int64_t u = u0; u < u1; u++) {
        const int bt = a->mask_shared ? 0 : (int)(u / a->nrb), rb = (int)(u % a->nrb);
        float *mp = a->mp + (size_t)u * MBT * a->NKT * T;
        for (int i = 0; i < MBT; i++)
            for (int r = 0; r < TM; r++) {
                const int m = rb * a->RB + i * TM + r;
                const float *mr = m < a->M ? mk + a->mrow[(int64_t)bt * a->M + m] : NULL;
                for (int n = 0; n < a->Npad; n++) {
                    const float val = n >= a->N ? -INFINITY : mr ? mr[n * a->msn] * 1.44269504088896341f : 0.0f;
                    mp[((size_t)i * a->NKT + n / TN) * T + r * TN + n % TN] = val;
                }
            }
    }
}

static inline void store_rows(float *dst, int64_t ld, const float *y, int rows, int cols)
{
    const int even = !(ld & 1) && !(((uintptr_t)dst) & 7) && cols == TN;
    for (int r = 0; r < rows; r++) {
        if (even) {
            uint64_t *d = (uint64_t *)(dst + r * ld);
            const uint64_t *s = (const uint64_t *)(y + r * TN);
            for (int e = 0; e < TN / 2; e++) d[e] = s[e];
        } else {
            uint32_t *d = (uint32_t *)(dst + r * ld);
            const uint32_t *s = (const uint32_t *)(y + r * TN);
            for (int e = 0; e < cols; e++) d[e] = s[e];
        }
    }
}

/* the 2x4-blocked IME GEMM over ks slabs: C tiles [i][j][T] int32 to cb */
#define IME_BLOCK(OP, A, B, KS, CB) do {                                                                       \
    size_t vl32_ = __riscv_vsetvl_e32m2(T);                                                                    \
    vint32m2_t c00 = __riscv_vmv_v_x_i32m2(0, vl32_), c01 = c00, c02 = c00, c03 = c00;                         \
    vint32m2_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;                                                     \
    const int8_t *pa_ = (A), *pb_ = (B);                                                                       \
    for (int s_ = 0; s_ < (KS); s_++, pa_ += MBT * tt, pb_ += NBT * bt_) {                                     \
        size_t vl8_ = __riscv_vsetvl_e8m1(tt);                                                                 \
        vint8m1_t a0 = __riscv_vle8_v_i8m1(pa_, vl8_), a1 = __riscv_vle8_v_i8m1(pa_ + tt, vl8_);               \
        vint8m1_t b0 = __riscv_vle8_v_i8m1(pb_, vl8_), b1 = __riscv_vle8_v_i8m1(pb_ + bt_, vl8_);              \
        OP(c00, a0, b0); OP(c10, a1, b0); OP(c01, a0, b1); OP(c11, a1, b1);                                    \
        vint8m1_t b2 = __riscv_vle8_v_i8m1(pb_ + 2 * bt_, vl8_), b3 = __riscv_vle8_v_i8m1(pb_ + 3 * bt_, vl8_); \
        OP(c02, a0, b2); OP(c12, a1, b2); OP(c03, a0, b3); OP(c13, a1, b3);                                    \
    }                                                                                                          \
    vl32_ = __riscv_vsetvl_e32m2(T);                                                                           \
    int32_t *cb_ = (CB);                                                                                       \
    __riscv_vse32_v_i32m2(cb_ + 0 * T, c00, vl32_); __riscv_vse32_v_i32m2(cb_ + 1 * T, c01, vl32_);           \
    __riscv_vse32_v_i32m2(cb_ + 2 * T, c02, vl32_); __riscv_vse32_v_i32m2(cb_ + 3 * T, c03, vl32_);           \
    __riscv_vse32_v_i32m2(cb_ + 4 * T, c10, vl32_); __riscv_vse32_v_i32m2(cb_ + 5 * T, c11, vl32_);           \
    __riscv_vse32_v_i32m2(cb_ + 6 * T, c12, vl32_); __riscv_vse32_v_i32m2(cb_ + 7 * T, c13, vl32_);           \
} while (0)

static void i8_unit(void *vc, int64_t u0, int64_t u1, int wid)
{
    tiles();
    actx *c = vc;
    attn_d *a = c->a;
    k3_graph *g = c->g;
    const int T = TM * TN, NKT = a->NKT, ksq = a->ksq, ksp = a->ksp, RB = a->RB, tt = TM * TK, bt_ = TN * TK;
    const int p16 = a->mode == 3 || a->mode == 6;           /* P as 16 bits = two u8 IME passes */
    const int q16 = a->mode == 6;                           /* Q as 16 bits too (s8 high byte, u8 low byte) */
    const float *q = k3_ptr(g, a->qt);
    float *o = k3_ptr(g, a->ot);
    int8_t *ap = scratch(1, 4 * (size_t)RB * a->Dp + 64), *qrows = ap + (size_t)RB * a->Dp;
    int8_t *aplo = qrows + (size_t)RB * a->Dp, *qlo = aplo + (size_t)RB * a->Dp;
    int32_t *sacc = scratch(2, sizeof(int32_t) * (size_t)MBT * NKT * T + 64);
    uint8_t *p8 = scratch(3, 2 * (size_t)MBT * NKT * T + 64), *p8lo = p8 + (size_t)MBT * NKT * T;
    int8_t *ap2 = scratch(4, 2 * (size_t)RB * a->Npad + 64), *ap2lo = ap2 + (size_t)RB * a->Npad;
    float *pat = scratch(5, sizeof(float) * 8 * T + 64);
    float *sq = scratch(6, sizeof(float) * 3 * (RB + 16)), *iq = sq + RB + 16, *amx = iq + RB + 16;
    int32_t *cb = scratch(7, sizeof(int32_t) * 2 * MBT * NBT * T + 64);
    for (int64_t u = u0; u < u1; u++) {
        const int bt = (int)(u / a->nrb), rb = (int)(u % a->nrb), m0 = rb * RB;
        const int nrow = a->M - m0 < RB ? a->M - m0 : RB;
        uint64_t p0 = k3_now_ns(), p1;
#define PH(k) do { p1 = k3_now_ns(); g_ph[wid & 15][k] += p1 - p0; p0 = p1; } while (0)
        /* 1. Q rows -> int8 per-row scale, A layout [ksq][MBT][TM][TK] */
        const float *Q = q + qbase(a, bt) + (int64_t)m0 * a->qsm;
        rows_absmax(Q, a->qsm, nrow, a->D, NULL, amx);
        scales(amx, nrow, sq, iq);
        for (int r = nrow; r < RB; r++) sq[r] = 0.0f;
        memset(qrows, 0, (size_t)RB * a->Dp);
        if (q16) {
            memset(qlo, 0, (size_t)RB * a->Dp);
            for (int r = 0; r < nrow; r++) {
                sq[r] *= 127.0f / 32767.0f;
                quant_row16(Q + (int64_t)r * a->qsm, a->D, 1.0f / sq[r], qrows + (size_t)r * a->Dp, (uint8_t *)qlo + (size_t)r * a->Dp);
            }
        } else
            for (int r = 0; r < nrow; r++) quant_row(Q + (int64_t)r * a->qsm, a->D, NULL, iq[r], NULL, qrows + (size_t)r * a->Dp);
        for (int h = 0; h <= q16; h++)
            for (int r = 0; r < RB; r++)
                for (int sl = 0; sl < ksq; sl++) {
                    int8_t *d = (h ? aplo : ap) + ((size_t)sl * MBT + r / TM) * tt + (r % TM) * TK;
                    const int8_t *s = (h ? qlo : qrows) + (size_t)r * a->Dp + sl * TK;
                    cp8(d, s);
                    if (TK == 16) cp8(d + 8, s + 8);
                }
        PH(0);
        /* 2. scores on the IME -> sacc [MBT][NKT][T] */
        const int8_t *kb = a->k8 + (size_t)bt * a->Npad * a->Dp;
        for (int nb = 0; nb < a->Npad / (NBT * TN); nb++) {
            IME_BLOCK(VMADOT, ap, kb + (size_t)nb * ksq * NBT * bt_, ksq, cb);
            if (q16) {                                         /* acc = 256 acc_hi + acc_lo, exact in int32 */
                int32_t *cl = cb + MBT * NBT * T;
                IME_BLOCK(VMADOTUS, aplo, kb + (size_t)nb * ksq * NBT * bt_, ksq, cl);
                for (int e = 0; e < MBT * NBT * T;) {
                    size_t vl = __riscv_vsetvl_e32m4(MBT * NBT * T - e);
                    __riscv_vse32_v_i32m4(cb + e, __riscv_vadd_vv_i32m4(__riscv_vsll_vx_i32m4(__riscv_vle32_v_i32m4(cb + e, vl), 8, vl), __riscv_vle32_v_i32m4(cl + e, vl), vl), vl);
                    e += (int)vl;
                }
            }
            for (int i = 0; i < MBT; i++) memcpy(sacc + ((size_t)i * NKT + nb * NBT) * T, cb + (size_t)i * NBT * T, sizeof(int32_t) * NBT * T);
        }
        PH(1);
        /* 3. per m-tile: t = acc sq sk c log2e + mask; base-2 softmax; P quantised in the tile layout */
        const float *skp = a->skp + (size_t)bt * NKT * T;
        const float *mpb = a->mt < 0 ? NULL : a->mp + (size_t)((a->mask_shared ? 0 : bt) * a->nrb + rb) * MBT * NKT * T;
        float spv[16];
        const float pq = p16 ? 65535.0f : 255.0f;
        for (int i = 0; i < MBT; i++) {
            float *sqp = pat, *rmp = pat + 2 * T, *red = pat + 4 * T;
            for (int e = 0; e < 2 * T; e++) sqp[e] = sq[i * TM + (e % T) / TN];
            const float *mpi = mpb ? mpb + (size_t)i * NKT * T : a->mp;
            float *st = (float *)(sacc + (size_t)i * NKT * T);
            const size_t vl = __riscv_vsetvl_e32m4(2 * T);
            const vfloat32m4_t vsq = __riscv_vle32_v_f32m4(sqp, vl);
            vfloat32m4_t vmx0 = __riscv_vfmv_v_f_f32m4(-INFINITY, vl), vmx1 = vmx0;
            for (int e = 0; e < NKT * T; e += 4 * T) {     /* NKT is a multiple of 4: two chunks per step */
                vfloat32m4_t t0 = __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4((const int32_t *)st + e, vl), vl);
                vfloat32m4_t t1 = __riscv_vfcvt_f_x_v_f32m4(__riscv_vle32_v_i32m4((const int32_t *)st + e + 2 * T, vl), vl);
                t0 = __riscv_vfmul_vv_f32m4(t0, __riscv_vfmul_vv_f32m4(vsq, __riscv_vle32_v_f32m4(skp + e, vl), vl), vl);
                t1 = __riscv_vfmul_vv_f32m4(t1, __riscv_vfmul_vv_f32m4(vsq, __riscv_vle32_v_f32m4(skp + e + 2 * T, vl), vl), vl);
                if (mpb || e + 4 * T > a->N / TN * T) {      /* no mask: only the padded key tiles */
                    t0 = __riscv_vfadd_vv_f32m4(t0, __riscv_vle32_v_f32m4(mpi + e, vl), vl);
                    t1 = __riscv_vfadd_vv_f32m4(t1, __riscv_vle32_v_f32m4(mpi + e + 2 * T, vl), vl);
                }
                __riscv_vse32_v_f32m4(st + e, t0, vl);
                __riscv_vse32_v_f32m4(st + e + 2 * T, t1, vl);
                vmx0 = __riscv_vfmax_vv_f32m4(vmx0, t0, vl);
                vmx1 = __riscv_vfmax_vv_f32m4(vmx1, t1, vl);
            }
            __riscv_vse32_v_f32m4(red, __riscv_vfmax_vv_f32m4(vmx0, vmx1, vl), vl);
            float rm[8];
            for (int r = 0; r < TM; r++) {
                float m = -INFINITY;
                for (int h = 0; h < 2; h++)
                    for (int cc = 0; cc < TN; cc++) m = red[h * T + r * TN + cc] > m ? red[h * T + r * TN + cc] : m;
                rm[r] = m > -126.0f ? m : -126.0f;
            }
            for (int e = 0; e < 2 * T; e++) rmp[e] = rm[(e % T) / TN];
            const vfloat32m4_t vrm = __riscv_vle32_v_f32m4(rmp, vl);
            vfloat32m4_t vs0 = __riscv_vfmv_v_f_f32m4(0.0f, vl), vs1 = vs0;
            uint8_t *pi = p8 + (size_t)i * NKT * T, *pl = p8lo + (size_t)i * NKT * T;
            for (int e = 0; e < NKT * T; e += 4 * T) {
                vfloat32m4_t x0 = vexp2_fast(__riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(st + e, vl), vrm, vl), vl);
                vfloat32m4_t x1 = vexp2_fast(__riscv_vfsub_vv_f32m4(__riscv_vle32_v_f32m4(st + e + 2 * T, vl), vrm, vl), vl);
                vs0 = __riscv_vfadd_vv_f32m4(vs0, x0, vl);
                vs1 = __riscv_vfadd_vv_f32m4(vs1, x1, vl);
                vuint16m2_t h0 = __riscv_vfncvt_xu_f_w_u16m2_rm(__riscv_vfmul_vf_f32m4(x0, pq, vl), __RISCV_FRM_RNE, vl);
                vuint16m2_t h1 = __riscv_vfncvt_xu_f_w_u16m2_rm(__riscv_vfmul_vf_f32m4(x1, pq, vl), __RISCV_FRM_RNE, vl);
                if (p16) {
                    __riscv_vse8_v_u8m1(pi + e, __riscv_vnsrl_wx_u8m1(h0, 8, vl), vl);
                    __riscv_vse8_v_u8m1(pi + e + 2 * T, __riscv_vnsrl_wx_u8m1(h1, 8, vl), vl);
                    __riscv_vse8_v_u8m1(pl + e, __riscv_vnsrl_wx_u8m1(h0, 0, vl), vl);
                    __riscv_vse8_v_u8m1(pl + e + 2 * T, __riscv_vnsrl_wx_u8m1(h1, 0, vl), vl);
                } else {
                    __riscv_vse8_v_u8m1(pi + e, __riscv_vnsrl_wx_u8m1(h0, 0, vl), vl);
                    __riscv_vse8_v_u8m1(pi + e + 2 * T, __riscv_vnsrl_wx_u8m1(h1, 0, vl), vl);
                }
            }
            __riscv_vse32_v_f32m4(red, __riscv_vfadd_vv_f32m4(vs0, vs1, vl), vl);
            for (int r = 0; r < TM; r++) {
                float s = 0.0f;
                for (int h = 0; h < 2; h++)
                    for (int cc = 0; cc < TN; cc++) s += red[h * T + r * TN + cc];
                spv[i * TM + r] = s > 0.0f ? 1.0f / (pq * s) : 0.0f;
            }
        }
        PH(2);
        /* 4. P tiles -> A layout [ksp][MBT][TM][TK] (a slab = TK / TN = 2 key tiles side by side) */
        for (int h = 0; h <= p16; h++) {
            const uint8_t *src = h ? p8lo : p8;
            int8_t *dst = h ? ap2lo : ap2;
            for (int sl = 0; sl < ksp; sl++)
                for (int i = 0; i < MBT; i++)
                    for (int r = 0; r < TM; r++) {
                        int8_t *d = dst + (((size_t)sl * MBT + i) * TM + r) * TK;
                        const uint8_t *s0 = src + ((size_t)i * NKT + sl * 2) * T + r * TN;
                        if (TN == 8) { cp8(d, s0); cp8(d + 8, s0 + T); }
                        else { cp4(d, s0); cp4(d + 4, s0 + T); }
                    }
        }
        PH(3);
        /* 5. O = P V on the IME: y = (256 acc_hi + acc_lo) sp sv */
        const int8_t *vb = a->v8 + (size_t)bt * a->DVp * a->Npad;
        const float *svp = a->svp + (size_t)bt * (a->DVp / TN) * T;
        for (int e = 0; e < 2 * T; e++) pat[e] = spv[(e / T) * TM + (e % T) / TN];
        float *O = o + obase(a, bt) + (int64_t)m0 * a->osm, *yb = pat + 4 * T;
        for (int dvb = 0; dvb < a->ndvb; dvb++) {
            const int8_t *bb = vb + (size_t)dvb * ksp * NBT * bt_;
            IME_BLOCK(VMADOTUS, ap2, bb, ksp, cb);
            if (p16) IME_BLOCK(VMADOTUS, ap2lo, bb, ksp, cb + MBT * NBT * T);
            const size_t vl32 = __riscv_vsetvl_e32m2(T);
            for (int i = 0; i < MBT; i++) {
                const int rows = nrow - i * TM < TM ? nrow - i * TM : TM;
                if (rows <= 0) break;
                const vfloat32m2_t vsp = __riscv_vle32_v_f32m2(pat + i * T, vl32);
                for (int j = 0; j < NBT; j++) {
                    const int dv0 = (dvb * NBT + j) * TN;
                    if (dv0 >= a->DV) break;
                    const int32_t *ch = cb + (size_t)(i * NBT + j) * T;
                    vfloat32m2_t y = __riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(ch, vl32), vl32);
                    if (p16) y = __riscv_vfmacc_vf_f32m2(__riscv_vfcvt_f_x_v_f32m2(__riscv_vle32_v_i32m2(ch + MBT * NBT * T, vl32), vl32), 256.0f, y, vl32);
                    y = __riscv_vfmul_vv_f32m2(y, __riscv_vfmul_vv_f32m2(vsp, __riscv_vle32_v_f32m2(svp + (size_t)(dv0 / TN) * T, vl32), vl32), vl32);
                    __riscv_vse32_v_f32m2(yb, y, vl32);
                    store_rows(O + (int64_t)(i * TM) * a->osm + dv0, a->osm, yb, rows, a->DV - dv0 < TN ? a->DV - dv0 : TN);
                }
            }
        }
        PH(4);
#undef PH
    }
}
#endif

/* ======================================================================== run */
static void attn_run32(k3_graph *g, k3_op *op);
void k3_attn_run(k3_graph *g, k3_op *op)
{
    attn_d *a = op->prep.wpack;
    if (!a->f16) { attn_run32(g, op); return; }
    void *orig[5];
    for (int h = 0; h < a->nh; h++) {             /* O too: elements the chain does not write round-trip exactly */
        k3_tensor *t = &g->t[a->hid[h]];
        orig[h] = t->data;
        k3_f16_convert(t->data, a->hbuf[h], t->nelem, 0);
        t->data = a->hbuf[h];
        t->dtype = DT_F32;
    }
    attn_run32(g, op);
    for (int h = 0; h < a->nh; h++) {
        k3_tensor *t = &g->t[a->hid[h]];
        if (a->hid[h] == a->ot) k3_f16_convert(t->data, orig[h], t->nelem, 1);
        t->data = orig[h];
        t->dtype = DT_F16;
    }
}
static void attn_run32(k3_graph *g, k3_op *op)
{
    tiles();
    attn_d *a = op->prep.wpack;
    actx c = {g, a};
    const int nw = k3_pool_size();
    const int64_t nu = (int64_t)a->NB * a->nrb;
    if (g_dbg < 0) { const char *e = getenv("K3_ATTN_DBG"); g_dbg = e ? atoi(e) : 0; }
    uint64_t t0 = k3_now_ns(), t1, t2;
    g_ncall++;
#define PAR(n, fn) do { if (nw == 1 || (n) < 2) fn(&c, 0, (n), 0); else k3_parallel((n), 1, fn, &c); } while (0)
#ifndef K3_NO_IME
    if (a->mode >= 3) {
        PAR(a->NB, i8_prepA);
        PAR((int64_t)a->NB * (a->Npad / (NBT * TN)), i8_prepB);
        t1 = k3_now_ns();
        if (a->mt >= 0) PAR(a->nmask, i8_mask);
        t2 = k3_now_ns();
        PAR(nu, i8_unit);
        g_tprep += t1 - t0; g_tmask += t2 - t1; g_tunit += k3_now_ns() - t2;
        return;
    }
#endif
    PAR(a->NB, f32_tr);
    t1 = k3_now_ns();
    PAR(nu, f32_unit);
    g_tprep += t1 - t0; g_tunit += k3_now_ns() - t1;
#undef PAR
}

/* ======================================================================== --attn-check */
static double g_worst_rel = 0.0, g_worst_l2 = 0.0;
static long g_nverify = 0;
void k3_attn_verify(k3_graph *g, k3_op *op)
{
    attn_d *a = op->prep.wpack;
    k3_tensor *ot = &g->t[a->ot];
    const int h16 = ot->dtype == DT_F16;
    const size_t nb = (size_t)ot->nelem * k3_esz(ot->dtype);
    void *fused_raw = malloc(nb);
    memcpy(fused_raw, ot->data, nb);
    for (int e = 0; e < a->nops; e++) {          /* the unfused chain, original kernels */
        k3_op *c = &g->ops[a->ops[e]];
        if (a->oimpl[e] == 5) k3_f16_run(g, c);
        else if (a->oimpl[e]) k3_opt_run(g, c);
        else k3_ref_run(g, c);
    }
    float *fused = fused_raw, *o = ot->data;
    if (h16) {                                     /* compare in fp32 (exact widening) */
        fused = malloc(sizeof(float) * (size_t)ot->nelem);
        o = malloc(sizeof(float) * (size_t)ot->nelem);
        k3_f16_convert(fused_raw, fused, ot->nelem, 0);
        k3_f16_convert(ot->data, o, ot->nelem, 0);
    }
    /* compare only the elements the chain writes: rows of O (B0 x B1 x M rows of DV) */
    double mx = 0, me = 0, se = 0, sr = 0;
    for (int bt = 0; bt < a->B0 * a->B1; bt++)
        for (int m = 0; m < a->M; m++) {
            const int64_t base = obase(a, bt) + (int64_t)m * a->osm;
            for (int j = 0; j < a->DV; j++) {
                const double r = o[base + j], f = fused[base + j], d = fabs(r - f);
                mx = fabs(r) > mx ? fabs(r) : mx;
                me = d > me ? d : me;
                se += d * d;
                sr += r * r;
            }
        }
    const double rel = mx > 0 ? me / mx : me, l2 = sr > 0 ? sqrt(se / sr) : sqrt(se);
    if (rel > g_worst_rel) g_worst_rel = rel;
    if (l2 > g_worst_l2) g_worst_l2 = l2;
    g_nverify++;
    memcpy(ot->data, fused_raw, nb);               /* downstream sees the fused result */
    if (h16) { free(fused); free(o); }
    free(fused_raw);
}

void k3_attn_summary(void)
{
    if (g_ncall)
        printf("ATTN_TIME calls=%ld prep_ms=%.3f mask_ms=%.3f units_ms=%.3f (totals) dbg=%d\n", g_ncall, g_tprep / 1e6,
               g_tmask / 1e6, g_tunit / 1e6, g_dbg);
    if (g_ncall) {
        double t[5] = {0};
        for (int w = 0; w < 16; w++) for (int k = 0; k < 5; k++) t[k] += g_ph[w][k] / 1e6;
        printf("ATTN_PHASES (cpu-ms summed over workers) q=%.2f qk=%.2f softmax=%.2f ppack=%.2f pv=%.2f\n", t[0], t[1], t[2], t[3], t[4]);
    }
    if (g_nverify)
        printf("ATTN_CHECK groups_run=%ld worst_maxabs_over_maxref=%.3e worst_rel_l2=%.3e\n", g_nverify, g_worst_rel,
               g_worst_l2);
}
