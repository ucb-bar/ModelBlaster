/* K3 IR runtime: loader, executor, thread pool and the command-line driver.
 *
 *   k3ir run   G.plan G.wblob G.cblob [opts] --in in.bin [--out out.bin]
 *   k3ir feed  G.plan G.wblob G.cblob [opts] --feed feed.bin --dump dump.bin   (every op on the reference
 *              walk's inputs: the isolation gate)
 *   k3ir chunk V.plan V.wblob V.cblob P.plan P.wblob P.cblob S.plan S.wblob S.cblob [opts]
 *              --obs obs.bin --temb temb.bin --steps 5 [--out actions.bin]       (SmolVLA chunk)
 * opts: --cpus LIST (first = the calling thread) --backend ref|a100|x100 --iters N --warm N --prof
 *       --opt-mask HEX (kinds that may use optimised kernels; default all) --check (each optimised
 *       int8 op is re-run with the reference kernel on the same inputs and compared byte for byte)
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "k3ir.h"
#include "mb_placement.h"

const char *k3_kind_names[K_NKINDS] = {
    "view", "strided_copy", "catn", "ewise1", "ewise2", "mul_scalar", "add", "softmax", "layer_norm",
    "rms_norm", "rope_half", "silu_mul", "gelu_exact", "linear", "bmm_strided", "conv2d", "index_select",
    "quantize_rows_s8", "linear_dq_s8", "linear3_dq_s8", "linear_dq_add_s8", "gelu", "ewise_scalar",
    "silu", "relu", "matmul", "layer_norm_noaffine", "attention", "softmax_dim", "mean_dim", "sum_dim",
    "softplus", "tanh", "mul", "sdpa", "conv_transpose2d", "sigmoid", "im2col",
    "cast", "maxpool2d", "upsample_nearest", "gather", "conv2d_grouped", "cumsum", "strided_scatter", "reduce"};

int k3_backend = BK_REF;
int k3_opt_mask = -1;
static int g_check = 0;

uint64_t k3_host_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ---- pools ---------------------------------------------------------------- */
/* One pool per cluster. Pool 0's worker 0 is the main thread; pool 1 (--cpus2) has a leader thread of
 * its own that runs whole ops posted by the main thread (k3_remote). Every thread is placed on its CPU
 * before it builds any frame with RVV spill slots and never migrates. k3_parallel splits over the pool
 * of the calling thread. Dispatch: --pool 0 dynamic (shared atomic counter), --pool 1 static contiguous
 * split with per-worker done words in their own cache lines. */
#define MAXW 16
int k3_pool_mode = 0, k3_pool2_mode = 0;   /* --pool (pool 0), --pool2 (pool 1) */
typedef struct k3_pool {
    int nw, id, mode;               /* mode: 0 dynamic, 1 static split */
    int cpus[MAXW];
    pthread_t th[MAXW];
    _Atomic uint64_t gen __attribute__((aligned(64)));
    _Atomic int64_t next __attribute__((aligned(64)));
    _Atomic int done __attribute__((aligned(64)));
    _Atomic int stop;
    int64_t n, grain;
    k3_range_fn fn;
    void *ctx;
    struct { _Atomic uint64_t v; char pad[56]; } __attribute__((aligned(64))) wdone[MAXW];
    /* remote ops (pool 1 leader) */
    _Atomic uint64_t rseq __attribute__((aligned(64)));
    _Atomic uint64_t rdone __attribute__((aligned(64)));
    void (*rfn)(void *);
    void *rarg;
} k3_pool;
static k3_pool g_pool[2];
static __thread k3_pool *tp = &g_pool[0];
static int g_npools = 1;

static void run_static(k3_pool *P, int wid)
{
    const int64_t n = P->n, nw = P->nw;
    const int64_t u0 = n * wid / nw, u1 = n * (wid + 1) / nw;
    if (u1 > u0) P->fn(P->ctx, u0, u1, wid);
}

static void run_units(k3_pool *P, int wid)
{
    for (;;) {
        int64_t u = atomic_fetch_add_explicit(&P->next, P->grain, memory_order_relaxed);
        if (u >= P->n) break;
        int64_t e = u + P->grain < P->n ? u + P->grain : P->n;
        P->fn(P->ctx, u, e, wid);
    }
}

static uint64_t g_spin_max = 1ull << 22;   /* --spin-log2 N: spins before an idle worker sleeps */
static void spin_wait(_Atomic uint64_t *w, uint64_t seen, _Atomic int *stop, uint64_t *gnow)
{
    uint64_t spins = 0;
    while ((*gnow = atomic_load_explicit(w, memory_order_acquire)) == seen) {
        if (stop && atomic_load_explicit(stop, memory_order_relaxed)) { *gnow = seen; return; }
        if (++spins > g_spin_max) {
            struct timespec ts = {0, 20000};
            nanosleep(&ts, NULL);
        }
    }
}

typedef struct { k3_pool *P; int wid; } warg;
static void *worker(void *arg)
{
    warg a = *(warg *)arg;
    free(arg);
    k3_pool *P = a.P;
    const int wid = a.wid;
    if (mb_place_current_thread(P->cpus[wid])) {
        fprintf(stderr, "k3ir: worker %d could not be placed on cpu %d\n", wid, P->cpus[wid]);
        exit(4);
    }
    tp = P;
    uint64_t seen = 0, gnow;
    for (;;) {
        spin_wait(&P->gen, seen, &P->stop, &gnow);
        if (gnow == seen) return NULL;              /* stop */
        seen = gnow;
        if (P->mode == 1) {
            run_static(P, wid);
            atomic_store_explicit(&P->wdone[wid].v, gnow, memory_order_release);
        } else {
            run_units(P, wid);
            atomic_fetch_add_explicit(&P->done, 1, memory_order_release);
        }
    }
}

static void spawn_workers(k3_pool *P)
{
    for (int i = 1; i < P->nw; i++) {
        warg *a = malloc(sizeof *a);
        a->P = P;
        a->wid = i;
        pthread_create(&P->th[i], NULL, worker, a);
    }
}

int k3_pool_start(const int *cpus, int n)
{
    k3_pool *P = &g_pool[0];
    P->mode = k3_pool_mode;
    P->nw = n < 1 ? 1 : (n > MAXW ? MAXW : n);
    for (int i = 0; i < P->nw; i++) P->cpus[i] = cpus ? cpus[i] : -1;
    /* worker 0 (the calling thread) was placed by main before any frame with RVV stack slots was
     * built: see main() */
    spawn_workers(P);
    return 0;
}

int k3_pool_size(void) { return tp->nw; }

void k3_parallel(int64_t n, int64_t grain, k3_range_fn fn, void *ctx)
{
    k3_pool *P = tp;
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    if (P->nw == 1 || n <= grain) {
        fn(ctx, 0, n, 0);
        return;
    }
    P->n = n;
    P->grain = grain;
    P->fn = fn;
    P->ctx = ctx;
    if (P->mode == 1) {
        const uint64_t gen = atomic_load_explicit(&P->gen, memory_order_relaxed) + 1;
        atomic_store_explicit(&P->gen, gen, memory_order_release);
        run_static(P, 0);
        for (int w = 1; w < P->nw; w++)
            while (atomic_load_explicit(&P->wdone[w].v, memory_order_acquire) != gen) {
            }
        return;
    }
    atomic_store_explicit(&P->next, 0, memory_order_relaxed);
    atomic_store_explicit(&P->done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&P->gen, 1, memory_order_release);
    run_units(P, 0);
    while (atomic_load_explicit(&P->done, memory_order_acquire) < P->nw - 1) {
    }
}

/* pool 1 leader: runs remote calls (whole ops, prepares) on its cluster */
static void *leader1(void *arg)
{
    (void)arg;
    k3_pool *P = &g_pool[1];
    if (mb_place_current_thread(P->cpus[0])) {
        fprintf(stderr, "k3ir: pool 1 leader could not be placed on cpu %d\n", P->cpus[0]);
        exit(4);
    }
    tp = P;
    spawn_workers(P);
    uint64_t seen = 0, gnow;
    for (;;) {
        spin_wait(&P->rseq, seen, &P->stop, &gnow);
        if (gnow == seen) return NULL;
        seen = gnow;
        P->rfn(P->rarg);
        atomic_store_explicit(&P->rdone, gnow, memory_order_release);
    }
}

int k3_pool2_start(const int *cpus, int n)
{
    k3_pool *P = &g_pool[1];
    P->id = 1;
    { extern int k3_pool2_mode; P->mode = k3_pool2_mode; }
    P->nw = n < 1 ? 1 : (n > MAXW ? MAXW : n);
    for (int i = 0; i < P->nw; i++) P->cpus[i] = cpus[i];
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 64u << 20);
    pthread_create(&P->th[0], &at, leader1, NULL);
    g_npools = 2;
    return 0;
}

/* run fn(arg) on pool `pl` (its leader for pool 1) and wait */
void k3_remote(int pl, void (*fn)(void *), void *arg)
{
    if (pl == 0 || g_npools < 2 || tp == &g_pool[pl]) { fn(arg); return; }
    k3_pool *P = &g_pool[pl];
    P->rfn = fn;
    P->rarg = arg;
    const uint64_t s = atomic_load_explicit(&P->rseq, memory_order_relaxed) + 1;
    atomic_store_explicit(&P->rseq, s, memory_order_release);
    while (atomic_load_explicit(&P->rdone, memory_order_acquire) != s) {
    }
}

void k3_pool_stop(void)
{
    for (int p = g_npools - 1; p >= 0; p--) {
        k3_pool *P = &g_pool[p];
        atomic_store(&P->stop, 1);
        for (int i = (p == 0); i < P->nw; i++) pthread_join(P->th[i], NULL);
    }
    g_pool[0].nw = 1;
    g_npools = 1;
}

/* ---- loading -------------------------------------------------------------- */
static void *read_file(const char *path, int64_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "k3ir: cannot open %s: %s\n", path, strerror(errno));
        exit(2);
    }
    fseek(f, 0, SEEK_END);
    int64_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *p = aligned_alloc(64, (size_t)((n + 63) & ~63ll) + 64);
    if (n && fread(p, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "k3ir: short read %s\n", path);
        exit(2);
    }
    fclose(f);
    if (size) *size = n;
    return p;
}

static void load_graph(k3_graph *g, const char *plan, const char *wb, const char *cb)
{
    int64_t nbytes;
    int64_t *w = read_file(plan, &nbytes);
    if (w[0] != 0x4D424B3349520001ll) {
        fprintf(stderr, "k3ir: %s is not a plan\n", plan);
        exit(2);
    }
    g->n_tensors = (int)w[1];
    g->n_ops = (int)w[2];
    g->n_in = (int)w[3];
    g->n_out = (int)w[4];
    int64_t wbytes = w[5], cbytes = w[6];
    int64_t *p = w + 7;
    g->wblob = read_file(wb, &g->wbytes);
    g->cblob = read_file(cb, &g->cbytes);
    if (g->wbytes != wbytes || g->cbytes != cbytes) {
        fprintf(stderr, "k3ir: blob sizes %lld/%lld do not match the plan %lld/%lld\n", (long long)g->wbytes,
                (long long)g->cbytes, (long long)wbytes, (long long)cbytes);
        exit(2);
    }
    g->t = calloc(g->n_tensors, sizeof(k3_tensor));
    for (int i = 0; i < g->n_tensors; i++, p += 5) {
        k3_tensor *t = &g->t[i];
        t->dtype = (int)p[0];
        t->kind = (int)p[1];
        t->nelem = p[2];
        t->a = p[3];
        t->b = p[4];
    }
    g->inputs = malloc(sizeof(int) * g->n_in);
    g->outputs = malloc(sizeof(int) * g->n_out);
    for (int i = 0; i < g->n_in; i++) g->inputs[i] = (int)*p++;
    for (int i = 0; i < g->n_out; i++) g->outputs[i] = (int)*p++;
    for (int i = 0; i < g->n_tensors; i++) {     /* buffers and consts first, aliases after */
        k3_tensor *t = &g->t[i];
        if (t->kind == TK_BUF) {
            size_t sz = ((size_t)t->nelem * k3_esz(t->dtype) + 127) & ~(size_t)63;
            t->data = aligned_alloc(64, sz);
            memset(t->data, 0, sz);
        } else if (t->kind == TK_CONST) {
            t->data = g->cblob + t->a;
        }
    }
    for (int i = 0; i < g->n_tensors; i++) {
        k3_tensor *t = &g->t[i];
        if (t->kind == TK_ALIAS)
            t->data = (uint8_t *)g->t[t->a].data + t->b * k3_esz(g->t[t->a].dtype);
    }
    g->ops = calloc(g->n_ops, sizeof(k3_op));
    for (int i = 0; i < g->n_ops; i++, p += K3_OPW) {
        k3_op *op = &g->ops[i];
        const int64_t *q = p;
        op->kind = (int)q[0];
        op->nin = (int)q[1];
        op->nout = (int)q[2];
        q += 3;
        for (int j = 0; j < K3_NIN; j++) op->in[j] = (int)q[j];
        q += K3_NIN;
        for (int j = 0; j < K3_NOUT; j++) op->out[j] = (int)q[j];
        q += K3_NOUT;
        for (int j = 0; j < K3_NW; j++) {
            op->w[j] = q[j];
            op->wp[j] = q[j] >= 0 ? g->wblob + q[j] : NULL;
        }
        q += K3_NW;
        memcpy(op->ip, q, sizeof(op->ip));
        q += K3_NIP;
        memcpy(op->fp, q, sizeof(op->fp));
    }
    free(w);
}

static int g_prof_on;
/* placement over the two pools: --place-graphs vision=1,prefix=0,step=0 (whole graphs; tags vision /
 * prefix / step / graph) or --place-ops gemm (int8 GEMMs, conv, bmm and fused attention on pool 1, every
 * other op on pool 0) */
static const char *g_place_graphs, *g_place_ops;
static int graph_pool(const char *tag)
{
    const char *nm = strstr(tag, "Pg") ? "prefix" : strchr(tag, 'V') ? "vision" : strchr(tag, 'S') ? "step" : "graph";
    if (!g_place_graphs) return 0;
    const char *p = strstr(g_place_graphs, nm);
    return p && p[strlen(nm)] == '=' ? atoi(p + strlen(nm) + 1) : 0;
}
static int is_gemm_kind(int k)
{
    return k == K_LINEAR_DQ_S8 || k == K_LINEAR3_DQ_S8 || k == K_LINEAR_DQ_ADD_S8 || k == K_CONV2D || k == K_BMM_STRIDED ||
           k == K_SDPA || k == K_MATMUL || k == K_LINEAR || k == K_CONV_T2D;
}
typedef struct { k3_graph *g; k3_op *op; int r; } prep_arg;
static void prep_one(void *a_)
{
    prep_arg *a = a_;
    a->r = a->op->f16 ? (k3_f16_prepare(a->g, a->op) ? 5 : 0) : k3_opt_prepare(a->g, a->op);
}
static void fuse_attn(void *g) { k3_attn_fuse(g); }
static void prepare_tag(k3_graph *g, const char *tag)
{
    const int gp = graph_pool(tag);
    for (int i = 0; i < g->n_ops; i++) {
        k3_op *op = &g->ops[i];
        op->impl = 0;
        op->cl = g_place_ops ? is_gemm_kind(op->kind) : gp;
        op->f16 = k3_op_is_f16(g, op);
        if (k3_backend != BK_REF && ((k3_opt_mask >> op->kind) & 1)) {
            prep_arg a = {g, op, 0};
            k3_remote(op->cl, prep_one, &a);        /* packed layouts follow the owning cluster's vlenb */
            op->impl = a.r;
        }
    }
    if (k3_backend != BK_REF && k3_attn_mode) {
        k3_remote(g_place_ops ? 1 : gp, fuse_attn, g);
        if (g_place_ops)
            for (int i = 0; i < g->n_ops; i++) if (g->ops[i].impl == 3) g->ops[i].cl = 1;
        k3_attn_report(g, tag);
    }
    if (k3_backend != BK_REF && k3_rowq_mode) printf("ROWQ %s fused=%d\n", tag, k3_rowq_fuse(g));
    g->prof = calloc(g->n_ops, sizeof(uint64_t));
}
#define prepare(g) prepare_tag(g, #g)

/* ---- execution ------------------------------------------------------------ */
static long g_check_bad;
int k3_attn_check;                    /* --attn-check: each fused attention re-run unfused and compared */

static int is_s8_kind(int k)
{
    return k == K_QUANTIZE_ROWS_S8 || k == K_LINEAR_DQ_S8 || k == K_LINEAR3_DQ_S8 || k == K_LINEAR_DQ_ADD_S8;
}

static void check_op(k3_graph *g, k3_op *op, int idx) __attribute__((unused));
static void check_op(k3_graph *g, k3_op *op, int idx)
{
    /* re-run the reference kernel into scratch outputs and compare byte for byte */
    void *save[K3_NOUT];
    void *tmp[K3_NOUT];
    for (int j = 0; j < op->nout; j++) {
        k3_tensor *t = &g->t[op->out[j]];
        size_t sz = (size_t)t->nelem * k3_esz(t->dtype);
        save[j] = t->data;
        tmp[j] = malloc(sz + 64);
        t->data = tmp[j];
    }
    k3_ref_run(g, op);
    for (int j = 0; j < op->nout; j++) {
        k3_tensor *t = &g->t[op->out[j]];
        size_t sz = (size_t)t->nelem * k3_esz(t->dtype);
        t->data = save[j];
        if (memcmp(save[j], tmp[j], sz)) {
            long nb = 0;
            for (size_t b = 0; b < sz; b++) nb += ((uint8_t *)save[j])[b] != ((uint8_t *)tmp[j])[b];
            if (g_check_bad < 20)
                fprintf(stderr, "CHECK_MISMATCH op %d (%s) out %d: %ld bytes differ\n", idx,
                        k3_kind_names[op->kind], j, nb);
            g_check_bad++;
        }
        free(tmp[j]);
    }
}

static volatile int g_cur_op = -1, g_cur_kind = -1;
#include <ucontext.h>
static void on_fault(int sig, siginfo_t *si, void *uc_)
{
    char b[192];
    unsigned long pc = 0;
#if defined(__riscv)
    pc = ((ucontext_t *)uc_)->uc_mcontext.__gregs[0];
#else
    (void)uc_;
#endif
    int n = snprintf(b, sizeof b, "k3ir: signal %d code %d addr %p pc %#lx in op %d (%s)\n", sig, si->si_code,
                     si->si_addr, pc, g_cur_op, g_cur_kind >= 0 ? k3_kind_names[g_cur_kind] : "-");
    if (write(2, b, n)) {}
    _exit(128 + sig);
}

static void run_op_here(k3_graph *g, k3_op *op, int idx);
typedef struct { k3_graph *g; k3_op *op; int idx; } run_arg;
static void run_remote(void *a_) { run_arg *a = a_; run_op_here(a->g, a->op, a->idx); }
static void run_op(k3_graph *g, k3_op *op, int idx)
{
    if (op->impl == 2) return;
    if (op->cl && g_npools > 1) {
        run_arg a = {g, op, idx};
        uint64_t t0 = g_prof_on ? k3_now_ns() : 0;
        const int pon = g_prof_on;
        g_prof_on = 0;
        k3_remote(op->cl, run_remote, &a);
        g_prof_on = pon;
        if (pon) g->prof[idx] += k3_now_ns() - t0;
        return;
    }
    run_op_here(g, op, idx);
}
static void run_op_here(k3_graph *g, k3_op *op, int idx)
{
    g_cur_op = idx;
    g_cur_kind = op->kind;
    if (op->impl == 2) return;                    /* fused into a later op */
    uint64_t t0 = g_prof_on ? k3_now_ns() : 0;
    if (op->impl == 3) {
        k3_attn_run(g, op);
        if (k3_attn_check) { extern void k3_attn_verify(k3_graph *, k3_op *); k3_attn_verify(g, op); }
    } else if (op->impl == 4) {
        k3_rowq_run(g, op);
    } else if (op->impl == 5) k3_f16_run(g, op);
    else if (op->impl) k3_opt_run(g, op);
    else k3_ref_run(g, op);
    if (g_prof_on) g->prof[idx] += k3_now_ns() - t0;
    if (g_check && op->impl == 4) check_op(g, &g->ops[op->prep.npad], op->prep.npad);
    if (g_check && ((op->impl == 1 && is_s8_kind(op->kind)) || (op->impl == 5 && k3_f16_exact_kind(op->kind))))
        check_op(g, op, idx);
}

static void run_graph(k3_graph *g)
{
    for (int i = 0; i < g->n_ops; i++) run_op(g, &g->ops[i], i);
}

static size_t tbytes(k3_graph *g, int id) { return (size_t)g->t[id].nelem * k3_esz(g->t[id].dtype); }

static void set_inputs(k3_graph *g, const uint8_t *src)
{
    for (int i = 0; i < g->n_in; i++) {
        memcpy(g->t[g->inputs[i]].data, src, tbytes(g, g->inputs[i]));
        src += tbytes(g, g->inputs[i]);
    }
}

static size_t in_bytes(k3_graph *g)
{
    size_t s = 0;
    for (int i = 0; i < g->n_in; i++) s += tbytes(g, g->inputs[i]);
    return s;
}

static size_t out_bytes(k3_graph *g)
{
    size_t s = 0;
    for (int i = 0; i < g->n_out; i++) s += tbytes(g, g->outputs[i]);
    return s;
}

static void get_outputs(k3_graph *g, uint8_t *dst)
{
    for (int i = 0; i < g->n_out; i++) {
        memcpy(dst, g->t[g->outputs[i]].data, tbytes(g, g->outputs[i]));
        dst += tbytes(g, g->outputs[i]);
    }
}

static int cmp_u64(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static uint64_t median(uint64_t *v, int n)
{
    qsort(v, n, sizeof(uint64_t), cmp_u64);
    return n ? v[n / 2] : 0;
}

static void print_prof(k3_graph *g, const char *tag, int iters)
{
    double kt[K_NKINDS] = {0};
    int kn[K_NKINDS] = {0};
    double tot = 0;
    for (int i = 0; i < g->n_ops; i++) {
        if (g->ops[i].impl == 2) continue;
        const int kk = g->ops[i].impl == 3 ? K_ATTENTION : g->ops[i].kind;
        kt[kk] += g->prof[i] / (double)iters;
        kn[kk]++;
        tot += g->prof[i] / (double)iters;
    }
    printf("PROF_BEGIN %s total_ms=%.3f\n", tag, tot / 1e6);
    {
        double kf[K_NKINDS] = {0};
        for (int i = 0; i < g->n_ops; i++)
            if (g->ops[i].f16 && g->ops[i].impl != 2) kf[g->ops[i].impl == 3 ? K_ATTENTION : g->ops[i].kind] += g->prof[i] / (double)iters;
        for (int k = 0; k < K_NKINDS; k++)
            if (kf[k] > 0) printf("PROF_KIND_F16 %s %s ms=%.3f\n", tag, k3_kind_names[k], kf[k] / 1e6);
    }
    for (int k = 0; k < K_NKINDS; k++)
        if (kn[k]) printf("PROF_KIND %s %s n=%d ms=%.3f pct=%.1f\n", tag, k3_kind_names[k], kn[k], kt[k] / 1e6,
                          100 * kt[k] / tot);
    for (int i = 0; i < g->n_ops; i++)
        printf("PROF_OP %s %d %s impl=%d f16=%d us=%.1f\n", tag, i, k3_kind_names[g->ops[i].kind], g->ops[i].impl,
               g->ops[i].f16, g->prof[i] / (double)iters / 1e3);
    printf("PROF_END %s\n", tag);
}

/* ---- CLI ------------------------------------------------------------------ */
static int parse_cpus(const char *s, int *cpus)
{
    int n = 0;
    while (*s && n < MAXW) {
        char *e;
        long lo = strtol(s, &e, 10), hi = lo;
        if (*e == '-') hi = strtol(e + 1, &e, 10);
        for (long c = lo; c <= hi && n < MAXW; c++) cpus[n++] = (int)c;
        s = *e == ',' ? e + 1 : e;
        if (*e && *e != ',') break;
    }
    return n;
}

static const char *argval(int argc, char **argv, const char *k, const char *def)
{
    for (int i = 0; i < argc - 1; i++)
        if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}

static int argflag(int argc, char **argv, const char *k)
{
    for (int i = 0; i < argc; i++)
        if (!strcmp(argv[i], k)) return 1;
    return 0;
}

static void write_file(const char *path, const void *p, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(p, 1, n, f) != n) {
        fprintf(stderr, "k3ir: cannot write %s\n", path);
        exit(2);
    }
    fclose(f);
}

static void setup(int argc, char **argv)
{
    const char *be = argval(argc, argv, "--backend", "ref");
    k3_backend = !strcmp(be, "a100") ? BK_A100 : !strcmp(be, "x100") ? BK_X100 : BK_REF;
    k3_opt_mask = (int)strtol(argval(argc, argv, "--opt-mask", "-1"), NULL, 0);
    g_check = argflag(argc, argv, "--check");
    g_prof_on = argflag(argc, argv, "--prof");
    extern int k3_par_min, k3_prefetch, k3_cbozero, k3_skip_epilogue;
    k3_par_min = atoi(argval(argc, argv, "--par-min", "32768"));
    k3_skip_epilogue = argflag(argc, argv, "--no-epilogue") ? 1 : atoi(argval(argc, argv, "--epi-mode", "0"));
    k3_prefetch = argflag(argc, argv, "--prefetch");
    k3_cbozero = !argflag(argc, argv, "--no-cbozero");
    extern int k3_vperm, k3_hg_cfg;
    k3_hg_cfg = atoi(argval(argc, argv, "--hg-cfg", "0"));
    k3_vperm = argflag(argc, argv, "--no-vperm") ? 0 : argflag(argc, argv, "--vperm") ? 1 : -1;
    k3_attn_mode = atoi(argval(argc, argv, "--attn", "0"));
    k3_pool_mode = atoi(argval(argc, argv, "--pool", "0"));
    k3_pool2_mode = atoi(argval(argc, argv, "--pool2", "0"));
    g_spin_max = 1ull << atoi(argval(argc, argv, "--spin-log2", "22"));
    { extern int k3_attn_i8_min; k3_attn_i8_min = atoi(argval(argc, argv, "--attn-i8-min", "128")); }
    k3_rowq_mode = argflag(argc, argv, "--fuse-rowq");
    g_place_graphs = argval(argc, argv, "--place-graphs", NULL);
    g_place_ops = argval(argc, argv, "--place-ops", NULL);
    { extern int k3_vstore, k3_bpf, k3_mtail; k3_vstore = atoi(argval(argc, argv, "--vstore", "-1"));
      k3_bpf = atoi(argval(argc, argv, "--bpf", "0")); k3_mtail = argflag(argc, argv, "--mtail"); }
    k3_attn_check = argflag(argc, argv, "--attn-check");
    int cpus[MAXW], n = 0;
    const char *cs = argval(argc, argv, "--cpus", NULL);
    if (cs) n = parse_cpus(cs, cpus);
    if (k3_pool_start(n ? cpus : NULL, n ? n : 1)) {
        fprintf(stderr, "k3ir: placement failed\n");
        exit(4);
    }
    const char *c2 = argval(argc, argv, "--cpus2", NULL);
    if (c2) {
        int cp2[MAXW], n2 = parse_cpus(c2, cp2);
        k3_pool2_start(cp2, n2);
        printf("K3IR pool1 cpus=%s workers=%d place_graphs=%s place_ops=%s\n", c2, n2, g_place_graphs ? g_place_graphs : "-",
               g_place_ops ? g_place_ops : "-");
    }
    printf("K3IR backend=%s workers=%d cpus=%s check=%d attn=%d\n", be, k3_pool_size(), cs ? cs : "-", g_check,
           k3_attn_mode);
}

static int cmd_run(int argc, char **argv)
{
    k3_graph g = {0};
    load_graph(&g, argv[2], argv[3], argv[4]);
    setup(argc, argv);
    prepare(&g);
    const int iters = atoi(argval(argc, argv, "--iters", "5")), warm = atoi(argval(argc, argv, "--warm", "1"));
    int64_t n;
    uint8_t *in = read_file(argval(argc, argv, "--in", NULL), &n);
    if (!n || (size_t)n % in_bytes(&g)) {
        fprintf(stderr, "k3ir: input is %lld bytes, graph wants (a multiple of) %zu\n", (long long)n, in_bytes(&g));
        return 2;
    }
    const int nrec = (int)((size_t)n / in_bytes(&g));
        if (nrec > 1 && !argval(argc, argv, "--iters", NULL)) {
        /* several records and no --iters (k3-kernopt): each record runs once, the outputs are concatenated;
         * with --iters (k3-fp16): latency on the first record, then the outputs of all */
        uint8_t *o = malloc(out_bytes(&g) * nrec);
        uint64_t *tr = calloc(nrec, 8);
        for (int r = 0; r < nrec; r++) {
            set_inputs(&g, in + (size_t)r * in_bytes(&g));
            uint64_t t0 = k3_now_ns();
            run_graph(&g);
            tr[r] = k3_now_ns() - t0;
            get_outputs(&g, o + (size_t)r * out_bytes(&g));
        }
        printf("RESULT records=%d wall_ms_median=%.3f check_mismatches=%ld\n", nrec, median(tr, nrec) / 1e6, g_check_bad);
        const char *outp = argval(argc, argv, "--out", NULL);
        if (outp) write_file(outp, o, out_bytes(&g) * nrec);
        k3_pool_stop();
        return g_check_bad ? 1 : 0;
    }
    int pon = g_prof_on;
    g_prof_on = 0;
    for (int i = 0; i < warm; i++) { set_inputs(&g, in); run_graph(&g); }
    uint64_t *wall = calloc(iters, sizeof(uint64_t));
    g_prof_on = pon;
    for (int i = 0; i < iters; i++) {
        set_inputs(&g, in);
        uint64_t t0 = k3_now_ns();
        run_graph(&g);
        wall[i] = k3_now_ns() - t0;
    }
    if (g_prof_on) print_prof(&g, "graph", iters);
    { extern void k3_attn_summary(void); k3_attn_summary(); }
    uint64_t mn = wall[0], mx = wall[0];
    for (int i = 0; i < iters; i++) { mn = wall[i] < mn ? wall[i] : mn; mx = wall[i] > mx ? wall[i] : mx; }
    printf("RESULT wall_ms_median=%.3f min=%.3f max=%.3f iters=%d check_mismatches=%ld\n",
           median(wall, iters) / 1e6, mn / 1e6, mx / 1e6, iters, g_check_bad);
    const char *outp = argval(argc, argv, "--out", NULL);
    if (outp) {
        uint8_t *o = malloc(out_bytes(&g) * (size_t)nrec);
        get_outputs(&g, o);
        for (int r = 1; r < nrec; r++) {
            set_inputs(&g, in + (size_t)r * in_bytes(&g));
            run_graph(&g);
            get_outputs(&g, o + (size_t)r * out_bytes(&g));
        }
        write_file(outp, o, out_bytes(&g) * (size_t)nrec);
    }
    if (nrec > 1) printf("RESULT_RECORDS %d\n", nrec);
    k3_pool_stop();
    return g_check_bad ? 1 : 0;
}

/* feed: every op runs on the reference walk's values of its inputs; every op's outputs are dumped.
 * feed format: int64 count padded to 64 B, then count x [int64 tensor id, int64 nbytes, pad to 64 B, bytes
 * padded to 64 B]; dump format: int64 count, then count x [int64 id, int64 nbytes, bytes] (unpadded). */
static int cmd_feed(int argc, char **argv)
{
    k3_graph g = {0};
    load_graph(&g, argv[2], argv[3], argv[4]);
    int64_t n;
    uint8_t *f = read_file(argval(argc, argv, "--feed", NULL), &n);   /* before placement: see read_file */
    setup(argc, argv);
    prepare(&g);
    void **ref = calloc(g.n_tensors, sizeof(void *));
    int64_t cnt = *(int64_t *)f;
    uint8_t *p = f + 64;
    for (int64_t i = 0; i < cnt; i++) {
        int64_t id, nb;                    /* entries are not 8-byte aligned: no int64 derefs */
        memcpy(&id, p, 8);
        memcpy(&nb, p + 8, 8);
        p += 64;
        if (nb != (int64_t)tbytes(&g, (int)id)) {
            fprintf(stderr, "k3ir feed: tensor %lld has %lld bytes, wants %zu\n", (long long)id, (long long)nb,
                    tbytes(&g, (int)id));
            return 2;
        }
        ref[id] = p;
        p += (nb + 63) & ~63ll;            /* entries padded to 64 bytes (A100 faults on misaligned memcpy) */
    }
    FILE *d = fopen(argval(argc, argv, "--dump", "dump.bin"), "wb");
    int64_t nout = 0;
    for (int i = 0; i < g.n_ops; i++) nout += g.ops[i].nout;
    fwrite(&nout, 8, 1, d);
    for (int i = 0; i < g.n_ops; i++) {
        k3_op *op = &g.ops[i];
        for (int j = 0; j < op->nin; j++) {
            int id = op->in[j];
            if (g.t[id].kind == TK_CONST) continue;
            if (!ref[id]) {
                fprintf(stderr, "k3ir feed: no reference value for input %d of op %d\n", id, i);
                return 2;
            }
            memcpy(g.t[id].data, ref[id], tbytes(&g, id));
        }
        run_op(&g, op, i);
        for (int j = 0; j < op->nout; j++) {
            int64_t id = op->out[j], nb = (int64_t)tbytes(&g, op->out[j]);
            fwrite(&id, 8, 1, d);
            fwrite(&nb, 8, 1, d);
            fwrite(g.t[id].data, 1, (size_t)nb, d);
        }
    }
    fclose(d);
    printf("RESULT feed_ops=%d check_mismatches=%ld\n", g.n_ops, g_check_bad);
    k3_pool_stop();
    return g_check_bad ? 1 : 0;
}

/* SmolVLA chunk: vision -> prefix -> steps x flow step, Euler x += dt v (dt = -1/steps), as
 * scripts/smolvla_int8/validate.py e2e. obs.bin: count x [pixels, lang, state, noise] (f32);
 * temb.bin: steps x 720 f32 (the sinusoidal time embedding at t = 1 + s dt). */
static int cmd_chunk(int argc, char **argv)
{
    k3_graph V = {0}, Pg = {0}, S = {0};
    load_graph(&V, argv[2], argv[3], argv[4]);
    load_graph(&Pg, argv[5], argv[6], argv[7]);
    load_graph(&S, argv[8], argv[9], argv[10]);
    setup(argc, argv);
    prepare(&V);
    prepare(&Pg);
    prepare(&S);
    const int steps = atoi(argval(argc, argv, "--steps", "5"));
    const int iters = atoi(argval(argc, argv, "--iters", "1")), warm = atoi(argval(argc, argv, "--warm", "0"));
    const int nobs_max = atoi(argval(argc, argv, "--nobs", "1000000"));
    int64_t nb, tb;
    uint8_t *obs = read_file(argval(argc, argv, "--obs", NULL), &nb);
    float *temb = read_file(argval(argc, argv, "--temb", NULL), &tb);
    const size_t pix = tbytes(&V, V.inputs[0]), lang = tbytes(&Pg, Pg.inputs[1]),
                 state = tbytes(&Pg, Pg.inputs[2]), noise = tbytes(&S, S.inputs[1]);
    const size_t rec = pix + lang + state + noise;
    if (nb % rec || tb != (int64_t)(steps * tbytes(&S, S.inputs[2]))) {
        fprintf(stderr, "k3ir chunk: obs %lld bytes not a multiple of %zu, or temb %lld bytes\n", (long long)nb,
                rec, (long long)tb);
        return 2;
    }
    if (tbytes(&V, V.outputs[0]) != tbytes(&Pg, Pg.inputs[0]) || tbytes(&Pg, Pg.outputs[0]) != tbytes(&S, S.inputs[0])) {
        fprintf(stderr, "k3ir chunk: graph interfaces do not match\n");
        return 2;
    }
    int nobs = (int)(nb / rec);
    if (nobs > nobs_max) nobs = nobs_max;
    const int nx = (int)(noise / 4);
    float *x = malloc(noise), *acts = malloc(noise * (size_t)nobs);
    const float dt = -1.0f / steps;
    uint64_t *tv = calloc((size_t)iters * nobs, 8), *tp = calloc((size_t)iters * nobs, 8),
             *ts = calloc((size_t)iters * nobs * steps, 8), *tc = calloc((size_t)iters * nobs, 8);
    int nt = 0, nts = 0;
    const int pon = g_prof_on;
    for (int o = 0; o < nobs; o++) {
        const uint8_t *r = obs + (size_t)o * rec;
        for (int it = -warm; it < iters; it++) {
            g_prof_on = pon && it >= 0 && o == 0;
            uint64_t c0 = k3_now_ns();
            memcpy(V.t[V.inputs[0]].data, r, pix);
            run_graph(&V);
            uint64_t c1 = k3_now_ns();
            memcpy(Pg.t[Pg.inputs[0]].data, V.t[V.outputs[0]].data, tbytes(&V, V.outputs[0]));
            memcpy(Pg.t[Pg.inputs[1]].data, r + pix, lang);
            memcpy(Pg.t[Pg.inputs[2]].data, r + pix + lang, state);
            run_graph(&Pg);
            uint64_t c2 = k3_now_ns();
            memcpy(S.t[S.inputs[0]].data, Pg.t[Pg.outputs[0]].data, tbytes(&Pg, Pg.outputs[0]));
            memcpy(x, r + pix + lang + state, noise);
            for (int s = 0; s < steps; s++) {
                uint64_t s0 = k3_now_ns();
                memcpy(S.t[S.inputs[1]].data, x, noise);
                memcpy(S.t[S.inputs[2]].data, temb + (size_t)s * (tbytes(&S, S.inputs[2]) / 4), tbytes(&S, S.inputs[2]));
                run_graph(&S);
                const float *v = S.t[S.outputs[0]].data;
                for (int i = 0; i < nx; i++) x[i] = x[i] + dt * v[i];
                if (it >= 0) ts[nts++] = k3_now_ns() - s0;
            }
            uint64_t c3 = k3_now_ns();
            if (it >= 0) {
                tv[nt] = c1 - c0;
                tp[nt] = c2 - c1;
                tc[nt] = c3 - c0;
                nt++;
            }
        }
        memcpy(acts + (size_t)o * nx, x, noise);
    }
    if (pon) {
        print_prof(&V, "vision", iters);
        print_prof(&Pg, "prefix", iters);
        print_prof(&S, "step", iters * steps);
    }
    { extern void k3_attn_summary(void); k3_attn_summary(); }
    const double mv = median(tv, nt) / 1e6, mp = median(tp, nt) / 1e6, ms = median(ts, nts) / 1e6,
                 mc = median(tc, nt) / 1e6, mcmin = tc[0] / 1e6, mcmax = tc[nt - 1] / 1e6;
    printf("RESULT chunk nobs=%d iters=%d vision_ms=%.3f prefix_ms=%.3f step_ms=%.3f chunk_ms=%.3f "
           "chunk_min_ms=%.3f chunk_max_ms=%.3f check_mismatches=%ld\n",
           nobs, iters, mv, mp, ms, mc, mcmin, mcmax, g_check_bad);
    const char *outp = argval(argc, argv, "--out", NULL);
    if (outp) write_file(outp, acts, noise * (size_t)nobs);
    k3_pool_stop();
    return g_check_bad ? 1 : 0;
}

/* A thread must never change cluster (X100 vlenb 32 <-> A100 vlenb 128) while a frame that GCC sized
 * with `csrr vlenb` (scalable RVV spill slots) is live: the frame's offsets are recomputed with the new
 * vlenb and the function then reads and writes the wrong stack slots (seen as corrupted locals and a
 * return to pc 0). So main places a fresh thread first and runs everything in frames built after. */
/* bench M K N: one synthetic op of the given kind (default linear_dq_s8) on random data */
static void null_fn(void *ctx, int64_t u0, int64_t u1, int wid) { (void)ctx; (void)u0; (void)u1; (void)wid; }
static int cmd_bench(int argc, char **argv)
{
    if (!strcmp(argval(argc, argv, "--kind", ""), "null")) {     /* dispatch + barrier cost of k3_parallel */
        const int units = atoi(argv[2]), iters = atoi(argval(argc, argv, "--iters", "2000"));
        setup(argc, argv);
        uint64_t *t = calloc(iters, 8);
        for (int i = -100; i < iters; i++) {
            uint64_t t0 = k3_now_ns();
            k3_parallel(units, 1, null_fn, NULL);
            if (i >= 0) t[i] = k3_now_ns() - t0;
        }
        printf("RESULT bench null units=%d pool=%d workers=%d us_median=%.3f\n", units, k3_pool_mode, k3_pool_size(),
               median(t, iters) / 1e3);
        k3_pool_stop();
        return 0;
    }
    const int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
    const int iters = atoi(argval(argc, argv, "--iters", "20"));
    const char *kind = argval(argc, argv, "--kind", "linear_dq_s8");
    k3_graph g = {0};
    g.n_tensors = 4;
    g.t = calloc(4, sizeof(k3_tensor));
    int f16 = !strcmp(kind, "linear_f16");
    int f32 = !strcmp(kind, "linear") || f16;
    int64_t ne[4] = {(int64_t)M * K, M, (int64_t)M * N, 0};
    int dt[4] = {f16 ? DT_F16 : f32 ? DT_F32 : DT_I8, DT_F32, f16 ? DT_F16 : DT_F32, DT_F32};
    for (int i = 0; i < 3; i++) {
        g.t[i].dtype = dt[i];
        g.t[i].nelem = ne[i];
        g.t[i].data = aligned_alloc(64, ((size_t)ne[i] * 4 + 127) & ~63ull);
    }
    srand(1);
    int8_t *x = g.t[0].data;
    float *xf = g.t[0].data;
    for (int64_t i = 0; i < ne[0]; i++) {
        if (f16) ((_Float16 *)xf)[i] = (_Float16)((rand() % 255 - 127) / 64.0f);
        else if (f32) xf[i] = (rand() % 255 - 127) / 64.0f;
        else x[i] = (int8_t)(rand() % 255 - 127);
    }
    for (int i = 0; i < M; i++) ((float *)g.t[1].data)[i] = 1e-3f * (1 + rand() % 100);
    k3_op op = {0};
    op.kind = f32 ? K_LINEAR : K_LINEAR_DQ_S8;
    op.nin = f32 ? 1 : 2;
    op.nout = 1;
    op.in[0] = 0; op.in[1] = 1; op.out[0] = 2;
    op.ip[0] = M; op.ip[1] = K; op.ip[2] = N;
    void *w = aligned_alloc(64, (size_t)N * K * 4 + 64);
    float *ws = malloc(4 * (size_t)N), *b = malloc(4 * (size_t)N);
    for (int64_t i = 0; i < (int64_t)N * K; i++) {
        if (f16) ((_Float16 *)w)[i] = (_Float16)((rand() % 255 - 127) / 64.0f);
        else if (f32) ((float *)w)[i] = (rand() % 255 - 127) / 64.0f;
        else ((int8_t *)w)[i] = (int8_t)(rand() % 255 - 127);
    }
    for (int i = 0; i < N; i++) {
        ws[i] = 1e-3f * (1 + rand() % 100);
        b[i] = 0.01f * (rand() % 100);
        if (f16) ((_Float16 *)b)[i] = (_Float16)b[i];
    }
    op.wp[0] = w; op.wp[1] = b; op.wp[2] = ws;
    if (f16) {
        op.ip[K3_IP_WN] = (int64_t)N * K;
        op.ip[K3_IP_WN + 1] = N;
        op.ip[K3_IP_WF16] = 3;
        op.f16 = 1;
    }
    setup(argc, argv);
    op.impl = k3_backend == BK_REF ? 0 : f16 ? (k3_f16_prepare(&g, &op) ? 5 : 0) : k3_opt_prepare(&g, &op);
    uint64_t *t = calloc(iters, 8);
    for (int i = -2; i < iters; i++) {
        uint64_t t0 = k3_now_ns();
        if (op.impl == 5) k3_f16_run(&g, &op); else if (op.impl) k3_opt_run(&g, &op); else k3_ref_run(&g, &op);
        if (i >= 0) t[i] = k3_now_ns() - t0;
    }
    if (g_check) check_op(&g, &op, 0);
    const double ms = median(t, iters) / 1e6;
    printf("RESULT bench %s M=%d K=%d N=%d ms=%.4f mac_per_ns=%.1f check_mismatches=%ld\n", kind, M, K, N, ms,
           (double)M * K * N / (ms * 1e6), g_check_bad);
    k3_pool_stop();
    return 0;
}

/* DDPM loop for the diffusion student: x <- step(eps(x, temb[t], cond), t, x) for the host-given
 * schedule. ddpm.bin per window: cond [Tc*Dc], x_init [n], noise [steps][n]; sched.bin: steps x
 * [temb row index, sqrt(1-a_t), sqrt(a_t), c_x0, c_x, sigma] (float64, the python floats of policies.DDPM),
 * temb.bin: rows of the timestep sinusoid. Same fp32 arithmetic order as policies.DDPM.step. */
static int cmd_ddpm(int argc, char **argv)
{
    k3_graph G = {0};
    load_graph(&G, argv[2], argv[3], argv[4]);
    int64_t nb, sb, tb;
    uint8_t *win = read_file(argval(argc, argv, "--win", NULL), &nb);
    double *sch = read_file(argval(argc, argv, "--sched", NULL), &sb);
    float *temb = read_file(argval(argc, argv, "--temb", NULL), &tb);
    setup(argc, argv);
    prepare(&G);
    const int steps = (int)(sb / (6 * 8)), iters = atoi(argval(argc, argv, "--iters", "1"));
    const int ix = G.inputs[0], it_ = G.inputs[1], ic = G.inputs[2];
    const size_t n = (size_t)G.t[ix].nelem, nc = (size_t)G.t[ic].nelem, ne = (size_t)G.t[it_].nelem;
    const size_t rec = (nc + n + (size_t)steps * n) * 4;
    const int nw = (int)(nb / rec);
    float *x = malloc(n * 4), *out = malloc(n * 4 * (size_t)nw);
    uint64_t *ts = calloc((size_t)nw * iters * steps + 1, 8), *ta = calloc((size_t)nw * iters + 1, 8);
    int nts = 0, nta = 0;
    const int pon = g_prof_on;
    g_prof_on = 0;
    for (int w = 0; w < nw; w++) {
        const float *r = (const float *)(win + (size_t)w * rec);
        for (int it = 0; it < iters; it++) {
            g_prof_on = pon && w == 0;
            uint64_t a0 = k3_now_ns();
            memcpy(G.t[ic].data, r, nc * 4);
            memcpy(x, r + nc, n * 4);
            for (int s = 0; s < steps; s++) {
                const double *c = sch + s * 6;
                uint64_t s0 = k3_now_ns();
                memcpy(G.t[ix].data, x, n * 4);
                memcpy(G.t[it_].data, temb + (size_t)c[0] * ne, ne * 4);
                run_graph(&G);
                ts[nts++] = k3_now_ns() - s0;
                const float *eps = G.t[G.outputs[0]].data, *nz = r + nc + n + (size_t)s * n;
                for (size_t i = 0; i < n; i++) {
                    float x0 = (x[i] - (float)c[1] * eps[i]) / (float)c[2];
                    x0 = x0 < -1.0f ? -1.0f : (x0 > 1.0f ? 1.0f : x0);
                    float m = (float)c[3] * x0 + (float)c[4] * x[i];
                    if (c[5] > 0) m = m + (float)c[5] * nz[i];
                    x[i] = m;
                }
            }
            ta[nta++] = k3_now_ns() - a0;
        }
        memcpy(out + (size_t)w * n, x, n * 4);
    }
    if (pon) print_prof(&G, "step", iters * steps);
    { extern void k3_attn_summary(void); k3_attn_summary(); }
    const double mstep = median(ts, nts) / 1e6, mact = median(ta, nta) / 1e6;
    printf("RESULT ddpm windows=%d steps=%d iters=%d step_ms=%.3f action_ms=%.3f\n", nw, steps, iters, mstep, mact);
    const char *outp = argval(argc, argv, "--out", NULL);
    if (outp) write_file(outp, out, n * 4 * (size_t)nw);
    k3_pool_stop();
    return 0;
}

/* peak resident set (VmHWM) and the loaded blobs, for the memory footprint */
static void print_mem(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long hwm = -1, rss = -1;
    if (f) {
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "VmHWM:", 6)) hwm = strtol(line + 6, NULL, 10);
            if (!strncmp(line, "VmRSS:", 6)) rss = strtol(line + 6, NULL, 10);
        }
        fclose(f);
    }
    printf("MEM vmhwm_kb=%ld vmrss_kb=%ld\n", hwm, rss);
}

static int dispatch(int argc, char **argv) __attribute__((noinline));
static int dispatch(int argc, char **argv)
{
    if (!strcmp(argv[1], "run")) return cmd_run(argc, argv);
    if (!strcmp(argv[1], "feed")) return cmd_feed(argc, argv);
    if (!strcmp(argv[1], "chunk")) return cmd_chunk(argc, argv);
    if (!strcmp(argv[1], "bench")) return cmd_bench(argc, argv);
    if (!strcmp(argv[1], "ddpm")) return cmd_ddpm(argc, argv);
    fprintf(stderr, "k3ir: unknown command %s\n", argv[1]);
    return 2;
}

static int g_argc, g_rc;
static char **g_argv;
static void *main_thread(void *arg)
{
    (void)arg;
    const char *cs = argval(g_argc, g_argv, "--cpus", NULL);
    int cpus[MAXW];
    if (cs && parse_cpus(cs, cpus) && mb_place_current_thread(cpus[0])) {
        fprintf(stderr, "k3ir: placement on cpu %d failed\n", cpus[0]);
        g_rc = 4;
        return NULL;
    }
    g_rc = dispatch(g_argc, g_argv);
    print_mem();
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: k3ir run|feed|chunk ...\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    g_argc = argc;
    g_argv = argv;
#ifdef K3_NO_IME
    /* Spike + pk build: no clone() under pk, and no cluster migration to guard against */
    return dispatch(argc, argv);
#endif
    pthread_attr_t at;
    pthread_attr_init(&at);
    pthread_attr_setstacksize(&at, 64u << 20);
    pthread_t t;
    pthread_create(&t, &at, main_thread, NULL);
    pthread_join(t, NULL);
    return g_rc;
}
