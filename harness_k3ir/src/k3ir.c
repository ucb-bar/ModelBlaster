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
    "softplus", "tanh", "mul", "sdpa", "conv_transpose2d", "sigmoid", "im2col"};

int k3_backend = BK_REF;
int k3_opt_mask = -1;
static int g_check = 0;

uint64_t k3_host_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

/* ---- pool ---------------------------------------------------------------- */
#define MAXW 16
static int g_nw = 1;
static int g_cpus[MAXW];
static pthread_t g_th[MAXW];
static struct {
    _Atomic uint64_t gen;
    _Atomic int64_t next;
    _Atomic int done;
    _Atomic int stop;
    int64_t n, grain;
    k3_range_fn fn;
    void *ctx;
} job;

static void run_units(int wid)
{
    for (;;) {
        int64_t u = atomic_fetch_add_explicit(&job.next, job.grain, memory_order_relaxed);
        if (u >= job.n) break;
        int64_t e = u + job.grain < job.n ? u + job.grain : job.n;
        job.fn(job.ctx, u, e, wid);
    }
}

static void *worker(void *arg)
{
    const int wid = (int)(intptr_t)arg;
    if (mb_place_current_thread(g_cpus[wid])) {
        fprintf(stderr, "k3ir: worker %d could not be placed on cpu %d\n", wid, g_cpus[wid]);
        exit(4);
    }
    uint64_t seen = 0;
    for (;;) {
        uint64_t spins = 0, gnow;
        while ((gnow = atomic_load_explicit(&job.gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&job.stop, memory_order_relaxed)) return NULL;
            if (++spins > (1u << 22)) {
                struct timespec ts = {0, 20000};
                nanosleep(&ts, NULL);
            }
        }
        seen = gnow;
        run_units(wid);
        atomic_fetch_add_explicit(&job.done, 1, memory_order_release);
    }
}

int k3_pool_start(const int *cpus, int n)
{
    g_nw = n < 1 ? 1 : (n > MAXW ? MAXW : n);
    for (int i = 0; i < g_nw; i++) g_cpus[i] = cpus ? cpus[i] : -1;
    /* worker 0 (the calling thread) was placed by main before any frame with RVV stack slots was
     * built: see main() */
    for (int i = 1; i < g_nw; i++) pthread_create(&g_th[i], NULL, worker, (void *)(intptr_t)i);
    return 0;
}

int k3_pool_size(void) { return g_nw; }

void k3_parallel(int64_t n, int64_t grain, k3_range_fn fn, void *ctx)
{
    if (n <= 0) return;
    if (grain < 1) grain = 1;
    if (g_nw == 1 || n <= grain) {
        fn(ctx, 0, n, 0);
        return;
    }
    job.n = n;
    job.grain = grain;
    job.fn = fn;
    job.ctx = ctx;
    atomic_store_explicit(&job.next, 0, memory_order_relaxed);
    atomic_store_explicit(&job.done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&job.gen, 1, memory_order_release);
    run_units(0);
    while (atomic_load_explicit(&job.done, memory_order_acquire) < g_nw - 1) {
    }
}

void k3_pool_stop(void)
{
    atomic_store(&job.stop, 1);
    for (int i = 1; i < g_nw; i++) pthread_join(g_th[i], NULL);
    g_nw = 1;
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

static void prepare(k3_graph *g)
{
    for (int i = 0; i < g->n_ops; i++) {
        k3_op *op = &g->ops[i];
        op->impl = 0;
        if (k3_backend != BK_REF && ((k3_opt_mask >> op->kind) & 1)) op->impl = k3_opt_prepare(g, op);
    }
}

/* ---- execution ------------------------------------------------------------ */
static uint64_t *g_prof;              /* per-op ns, accumulated, for the graph being profiled */
static int g_prof_on;
static long g_check_bad;

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

static void run_op(k3_graph *g, k3_op *op, int idx)
{
    g_cur_op = idx;
    g_cur_kind = op->kind;
    uint64_t t0 = g_prof_on ? k3_now_ns() : 0;
    if (op->impl) k3_opt_run(g, op);
    else k3_ref_run(g, op);
    if (g_prof_on) g_prof[idx] += k3_now_ns() - t0;
    if (g_check && op->impl && is_s8_kind(op->kind)) check_op(g, op, idx);
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
        kt[g->ops[i].kind] += g_prof[i] / (double)iters;
        kn[g->ops[i].kind]++;
        tot += g_prof[i] / (double)iters;
    }
    printf("PROF_BEGIN %s total_ms=%.3f\n", tag, tot / 1e6);
    for (int k = 0; k < K_NKINDS; k++)
        if (kn[k]) printf("PROF_KIND %s %s n=%d ms=%.3f pct=%.1f\n", tag, k3_kind_names[k], kn[k], kt[k] / 1e6,
                          100 * kt[k] / tot);
    for (int i = 0; i < g->n_ops; i++)
        printf("PROF_OP %s %d %s impl=%d us=%.1f\n", tag, i, k3_kind_names[g->ops[i].kind], g->ops[i].impl,
               g_prof[i] / (double)iters / 1e3);
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
    extern int k3_vperm;
    k3_vperm = argflag(argc, argv, "--no-vperm") ? 0 : argflag(argc, argv, "--vperm") ? 1 : -1;
    int cpus[MAXW], n = 0;
    const char *cs = argval(argc, argv, "--cpus", NULL);
    if (cs) n = parse_cpus(cs, cpus);
    if (k3_pool_start(n ? cpus : NULL, n ? n : 1)) {
        fprintf(stderr, "k3ir: placement failed\n");
        exit(4);
    }
    printf("K3IR backend=%s workers=%d cpus=%s check=%d\n", be, k3_pool_size(), cs ? cs : "-", g_check);
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
    if ((size_t)n != in_bytes(&g)) {
        fprintf(stderr, "k3ir: input is %lld bytes, graph wants %zu\n", (long long)n, in_bytes(&g));
        return 2;
    }
    g_prof = calloc(g.n_ops, sizeof(uint64_t));
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
    uint64_t mn = wall[0], mx = wall[0];
    for (int i = 0; i < iters; i++) { mn = wall[i] < mn ? wall[i] : mn; mx = wall[i] > mx ? wall[i] : mx; }
    printf("RESULT wall_ms_median=%.3f min=%.3f max=%.3f iters=%d check_mismatches=%ld\n",
           median(wall, iters) / 1e6, mn / 1e6, mx / 1e6, iters, g_check_bad);
    const char *outp = argval(argc, argv, "--out", NULL);
    if (outp) {
        uint8_t *o = malloc(out_bytes(&g));
        get_outputs(&g, o);
        write_file(outp, o, out_bytes(&g));
    }
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
    g_prof = calloc(g.n_ops, sizeof(uint64_t));
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
    for (int o = 0; o < nobs; o++) {
        const uint8_t *r = obs + (size_t)o * rec;
        for (int it = -warm; it < iters; it++) {
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
static int cmd_bench(int argc, char **argv)
{
    const int M = atoi(argv[2]), K = atoi(argv[3]), N = atoi(argv[4]);
    const int iters = atoi(argval(argc, argv, "--iters", "20"));
    const char *kind = argval(argc, argv, "--kind", "linear_dq_s8");
    k3_graph g = {0};
    g.n_tensors = 4;
    g.t = calloc(4, sizeof(k3_tensor));
    int f32 = !strcmp(kind, "linear");
    int64_t ne[4] = {(int64_t)M * K, M, (int64_t)M * N, 0};
    int dt[4] = {f32 ? DT_F32 : DT_I8, DT_F32, DT_F32, DT_F32};
    for (int i = 0; i < 3; i++) {
        g.t[i].dtype = dt[i];
        g.t[i].nelem = ne[i];
        g.t[i].data = aligned_alloc(64, ((size_t)ne[i] * 4 + 127) & ~63ull);
    }
    srand(1);
    int8_t *x = g.t[0].data;
    float *xf = g.t[0].data;
    for (int64_t i = 0; i < ne[0]; i++) { if (f32) xf[i] = (rand() % 255 - 127) / 64.0f; else x[i] = (int8_t)(rand() % 255 - 127); }
    for (int i = 0; i < M; i++) ((float *)g.t[1].data)[i] = 1e-3f * (1 + rand() % 100);
    k3_op op = {0};
    op.kind = f32 ? K_LINEAR : K_LINEAR_DQ_S8;
    op.nin = f32 ? 1 : 2;
    op.nout = 1;
    op.in[0] = 0; op.in[1] = 1; op.out[0] = 2;
    op.ip[0] = M; op.ip[1] = K; op.ip[2] = N;
    void *w = aligned_alloc(64, (size_t)N * K * 4 + 64);
    float *ws = malloc(4 * (size_t)N), *b = malloc(4 * (size_t)N);
    for (int64_t i = 0; i < (int64_t)N * K; i++) { if (f32) ((float *)w)[i] = (rand() % 255 - 127) / 64.0f; else ((int8_t *)w)[i] = (int8_t)(rand() % 255 - 127); }
    for (int i = 0; i < N; i++) { ws[i] = 1e-3f * (1 + rand() % 100); b[i] = 0.01f * (rand() % 100); }
    op.wp[0] = w; op.wp[1] = b; op.wp[2] = ws;
    setup(argc, argv);
    op.impl = k3_backend != BK_REF ? k3_opt_prepare(&g, &op) : 0;
    uint64_t *t = calloc(iters, 8);
    for (int i = -2; i < iters; i++) {
        uint64_t t0 = k3_now_ns();
        if (op.impl) k3_opt_run(&g, &op); else k3_ref_run(&g, &op);
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
    for (int w = 0; w < nw; w++) {
        const float *r = (const float *)(win + (size_t)w * rec);
        for (int it = 0; it < iters; it++) {
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
    const double mstep = median(ts, nts) / 1e6, mact = median(ta, nta) / 1e6;
    printf("RESULT ddpm windows=%d steps=%d iters=%d step_ms=%.3f action_ms=%.3f\n", nw, steps, iters, mstep, mact);
    const char *outp = argval(argc, argv, "--out", NULL);
    if (outp) write_file(outp, out, n * 4 * (size_t)nw);
    k3_pool_stop();
    return 0;
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
