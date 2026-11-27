/* K3 IR runtime: executes a ModelBlaster IR packed by harness_k3ir/pack_ir.py.
 * One plan = one graph (e.g. SmolVLA vision / prefix / one flow step). */
#ifndef K3IR_H
#define K3IR_H
#include <stddef.h>
#include <stdint.h>

#define K3_NIN 8
#define K3_NOUT 4
#define K3_NW 6
#define K3_NIP 40
#define K3_NFP 4
#define K3_OPW (3 + K3_NIN + K3_NOUT + K3_NW + K3_NIP + K3_NFP)

enum { DT_F32 = 0, DT_I8 = 1, DT_I32 = 2 };
enum { TK_BUF = 0, TK_CONST = 1, TK_ALIAS = 2 };

enum {
    K_VIEW, K_STRIDED_COPY, K_CATN, K_EWISE1, K_EWISE2, K_MUL_SCALAR, K_ADD, K_SOFTMAX, K_LAYER_NORM,
    K_RMS_NORM, K_ROPE_HALF, K_SILU_MUL, K_GELU_EXACT, K_LINEAR, K_BMM_STRIDED, K_CONV2D, K_INDEX_SELECT,
    K_QUANTIZE_ROWS_S8, K_LINEAR_DQ_S8, K_LINEAR3_DQ_S8, K_LINEAR_DQ_ADD_S8, K_GELU, K_EWISE_SCALAR,
    K_SILU, K_RELU, K_MATMUL, K_LAYER_NORM_NOAFFINE, K_ATTENTION, K_SOFTMAX_DIM, K_MEAN_DIM, K_SUM_DIM,
    K_SOFTPLUS, K_TANH, K_MUL, K_SDPA, K_CONV_T2D, K_SIGMOID, K_IM2COL, K_NKINDS
};
extern const char *k3_kind_names[K_NKINDS];

typedef struct {
    int dtype, kind;
    int64_t nelem, a, b;
    void *data;
} k3_tensor;

struct k3_op;
typedef struct k3_prep {          /* per-op state built at load time by the optimised backends */
    void *wpack;                  /* packed weights */
    double *wsd;                  /* w_scale as double, padded */
    float *biasp;                 /* bias, padded (zeros when absent) */
    int npad, kpad;
} k3_prep;

typedef struct k3_op {
    int kind, nin, nout;
    int in[K3_NIN], out[K3_NOUT];
    int64_t w[K3_NW];
    int64_t ip[K3_NIP];
    double fp[K3_NFP];
    const void *wp[K3_NW];        /* resolved weight pointers (NULL when absent) */
    k3_prep prep;
    int impl;                     /* 0 = reference, 1 = optimised */
} k3_op;

typedef struct {
    int n_tensors, n_ops, n_in, n_out;
    k3_tensor *t;
    k3_op *ops;
    int *inputs, *outputs;
    uint8_t *wblob, *cblob;
    int64_t wbytes, cbytes;
} k3_graph;

static inline size_t k3_esz(int dt) { return dt == DT_I8 ? 1 : 4; }
static inline void *k3_ptr(k3_graph *g, int id) { return g->t[id].data; }

/* thread pool ------------------------------------------------------------- */
typedef void (*k3_range_fn)(void *ctx, int64_t u0, int64_t u1, int worker);
int k3_pool_start(const int *cpus, int n);      /* worker 0 = calling thread, placed on cpus[0] */
int k3_pool_size(void);
void k3_parallel(int64_t nunits, int64_t grain, k3_range_fn fn, void *ctx);
void k3_pool_stop(void);

/* backends ---------------------------------------------------------------- */
enum { BK_REF = 0, BK_A100 = 1, BK_X100 = 2 };
extern int k3_backend;              /* which optimised kernels to use */
extern int k3_opt_mask;             /* bit per kind: 1 = use optimised kernel when available */

void k3_ref_run(k3_graph *g, k3_op *op);                 /* reference (scalar C) kernels */
int k3_opt_prepare(k3_graph *g, k3_op *op);              /* 1 if an optimised kernel takes the op */
void k3_opt_run(k3_graph *g, k3_op *op);

static inline uint64_t k3_now_ns(void)
{
#if defined(__riscv)
    uint64_t t;
    __asm__ volatile("rdtime %0" : "=r"(t));
    return t * 125 / 3;            /* 24 MHz timebase on the K3 */
#else
    extern uint64_t k3_host_ns(void);
    return k3_host_ns();
#endif
}
#endif
