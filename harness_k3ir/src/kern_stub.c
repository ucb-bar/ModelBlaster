/* host build without optimised kernels */
#include "k3ir.h"
int k3_opt_prepare(k3_graph *g, k3_op *op) { (void)g; (void)op; return 0; }
void k3_opt_run(k3_graph *g, k3_op *op) { k3_ref_run(g, op); }
int k3_par_min, k3_skip_epilogue, k3_prefetch, k3_cbozero, k3_vperm;
int k3_attn_mode;
int k3_attn_fuse(k3_graph *g) { (void)g; return 0; }
void k3_attn_run(k3_graph *g, k3_op *op) { (void)g; (void)op; }
void k3_attn_report(k3_graph *g, const char *tag) { (void)g; (void)tag; }
void k3_attn_verify(k3_graph *g, k3_op *op) { (void)g; (void)op; }
void k3_attn_summary(void) {}
int k3_rowq_mode;
int k3_rowq_fuse(k3_graph *g) { (void)g; return 0; }
void k3_rowq_run(k3_graph *g, k3_op *op) { (void)g; (void)op; }
int k3_vstore;
int k3_bpf, k3_mtail;
int k3_attn_i8_min;
int k3_f16_prepare(k3_graph *g, k3_op *op) { (void)g; (void)op; return 0; }
void k3_f16_run(k3_graph *g, k3_op *op) { k3_ref_run(g, op); }
int k3_f16_exact_kind(int kind) { (void)kind; return 0; }
int k3_hg_cfg;
