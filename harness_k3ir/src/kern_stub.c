/* host build without optimised kernels */
#include "k3ir.h"
int k3_opt_prepare(k3_graph *g, k3_op *op) { (void)g; (void)op; return 0; }
void k3_opt_run(k3_graph *g, k3_op *op) { k3_ref_run(g, op); }
int k3_par_min, k3_skip_epilogue, k3_prefetch, k3_cbozero, k3_vperm;
