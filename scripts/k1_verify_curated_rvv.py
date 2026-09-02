#!/usr/bin/env python3
"""Bit-exact verify of curated RVV kernels, on the K1 itself.

Same construction and the same reasons as k1_verify_fused_conv_rvv.py, which
did this for the fused convolutions; this covers the ops that were still
falling through to the scalar reference after those landed -- lstm_s8,
add_s8, avgpool2d_s8, elu_s8, leaky_relu_s8, sigmoid_s8.

Why on the board and not on the host
------------------------------------
rvv_x60's verify_method is VERIFY_CROSS_COMPILE: generation builds the
candidate for the target ISA but does not run it, because the host is x86 and
cannot execute rv64gcv. (It used to declare VERIFY_HOST_CTYPES, which could not
even COMPILE an RVV candidate and so rejected every one of them -- see
tests/test_cross_compile_verify.py.) A kernel that cross-compiles is still
numerically unverified, and this script is where that gets settled. Spike bakes VLEN at build time and does not model this toolchain's
codegen, and the properties at issue are exactly VLEN- and codegen-dependent
(vl-tail handling, whether frm actually reaches vfcvt, whether GCC contracts a
multiply and an add into an fma).

So the curated kernel and the op's own scalar reference_impl are cross-compiled
into ONE binary with the real backend flags, executed on the board over the
(shape, quant) tuples taken from the real model graphs, and compared
element-wise on-device. Acceptance is max_abs_err == 0.

    scripts/k1_verify_curated_rvv.py                  # every op, every model
    scripts/k1_verify_curated_rvv.py --op lstm_s8

Two data regimes are run for every case. "full" fills int8 across its whole
range; "small" keeps magnitudes low so the nonlinearities stay off their
saturated tails -- which is the regime where a rounding difference would
actually show, since a saturated sigmoid hides one.

This checks kernel-vs-reference, not the weight LAYOUT. A packing mismatch is
invisible here and is caught by the end-to-end golden compare in
validation/k1_runner.py instead.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402

CROSS = os.environ.get(
    "CROSS",
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-",
)
HOST = os.environ.get("MODELBLASTER_K1_HOST", "k1")
REMOTE_ROOT = os.environ.get("MODELBLASTER_K1_REMOTE_ROOT", "/root/mb_k1")
MARCH = ["-march=rv64gcv_zvl256b", "-mabi=lp64d"]

MODELS = ["vitfly_lstm", "lstm_tiny", "dronet", "yolov8_nano",
          "vitfly_frontend", "mlp_control", "attn_block", "norm_block", "ffn_block"]

CURATED = {
    "lstm_s8": "kernels/rvv/rvv_lstm_s8_rvv_gate_int_dot.c",
    "add_s8": "kernels/rvv/rvv_add_s8_rvv_frm_rmm.c",
    "avgpool2d_s8": "kernels/rvv/rvv_avgpool2d_s8_rvv_ow_lanes.c",
    "elu_s8": "kernels/rvv/rvv_elu_s8_rvv_memo_lut_gather.c",
    "leaky_relu_s8": "kernels/rvv/rvv_leaky_relu_s8_rvv_frm_rmm.c",
    "sigmoid_s8": "kernels/rvv/rvv_sigmoid_s8_rvv_memo_lut_gather.c",
    "matmul_s8": "kernels/rvv/rvv_matmul_s8_rvv_k_reduce_n_lanes.c",
    "linear_s8": "kernels/rvv/rvv_linear_s8_direct.c",
    # RoPE and the transformer blocks. sin/cos were verified with --kernel
    # when they landed and were never added here, so the default full sweep
    # -- the thing that would catch a later edit to one of them -- did not
    # cover them. It does now.
    "sin_s8": "kernels/rvv/rvv_sin_s8_rvv_memo_lut_gather.c",
    "cos_s8": "kernels/rvv/rvv_cos_s8_rvv_memo_lut_gather.c",
    "mul_s8": "kernels/rvv/rvv_mul_s8_rvv_frm_rmm.c",
    "gelu_s8": "kernels/rvv/rvv_gelu_s8_rvv_memo_lut_gather.c",
    "softmax_s8": "kernels/rvv/rvv_softmax_s8_rvv_cached_exp.c",
    "layernorm_s8": "kernels/rvv/rvv_layernorm_s8_rvv_f64_tail.c",
    "adaptive_avg_pool2d_s8":
        "kernels/rvv/rvv_adaptive_avg_pool2d_s8_rvv_window_sum.c",
    "depthwise_conv2d_s8":
        "kernels/rvv/rvv_depthwise_conv2d_s8_rvv_ow_lanes_taps.c",
}

#: Shapes to check beyond the ones the model graphs happen to contain.
#: attn_block's own matmuls are M=8,K=32,N=8 and M=8,K=8,N=32 -- correctness
#: shapes, but K=32 is exactly one e8m1 strip at VLEN=256, so they never
#: exercise a k-tail, a multi-strip accumulation, or an N-strip wider than one
#: vector. A kernel that is wrong in exactly those places passes on the model
#: shapes alone. These come from matmul_s8's own KernelSpec.extra_shapes plus
#: two deliberate off-by-one strip boundaries.
#: Complete cases for ops NO MODEL IN THIS TREE EMITS. `EXTRA_CASES` above
#: extends a donor case borrowed from a real graph, so it cannot help here --
#: there is no donor. These carry their own quant values.
#:
#: Both entries are ViNT ops. ViNT cannot be built in this checkout (its int8
#: calibration needs the IDSIA stills and the IsaacLab forest renders, neither
#: of which is present), so these kernels are verified against the reference
#: and NOT yet exercised by a real model. The shapes are chosen to hit the
#: cases the kernels actually branch on rather than to look like ViNT:
#: padded and unpadded, stride 1 and 2, an odd width that leaves a vector
#: tail, and a channel count above one vector.
SYNTHETIC_CASES = {
    "depthwise_conv2d_s8": [
        # 3x3 stride 1 same-padding: the common mobile block, and the case
        # where every tap has a different valid output range.
        dict(N=1, C=16, IH=14, IW=14, KH=3, KW=3, SH=1, SW=1, PH=1, PW=1),
        # stride 2, so the strided load path runs and the tap ranges are not
        # simply shifted by one.
        dict(N=1, C=16, IH=14, IW=14, KH=3, KW=3, SH=2, SW=2, PH=1, PW=1),
        # unpadded: every tap in bounds for every column, the easy path.
        dict(N=1, C=8,  IH=14, IW=14, KH=3, KW=3, SH=1, SW=1, PH=0, PW=0),
        # odd width, so the last tile is a partial vector.
        dict(N=1, C=8,  IH=9,  IW=13, KH=3, KW=3, SH=1, SW=1, PH=1, PW=1),
        # 5x5, more taps than a tile is wide at small IW.
        dict(N=1, C=4,  IH=11, IW=11, KH=5, KW=5, SH=1, SW=1, PH=2, PW=2),
        # 1x1: degenerate, no padding logic at all.
        dict(N=1, C=32, IH=7,  IW=7,  KH=1, KW=1, SH=1, SW=1, PH=0, PW=0),
    ],
    "adaptive_avg_pool2d_s8": [
        # output_size=(1,1) -- the only form the extractor emits, and a
        # whole-plane reduction.
        dict(N=1, C=32, IH=8,  IW=13, OH=1, OW=1),
        dict(N=1, C=10, IH=14, IW=14, OH=1, OW=1),
        # a run shorter than MB_AAP_VEC_MIN, so the scalar arm runs
        dict(N=1, C=8,  IH=3,  IW=5,  OH=1, OW=1),
        # non-degenerate output: uneven windows, which is where the
        # ih0/ih1/iw0/iw1 arithmetic actually differs per output
        dict(N=1, C=8,  IH=13, IW=13, OH=3, OW=3),
        dict(N=1, C=4,  IH=16, IW=16, OH=4, OW=4),
    ],
}

#: Quant tuples for the synthetic cases, by op. Realistic rather than 1.0, so
#: the requantize tail is exercised instead of short-circuited.
SYNTHETIC_QUANT = {
    "depthwise_conv2d_s8": dict(input_offset=0, filter_offset=0,
                                output_offset=0, output_multiplier=1207959552,
                                output_shift=7, activation_min=-128,
                                activation_max=127),
    "adaptive_avg_pool2d_s8": dict(scale_in=0.0235, scale_out=0.0180,
                                   activation_min=-128, activation_max=127),
}

EXTRA_CASES = {
    "matmul_s8": [
        {"M": 7, "K": 64,  "N": 7,   "transpose_b": 1, "scale_div": 8.0},
        {"M": 7, "K": 7,   "N": 64,  "transpose_b": 0, "scale_div": 1.0},
        {"M": 7, "K": 512, "N": 512, "transpose_b": 0, "scale_div": 1.0},
        {"M": 3, "K": 33,  "N": 5,   "transpose_b": 1, "scale_div": 1.0},
        {"M": 3, "K": 31,  "N": 33,  "transpose_b": 0, "scale_div": 1.0},
        # M in the hundreds. The IME micro-tile is 4x4x8 and hardware-forced,
        # so at M=7 the second row-tile is half padding and the comparison is
        # unfair to it. These are the shapes where the tile is fully used --
        # a real transformer's MLP block, not attention's tiny M.
        {"M": 64,  "K": 512, "N": 512, "transpose_b": 0, "scale_div": 1.0},
        {"M": 128, "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
        # The IME crossover sweep. K and N held fixed, M swept across the
        # hardware-forced 4-row micro-tile, because M is what decides whether
        # the MAC unit is fed or padded. The scheduler needs to know WHERE the
        # IME cost cell stops being the cheaper one, and interpolating between
        # M=7 and M=64 would invent the answer.
        {"M": 4,  "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
        {"M": 8,  "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
        {"M": 16, "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
        {"M": 32, "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
        {"M": 64, "K": 256, "N": 256, "transpose_b": 1, "scale_div": 1.0},
    ],
}


def collect_cases(op_name, models):
    """Unique (shape, quant) tuples for `op_name` across the given models."""
    cases, seen = [], set()
    for model in models:
        for root in ("k1", "k1_xpurt"):
            graph = os.path.join(REPO, "build", root, model, "int8",
                                 "graph.json")
            if os.path.exists(graph):
                break
        else:
            continue
        ir = json.load(open(graph))
        for node in ir["ops"]:
            if node["op"] != op_name:
                continue
            case = dict(node.get("shape") or {})
            case.update(node.get("quant") or {})
            key = tuple(sorted(case.items()))
            if key in seen:
                continue
            seen.add(key)
            case["tag"] = f"{model}:{node['name']}"
            cases.append(case)

    # Shapes the graphs do not happen to contain but the kernel must survive:
    # strip boundaries and a large K. Quant values are borrowed from the first
    # real case so the requantize tail sees realistic scales rather than 1.0.
    # Ops no model emits get their cases outright -- there is no donor to
    # extend, so EXTRA_CASES below cannot reach them.
    for i, extra in enumerate(SYNTHETIC_CASES.get(op_name, [])):
        case = dict(extra)
        case.update(SYNTHETIC_QUANT.get(op_name, {}))
        key = tuple(sorted(case.items()))
        if key in seen:
            continue
        seen.add(key)
        case["tag"] = f"synthetic:{op_name}#{i}"
        cases.append(case)

    donor = dict(cases[0]) if cases else {}
    for extra in EXTRA_CASES.get(op_name, []):
        case = {k: v for k, v in donor.items() if k not in ("tag",)}
        case.update(extra)
        key = tuple(sorted(case.items()))
        if key in seen:
            continue
        seen.add(key)
        case["tag"] = ("synthetic:M%(M)dK%(K)dN%(N)dtb%(transpose_b)d" % case)
        cases.append(case)
    return cases


# --------------------------------------------------------------------------
# Per-op harness emitters. Each returns the C main() that allocates the
# buffers, fills them in both regimes, calls kernel_cand and kernel_ref, and
# prints a per-case max_abs_err.
# --------------------------------------------------------------------------

PRELUDE = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static uint64_t rs;
static uint32_t xr(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}
static int8_t rnd8(int lim) { return (int8_t)((int)(xr() % (2*lim)) - lim); }
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static int worst = 0;
static void report(const char *tag, const char *regime, const char *dims,
                   const int8_t *cand, const int8_t *ref, long n,
                   double tc, double tr) {
    int maxerr = 0; long ndiff = 0;
    for (long k = 0; k < n; k++) {
        int d = cand[k] - ref[k];
        if (d < 0) d = -d;
        if (d) ndiff++;
        if (d > maxerr) maxerr = d;
    }
    if (maxerr > worst) worst = maxerr;
    printf("CASE %-34s regime=%-5s %-34s n=%7ld max_abs_err=%d n_diff=%ld "
           "cand=%.3fms ref=%.3fms speedup=%.2fx\n",
           tag, regime, dims, n, maxerr, ndiff, tc, tr,
           tr / (tc > 0 ? tc : 1e-9));
    fflush(stdout);
}
"""


def emit_lstm(cases):
    max_is = max(c["input_size"] for c in cases)
    max_h = max(c["hidden_size"] for c in cases)
    rows = []
    for c in cases:
        rows.append(
            '    { "%(tag)s", %(input_size)d, %(hidden_size)d, %(scale_in).9ef,'
            ' %(scale_w_ih).9ef, %(scale_w_hh).9ef, %(scale_b).9ef,'
            ' %(scale_h).9ef, %(scale_c).9ef, %(has_bias)d },' % c)
    return PRELUDE + f"""
#define MAX_IS {max_is}
#define MAX_H  {max_h}

typedef struct {{
    const char *tag;
    int input_size, hidden_size;
    float scale_in, scale_w_ih, scale_w_hh, scale_b, scale_h, scale_c;
    int has_bias;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, const int8_t *,
                        const int32_t *, const int32_t *, int8_t *, int8_t *,
                        int8_t *, int, int, float, float, float, float, float,
                        float, int);
extern void kernel_ref(const int8_t *, const int8_t *, const int8_t *,
                       const int32_t *, const int32_t *, int8_t *, int8_t *,
                       int8_t *, int, int, float, float, float, float, float,
                       float, int);

static int8_t  IN[MAX_IS];
static int8_t  W_IH[4*MAX_H*MAX_IS];
static int8_t  W_HH[4*MAX_H*MAX_H];
static int32_t B_IH[4*MAX_H], B_HH[4*MAX_H];
static int8_t  H0[MAX_H], C0[MAX_H];
static int8_t  HC[MAX_H], CC[MAX_H], OC_[MAX_H];
static int8_t  HR[MAX_H], CR[MAX_H], OR_[MAX_H];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        const int IS = c->input_size, H = c->hidden_size;
        char dims[64];
        snprintf(dims, sizeof dims, "IS%d H%d bias%d", IS, H, c->has_bias);
        for (int regime = 0; regime < 2; regime++) {{
            /* "full" saturates the gates -- a saturated sigmoid HIDES a
             * rounding difference, so "small" keeps the pre-activations in
             * the linear part of the curve where a difference would show. */
            int wlim = regime ? 8 : 128;
            int ilim = regime ? 32 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < IS; k++) IN[k] = rnd8(ilim);
            for (long k = 0; k < 4L*H*IS; k++) W_IH[k] = rnd8(wlim);
            for (long k = 0; k < 4L*H*H;  k++) W_HH[k] = rnd8(wlim);
            for (int k = 0; k < 4*H; k++) {{
                B_IH[k] = (int32_t)((int)(xr() % 8192) - 4096);
                B_HH[k] = (int32_t)((int)(xr() % 8192) - 4096);
            }}
            /* Non-zero recurrent state: a zeroed h/c would leave the w_hh
             * reduction multiplying by zero, which is exactly the half of
             * the kernel this test exists for. */
            for (int k = 0; k < H; k++) {{ H0[k] = rnd8(ilim); C0[k] = rnd8(ilim); }}
            memcpy(HC, H0, H); memcpy(CC, C0, H);
            memcpy(HR, H0, H); memcpy(CR, C0, H);
            memset(OC_, 0, H); memset(OR_, 0, H);

            double t0 = now_ms();
            kernel_cand(IN, W_IH, W_HH, B_IH, B_HH, HC, CC, OC_, IS, H,
                        c->scale_in, c->scale_w_ih, c->scale_w_hh, c->scale_b,
                        c->scale_h, c->scale_c, c->has_bias);
            double t1 = now_ms();
            kernel_ref(IN, W_IH, W_HH, B_IH, B_HH, HR, CR, OR_, IS, H,
                       c->scale_in, c->scale_w_ih, c->scale_w_hh, c->scale_b,
                       c->scale_h, c->scale_c, c->has_bias);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OC_, OR_, H,
                   t1 - t0, t2 - t1);
            /* The updated state is an output too, and a kernel that got
             * `output` right while corrupting c_state would pass on h
             * alone for one timestep. */
            report(c->tag, regime ? "small" : "full", "c_state", CC, CR, H,
                   t1 - t0, t2 - t1);
        }}
        /* has_bias is 0 in every graph here (VitFly's LSTM is built with
         * bias=False), so exercise the bias path explicitly rather than
         * shipping a branch nothing ever ran. */
        if (!c->has_bias) {{
            rs = 0xD1B54A32D192ED03ull ^ ((uint64_t)i << 8);
            for (int k = 0; k < IS; k++) IN[k] = rnd8(32);
            for (long k = 0; k < 4L*H*IS; k++) W_IH[k] = rnd8(8);
            for (long k = 0; k < 4L*H*H;  k++) W_HH[k] = rnd8(8);
            for (int k = 0; k < 4*H; k++) {{
                B_IH[k] = (int32_t)((int)(xr() % 8192) - 4096);
                B_HH[k] = (int32_t)((int)(xr() % 8192) - 4096);
            }}
            for (int k = 0; k < H; k++) {{ H0[k] = rnd8(32); C0[k] = rnd8(32); }}
            memcpy(HC, H0, H); memcpy(CC, C0, H);
            memcpy(HR, H0, H); memcpy(CR, C0, H);
            kernel_cand(IN, W_IH, W_HH, B_IH, B_HH, HC, CC, OC_, IS, H,
                        c->scale_in, c->scale_w_ih, c->scale_w_hh, c->scale_b,
                        c->scale_h, c->scale_c, 1);
            kernel_ref(IN, W_IH, W_HH, B_IH, B_HH, HR, CR, OR_, IS, H,
                       c->scale_in, c->scale_w_ih, c->scale_w_hh, c->scale_b,
                       c->scale_h, c->scale_c, 1);
            report(c->tag, "bias", dims, OC_, OR_, H, 1, 1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_add(cases):
    max_n = max(c["n"] for c in cases)
    rows = ['    { "%(tag)s", %(n)d, %(scale_a).9ef, %(scale_b).9ef,'
            ' %(scale_out).9ef, %(activation_min)d, %(activation_max)d },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_N {max_n}
typedef struct {{
    const char *tag; int n;
    float scale_a, scale_b, scale_out; int amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, int8_t *, int,
                        float, float, float, int, int);
extern void kernel_ref(const int8_t *, const int8_t *, int8_t *, int,
                       float, float, float, int, int);

static int8_t A[MAX_N], B[MAX_N], OUTC[MAX_N], OUTR[MAX_N];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[64];
        snprintf(dims, sizeof dims, "n=%d act[%d,%d]", c->n, c->amin, c->amax);
        for (int regime = 0; regime < 2; regime++) {{
            /* "full" drives the sum past the output clamp on both sides;
             * "small" keeps it inside, so the rounding of the divide is
             * what decides the answer instead of the clamp. */
            int lim = regime ? 24 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < c->n; k++) {{ A[k] = rnd8(lim); B[k] = rnd8(lim); }}
            memset(OUTC, 0, c->n); memset(OUTR, 0, c->n);
            double t0 = now_ms();
            kernel_cand(A, B, OUTC, c->n, c->scale_a, c->scale_b,
                        c->scale_out, c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref(A, B, OUTR, c->n, c->scale_a, c->scale_b,
                       c->scale_out, c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   c->n, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_avgpool(cases):
    max_in = max(c["N"]*c["C"]*c["IH"]*c["IW"] for c in cases)
    max_out = max(c["N"]*c["C"]*c["OH"]*c["OW"] for c in cases)
    rows = ['    { "%(tag)s", %(N)d,%(C)d,%(IH)d,%(IW)d,%(KH)d,%(KW)d,'
            '%(SH)d,%(SW)d,%(PH)d,%(PW)d,%(count_include_pad)d },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_IN  {max_in}
#define MAX_OUT {max_out}
typedef struct {{
    const char *tag;
    int N, C, IH, IW, KH, KW, SH, SW, PH, PW, cip;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, int8_t *, int, int, int, int,
                        int, int, int, int, int, int, int);
extern void kernel_ref(const int8_t *, int8_t *, int, int, int, int,
                       int, int, int, int, int, int, int);

static int8_t IN[MAX_IN], OUTC[MAX_OUT], OUTR[MAX_OUT];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        int OH = (c->IH + 2*c->PH - c->KH) / c->SH + 1;
        int OW = (c->IW + 2*c->PW - c->KW) / c->SW + 1;
        long n_in = (long)c->N*c->C*c->IH*c->IW;
        long n_out = (long)c->N*c->C*OH*OW;
        char dims[80];
        snprintf(dims, sizeof dims, "C%d %dx%d K%d S%d P%d cip%d",
                 c->C, c->IH, c->IW, c->KH, c->SH, c->PH, c->cip);
        for (int regime = 0; regime < 2; regime++) {{
            /* "full" reaches the +-128 output clamp; "small" keeps the mean
             * inside it so the half-away-from-zero rounding of the divide
             * is what the comparison is actually testing. */
            int lim = regime ? 20 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (long k = 0; k < n_in; k++) IN[k] = rnd8(lim);
            memset(OUTC, 0, n_out); memset(OUTR, 0, n_out);
            double t0 = now_ms();
            kernel_cand(IN, OUTC, c->N, c->C, c->IH, c->IW, c->KH, c->KW,
                        c->SH, c->SW, c->PH, c->PW, c->cip);
            double t1 = now_ms();
            kernel_ref(IN, OUTR, c->N, c->C, c->IH, c->IW, c->KH, c->KW,
                       c->SH, c->SW, c->PH, c->PW, c->cip);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   n_out, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def _emit_unary(cases, extra_field, extra_c_type, call_extra):
    max_n = max(c["n"] for c in cases)
    rows = []
    for c in cases:
        row = ('    { "%(tag)s", %(n)d, %(scale_in).9ef, %(scale_out).9ef,'
               ' %(activation_min)d, %(activation_max)d' % c)
        if extra_field:
            row += ", %.9ef" % c[extra_field]
        rows.append(row + " },")
    field = f"    {extra_c_type} {extra_field};\n" if extra_field else ""
    return PRELUDE + f"""
#define MAX_N {max_n}
typedef struct {{
    const char *tag; int n;
    float scale_in, scale_out; int amin, amax;
{field}}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, int8_t *, int, float, float, int, int
                        {', float' if extra_field else ''});
extern void kernel_ref(const int8_t *, int8_t *, int, float, float, int, int
                       {', float' if extra_field else ''});

static int8_t IN[MAX_N], OUTC[MAX_N], OUTR[MAX_N];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[64];
        snprintf(dims, sizeof dims, "n=%d act[%d,%d]", c->n, c->amin, c->amax);
        for (int regime = 0; regime < 2; regime++) {{
            /* "small" concentrates the input into a narrow band, which is
             * both the sensitive part of the curve and the case the
             * memoized table is built for -- few distinct bytes, many
             * repeats. "full" spans all 256. */
            int lim = regime ? 12 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < c->n; k++) IN[k] = rnd8(lim);
            memset(OUTC, 0, c->n); memset(OUTR, 0, c->n);
            double t0 = now_ms();
            kernel_cand(IN, OUTC, c->n, c->scale_in, c->scale_out,
                        c->amin, c->amax{call_extra});
            double t1 = now_ms();
            kernel_ref(IN, OUTR, c->n, c->scale_in, c->scale_out,
                       c->amin, c->amax{call_extra});
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   c->n, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_matmul(cases):
    """out[M,N] = a[M,K] @ (transpose_b ? b[N,K]^T : b[K,N]), int8.

    Both regimes matter here for a specific reason. "full" drives the int32
    accumulator hard, so a widening mistake (accumulating in i16, or an LMUL
    that cannot hold the sum) shows as saturation. "small" keeps the products
    low so the FLOAT REQUANTIZE decides the answer -- which is where a
    vectorized tail would diverge from the reference's roundf, since RVV float
    conversion rounds to nearest-even and roundf rounds half away from zero.
    A kernel that vectorized the tail passes "full" and fails "small".
    """
    max_a = max(c["M"] * c["K"] for c in cases)
    max_b = max(c["K"] * c["N"] for c in cases)
    max_o = max(c["M"] * c["N"] for c in cases)
    rows = ['    { "%(tag)s", %(M)d, %(K)d, %(N)d, %(transpose_b)d,'
            ' %(scale_a).9ef, %(scale_b).9ef, %(scale_out).9ef,'
            ' %(scale_div).9ef, %(activation_min)d, %(activation_max)d },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_A {max_a}
#define MAX_B {max_b}
#define MAX_O {max_o}
typedef struct {{
    const char *tag; int M, K, N, transpose_b;
    float scale_a, scale_b, scale_out, scale_div; int amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, int8_t *,
                        int, int, int, float, float, float, int, float,
                        int, int);
extern void kernel_ref(const int8_t *, const int8_t *, int8_t *,
                       int, int, int, float, float, float, int, float,
                       int, int);

static int8_t A[MAX_A], B[MAX_B], OUTC[MAX_O], OUTR[MAX_O];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[96];
        snprintf(dims, sizeof dims, "M=%d K=%d N=%d tb=%d act[%d,%d]",
                 c->M, c->K, c->N, c->transpose_b, c->amin, c->amax);
        int na = c->M * c->K, nb = c->K * c->N, no = c->M * c->N;
        for (int regime = 0; regime < 2; regime++) {{
            int lim = regime ? 8 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < na; k++) A[k] = rnd8(lim);
            for (int k = 0; k < nb; k++) B[k] = rnd8(lim);
            memset(OUTC, 0, no); memset(OUTR, 0, no);
            double t0 = now_ms();
            kernel_cand(A, B, OUTC, c->M, c->K, c->N, c->scale_a, c->scale_b,
                        c->scale_out, c->transpose_b, c->scale_div,
                        c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref(A, B, OUTR, c->M, c->K, c->N, c->scale_a, c->scale_b,
                       c->scale_out, c->transpose_b, c->scale_div,
                       c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   no, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_softmax(cases):
    """out[m,:] = softmax(in[m,:] * scale_in) quantized to int8.

    BIT-EXACTNESS IS GENUINELY HARD HERE and the number this reports is the
    interesting part. The reference calls `expf` per element and accumulates
    the denominator in float, sequentially. Float addition is not associative,
    so ANY vector reduction gives a different sum, and a vectorized exp differs
    from libm's `expf` in the last ulp. A kernel that keeps the row sum scalar
    and LUTs the exponential can be exact; one that reduces in vector form
    cannot. Report what it actually is rather than assuming either.
    """
    max_n = max(c["M"] * c["K"] for c in cases)
    rows = ['    { "%(tag)s", %(M)d, %(K)d, %(scale_in).9ef, %(scale_out).9ef },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_N {max_n}
typedef struct {{ const char *tag; int M, K; float scale_in, scale_out; }} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, int8_t *, int, int, float, float);
extern void kernel_ref(const int8_t *, int8_t *, int, int, float, float);

static int8_t A[MAX_N], OUTC[MAX_N], OUTR[MAX_N];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[64];
        snprintf(dims, sizeof dims, "M=%d K=%d", c->M, c->K);
        int n = c->M * c->K;
        for (int regime = 0; regime < 2; regime++) {{
            /* "small" keeps the row range narrow so no single element
             * dominates the softmax -- that is where the denominator's
             * rounding decides the output rather than one saturating term. */
            int lim = regime ? 8 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < n; k++) A[k] = rnd8(lim);
            memset(OUTC, 0, n); memset(OUTR, 0, n);
            double t0 = now_ms();
            kernel_cand(A, OUTC, c->M, c->K, c->scale_in, c->scale_out);
            double t1 = now_ms();
            kernel_ref(A, OUTR, c->M, c->K, c->scale_in, c->scale_out);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   n, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_layernorm(cases):
    """Row-wise layernorm with float gamma/beta.

    The reference accumulates mean and variance in DOUBLE and takes a double
    `sqrt`. Reproducing that bit-exactly in vector form means reproducing the
    summation ORDER, so the same caveat as softmax applies -- and more sharply,
    because two reductions have to match, not one.
    """
    max_n = max(c["M"] * c["K"] for c in cases)
    max_k = max(c["K"] for c in cases)
    rows = ['    { "%(tag)s", %(M)d, %(K)d, %(scale_in).9ef, %(scale_out).9ef,'
            ' %(eps).9ef, %(activation_min)d, %(activation_max)d },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_N {max_n}
#define MAX_K {max_k}
typedef struct {{
    const char *tag; int M, K; float scale_in, scale_out, eps; int amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const float *, const float *, int8_t *,
                        int, int, float, float, float, int, int);
extern void kernel_ref(const int8_t *, const float *, const float *, int8_t *,
                       int, int, float, float, float, int, int);

static int8_t A[MAX_N], OUTC[MAX_N], OUTR[MAX_N];
static float GAMMA[MAX_K], BETA[MAX_K];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[80];
        snprintf(dims, sizeof dims, "M=%d K=%d act[%d,%d]",
                 c->M, c->K, c->amin, c->amax);
        int n = c->M * c->K;
        for (int regime = 0; regime < 2; regime++) {{
            int lim = regime ? 8 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < n; k++) A[k] = rnd8(lim);
            /* gamma near 1 and beta near 0, as a trained net has them: an
             * arbitrary gamma would swamp the normalization and hide a
             * variance bug behind a large multiply. */
            for (int k = 0; k < c->K; k++) {{
                GAMMA[k] = 1.0f + (float)(rnd8(32)) / 256.0f;
                BETA[k]  = (float)(rnd8(32)) / 512.0f;
            }}
            memset(OUTC, 0, n); memset(OUTR, 0, n);
            double t0 = now_ms();
            kernel_cand(A, GAMMA, BETA, OUTC, c->M, c->K, c->scale_in,
                        c->scale_out, c->eps, c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref(A, GAMMA, BETA, OUTR, c->M, c->K, c->scale_in,
                       c->scale_out, c->eps, c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   n, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_linear(cases):
    """out[M,N] = requant(bias[n] + sum_k in[m,k]*w[n,k]), int8.

    The requantize here is a Q0.31 multiply plus a rounding shift -- INTEGER,
    unlike matmul_s8's float scale. So bit-exactness is a question of getting
    the int64 arithmetic right rather than of matching a float rounding mode,
    and both regimes test the same thing: "full" drives the int32 accumulator
    toward saturation, "small" keeps it where the rounding shift decides the
    last bit.
    """
    max_i = max(c["M"] * c["K"] for c in cases)
    max_w = max(c["K"] * c["N"] for c in cases)
    max_o = max(c["M"] * c["N"] for c in cases)
    max_n = max(c["N"] for c in cases)
    rows = ['    { "%(tag)s", %(M)d, %(K)d, %(N)d, %(input_offset)d,'
            ' %(filter_offset)d, %(output_offset)d, %(output_multiplier)d,'
            ' %(output_shift)d, %(activation_min)d, %(activation_max)d },' % c
            for c in cases]
    return PRELUDE + f"""
#define MAX_I {max_i}
#define MAX_W {max_w}
#define MAX_O {max_o}
#define MAX_N {max_n}
typedef struct {{
    const char *tag; int M, K, N;
    int in_off, filt_off, out_off, out_mult, out_shift, amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, const int32_t *,
                        int8_t *, int, int, int, int, int, int, int, int,
                        int, int);
extern void kernel_ref(const int8_t *, const int8_t *, const int32_t *,
                       int8_t *, int, int, int, int, int, int, int, int,
                       int, int);

static int8_t A[MAX_I], W[MAX_W], OUTC[MAX_O], OUTR[MAX_O];
static int32_t BIAS[MAX_N];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        char dims[96];
        snprintf(dims, sizeof dims, "M=%d K=%d N=%d act[%d,%d]",
                 c->M, c->K, c->N, c->amin, c->amax);
        int ni = c->M * c->K, nw = c->K * c->N, no = c->M * c->N;
        for (int regime = 0; regime < 2; regime++) {{
            int lim = regime ? 8 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (int k = 0; k < ni; k++) A[k] = rnd8(lim);
            for (int k = 0; k < nw; k++) W[k] = rnd8(lim);
            for (int k = 0; k < c->N; k++) BIAS[k] = (int32_t)rnd8(64) * 37;
            memset(OUTC, 0, no); memset(OUTR, 0, no);
            double t0 = now_ms();
            kernel_cand(A, W, BIAS, OUTC, c->M, c->K, c->N, c->in_off,
                        c->filt_off, c->out_off, c->out_mult, c->out_shift,
                        c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref(A, W, BIAS, OUTR, c->M, c->K, c->N, c->in_off,
                       c->filt_off, c->out_off, c->out_mult, c->out_shift,
                       c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   no, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_adaptive_avg_pool(cases):
    max_in = max(c["N"]*c["C"]*c["IH"]*c["IW"] for c in cases)
    max_out = max(c["N"]*c["C"]*c["OH"]*c["OW"] for c in cases)
    rows = ['    { "%(tag)s", %(N)d,%(C)d,%(IH)d,%(IW)d,%(OH)d,%(OW)d,'
            '%(scale_in).9ef,%(scale_out).9ef,%(activation_min)d,'
            '%(activation_max)d },' % c for c in cases]
    return PRELUDE + f"""
#define MAX_IN  {max_in}
#define MAX_OUT {max_out}
typedef struct {{
    const char *tag;
    int N, C, IH, IW, OH, OW;
    float scale_in, scale_out;
    int amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, int8_t *, int, int, int, int, int, int,
                        float, float, int, int);
extern void kernel_ref (const int8_t *, int8_t *, int, int, int, int, int, int,
                        float, float, int, int);

static int8_t IN[MAX_IN], OUTC[MAX_OUT], OUTR[MAX_OUT];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        long n_in  = (long)c->N*c->C*c->IH*c->IW;
        long n_out = (long)c->N*c->C*c->OH*c->OW;
        char dims[96];
        snprintf(dims, sizeof dims, "C%d %dx%d -> %dx%d",
                 c->C, c->IH, c->IW, c->OH, c->OW);
        for (int regime = 0; regime < 2; regime++) {{
            int lim = regime ? 20 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (long k = 0; k < n_in; k++) IN[k] = rnd8(lim);
            memset(OUTC, 0, n_out); memset(OUTR, 0, n_out);
            double t0 = now_ms();
            kernel_cand(IN, OUTC, c->N, c->C, c->IH, c->IW, c->OH, c->OW,
                        c->scale_in, c->scale_out, c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref (IN, OUTR, c->N, c->C, c->IH, c->IW, c->OH, c->OW,
                        c->scale_in, c->scale_out, c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   n_out, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def emit_depthwise(cases):
    max_in = max(c["N"]*c["C"]*c["IH"]*c["IW"] for c in cases)
    def _o(c, d):
        return ((c[d] + 2*c["P"+("H" if d=="IH" else "W")]
                 - c["K"+("H" if d=="IH" else "W")])
                // c["S"+("H" if d=="IH" else "W")]) + 1
    max_out = max(c["N"]*c["C"]*_o(c,"IH")*_o(c,"IW") for c in cases)
    max_w = max(c["C"]*c["KH"]*c["KW"] for c in cases)
    rows = ['    { "%(tag)s", %(N)d,%(C)d,%(IH)d,%(IW)d,%(KH)d,%(KW)d,'
            '%(SH)d,%(SW)d,%(PH)d,%(PW)d,%(input_offset)d,%(filter_offset)d,'
            '%(output_offset)d,%(output_multiplier)d,%(output_shift)d,'
            '%(activation_min)d,%(activation_max)d },' % c for c in cases]
    return PRELUDE + f"""
#define MAX_IN  {max_in}
#define MAX_OUT {max_out}
#define MAX_W   {max_w}
typedef struct {{
    const char *tag;
    int N, C, IH, IW, KH, KW, SH, SW, PH, PW;
    int input_offset, filter_offset, output_offset;
    int output_multiplier, output_shift, amin, amax;
}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, const int32_t *,
                        int8_t *, int, int, int, int, int, int, int, int,
                        int, int, int, int, int, int, int, int, int);
extern void kernel_ref (const int8_t *, const int8_t *, const int32_t *,
                        int8_t *, int, int, int, int, int, int, int, int,
                        int, int, int, int, int, int, int, int, int);

static int8_t IN[MAX_IN], W[MAX_W], OUTC[MAX_OUT], OUTR[MAX_OUT];
static int32_t B[MAX_W];

int main(void) {{
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        int OH = (c->IH + 2*c->PH - c->KH) / c->SH + 1;
        int OW = (c->IW + 2*c->PW - c->KW) / c->SW + 1;
        long n_in  = (long)c->N*c->C*c->IH*c->IW;
        long n_out = (long)c->N*c->C*OH*OW;
        long n_w   = (long)c->C*c->KH*c->KW;
        char dims[110];
        snprintf(dims, sizeof dims, "C%d %dx%d K%dx%d S%d P%d -> %dx%d",
                 c->C, c->IH, c->IW, c->KH, c->KW, c->SH, c->PH, OH, OW);
        for (int regime = 0; regime < 2; regime++) {{
            int lim = regime ? 20 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (long k = 0; k < n_in; k++) IN[k] = rnd8(lim);
            for (long k = 0; k < n_w;  k++) W[k]  = rnd8(lim);
            /* Bias in the range a real per-channel int32 bias occupies. */
            for (long k = 0; k < c->C; k++) B[k] = (int32_t)rnd8(64) * 137;
            memset(OUTC, 0, n_out); memset(OUTR, 0, n_out);
            double t0 = now_ms();
            kernel_cand(IN, W, B, OUTC, c->N, c->C, c->IH, c->IW,
                        c->KH, c->KW, c->SH, c->SW, c->PH, c->PW,
                        c->input_offset, c->filter_offset, c->output_offset,
                        c->output_multiplier, c->output_shift,
                        c->amin, c->amax);
            double t1 = now_ms();
            kernel_ref (IN, W, B, OUTR, c->N, c->C, c->IH, c->IW,
                        c->KH, c->KW, c->SH, c->SW, c->PH, c->PW,
                        c->input_offset, c->filter_offset, c->output_offset,
                        c->output_multiplier, c->output_shift,
                        c->amin, c->amax);
            double t2 = now_ms();
            report(c->tag, regime ? "small" : "full", dims, OUTC, OUTR,
                   n_out, t1 - t0, t2 - t1);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


EMITTERS = {
    "lstm_s8": emit_lstm,
    "add_s8": emit_add,
    "avgpool2d_s8": emit_avgpool,
    "elu_s8": lambda cs: _emit_unary(cs, "alpha", "float", ", c->alpha"),
    "leaky_relu_s8": lambda cs: _emit_unary(cs, "negative_slope", "float",
                                            ", c->negative_slope"),
    "sigmoid_s8": lambda cs: _emit_unary(cs, None, None, ""),
    "matmul_s8": emit_matmul,
    # mul_s8 is signature-identical to add_s8, and gelu_s8 to
    # sigmoid_s8, so both reuse the existing harnesses.
    "mul_s8": emit_add,
    "gelu_s8": lambda cs: _emit_unary(cs, None, None, ""),
    # RoPE pair: same signature as sigmoid_s8, so same harness.
    "sin_s8": lambda cs: _emit_unary(cs, None, None, ""),
    "cos_s8": lambda cs: _emit_unary(cs, None, None, ""),
    "softmax_s8": emit_softmax,
    "layernorm_s8": emit_layernorm,
    "linear_s8": emit_linear,
    # ViNT ops that no model in this tree emits; their cases come from
    # SYNTHETIC_CASES rather than from a graph.
    "adaptive_avg_pool2d_s8": emit_adaptive_avg_pool,
    "depthwise_conv2d_s8": emit_depthwise,
}


def run_op(op_name, models, keep_dir=None, kernel_path=None):
    cases = collect_cases(op_name, models)
    if not cases:
        print(f"[{op_name}] no cases found in {models}")
        return None
    print(f"[{op_name}] {len(cases)} unique (shape, quant) case(s)")
    workdir = keep_dir or tempfile.mkdtemp(prefix=f"k1v_{op_name}_")
    os.makedirs(workdir, exist_ok=True)

    prologue = ("#include <stddef.h>\n#include <float.h>\n#include <stdint.h>\n"
                "#include <string.h>\n#include <math.h>\n"
                "#include <riscv_vector.h>\n"
                '#include "mb_rvv_vxrm_compat.h"\n')
    cand_c = os.path.join(workdir, "cand.c")
    with open(cand_c, "w") as f:
        f.write(prologue)
        f.write(open(kernel_path or os.path.join(
            REPO, CURATED[op_name])).read())
    ref_c = os.path.join(workdir, "ref.c")
    with open(ref_c, "w") as f:
        f.write("#include <stddef.h>\n#include <stdint.h>\n"
                "#include <string.h>\n#include <math.h>\n")
        f.write(KERNEL_SPECS[op_name].reference_impl)
    main_c = os.path.join(workdir, "main.c")
    with open(main_c, "w") as f:
        f.write(EMITTERS[op_name](cases))

    elf = os.path.join(workdir, f"verify_{op_name}")
    common = [f"{CROSS}gcc", "-O2", "-static", *MARCH,
              f"-I{os.path.join(REPO, 'kernels', 'rvv')}"]
    for src, sym, obj in ((cand_c, "kernel_cand", "cand.o"),
                          (ref_c, "kernel_ref", "ref.o")):
        subprocess.run(common + [f"-Dkernel_{op_name}={sym}", "-c", src,
                                 "-o", os.path.join(workdir, obj)], check=True)
    subprocess.run(common + ["-c", main_c, "-o", os.path.join(workdir, "main.o")],
                   check=True)
    subprocess.run(common + [os.path.join(workdir, "main.o"),
                             os.path.join(workdir, "cand.o"),
                             os.path.join(workdir, "ref.o"),
                             "-o", elf, "-lm"], check=True)

    # vtype gate on the linked binary, the same check the deploy path applies.
    vt = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "check_rvv_vtype.py"),
         "--objdump", f"{CROSS}objdump", elf],
        capture_output=True, text=True)
    print(vt.stdout.strip() or vt.stderr.strip())
    if vt.returncode != 0:
        print(f"[{op_name}] vtype check FAILED; not deploying")
        return False

    subprocess.run(["ssh", HOST, f"mkdir -p {REMOTE_ROOT}/verify"], check=True)
    subprocess.run(["scp", "-q", elf, f"{HOST}:{REMOTE_ROOT}/verify/"],
                   check=True)
    proc = subprocess.run(
        ["ssh", HOST,
         f"cd {REMOTE_ROOT}/verify && taskset -c 3 ./verify_{op_name}"],
        capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.stderr.strip():
        print(proc.stderr, file=sys.stderr)
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--op", action="append", default=[])
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--kernel", default=None,
                    help="verify THIS file against the op's reference instead "
                         "of the one wired into CURATED. Needed for a second "
                         "implementation of an op that already has one -- the "
                         "IME matmul against the RVV matmul, say, where both "
                         "are matmul_s8 and only one can be the curated pick. "
                         "Requires exactly one --op.")
    args = ap.parse_args()
    if args.kernel and len(args.op) != 1:
        ap.error("--kernel names one file, so it needs exactly one --op")
    models = args.model or MODELS
    ops = args.op or list(CURATED)
    ok = True
    for op in ops:
        res = run_op(op, models,
                     keep_dir=(os.path.join(args.workdir, op)
                               if args.workdir else None),
                     kernel_path=args.kernel)
        if res is False:
            ok = False
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
