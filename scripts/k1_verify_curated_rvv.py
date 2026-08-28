#!/usr/bin/env python3
"""Bit-exact verify of curated RVV kernels, on the K1 itself.

Same construction and the same reasons as k1_verify_fused_conv_rvv.py, which
did this for the fused convolutions; this covers the ops that were still
falling through to the scalar reference after those landed -- lstm_s8,
add_s8, avgpool2d_s8, elu_s8, leaky_relu_s8, sigmoid_s8.

Why on the board and not on the host
------------------------------------
rvv_x60's declared verify_method is VERIFY_HOST_CTYPES: it compiles the
candidate with the HOST compiler and dlopen()s it. That works for a scalar
kernel and cannot work for one written in RVV intrinsics, because the host is
x86. Spike bakes VLEN at build time and does not model this toolchain's
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
          "vitfly_frontend", "mlp_control"]

CURATED = {
    "lstm_s8": "kernels/rvv/rvv_lstm_s8_rvv_gate_int_dot.c",
    "add_s8": "kernels/rvv/rvv_add_s8_rvv_frm_rmm.c",
    "avgpool2d_s8": "kernels/rvv/rvv_avgpool2d_s8_rvv_ow_lanes.c",
    "elu_s8": "kernels/rvv/rvv_elu_s8_rvv_memo_lut_gather.c",
    "leaky_relu_s8": "kernels/rvv/rvv_leaky_relu_s8_rvv_frm_rmm.c",
    "sigmoid_s8": "kernels/rvv/rvv_sigmoid_s8_rvv_memo_lut_gather.c",
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


EMITTERS = {
    "lstm_s8": emit_lstm,
    "add_s8": emit_add,
    "avgpool2d_s8": emit_avgpool,
    "elu_s8": lambda cs: _emit_unary(cs, "alpha", "float", ", c->alpha"),
    "leaky_relu_s8": lambda cs: _emit_unary(cs, "negative_slope", "float",
                                            ", c->negative_slope"),
    "sigmoid_s8": lambda cs: _emit_unary(cs, None, None, ""),
}


def run_op(op_name, models, keep_dir=None):
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
        f.write(open(os.path.join(REPO, CURATED[op_name])).read())
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
    args = ap.parse_args()
    models = args.model or MODELS
    ops = args.op or list(CURATED)
    ok = True
    for op in ops:
        res = run_op(op, models,
                     keep_dir=(os.path.join(args.workdir, op)
                               if args.workdir else None))
        if res is False:
            ok = False
    print("VERIFY", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
