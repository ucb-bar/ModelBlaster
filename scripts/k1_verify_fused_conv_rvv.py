#!/usr/bin/env python3
"""Bit-exact verify of the curated RVV fused-conv kernels, on the K1 itself.

Why on the board and not on the host
------------------------------------
rvv_x60's declared verify_method is VERIFY_HOST_CTYPES, which compiles the
candidate with the HOST compiler and dlopen()s it. That works for a scalar
kernel; it cannot work for one written in RVV intrinsics, because the host is
x86. Spike is the other option, but spike bakes VLEN at build time and does not
model the SpaceMiT toolchain's codegen -- and the properties at issue here are
exactly VLEN- and codegen-dependent (vl-tail handling, vsmul/vnclip rounding,
whether GCC contracts `bn_s * fv + bn_b` into an fma).

So this runs the comparison where the answer counts: the curated kernel and the
op's scalar reference_impl, cross-compiled into ONE binary with the real
backend flags, executed on the board over the real (shape, quant) tuples taken
from the model graphs, with the outputs compared element-wise on-device.
Acceptance is max_abs_err == 0.

    scripts/k1_verify_fused_conv_rvv.py --model dronet --model yolov8_nano

Note this checks kernel-vs-reference, not the weight LAYOUT: both sides index
IHWOC because both are compiled with -DMODELBLASTER_RVV_IHWOC_WEIGHTS, exactly
as the real build does. A packing mismatch would be invisible here and is
caught by the end-to-end golden compare in validation/k1_runner.py instead.
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

from modelblaster.pipeline.reference_kernels import (  # noqa: E402
    CONV2D_BATCHNORM2D_S8, CONV2D_BATCHNORM2D_SILU_S8,
)

CROSS = os.environ.get(
    "CROSS",
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-",
)
HOST = os.environ.get("MODELBLASTER_K1_HOST", "k1")
REMOTE_ROOT = os.environ.get("MODELBLASTER_K1_REMOTE_ROOT", "/root/mb_k1")
MARCH = ["-march=rv64gcv_zvl256b", "-mabi=lp64d"]

OPS = {
    "conv2d_batchnorm2d_s8": dict(
        spec=CONV2D_BATCHNORM2D_S8,
        curated="kernels/rvv/rvv_conv2d_batchnorm2d_s8_rvv_oc_blocked_bn_epilogue.c",
        silu=False,
    ),
    "conv2d_batchnorm2d_silu_s8": dict(
        spec=CONV2D_BATCHNORM2D_SILU_S8,
        curated="kernels/rvv/rvv_conv2d_batchnorm2d_silu_s8"
                "_rvv_oc_blocked_bn_silu_epilogue.c",
        silu=True,
    ),
}


def collect_cases(models, op_name):
    """Unique (shape, quant) tuples for `op_name` across the given models."""
    cases, seen = [], set()
    for model in models:
        graph = os.path.join(REPO, "build", "k1", model, "int8", "graph.json")
        if not os.path.exists(graph):
            graph = os.path.join(REPO, "build", "k1_xpurt", model, "int8",
                                 "graph.json")
        if not os.path.exists(graph):
            print(f"  [{model}] no extracted graph.json; skipping", flush=True)
            continue
        ir = json.load(open(graph))
        for op in ir["ops"]:
            if op["op"] != op_name:
                continue
            subs = {s["op"]: s for s in op["sub_ops"]}
            conv, bn = subs["conv2d_s8"], subs["batchnorm2d_s8"]
            silu = subs.get("silu_s8")
            s, cq, bq = conv["shape"], conv["quant"], bn["quant"]
            case = dict(
                tag=f"{model}:{op['name']}",
                N=s["N"], IC=s["IC"], IH=s["IH"], IW=s["IW"], OC=s["OC"],
                KH=s["KH"], KW=s["KW"], SH=s["SH"], SW=s["SW"],
                PH=s["PH"], PW=s["PW"],
                input_offset=cq["input_offset"],
                filter_offset=cq["filter_offset"],
                conv_output_offset=cq["output_offset"],
                conv_output_multiplier=cq["output_multiplier"],
                conv_output_shift=cq["output_shift"],
                conv_activation_min=cq["activation_min"],
                conv_activation_max=cq["activation_max"],
                bn_scale_in=bq["scale_in"], bn_scale_out=bq["scale_out"],
                bn_activation_min=bq["activation_min"],
                bn_activation_max=bq["activation_max"],
            )
            if silu is not None:
                sq = silu["quant"]
                case.update(silu_scale_in=sq["scale_in"],
                            silu_scale_out=sq["scale_out"],
                            silu_activation_min=sq["activation_min"],
                            silu_activation_max=sq["activation_max"])
            key = tuple(v for k, v in sorted(case.items()) if k != "tag")
            if key in seen:
                continue
            seen.add(key)
            cases.append(case)
    return cases


def _dims(c):
    OH = (c["IH"] + 2 * c["PH"] - c["KH"]) // c["SH"] + 1
    OW = (c["IW"] + 2 * c["PW"] - c["KW"]) // c["SW"] + 1
    return OH, OW


def emit_harness(cases, silu: bool) -> str:
    max_in = max(c["N"] * c["IC"] * c["IH"] * c["IW"] for c in cases)
    max_w = max(c["IC"] * c["KH"] * c["KW"] * c["OC"] for c in cases)
    max_oc = max(c["OC"] for c in cases)
    max_out = 0
    for c in cases:
        OH, OW = _dims(c)
        max_out = max(max_out, c["N"] * c["OC"] * OH * OW)

    sig_tail = (
        "float bn_scale_in, float bn_scale_out, "
        "int bn_activation_min, int bn_activation_max"
    )
    call_tail = ("c->bn_scale_in, c->bn_scale_out, "
                 "c->bn_activation_min, c->bn_activation_max")
    fields = ""
    if silu:
        sig_tail += (", float silu_scale_in, float silu_scale_out, "
                     "int silu_activation_min, int silu_activation_max")
        call_tail += (", c->silu_scale_in, c->silu_scale_out, "
                      "c->silu_activation_min, c->silu_activation_max")
        fields = ("    float silu_scale_in, silu_scale_out;\n"
                  "    int silu_activation_min, silu_activation_max;\n")

    rows = []
    for c in cases:
        row = ("    { \"%(tag)s\", %(N)d,%(IC)d,%(IH)d,%(IW)d,%(OC)d,"
               "%(KH)d,%(KW)d,%(SH)d,%(SW)d,%(PH)d,%(PW)d, "
               "%(input_offset)d,%(filter_offset)d,%(conv_output_offset)d,"
               "%(conv_output_multiplier)d,%(conv_output_shift)d,"
               "%(conv_activation_min)d,%(conv_activation_max)d, "
               "%(bn_scale_in).9gf,%(bn_scale_out).9gf,"
               "%(bn_activation_min)d,%(bn_activation_max)d") % c
        if silu:
            row += (", %(silu_scale_in).9gf,%(silu_scale_out).9gf,"
                    "%(silu_activation_min)d,%(silu_activation_max)d" % c)
        rows.append(row + " },")

    return f"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define MAX_IN  {max_in}
#define MAX_W   {max_w}
#define MAX_OUT {max_out}
#define MAX_OC  {max_oc}

typedef struct {{
    const char *tag;
    int N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW;
    int input_offset, filter_offset, conv_output_offset;
    int conv_output_multiplier, conv_output_shift;
    int conv_activation_min, conv_activation_max;
    float bn_scale_in, bn_scale_out;
    int bn_activation_min, bn_activation_max;
{fields}}} case_t;

static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))

extern void kernel_cand(const int8_t *, const int8_t *, const int32_t *,
                        const float *, const float *, int8_t *,
                        int, int, int, int, int, int, int, int, int, int, int,
                        int, int, int, int, int, int, int, {sig_tail});
extern void kernel_ref(const int8_t *, const int8_t *, const int32_t *,
                       const float *, const float *, int8_t *,
                       int, int, int, int, int, int, int, int, int, int, int,
                       int, int, int, int, int, int, int, {sig_tail});

static int8_t  IN[MAX_IN];
static int8_t  W[MAX_W];
static int32_t BIAS[MAX_OC];
static float   BNS[MAX_OC], BNB[MAX_OC];
static int8_t  OUT_CAND[MAX_OUT], OUT_REF[MAX_OUT];

static uint64_t rs;
static uint32_t xr(void) {{
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    return (uint32_t)(rs >> 32);
}}
static float urand(float lo, float hi) {{
    return lo + (hi - lo) * ((float)(xr() & 0xffffff) / 16777216.0f);
}}
static double now_ms(void) {{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}}

int main(void) {{
    int worst = 0;
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        int OH = (c->IH + 2*c->PH - c->KH) / c->SH + 1;
        int OW = (c->IW + 2*c->PW - c->KW) / c->SW + 1;
        long n_in  = (long)c->N * c->IC * c->IH * c->IW;
        long n_w   = (long)c->IC * c->KH * c->KW * c->OC;
        long n_out = (long)c->N * c->OC * OH * OW;

        /* Two data regimes. "full" drives the conv accumulator hard and
         * exercises the requantize clamps; "small" keeps it in range so the
         * BN and SiLU stages see the whole int8 domain instead of two
         * saturated values. */
        for (int regime = 0; regime < 2; regime++) {{
            int wlim = regime ? 5 : 128;   /* weight magnitude bound */
            int ilim = regime ? 33 : 128;  /* input magnitude bound */
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ regime;
            for (long k = 0; k < n_in; k++)
                IN[k] = (int8_t)((int)(xr() % (2*ilim)) - ilim);
            for (long k = 0; k < n_w; k++)
                W[k] = (int8_t)((int)(xr() % (2*wlim)) - wlim);
            for (int k = 0; k < c->OC; k++) {{
                BIAS[k] = (int32_t)((int)(xr() % 4096) - 2048);
                BNS[k] = urand(-1.5f, 1.5f);
                BNB[k] = urand(-1.0f, 1.0f);
            }}
            memset(OUT_CAND, 0, (size_t)n_out);
            memset(OUT_REF, 0, (size_t)n_out);

            double t0 = now_ms();
            kernel_cand(IN, W, BIAS, BNS, BNB, OUT_CAND,
                        c->N, c->IC, c->IH, c->IW, c->OC,
                        c->KH, c->KW, c->SH, c->SW, c->PH, c->PW,
                        c->input_offset, c->filter_offset,
                        c->conv_output_offset, c->conv_output_multiplier,
                        c->conv_output_shift, c->conv_activation_min,
                        c->conv_activation_max, {call_tail});
            double t1 = now_ms();
            kernel_ref(IN, W, BIAS, BNS, BNB, OUT_REF,
                       c->N, c->IC, c->IH, c->IW, c->OC,
                       c->KH, c->KW, c->SH, c->SW, c->PH, c->PW,
                       c->input_offset, c->filter_offset,
                       c->conv_output_offset, c->conv_output_multiplier,
                       c->conv_output_shift, c->conv_activation_min,
                       c->conv_activation_max, {call_tail});
            double t2 = now_ms();

            int maxerr = 0; long ndiff = 0;
            for (long k = 0; k < n_out; k++) {{
                int d = OUT_CAND[k] - OUT_REF[k];
                if (d < 0) d = -d;
                if (d) ndiff++;
                if (d > maxerr) maxerr = d;
            }}
            if (maxerr > worst) worst = maxerr;
            printf("CASE %2d %-28s regime=%s IC%d %dx%d->OC%d K%d S%d P%d "
                   "n=%ld max_abs_err=%d n_diff=%ld cand=%.2fms ref=%.2fms "
                   "speedup=%.2fx\\n",
                   i, c->tag, regime ? "small" : "full",
                   c->IC, c->IH, c->IW, c->OC, c->KH, c->SH, c->PH,
                   n_out, maxerr, ndiff, t1 - t0, t2 - t1,
                   (t2 - t1) / (t1 - t0 > 0 ? t1 - t0 : 1e-9));
            fflush(stdout);
        }}
    }}
    printf("WORST max_abs_err=%d over %d cases\\n", worst, N_CASES);
    return worst == 0 ? 0 : 1;
}}
"""


def run_op(op_name, models, keep_dir=None):
    info = OPS[op_name]
    cases = collect_cases(models, op_name)
    if not cases:
        print(f"[{op_name}] no cases found in {models}")
        return None
    print(f"[{op_name}] {len(cases)} unique (shape, quant) cases")
    workdir = keep_dir or tempfile.mkdtemp(prefix="k1verify_")
    os.makedirs(workdir, exist_ok=True)

    prologue = ("#include <stddef.h>\n#include <float.h>\n#include <stdint.h>\n"
                "#include <math.h>\n#include <riscv_vector.h>\n"
                '#include "mb_rvv_vxrm_compat.h"\n')
    cand_c = os.path.join(workdir, "cand.c")
    with open(cand_c, "w") as f:
        f.write(prologue)
        f.write(open(os.path.join(REPO, info["curated"])).read())
    ref_c = os.path.join(workdir, "ref.c")
    with open(ref_c, "w") as f:
        f.write("#include <stddef.h>\n#include <stdint.h>\n#include <math.h>\n")
        f.write(info["spec"].reference_impl)
    main_c = os.path.join(workdir, "main.c")
    with open(main_c, "w") as f:
        f.write(emit_harness(cases, info["silu"]))

    elf = os.path.join(workdir, f"verify_{op_name}")
    common = [f"{CROSS}gcc", "-O2", "-static", *MARCH,
              "-DMODELBLASTER_RVV_IHWOC_WEIGHTS=1",
              f"-I{os.path.join(REPO, 'kernels', 'rvv')}"]
    cmd = common + [
        f"-Dkernel_{op_name}=kernel_cand", "-c", cand_c,
        "-o", os.path.join(workdir, "cand.o")]
    subprocess.run(cmd, check=True)
    cmd = common + [
        f"-Dkernel_{op_name}=kernel_ref", "-c", ref_c,
        "-o", os.path.join(workdir, "ref.o")]
    subprocess.run(cmd, check=True)
    subprocess.run(common + ["-c", main_c, "-o", os.path.join(workdir, "main.o")],
                   check=True)
    subprocess.run(common + [os.path.join(workdir, "main.o"),
                             os.path.join(workdir, "cand.o"),
                             os.path.join(workdir, "ref.o"),
                             "-o", elf, "-lm"], check=True)

    # vtype gate on the linked binary, same check the deploy path applies.
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
    ap.add_argument("--model", action="append", default=[],
                    help="model name(s) whose graph.json supplies the shapes")
    ap.add_argument("--op", action="append", default=[],
                    help="restrict to these fused ops")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()
    models = args.model or ["dronet", "yolov8_nano"]
    ops = args.op or list(OPS)
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
