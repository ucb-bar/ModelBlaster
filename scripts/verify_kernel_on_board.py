#!/usr/bin/env python3
"""Per-kernel edit -> measure loop: bit-exact verify + timing on a real board core.

Generalises scripts/k1_verify_curated_rvv.py and k1_verify_fused_conv_rvv.py
(same construction, same acceptance: max_abs_err == 0 against the op's scalar
reference_impl) with the things a kernel-tuning loop needs and those scripts
lack:

  * the backend's REAL cflags (so rvv_k3_a100 is built at zvl1024b, rvv_k3 at
    zvl256b) instead of a hard-coded -march;
  * the kernel the picker would choose for that backend (own dir first, then
    the alias dirs), or --kernel-file to override;
  * pinning to ONE cpu, including K3 A100 cpus (8-15), which a thread can only
    reach after registering itself -- done by runtime/mb_placement.c inside the
    binary, never from a shell;
  * median/min wall time per shape from rdtime (24 MHz, so 1 tick = 1/24 us),
    and an old-vs-new comparison in the same process (--baseline-kernel /
    --compare-backend), so noise and cpu state are shared;
  * real shapes pulled from build/k3*/<model>/int8/graph.json.

Usage:
    scripts/verify_kernel_on_board.py --board k3 --backend rvv_k3_a100 --cpu 8 \\
        --op maxpool2d_s8 [--algo direct] [--kernel-file my.c] \\
        [--shapes-from build/k3/dronet/int8/graph.json ... | --shape 'N=1;C=32;...'] \\
        [--baseline-kernel old.c | --compare-backend rvv_k3] [--reps 21] [--json out.json]

Rules enforced (fail loud):
  * a backend whose name ends _a100 on an X100 cpu is refused: zvl1024b code is
    an illegal instruction stream on VLEN=256.
  * max_abs_err != 0 on any shape -> exit 1.

Board concurrency: if MODELBLASTER_BOARD_LOCK (or --board-lock) names a file,
the deploy+run section holds an exclusive flock on it, so several agents can
share the board. Do NOT also wrap this script in `flock` on the same file.
ssh uses MODELBLASTER_SSH_CONFIG (-F) and MODELBLASTER_K3_HOST like the runners.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, REPO)

from modelblaster.pipeline import backends as backends_mod  # noqa: E402
from modelblaster.pipeline import core_registry  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402
from modelblaster.validation import board_config  # noqa: E402

SHAPE_KEYS = ("N", "M", "K", "C", "IC", "OC", "IH", "IW", "KH", "KW", "SH", "PH")
CLOCK_MHZ = 24.0
SCALAR_MARCH = "-march=rv64gc_zfh"

# --------------------------------------------------------------------------
# Per-op knowledge: how big each pointer argument is, and how to read a case
# out of a graph.json node. Everything else is derived from the KernelSpec
# signature, so adding an op is two small entries here.
# --------------------------------------------------------------------------


def _o(i, p, k, s, d=1):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


def _sz_pool(c):
    OH = _o(c["IH"], c["PH"], c["KH"], c["SH"], c.get("DH", 1))
    OW = _o(c["IW"], c["PW"], c["KW"], c["SW"], c.get("DW", 1))
    return {"input": c["N"] * c["C"] * c["IH"] * c["IW"],
            "output": c["N"] * c["C"] * OH * OW}


def _sz_conv(c):
    OH = _o(c["IH"], c["PH"], c["KH"], c["SH"])
    OW = _o(c["IW"], c["PW"], c["KW"], c["SW"])
    return {"input": c["N"] * c["IC"] * c["IH"] * c["IW"],
            # IHWOC packing (-DMODELBLASTER_RVV_IHWOC_WEIGHTS): same count.
            "weight": c["IC"] * c["KH"] * c["KW"] * c["OC"],
            "bias": c["OC"], "bn_scale": c["OC"], "bn_bias": c["OC"],
            "output": c["N"] * c["OC"] * OH * OW}


def _sz_dw(c):
    OH = _o(c["IH"], c["PH"], c["KH"], c["SH"])
    OW = _o(c["IW"], c["PW"], c["KW"], c["SW"])
    return {"input": c["N"] * c["C"] * c["IH"] * c["IW"],
            "weight": c["C"] * c["KH"] * c["KW"], "bias": c["C"],
            "output": c["N"] * c["C"] * OH * OW}


def _sz_linear(c):
    return {"input": c["M"] * c["K"], "weight": c["K"] * c["N"],
            "bias": c["N"], "output": c["M"] * c["N"]}


def _sz_matmul(c):
    return {"a": c["M"] * c["K"], "b": c["K"] * c["N"],
            "output": c["M"] * c["N"]}


SIZES = {
    "maxpool2d_s8": _sz_pool, "avgpool2d_s8": _sz_pool,
    "conv2d_s8": _sz_conv, "conv2d_batchnorm2d_s8": _sz_conv,
    "conv2d_batchnorm2d_silu_s8": _sz_conv,
    "depthwise_conv2d_s8": _sz_dw, "linear_s8": _sz_linear,
    "matmul_s8": _sz_matmul,
}

#: Realistic quant for --shape cases that do not spell it out (a multiplier of
#: 1.0 would short-circuit the requantize tail).
DEFAULT_PARAMS = {
    "input_offset": 0, "filter_offset": 0, "output_offset": 0,
    "output_multiplier": 1207959552, "output_shift": 7,
    "activation_min": -128, "activation_max": 127,
    "conv_output_offset": 0, "conv_output_multiplier": 1207959552,
    "conv_output_shift": 7, "conv_activation_min": -128,
    "conv_activation_max": 127,
    "bn_scale_in": 0.05, "bn_scale_out": 0.05,
    "bn_activation_min": -128, "bn_activation_max": 127,
    "silu_scale_in": 0.05, "silu_scale_out": 0.05,
    "silu_activation_min": -128, "silu_activation_max": 127,
    "DH": 1, "DW": 1, "N": 1, "count_include_pad": 0,
    "scale_a": 0.05, "scale_b": 0.05, "scale_out": 0.05, "scale_div": 1.0,
    "transpose_b": 0,
}


def case_from_node(op, node):
    """One case dict (param name -> value) from a graph.json op node."""
    if op in ("conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8") \
            and node["op"] == op:
        subs = {s["op"]: s for s in node["sub_ops"]}
        conv, bn = subs["conv2d_s8"], subs["batchnorm2d_s8"]
        cq, bq = conv["quant"], bn["quant"]
        c = dict(conv["shape"])
        c.update(input_offset=cq["input_offset"],
                 filter_offset=cq["filter_offset"],
                 conv_output_offset=cq["output_offset"],
                 conv_output_multiplier=cq["output_multiplier"],
                 conv_output_shift=cq["output_shift"],
                 conv_activation_min=cq["activation_min"],
                 conv_activation_max=cq["activation_max"],
                 bn_scale_in=bq["scale_in"], bn_scale_out=bq["scale_out"],
                 bn_activation_min=bq["activation_min"],
                 bn_activation_max=bq["activation_max"])
        if op.endswith("silu_s8"):
            sq = subs["silu_s8"]["quant"]
            c.update(silu_scale_in=sq["scale_in"],
                     silu_scale_out=sq["scale_out"],
                     silu_activation_min=sq["activation_min"],
                     silu_activation_max=sq["activation_max"])
        return c
    c = dict(node.get("shape") or {})
    c.update(node.get("quant") or {})
    return c


def parse_signature(sig):
    """[(ctype, name, is_ptr, is_const)] from a KernelSpec C signature."""
    m = re.search(r"\((.*)\)\s*$", sig.strip(), re.S)
    out = []
    for p in m.group(1).split(","):
        p = p.strip()
        is_ptr = "*" in p
        is_const = p.startswith("const ")
        toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", p)
        name = toks[-1]
        ctype = " ".join(t for t in toks[:-1] if t != "const")
        out.append((ctype, name, is_ptr, is_const))
    return out


def graph_files(explicit):
    if explicit:
        return list(explicit)
    found = []
    for root in ("k3", "k3_xpurt"):
        base = os.path.join(REPO, "build", root)
        if not os.path.isdir(base):
            continue
        for model in sorted(os.listdir(base)):
            g = os.path.join(base, model, "int8", "graph.json")
            if os.path.exists(g):
                found.append(g)
    return found


def collect_cases(op, graphs, shape_strs):
    cases, seen = [], set()

    def add(c, tag):
        key = tuple(sorted((k, v) for k, v in c.items()))
        if key in seen:
            return
        seen.add(key)
        c = dict(c)
        c["_tag"] = tag
        cases.append(c)

    for s in shape_strs:
        c = {}
        for kv in s.split(";"):
            if kv.strip():
                k, v = kv.split("=", 1)
                c[k.strip()] = float(v) if "." in v or "e" in v.lower() else int(v)
        add(c, "cli")
    if not shape_strs:
        for g in graphs:
            ir = json.load(open(g))
            model = os.path.basename(os.path.dirname(os.path.dirname(g)))
            for node in ir["ops"]:
                if node["op"] == op:
                    add(case_from_node(op, node), f"{model}:{node['name']}")
    return cases


# --------------------------------------------------------------------------
# Kernel lookup, mirroring generate_kernels' curated probe order.
# --------------------------------------------------------------------------


def lookup_kernel(backend, op, algo=None):
    spec = KERNEL_SPECS[op]
    dirs = [backend.name, *getattr(backend, "curated_aliases", ())]
    algos = [a for a in spec.algorithms
             if not a.target_affinity or backend.name in a.target_affinity]
    if algo:
        algos = [a for a in spec.algorithms if a.name == algo]
        if not algos:
            raise SystemExit(f"op {op} has no algorithm {algo!r}; has "
                             f"{[a.name for a in spec.algorithms]}")
    tried = []
    for d in dirs:
        for a in algos:
            p = os.path.join(REPO, "kernels", d, f"{d}_{op}_{a.name}.c")
            tried.append(p)
            if os.path.exists(p):
                return p
    raise SystemExit(f"no curated kernel for {op} on backend {backend.name}; "
                     f"tried:\n  " + "\n  ".join(tried) +
                     "\n(pass --kernel-file to verify an arbitrary file)")


# --------------------------------------------------------------------------
# C harness
# --------------------------------------------------------------------------

C_PRELUDE = r"""
#define _GNU_SOURCE 1
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_placement.h"

static inline uint64_t rdtime_(void) {
    uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); return t;
}
static uint64_t rs;
static uint32_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                           return (uint32_t)(rs >> 32); }
static int8_t rnd8(int lim) { return (int8_t)((int)(xr() % (2*lim)) - lim); }
static float urand(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xr() & 0xffffff) / 16777216.0f);
}
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
"""


def emit_harness(op, sig, cases, has_base, tail_args):
    params = parse_signature(sig)
    scalars = [(t, n) for t, n, p, _ in params if not p]
    ptrs = [(t, n, c) for t, n, p, c in params if p]
    outs = [n for t, n, c in ptrs if not c]
    if len(outs) != 1:
        raise SystemExit(f"{op}: expected exactly one output pointer, got {outs}")
    out = outs[0]
    # Scalars that need a C field: every non-pointer param of the signature.
    fields = "".join(f"    {t} {n};\n" for t, n in scalars)
    szfields = "".join(f"    size_t sz_{n};\n" for _, n, _ in ptrs)
    rows = []
    for c in cases:
        sizes = SIZES[op](c)
        vals = []
        for t, n in scalars:
            v = c[n]
            vals.append(f"{float(v):.9g}f" if t == "float" else str(int(v)))
        for _, n, _ in ptrs:
            if n not in sizes:
                raise SystemExit(f"{op}: no size rule for pointer '{n}'")
            vals.append(str(sizes[n]))
        rows.append(f'    {{ "{c["_tag"]}", ' + ", ".join(vals) + " },")
    max_sz = {n: max(SIZES[op](c)[n] for c in cases) for _, n, _ in ptrs}

    decls, fills, zero = [], [], []
    for t, n, is_const in ptrs:
        ct = {"int8_t": "int8_t", "int32_t": "int32_t", "float": "float"}[t]
        if n == out:
            decls.append(f"static int8_t B_{n}_cand[{max_sz[n]}], "
                         f"B_{n}_ref[{max_sz[n]}], B_{n}_base[{max_sz[n]}];")
            continue
        decls.append(f"static {ct} B_{n}[{max_sz[n]}];")
        if t == "int8_t":
            lim = ("wlim" if n in ("weight", "b") else "ilim")
            fills.append(f"for (size_t k = 0; k < c->sz_{n}; k++) "
                         f"B_{n}[k] = rnd8({lim});")
        elif t == "int32_t":
            fills.append(f"for (size_t k = 0; k < c->sz_{n}; k++) "
                         f"B_{n}[k] = (int32_t)rnd8(64) * 137;")
        else:
            fills.append(f"for (size_t k = 0; k < c->sz_{n}; k++) "
                         f"B_{n}[k] = urand(-1.5f, 1.5f);")

    def call(sym, outbuf):
        args = []
        for t, n, p, _ in params:
            if p:
                args.append(outbuf if n == out else f"B_{n}")
            else:
                args.append(f"c->{n}")
        return f"{sym}({', '.join(args)});"

    proto = ", ".join(("const " if cst else "") + t + (" *" if p else "")
                      for t, n, p, cst in params)
    ext = (f"extern void kernel_cand({proto});\n"
           f"extern void kernel_ref({proto});\n")
    if has_base:
        ext += f"extern void kernel_base({proto});\n"

    base_blk = ""
    if has_base:
        base_blk = f"""
            memset(B_{out}_base, 0, c->sz_{out});
            {call("kernel_base", f"B_{out}_base")}
            {{ int e = 0; for (size_t k = 0; k < c->sz_{out}; k++) {{
                int d = B_{out}_base[k] - B_{out}_ref[k]; if (d < 0) d = -d;
                if (d > e) e = d; }} base_err = e; }}
            if (regime == 0) {{
                for (int w = 0; w < 2; w++) {{ {call("kernel_base", f"B_{out}_base")} }}
                for (int r = 0; r < reps; r++) {{
                    uint64_t t0 = rdtime_();
                    {call("kernel_base", f"B_{out}_base")}
                    ts[r] = rdtime_() - t0;
                }}
                qsort(ts, (size_t)reps, sizeof ts[0], cmp_u64);
                base_med = ts[reps / 2]; base_min = ts[0];
            }}"""

    return C_PRELUDE + f"""
typedef struct {{
    const char *tag;
{fields}{szfields}}} case_t;
static const case_t CASES[] = {{
{chr(10).join(rows)}
}};
#define N_CASES ((int)(sizeof(CASES)/sizeof(CASES[0])))
{ext}
{chr(10).join(decls)}
static uint64_t ts[4096];

int main(int argc, char **argv) {{
    if (argc < 3) {{ fprintf(stderr, "usage: %s cpu reps\\n", argv[0]); return 2; }}
    int cpu = atoi(argv[1]);
    int reps = atoi(argv[2]);
    if (reps < 1 || reps > 4096) {{ fprintf(stderr, "bad reps\\n"); return 2; }}
    /* Registers the thread with the kernel first if cpu is an A100 cpu, then
     * pins and verifies sched_getcpu(); failure is fatal -- timing from an
     * unplaced thread would be filed under the wrong core type. */
    if (mb_place_current_thread(cpu) != 0) return 3;
    int worst = 0;
    for (int i = 0; i < N_CASES; i++) {{
        const case_t *c = &CASES[i];
        for (int regime = 0; regime < 2; regime++) {{
            int ilim = regime ? 33 : 128, wlim = regime ? 5 : 128;
            rs = 0x9E3779B97F4A7C15ull ^ ((uint64_t)i << 8) ^ (uint64_t)regime;
            {chr(10).join("            " + f for f in fills)}
            memset(B_{out}_cand, 0, c->sz_{out});
            memset(B_{out}_ref, 0, c->sz_{out});
            {call("kernel_ref", f"B_{out}_ref")}
            {call("kernel_cand", f"B_{out}_cand")}
            int err = 0; long ndiff = 0;
            for (size_t k = 0; k < c->sz_{out}; k++) {{
                int d = B_{out}_cand[k] - B_{out}_ref[k]; if (d < 0) d = -d;
                if (d) ndiff++;
                if (d > err) err = d;
            }}
            uint64_t med = 0, mn = 0, base_med = 0, base_min = 0;
            int base_err = 0;
            if (regime == 0) {{
                /* Timing in the "full" regime only. 2 untimed warm runs, then
                 * `reps` timed; rdtime is 24 MHz so one tick is 41.7 ns. */
                for (int w = 0; w < 2; w++) {{ {call("kernel_cand", f"B_{out}_cand")} }}
                for (int r = 0; r < reps; r++) {{
                    uint64_t t0 = rdtime_();
                    {call("kernel_cand", f"B_{out}_cand")}
                    ts[r] = rdtime_() - t0;
                }}
                qsort(ts, (size_t)reps, sizeof ts[0], cmp_u64);
                med = ts[reps / 2]; mn = ts[0];
            }}{base_blk}
            if (err > worst) worst = err;
            printf("R|%s|%s|%zu|%d|%ld|%llu|%llu|%d|%llu|%llu\\n", c->tag,
                   regime ? "small" : "full", c->sz_{out}, err, ndiff,
                   (unsigned long long)med, (unsigned long long)mn, base_err,
                   (unsigned long long)base_med, (unsigned long long)base_min);
            fflush(stdout);
        }}
    }}
    printf("W|%d\\n", worst);
    return worst == 0 ? 0 : 1;
}}
"""


# --------------------------------------------------------------------------
# Build / deploy / run
# --------------------------------------------------------------------------


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def build(workdir, op, backend, cand_path, base_path, base_backend, harness_c,
          cross, ai_cpus):
    flags = ["-O2", "-static", *backend.resolved_kernel_cflags(REPO)]

    def wrap(src_text, be):
        inc = "".join(f"#include {i}\n" for i in be.kernel_includes)
        return ("#include <stddef.h>\n#include <float.h>\n#include <stdint.h>\n"
                "#include <string.h>\n#include <math.h>\n" + inc + src_text)

    objs = []

    def compile_obj(name, text, sym, be, scalar=False):
        src = os.path.join(workdir, name + ".c")
        with open(src, "w") as f:
            f.write(text)
        fl = ["-O2", "-static", *be.resolved_kernel_cflags(REPO)]
        if scalar:
            fl = [SCALAR_MARCH if f.startswith("-march=") else f for f in fl]
        obj = os.path.join(workdir, name + ".o")
        run([f"{cross}gcc", *fl, f"-Dkernel_{op}={sym}", "-c", src, "-o", obj])
        # Keep only the entry symbol global so cand/base/ref helpers of the
        # same name (file-scope tables, helpers) cannot collide at link time.
        run([f"{cross}objcopy", f"--keep-global-symbol={sym}", obj])
        objs.append(obj)

    compile_obj("cand", wrap(open(cand_path).read(), backend), "kernel_cand",
                backend)
    # Everything that is NOT the kernel under test is built WITHOUT the vector
    # extension. At -O2 GCC auto-vectorises harness loops, and its vector
    # spill/stack handling at a lower zvl than the running core's VLEN is
    # exactly the hazard rvv_k3_a100 exists to avoid; the reference and the
    # harness must be correct on every core, whatever its VLEN.
    compile_obj("ref", "#include <stddef.h>\n#include <stdint.h>\n"
                "#include <string.h>\n#include <math.h>\n"
                + KERNEL_SPECS[op].reference_impl, "kernel_ref", backend, scalar=True)
    if base_path:
        compile_obj("base", wrap(open(base_path).read(), base_backend),
                    "kernel_base", base_backend)

    main_c = os.path.join(workdir, "main.c")
    with open(main_c, "w") as f:
        f.write(harness_c)
    rt = os.path.join(REPO, "runtime")
    defs = ["-DMODELBLASTER_PLATFORM_LINUX=1"]
    if ai_cpus:
        defs.append(f'-DMODELBLASTER_AI_CPUS="{ai_cpus}"')
    elf = os.path.join(workdir, f"vk_{op}")
    flags = [SCALAR_MARCH if f.startswith("-march=") else f for f in flags]
    run([f"{cross}gcc", *flags, *defs, f"-I{rt}", main_c,
         os.path.join(rt, "mb_placement.c"), *objs, "-o", elf, "-lm"])

    # Same vtype gate the deploy path applies, on the linked binary.
    vt = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "check_rvv_vtype.py"),
         "--objdump", f"{cross}objdump", elf], capture_output=True, text=True)
    msg = (vt.stdout.strip() or vt.stderr.strip())
    if vt.returncode != 0:
        raise SystemExit(f"vtype check FAILED; not deploying\n{msg}")
    return elf


@contextlib.contextmanager
def board_lock(path):
    if not path:
        yield
        return
    with open(path, "a") as f:
        t0 = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        waited = time.time() - t0
        if waited > 1:
            print(f"  (waited {waited:.0f}s for the board lock)", file=sys.stderr)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def ssh_base(cfg):
    opts = ["-o", "ConnectTimeout=60"]
    cfgf = os.environ.get("MODELBLASTER_SSH_CONFIG")
    if cfgf:
        opts = ["-F", cfgf] + opts
    return opts


def retry(cmd, tries=4, **kw):
    for i in range(tries):
        r = subprocess.run(cmd, **kw)
        if r.returncode == 0 or i == tries - 1:
            return r
        time.sleep(5)


def deploy_and_run(cfg, elf, cpu, cpu_type, reps, lock):
    opts = ssh_base(cfg)
    tag = hashlib.sha1(f"{os.getpid()}{time.time()}".encode()).hexdigest()[:8]
    rdir = f"{cfg.remote_root}/verify"
    rbin = f"{rdir}/vk_{tag}"
    # A100 cpus: the process must START on the X100 (mask 0-7); the binary
    # registers the thread and moves itself. X100: just pin to the cpu.
    mask = cfg.launch_cpus if cpu_type == "a100" else str(cpu)
    with board_lock(lock):
        run(["ssh", *opts, cfg.host, f"mkdir -p {rdir}"])
        r = retry(["scp", *opts, "-q", elf, f"{cfg.host}:{rbin}"])
        if r.returncode != 0:
            raise SystemExit("scp to board failed")
        try:
            proc = subprocess.run(
                ["ssh", *opts, cfg.host,
                 (("flock " + os.environ["MODELBLASTER_BOARD_FLOCK"] + " ") if os.environ.get("MODELBLASTER_BOARD_FLOCK") else "") + f"taskset -c {mask} {rbin} {cpu} {reps}"],
                capture_output=True, text=True)
        finally:
            subprocess.run(["ssh", *opts, cfg.host, f"rm -f {rbin}"],
                           capture_output=True)
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--board", default="k3", choices=["k3"])
    ap.add_argument("--backend", required=True)
    ap.add_argument("--cpu", type=int, required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--algo", default=None)
    ap.add_argument("--kernel-file", default=None,
                    help="kernel to test instead of the backend's curated pick")
    ap.add_argument("--shapes-from", nargs="+", default=None,
                    help="graph.json file(s); default: every "
                         "build/k3*/<model>/int8/graph.json")
    ap.add_argument("--shape", action="append", default=[],
                    help="'k=v;k=v;...' explicit case (repeatable); missing "
                         "quant params get realistic defaults")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="keep only N cases, evenly spread over the work "
                         "range (largest..smallest) -- a quick inner-loop "
                         "subset; 0 = all")
    ap.add_argument("--reps", type=int, default=21)
    ap.add_argument("--json", default=None)
    ap.add_argument("--baseline-kernel", default=None,
                    help="also time this file (built with --backend's flags)")
    ap.add_argument("--compare-backend", default=None,
                    help="also time THIS backend's curated kernel, built with "
                         "its own flags (must be legal on --cpu)")
    ap.add_argument("--board-lock", default=os.environ.get(
        "MODELBLASTER_BOARD_LOCK"))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    if args.op not in KERNEL_SPECS:
        raise SystemExit(f"unknown op {args.op}")
    if args.op not in SIZES:
        raise SystemExit(f"op {args.op} has no buffer-size rule in "
                         f"{os.path.basename(__file__)}; supported: {sorted(SIZES)}")
    if args.baseline_kernel and args.compare_backend:
        raise SystemExit("--baseline-kernel and --compare-backend are exclusive")

    cfg = board_config.resolve(args.board)
    reg = core_registry.load(os.path.join(REPO, cfg.registry))
    cpu_type = board_config.core_type_of(cfg, [args.cpu])
    for be_name in filter(None, [args.backend, args.compare_backend]):
        if be_name.endswith("_a100") and cpu_type != "a100":
            raise SystemExit(
                f"backend {be_name} is built at zvl1024b and is ILLEGAL on "
                f"{cpu_type} cpu {args.cpu} (VLEN 256); use an A100 cpu 8-15")
    backend = backends_mod.get(args.backend)
    base_backend = backends_mod.get(args.compare_backend) \
        if args.compare_backend else backend

    cand_path = args.kernel_file or lookup_kernel(backend, args.op, args.algo)
    base_path = args.baseline_kernel or (
        lookup_kernel(base_backend, args.op, args.algo)
        if args.compare_backend else None)
    print(f"kernel   : {os.path.relpath(cand_path, REPO)}  (backend {backend.name})")
    if base_path:
        print(f"baseline : {os.path.relpath(base_path, REPO)}  "
              f"(flags of {base_backend.name})")

    graphs = graph_files(args.shapes_from)
    cases = collect_cases(args.op, graphs, args.shape)
    if not cases:
        raise SystemExit(f"no {args.op} cases found in {graphs or 'build/k3*'}; "
                         f"pass --shapes-from or --shape")
    if args.max_cases and len(cases) > args.max_cases:
        def work(c):
            return SIZES[args.op](c)["output"] * c.get("IC", c.get("K", 1)) \
                * c.get("KH", 1) * c.get("KW", 1)
        cases.sort(key=work, reverse=True)
        n = args.max_cases
        idx = sorted({round(i * (len(cases) - 1) / max(n - 1, 1)) for i in range(n)})
        cases = [cases[i] for i in idx]
    # Fill unspecified params from realistic defaults; fail loud on the rest.
    params = parse_signature(KERNEL_SPECS[args.op].signature)
    for c in cases:
        for _, n, p, _ in params:
            if not p and n not in c:
                if n in DEFAULT_PARAMS:
                    c[n] = DEFAULT_PARAMS[n]
                else:
                    raise SystemExit(f"case {c['_tag']}: missing param {n!r}")
    print(f"cases    : {len(cases)}  cpu {args.cpu} ({cpu_type})  reps {args.reps}")

    cross = os.environ.get("CROSS")
    if not cross:
        raise SystemExit("CROSS is not set; eval \"$(XPU-RT/scripts/"
                         "setup_spacemit_toolchain.sh)\"")
    workdir = args.workdir or tempfile.mkdtemp(prefix=f"vkob_{args.op}_")
    os.makedirs(workdir, exist_ok=True)
    harness = emit_harness(args.op, KERNEL_SPECS[args.op].signature, cases,
                           bool(base_path), None)
    elf = build(workdir, args.op, backend, cand_path, base_path, base_backend,
                harness, cross, reg.ai_cpus_str())
    proc = deploy_and_run(cfg, elf, args.cpu, cpu_type, args.reps,
                          args.board_lock)
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr)

    by_tag = {c["_tag"]: c for c in cases}
    rows = []
    for line in proc.stdout.splitlines():
        if line.startswith("R|"):
            _, tag, regime, n, err, nd, med, mn, berr, bmed, bmin = line.split("|")
            dims = " ".join(f"{k}{by_tag[tag][k]}" for k in SHAPE_KEYS
                            if k in by_tag.get(tag, {}))
            rows.append(dict(tag=tag, dims=dims, regime=regime, n_out=int(n),
                             max_abs_err=int(err), n_diff=int(nd),
                             med_us=int(med) / CLOCK_MHZ,
                             min_us=int(mn) / CLOCK_MHZ,
                             base_max_abs_err=int(berr),
                             base_med_us=int(bmed) / CLOCK_MHZ,
                             base_min_us=int(bmin) / CLOCK_MHZ))
    ok = proc.returncode == 0 and rows
    base = bool(base_path)
    print(f"\n{'shape':64s} {'err':>3s} {'med_us':>10s} {'min_us':>10s}"
          + (f" {'base_med':>10s} {'base_min':>10s} {'speedup':>8s}" if base else ""))
    tot = tot_b = 0.0
    for r in rows:
        if r["regime"] != "full":
            if r["max_abs_err"] or r["base_max_abs_err"] and base:
                print(f"{r['tag'][:44]:44s} ERR in small regime: "
                      f"max_abs_err={r['max_abs_err']}")
            continue
        tot += r["med_us"]
        tot_b += r["base_med_us"]
        line = (f"{(r['tag'][:30] + ' ' + r['dims'])[:64]:64s} {r['max_abs_err']:3d} "
                f"{r['med_us']:10.2f} {r['min_us']:10.2f}")
        if base:
            sp = r["base_med_us"] / r["med_us"] if r["med_us"] else 0
            line += (f" {r['base_med_us']:10.2f} {r['base_min_us']:10.2f} "
                     f"{sp:7.2f}x")
        print(line)
    worst = max((r["max_abs_err"] for r in rows), default=-1)
    summ = dict(board=args.board, backend=args.backend, cpu=args.cpu,
                core_type=cpu_type, op=args.op,
                kernel=os.path.relpath(cand_path, REPO),
                baseline=os.path.relpath(base_path, REPO) if base_path else None,
                baseline_backend=base_backend.name if base_path else None,
                cases=len([r for r in rows if r["regime"] == "full"]),
                max_abs_err=worst, total_med_us=tot,
                baseline_total_med_us=tot_b if base else None,
                reps=args.reps, returncode=proc.returncode)
    print(f"SUMMARY op={args.op} backend={args.backend} cpu={args.cpu} "
          f"cases={summ['cases']} max_abs_err={worst} "
          f"total_med_us={tot:.2f}"
          + (f" baseline_total_med_us={tot_b:.2f} speedup={tot_b / tot:.2f}x"
             if base and tot else ""))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(dict(summary=summ, rows=rows), f, indent=1)
    if not ok or worst != 0:
        print(f"VERIFY FAIL (board exit status {proc.returncode})"
              + ("" if rows else " (no output from board)"))
        if not rows:
            print(proc.stdout)
        return 1
    print("VERIFY PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
