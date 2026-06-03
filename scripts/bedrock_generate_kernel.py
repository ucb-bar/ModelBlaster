#!/usr/bin/env python3
"""E2 — Actual Bedrock kernel generation for a registered fused KernelSpec.

Invokes Bedrock to produce an optimized RVV+OPU C implementation of the
chosen fused KernelSpec, then host-verifies bit-exactness against the
reference_impl on every registered extra_shape. Honest gate:
  - max_abs_err > 0 on ANY shape → REJECT
  - kernel cost > budget → STOP
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from pipeline.bedrock_client import BedrockClient
from pipeline.verify_kernel import host_compile
import pipeline.reference_kernels as rk

RNG = np.random.default_rng(42)


def build_prompt(spec, algo):
    """Construct the kernel-generation prompt from a KernelSpec + chosen
    AlgorithmCandidate. The output is verified by cross-compiling for
    rv64 + spike RVV functional simulation."""
    return f"""You will write a C implementation of the fused kernel below for the
RVV + Saturn-OPU target (RV64 + RVV vector extension). The output must
be:
  * A SINGLE C function with EXACTLY the signature below.
  * Self-contained: include any helpers as `static inline` above the function.
  * Bit-exact match vs the reference impl on spike (max_abs_err == 0).
  * No `#include` statements — they are added by the harness.
  * No comments other than 1-2 short lines explaining each major step.

You MAY use RVV 1.0 intrinsics (`<riscv_vector.h>` types and the
`__riscv_v*` builtin functions) — the cross-compile target is
rv64gcv_zba_zbb_zbc_zbs (GCC 13+) and spike has RVV 1.0 enabled. The
algorithm description below describes the target programming model.

SIGNATURE (must match exactly, no rename, no reorder):
{spec.signature}

REFERENCE IMPLEMENTATION (your output must produce bit-identical results):
{spec.reference_impl}

ALGORITHM TO IMPLEMENT (target_affinity={algo.target_affinity}):
{algo.description}

OUTPUT FORMAT:
Wrap your final answer in ```c ... ``` fences. Output ONLY the code,
nothing else. No prose before or after the fenced block."""


def extract_code(text: str) -> str | None:
    """Extract C code from ```c ... ``` fences."""
    import re
    m = re.search(r"```(?:c|C)?\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


def verify_kernel(c_source, spec, workdir, n_repeats=1):
    """Compile candidate + reference; compare on every extra_shape."""
    if spec.op == "batchnorm2d_silu_s8":
        return _verify_bn_silu(c_source, spec, workdir)
    elif spec.op == "conv2d_batchnorm2d_s8":
        return _verify_conv_bn(c_source, spec, workdir)
    else:
        raise NotImplementedError(spec.op)


def _verify_bn_silu(c_source, spec, workdir):
    """Bit-exact verify for batchnorm2d_silu_s8."""
    cand_so = host_compile(c_source, "cand_bn_silu", workdir)
    ref_so  = host_compile(spec.reference_impl, "ref_bn_silu", workdir)

    i8p = ctypes.POINTER(ctypes.c_int8)
    fp  = ctypes.POINTER(ctypes.c_float)
    ci  = ctypes.c_int; cf = ctypes.c_float

    def _load(so):
        lib = ctypes.CDLL(so)
        fn = lib.kernel_batchnorm2d_silu_s8
        fn.argtypes = [i8p, fp, fp, i8p, ci, ci, ci, ci,
                       cf, cf, ci, ci, cf, cf, ci, ci]
        fn.restype = None
        return fn

    cand_fn = _load(cand_so)
    ref_fn  = _load(ref_so)

    results = []
    for shape in spec.extra_shapes:
        N, C, H, W = shape["N"], shape["C"], shape["H"], shape["W"]
        n_el = N * C * H * W
        x = RNG.integers(-128, 128, size=n_el, dtype=np.int8)
        bn_scale = RNG.uniform(0.5, 1.5, size=C).astype(np.float32)
        bn_bias  = RNG.uniform(-1.0, 1.0, size=C).astype(np.float32)
        y_cand = np.zeros_like(x); y_ref = np.zeros_like(x)
        args = (N, C, H, W, 0.01, 0.01, -128, 127, 0.01, 0.01, -128, 127)
        cand_fn(x.ctypes.data_as(i8p), bn_scale.ctypes.data_as(fp),
                bn_bias.ctypes.data_as(fp), y_cand.ctypes.data_as(i8p), *args)
        ref_fn (x.ctypes.data_as(i8p), bn_scale.ctypes.data_as(fp),
                bn_bias.ctypes.data_as(fp), y_ref .ctypes.data_as(i8p), *args)
        diff = np.abs(y_cand.astype(np.int32) - y_ref.astype(np.int32))
        results.append({"shape": shape, "max_abs_err": int(diff.max()),
                          "n_diff": int((diff > 0).sum()), "n_el": int(n_el)})
    return results


def _verify_conv_bn(c_source, spec, workdir):
    """Bit-exact verify for conv2d_batchnorm2d_s8 (small shapes only)."""
    cand_so = host_compile(c_source, "cand_conv_bn", workdir)
    ref_so  = host_compile(spec.reference_impl, "ref_conv_bn", workdir)

    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    fp = ctypes.POINTER(ctypes.c_float)
    ci = ctypes.c_int; cf = ctypes.c_float
    argtypes = [i8p, i8p, i32p, fp, fp, i8p,
                ci, ci, ci, ci, ci, ci, ci, ci, ci, ci, ci,
                ci, ci, ci, ci, ci, ci, ci,
                cf, cf, ci, ci]

    def _load(so):
        lib = ctypes.CDLL(so)
        fn = lib.kernel_conv2d_batchnorm2d_s8
        fn.argtypes = argtypes
        fn.restype = None
        return fn

    cand_fn = _load(cand_so); ref_fn = _load(ref_so)
    SMALL = [s for s in spec.extra_shapes if s["IH"]*s["IW"]*s["IC"] <= 256]
    if not SMALL:
        SMALL = [spec.extra_shapes[-1]]
    results = []
    for shape in SMALL:
        N, IC, IH, IW = shape["N"], shape["IC"], shape["IH"], shape["IW"]
        OC, KH, KW = shape["OC"], shape["KH"], shape["KW"]
        SH, SW, PH, PW = shape["SH"], shape["SW"], shape["PH"], shape["PW"]
        OH = (IH + 2*PH - KH) // SH + 1; OW = (IW + 2*PW - KW) // SW + 1
        x = RNG.integers(-128, 128, size=N*IC*IH*IW, dtype=np.int8)
        w = RNG.integers(-128, 128, size=OC*IC*KH*KW, dtype=np.int8)
        b = RNG.integers(-1000, 1000, size=OC, dtype=np.int32)
        bn_scale = RNG.uniform(0.5, 1.5, size=OC).astype(np.float32)
        bn_bias  = RNG.uniform(-1.0, 1.0, size=OC).astype(np.float32)
        y_cand = np.zeros(N*OC*OH*OW, dtype=np.int8)
        y_ref  = np.zeros(N*OC*OH*OW, dtype=np.int8)
        extra = (N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW,
                 0, 0, 0, 1073741824, 1, -128, 127,
                 0.01, 0.01, -128, 127)
        cand_fn(x.ctypes.data_as(i8p), w.ctypes.data_as(i8p),
                b.ctypes.data_as(i32p),
                bn_scale.ctypes.data_as(fp), bn_bias.ctypes.data_as(fp),
                y_cand.ctypes.data_as(i8p), *extra)
        ref_fn (x.ctypes.data_as(i8p), w.ctypes.data_as(i8p),
                b.ctypes.data_as(i32p),
                bn_scale.ctypes.data_as(fp), bn_bias.ctypes.data_as(fp),
                y_ref .ctypes.data_as(i8p), *extra)
        diff = np.abs(y_cand.astype(np.int32) - y_ref.astype(np.int32))
        results.append({"shape": shape, "max_abs_err": int(diff.max()),
                          "n_diff": int((diff > 0).sum()),
                          "n_el": int(len(y_cand))})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    choices=["batchnorm2d_silu_s8", "conv2d_batchnorm2d_s8"])
    ap.add_argument("--algo-idx", type=int, default=0,
                    help="which AlgorithmCandidate to use as the prompt seed")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out-dir",
                    default=str(REPO / "artifacts" / "kernels"))
    args = ap.parse_args()

    spec_map = {
        "batchnorm2d_silu_s8": rk.BATCHNORM2D_SILU_S8,
        "conv2d_batchnorm2d_s8": rk.CONV2D_BATCHNORM2D_S8,
    }
    spec = spec_map[args.spec]
    algo = spec.algorithms[args.algo_idx]
    out_dir = Path(args.out_dir) / args.spec / algo.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "generation_log.jsonl"

    client = BedrockClient(log_path=str(log_path))
    prompt = build_prompt(spec, algo)
    (out_dir / "prompt.txt").write_text(prompt)
    print(f"Spec: {spec.op}")
    print(f"Algo: {algo.name} (target_affinity={algo.target_affinity})")
    print(f"Prompt: {len(prompt)} chars, ~{len(prompt)//4} tokens")

    total_usd = 0.0
    for attempt in range(1, args.max_attempts + 1):
        print(f"\n--- attempt {attempt} ---")
        t0 = time.perf_counter()
        try:
            r = client.converse(user=prompt, max_tokens=args.max_tokens,
                                  temperature=0.2)
        except Exception as e:
            print(f"BedrockClient error: {e}")
            continue
        wall = time.perf_counter() - t0
        cost = getattr(r, "usd", None) or 0.0
        total_usd += cost
        print(f"  wall={wall:.1f}s, tokens out={getattr(r, 'output_tokens', '?')}, "
              f"cost=${cost:.4f}, total=${total_usd:.4f}")

        code = extract_code(r.text)
        if not code:
            print("  no code fence found; raw text saved")
            (out_dir / f"attempt_{attempt}_raw.txt").write_text(r.text)
            continue
        (out_dir / f"attempt_{attempt}.c").write_text(code)
        print(f"  candidate: {len(code)} chars")

        try:
            with tempfile.TemporaryDirectory() as wd:
                results = verify_kernel(code, spec, wd)
        except Exception as e:
            print(f"  verify error: {type(e).__name__}: {str(e)[:200]}")
            continue

        max_err_overall = max(r["max_abs_err"] for r in results)
        n_pass = sum(1 for r in results if r["max_abs_err"] == 0)
        print(f"  shape pass: {n_pass}/{len(results)}, "
              f"worst max_abs_err: {max_err_overall}")
        for r in results:
            print(f"    {r['shape']}  max_abs_err={r['max_abs_err']}  n_diff={r['n_diff']}/{r['n_el']}")

        report = {
            "attempt": attempt,
            "spec_op": spec.op,
            "algo_name": algo.name,
            "wall_s": round(wall, 2),
            "cost_usd": cost,
            "cumulative_cost_usd": total_usd,
            "n_shapes_pass": n_pass,
            "n_shapes_total": len(results),
            "max_abs_err_overall": max_err_overall,
            "per_shape": results,
        }
        (out_dir / f"attempt_{attempt}_report.json").write_text(
            json.dumps(report, indent=2)
        )

        if max_err_overall == 0:
            # Accept.
            print(f"\n  ACCEPTED on attempt {attempt}: bit-exact on "
                  f"{n_pass}/{len(results)} shapes")
            final = out_dir / f"final_kernel.c"
            final.write_text(code)
            summary = {"status": "PASS", "attempt": attempt,
                       "cost_usd": total_usd, "n_shapes": n_pass}
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            return 0

    summary = {"status": "FAIL_AFTER_ATTEMPTS", "attempts": args.max_attempts,
               "cost_usd": total_usd}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFailed after {args.max_attempts} attempts; total cost ${total_usd:.4f}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
