"""Harness 3: walk a ModelBlaster IR on the GPU, one CUDA kernel per op.

Every non-alias op is dispatched through ModelBlaster's own C ABI
(KernelSpec.signature) to a CUDA launcher in a shared library, on device
buffers. Output is checked against the io.npz golden, every intermediate is
diffed against walk_torch (to localise the first bad op), and each op is
timed with CUDA events.

    python walk_cuda.py IR_DIR --specs specs.json --hand lib_hand.so \
        [--seeds SEED_DIR] [--impl hand|seed|mixed] [--reps 20] [--json out.json]

An fp16 IR (`--quant fp16`, `_f16` ops) runs on fp16 buffers: tensors and
weights are stored as torch.float16 and the `_f16` kernels (lib built from
cuda/mb_cuda_kernels_f16.cu) read and write __half; the per-intermediate
oracle is walk_torch's fp16-storage walk and the bounds are cuda_dtypes'.

impl=hand   only hand-written kernels (fails on an op without one)
impl=seed   only auto-ported single-thread reference kernels (ref2cuda.py)
impl=mixed  hand kernel when available, else seed
"""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import walk_torch
from mbabi import bind, parse_signature
from mbir import IR, compare, cuda_dtypes, load_ir, strict_fp32

_TORCH_DTYPE = {"f16": torch.float16, "f32": torch.float32, "i64": torch.int64}   # i64: argmax/argmin indices


def _storage(ir: IR, t: str) -> torch.dtype:
    """Device dtype of IR tensor t (graph.json `dtype`: f16 / f32)."""
    return _TORCH_DTYPE.get(ir.graph["tensors"].get(t, {}).get("dtype"), torch.float32)


class KernelSet:
    def __init__(self, specs: dict, hand: str | None, seeds: str | None, impl: str):
        self.specs, self.impl = specs, impl
        self.hand = ctypes.CDLL(hand) if hand else None
        self.seed_dir = Path(seeds) if seeds else None
        self._seed_libs: dict[str, ctypes.CDLL] = {}

    def _seed(self, op):
        if self.seed_dir is None:
            return None
        if op not in self._seed_libs:
            p = self.seed_dir / f"lib_{op}.so"
            self._seed_libs[op] = ctypes.CDLL(str(p)) if p.exists() else None
        return self._seed_libs[op]

    def resolve(self, op: str):
        """-> (callable, params, source, lib) or raises KeyError."""
        name, params = parse_signature(self.specs[op]["signature"])
        order = {"hand": ["hand"], "seed": ["seed"], "mixed": ["hand", "seed"]}[self.impl]
        for src in order:
            lib = self.hand if src == "hand" else self._seed(op)
            if lib is not None and hasattr(lib, name):
                fn = getattr(lib, name)
                fn.argtypes = [p.ctypes_type() for p in params]
                fn.restype = None
                return fn, params, src, lib
        raise KeyError(f"no {self.impl} kernel for {op}")


def build_plan(ir: IR, ks: KernelSet):
    plan, missing = [], []
    for op in ir.ops:
        if op["op"] in ("view", "chunk2_c1"):
            plan.append((op, None))
            continue
        try:
            plan.append((op, ks.resolve(op["op"])))
        except KeyError:
            missing.append(op["op"])
    return plan, sorted(set(missing))


def run(ir: IR, plan, dev="cuda", time_ops=False, feed=None):
    """feed: an env (the PyTorch walk's) whose values replace every kernel's
    inputs, so each op is checked in isolation from upstream rounding."""
    env = {t: v.to(_storage(ir, t)) for t, v in ir.input_tensors(dev).items()}
    # folded constants / gather indices (int32, the kernels' `const int *`)
    env.update({t: v if not v.is_floating_point() else v.to(_storage(ir, t))
                for t, v in ir.const_tensors(dev, int_dtype=torch.int32).items()})
    weights = {}

    def wptr(key):
        if key not in weights:      # as stored: float16 weights for an fp16 IR
            weights[key] = torch.from_numpy(np.ascontiguousarray(ir.weights[key])).to(dev)
        return weights[key].data_ptr()

    per_op = []
    for op, k in plan:
        if op["op"] == "view":
            off, n = op.get("offset", 0), op["shape"]["n"]
            env[op["outputs"][0]] = env[op["inputs"][0]].reshape(-1)[off:off + n].view(
                ir.tshape(op["outputs"][0]))
            continue
        if op["op"] == "chunk2_c1":
            # ModelBlaster aliases the two halves at a channel offset; that is
            # only contiguous for N == 1, which is all the corpus uses.
            x = env[op["inputs"][0]]
            assert x.shape[0] == 1, "chunk2_c1 alias needs N==1"
            for i, t in enumerate(op["outputs"]):
                env[t] = x.narrow(1, i * op["shape"]["c_each"], op["shape"]["c_each"])
            continue
        fn, params, src, lib = k
        # the fed copies are this op's own: writing them into env would
        # overwrite the producer's output before it is compared
        fed = {}
        if feed is not None:
            for t in op["inputs"]:
                # constants (int32 gather indices) are already exact
                if t in feed and not ir.graph["tensors"].get(t, {}).get("const"):
                    fed[t] = feed[t].to(device=dev, dtype=env[t].dtype).contiguous().clone()
        for t in op["outputs"]:
            env[t] = torch.empty(ir.tshape(t), device=dev, dtype=_storage(ir, t))
        args = bind(op, params, lambda t: fed.get(t, env[t]).data_ptr(), wptr)
        if time_ops:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
        fn(*args)
        err = lib.mb_last_error()
        if err:
            raise RuntimeError(f"launch error {err} in {op['name']} ({op['op']}, {src})")
        if time_ops:
            e.record()
            per_op.append((op, src, s, e))
    out = torch.cat([env[t].reshape(-1) for t in ir.output_names()])
    return out, env, per_op


def evaluate(d: str, ks: KernelSet, reps: int) -> dict:
    ir = load_ir(d)
    plan, missing = build_plan(ir, ks)
    srcs = [k[2] for _, k in plan if k]
    rec = {"ir": ir.name, "impl": ks.impl, "dtype": ir.dtype, "n_kernel_ops": len(srcs),
           "from_hand": srcs.count("hand"), "from_seed": srcs.count("seed"), "missing": missing}
    if missing:
        rec["ok"] = False
        return rec
    with torch.no_grad():
        t0 = time.time()
        out, env, _ = run(ir, plan)
        torch.cuda.synchronize()
        rec["first_run_s"] = round(time.time() - t0, 3)
        rec.update(compare(out, ir.golden_out, ir.dtype))
        # Localise: diff every intermediate against the PyTorch walk. cuDNN
        # is off for this oracle: its fast conv algorithms (FFT/Winograd) are
        # ~2e-3 off fp64 on e.g. AlexNet conv2, where the naive kernel is 2e-6.
        with torch.backends.cudnn.flags(enabled=False):
            _, ref_env = walk_torch.walk(ir)
        # each kernel on the reference's inputs: a diff is that kernel's own
        # error, not rounding propagated through an ill-conditioned graph
        _, iso_env, _ = run(ir, plan, feed=ref_env)
        rec["output_vs_ref_walk"] = compare(out, torch.cat(
            [ref_env[t].reshape(-1) for t in ir.output_names()]).cpu().numpy(), ir.dtype)["rel_to_range"]
        worst, first_bad, by_kind = 0.0, None, {}
        for op, _k in plan:
            if _k is None:
                continue
            for t in op["outputs"]:
                if t not in ref_env:
                    continue
                a, b = iso_env[t].float(), ref_env[t].float()
                d = torch.where(a == b, torch.zeros_like(a), (a - b).abs())  # inf == inf
                # scale = largest finite magnitude among the op's output and
                # inputs, so a reduction that cancels to ~0 (mean of a
                # GroupNorm output) is judged against what it summed
                scale = 0.0
                for v in [b] + [ref_env[i] for i in op["inputs"] if i in ref_env]:
                    fin = v.float()[torch.isfinite(v.float())]
                    scale = max(scale, fin.abs().max().item() if fin.numel() else 0.0)
                e = d.max().item() / max(scale, 1e-12)
                worst = max(worst, e)
                by_kind[op["op"]] = max(by_kind.get(op["op"], 0.0), e)
                tol = cuda_dtypes.TOLERANCE[cuda_dtypes.op_dtype(op["op"])]
                bad = d.max().item() > tol["slot_rel"] * scale + tol["slot_abs"]   # abs: fp16 subnormals
                if first_bad is None and (bad or not (torch.isfinite(a) == torch.isfinite(b)).all()):
                    first_bad = f"{op['name']} ({op['op']}) rel_err={e:.2e}"
        rec["worst_intermediate_rel_err"] = worst
        rec["first_divergent_op"] = first_bad
        rec["intermediate_rel_err_by_op"] = {k: float(f"{v:.3g}") for k, v in sorted(by_kind.items())}
        # Timing: whole-model wall (events) + per-op breakdown.
        for _ in range(2):
            run(ir, plan)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            run(ir, plan)
        e.record()
        torch.cuda.synchronize()
        rec["model_ms"] = s.elapsed_time(e) / reps
        _, _, per_op = run(ir, plan, time_ops=True)
        torch.cuda.synchronize()
        by_type = {}
        for op, src, a, b in per_op:
            by_type[op["op"]] = by_type.get(op["op"], 0.0) + a.elapsed_time(b)
        rec["ms_by_op_type"] = {k: round(v, 4) for k, v in sorted(by_type.items(), key=lambda kv: -kv[1])}
        # PyTorch eager reference time for the same walk.
        s.record()
        for _ in range(reps):
            walk_torch.walk(ir)
        e.record()
        torch.cuda.synchronize()
        rec["torch_eager_ms"] = s.elapsed_time(e) / reps
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("irs", nargs="+")
    ap.add_argument("--specs", required=True)
    ap.add_argument("--hand")
    ap.add_argument("--seeds")
    ap.add_argument("--impl", default="mixed", choices=["hand", "seed", "mixed"])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json")
    a = ap.parse_args()
    strict_fp32()
    ks = KernelSet(json.load(open(a.specs)), a.hand, a.seeds, a.impl)
    results = []
    for d in a.irs:
        try:
            r = evaluate(d, ks, a.reps)
        except Exception as e:
            r = {"ir": d, "impl": a.impl, "ok": False, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        print(json.dumps(r), flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    sys.exit(0 if all(r.get("ok") for r in results) else 1)


if __name__ == "__main__":
    main()
