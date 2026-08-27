#!/usr/bin/env python3
"""Inventory SmolVLA's operator composition and ModelBlaster's coverage of it.

The plan's SmolVLA milestone is deliberately staged: export and inventory first,
identify the dominant ops, and only then decide what lowering to add. Full int8
SmolVLA is a separate milestone and must not block anything.

This produces the inventory. It reads the torch.export graph directly rather
than ModelBlaster's IR, because the point is to see *everything the model
contains*, including what the lowering path currently drops -- a walker that
emits 2 ops for a 450M model is reporting its own coverage, not the model.

Output: counts per aten op, which map to an existing ModelBlaster kernel, and
which do not, ordered so the biggest gaps come first.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys


# aten op -> the ModelBlaster kernel family that would serve it. Only entries
# that genuinely correspond; anything absent is reported as a gap.
ATEN_TO_KERNEL = {
    "addmm": "linear_s8", "mm": "matmul_s8", "bmm": "matmul_s8",
    "linear": "linear_s8", "matmul": "matmul_s8",
    "convolution": "conv2d_s8", "conv2d": "conv2d_s8",
    "relu": "relu_s8", "gelu": "gelu", "silu": "silu_s8",
    "sigmoid": "sigmoid_s8", "tanh": "tanh", "elu": "elu_s8",
    "leaky_relu": "leaky_relu_s8",
    "add": "add_s8", "mul": "mul_s8", "sub": "add_s8", "div": "mul_s8",
    "cat": "cat2_c1_s8", "max_pool2d": "maxpool2d_s8",
    "avg_pool2d": "avgpool2d_s8", "mean": "avgpool2d_s8",
    "softmax": "softmax", "_softmax": "softmax",
    "layer_norm": "layernorm", "native_layer_norm": "layernorm",
    "rms_norm": "rmsnorm",
    "batch_norm": "batchnorm2d_s8", "native_batch_norm": "batchnorm2d_s8",
    "embedding": "embedding",
    "lstm": "lstm_s8",
    # pure shape/view ops: free, no kernel needed
    "view": None, "reshape": None, "permute": None, "transpose": None,
    "expand": None, "slice": None, "select": None, "unsqueeze": None,
    "squeeze": None, "clone": None, "contiguous": None, "to": None,
    "_to_copy": None, "detach": None, "t": None, "flatten": None,
    "split": None, "chunk": None, "stack": None, "index": None,
    "full": None, "zeros": None, "ones": None, "arange": None,
    "empty": None, "scalar_tensor": None, "lift_fresh_copy": None,
    "empty_like": None, "zeros_like": None, "ones_like": None,
    # Bookkeeping the exporter inserts, not computation. Counting these as
    # gaps made coverage look far worse than it is: _assert_tensor_metadata
    # alone is 473 of 4379 nodes and costs nothing at runtime.
    "_assert_tensor_metadata": None, "_assert_scalar": None,
    "_local_scalar_dense": None, "sym_size": None, "sym_constrain_range": None,
    "getitem": None, "<built-in function getitem>": None,
    "copy_": None, "alias": None, "detach_": None, "clone_": None,
    "_unsafe_view": None, "expand_as": None, "view_as": None,
    # In-place variants of ops we already cover.
    "add_": "add_s8", "mul_": "mul_s8", "sub_": "add_s8", "div_": "mul_s8",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/smolvla_op_inventory.json")
    ap.add_argument("--model", default="smolvla")
    a = ap.parse_args()

    sys.path.insert(0, os.environ.get("LEROBOT_ROOT", "") + "/src")
    from modelblaster.pipeline import reference_kernels as rk
    from modelblaster.pipeline import extract_graph_export as ege

    model, sample = ege._load_model(a.model) if hasattr(ege, "_load_model") else (None, None)
    if model is None:
        from modelblaster.models import smolvla as model_mod
        model = model_mod.get_model()
        sample = model_mod.get_sample_input()

    import torch
    with torch.no_grad():
        ep = torch.export.export(model, (sample,) if not isinstance(sample, tuple) else sample)

    counts = collections.Counter()
    for n in ep.graph.nodes:
        if n.op != "call_function":
            continue
        name = getattr(n.target, "_schema", None)
        nm = (str(n.target).split(".")[-2] if "." in str(n.target)
              else str(n.target))
        nm = nm.replace("aten::", "")
        counts[nm] += 1

    have = set(rk.KERNEL_SPECS)
    compute, free, gaps = {}, {}, {}
    for op, c in counts.items():
        if op in ATEN_TO_KERNEL:
            k = ATEN_TO_KERNEL[op]
            if k is None:
                free[op] = c
            elif k in have:
                compute[op] = (c, k)
            else:
                gaps[op] = (c, k + " (mapped, kernel MISSING)")
        else:
            gaps[op] = (c, "unmapped")

    total = sum(counts.values())
    print(f"SmolVLA export: {total} call_function nodes, "
          f"{len(counts)} distinct aten ops\n")
    print(f"{'COVERED (kernel exists)':<44}{'count':>7}  kernel")
    for op, (c, k) in sorted(compute.items(), key=lambda x: -x[1][0]):
        print(f"  {op:<42}{c:>7}  {k}")
    print(f"\n{'GAPS (no kernel)':<44}{'count':>7}  note")
    for op, (c, k) in sorted(gaps.items(), key=lambda x: -x[1][0]):
        print(f"  {op:<42}{c:>7}  {k}")
    print(f"\nfree / shape-only / bookkeeping: {sum(free.values())} nodes "
          f"across {len(free)} kinds (no kernel needed)")
    cov = sum(c for c, _ in compute.values())
    gap = sum(c for c, _ in gaps.values())
    print(f"\ncompute nodes covered: {cov}   gaps: {gap}   "
          f"coverage of COMPUTE nodes: {100*cov/max(1, cov+gap):.1f}%")
    print("(free/bookkeeping nodes are excluded from both sides -- counting "
          "them as gaps understates coverage badly)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"total_nodes": total,
               "covered": {k: v[0] for k, v in compute.items()},
               "gaps": {k: v[0] for k, v in gaps.items()},
               "free": free,
               "coverage_pct": round(100*cov/max(1, cov+gap), 1)},
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
