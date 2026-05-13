"""Export-based graph extractor for agents pipeline.

Companion to ``extract_graph.py`` (FX-based). Use this path when
``torch.fx.symbolic_trace`` can't trace the model — e.g. ViNT, which
hits ``len(...)`` calls inside EfficientNet plus in-place index
assignments + nn.TransformerEncoder internals. ``torch.export`` traces
at aten-op granularity instead of fx-module granularity, so models
with dynamic-Python-shape branches lower cleanly.

Phase A scope (this commit): trace the model, classify every aten op,
print the work-list of new compute ops the int8 extractor will need to
handle. Does NOT emit ``graph.json`` / ``weights.npz`` / ``io.npz``
yet — that's the next slice of work; this is the discovery pass that
tells us exactly which ops to wire up.

Usage:

    conda activate xpurt   # ViNT/efficientnet deps live here
    python -m agents.pipeline.extract_graph_export \\
        --model vint --quant int8 --out-dir /tmp/vint_probe

Run via the ``xpurt`` env so the vendored vint_train + efficientnet
deps are importable; the agents/pipeline package is on sys.path
through ``zephyr-chipyard-sw/agents`` either way.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch


# Aten ops we already lower to integer kernels in extract_int8 today.
# Used to classify the work-list.
_SUPPORTED_COMPUTE = {
    "conv2d.default",           # incl. depthwise via groups arg (kernel branches)
    "linear.default",
    "relu.default",
    "add.Tensor",               # residual add → add_s8 (gemmini_q31 resadd)
    "max_pool2d.default",
    "max_pool2d_with_indices.default",
    "cat.default",              # → cat{2,3,4}_c1_s8
    "concat.default",
}

# New compute ops we need to add for ViNT (see notes/vint_zephyr_plan.md §1).
_NEW_COMPUTE = {
    "mul.Tensor":                  "mul_s8 (elementwise; SE gating + post-attention)",
    "sigmoid.default":             "sigmoid_s8 (LUT)",
    "gelu.default":                "gelu_s8 (LUT)",
    "layer_norm.default":          "layer_norm_s8 (row reduce + reciprocal)",
    "adaptive_avg_pool2d.default": "adaptive_avg_pool2d_s8 (SE pool + final pool)",
    # scaled_dot_product_attention will be DECOMPOSED into matmul_s8 +
    # softmax_s8 + matmul_s8 at extract time so the scheduler can place
    # the two matmuls independently.
    "scaled_dot_product_attention.default": "decompose → matmul_s8 + softmax_s8 + matmul_s8",
}

# Zero-compute aliases — the existing dispatch-id remap pattern in
# generate_skeleton already filters these from the codegen table.
_ALIAS = {
    "view.default", "transpose.int", "slice.Tensor", "select.int",
    "unsqueeze.default", "unflatten.int", "squeeze.dim",
    "contiguous.default", "permute.default", "reshape.default",
    "flatten.using_ints", "split.Tensor", "expand_as.default",
    "clone.default",
}

# Folded into the preceding op at quant time.
_FOLD = {
    "batch_norm.default":      "fold into preceding conv2d (scale + bias adjustments)",
    "pad.default":              "fold into conv2d padding args",
}

# Eval-time no-ops.
_NOOP = {
    "dropout.default":                                         "noop (eval mode)",
    "wrap_with_set_grad_enabled":                              "torch.export artifact",
    "<built-in function getitem>":                              "tuple/list indexing",
    "copy_.default":                                            "noop (in-place identity)",
}

# Tail post-process — small / scalar / one-off, runs once per inference
# in the C harness's main, not as a profiled dispatch.
_TAIL = {
    "cumsum.default":            "delta→absolute waypoints (one-off, scalar)",
    "linalg_vector_norm.default": "F.normalize angle (one-off, scalar)",
    "clamp_min.default":          "F.normalize angle (one-off, scalar)",
    "div.Tensor":                 "F.normalize angle (one-off, scalar)",
}


def _classify(name: str) -> str:
    if name in _SUPPORTED_COMPUTE: return "supported"
    if name in _NEW_COMPUTE:        return "new"
    if name in _ALIAS:              return "alias"
    if name in _FOLD:               return "fold"
    if name in _NOOP:               return "noop"
    if name in _TAIL:               return "tail"
    return "UNKNOWN"


def _op_name(node) -> str:
    if node.op == "call_function":
        return str(node.target).split('.OverloadPacket')[0].replace('aten.', '')
    if node.op == "call_method":
        return f"method:{node.target}"
    return node.op


def _load_model(name: str):
    if name == "vint":
        from agents.models import vint as model_mod
    else:
        # Defer to the FX extractor's choices for any non-export-only model.
        raise SystemExit(
            f"--model {name} doesn't need extract_graph_export; "
            f"use extract_graph.py (FX-based) instead."
        )
    return model_mod.get_model(), model_mod.get_sample_input()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=["vint"],
                   help="Model registered in agents/models/.")
    p.add_argument("--quant", default="int8", choices=["fp32", "int8"],
                   help="Quant mode. fp32 dumps the aten graph as-is; int8 "
                        "(default for this discovery pass) is the eventual "
                        "target.")
    p.add_argument("--out-dir", required=True,
                   help="Where to write the discovery report (and "
                        "graph.json/weights.npz/io.npz in a future slice).")
    args = p.parse_args()

    model, sample = _load_model(args.model)
    if isinstance(sample, torch.Tensor):
        sample = (sample,)

    print(f"[extract_export] tracing {args.model} via torch.export ...",
          flush=True)
    ep = torch.export.export(model, sample)
    print(f"[extract_export] traced: {len(list(ep.graph_module.graph.nodes))} "
          f"aten nodes", flush=True)

    by_class: dict[str, Counter] = defaultdict(Counter)
    for n in ep.graph_module.graph.nodes:
        if n.op in ("placeholder", "get_attr", "output"):
            continue
        opname = _op_name(n)
        by_class[_classify(opname)][opname] += 1

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "op_inventory.txt"
    with open(report_path, "w") as f:
        for cls in ("supported", "new", "fold", "alias", "noop", "tail", "UNKNOWN"):
            if not by_class[cls]:
                continue
            f.write(f"=== {cls} ({sum(by_class[cls].values())}) ===\n")
            for op, n in by_class[cls].most_common():
                note = _NEW_COMPUTE.get(op) or _FOLD.get(op) or _NOOP.get(op) or _TAIL.get(op) or ""
                f.write(f"  {n:5d}  {op:40s}  {note}\n")
            f.write("\n")
    print(f"[extract_export] wrote op inventory to {report_path}", flush=True)

    if by_class["UNKNOWN"]:
        print("\n!!! UNKNOWN ops below need a classification in extract_graph_export.py:")
        for op, n in by_class["UNKNOWN"].most_common():
            print(f"    {n:5d}  {op}")
        sys.exit(1)

    print("\nDiscovery summary (no IR emitted yet):")
    for cls in ("supported", "new", "fold", "alias", "noop", "tail"):
        n = sum(by_class[cls].values()); unique = len(by_class[cls])
        print(f"  {cls:10s} {n:5d} calls across {unique:3d} unique kinds")

    if by_class["new"]:
        print("\nWork-list (new compute ops to add to extract_int8 + "
              "reference_kernels + verify_kernel + skeleton emitter):")
        for op, n in by_class["new"].most_common():
            print(f"  • {op} ×{n} — {_NEW_COMPUTE[op]}")


if __name__ == "__main__":
    main()
