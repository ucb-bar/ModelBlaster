"""Emit an IREE-format `<model>_dispatch_graph.json` from our IR.

XPU-RT's `dispatch_deps_path` reader expects this shape:

    {
      "dot_file": "<optional .dot path>",
      "dispatches": {
        "dispatch_<id>[ _<ord>]": {
          "id":           <int>,
          "ordinal":      <int>,
          "total":        <int>,
          "dependencies": ["dispatch_<other_id>", ...]
        },
        ...
      }
    }

For static, single-shot IRs (no recurrent / multi-ordinal dispatches),
ordinal=1, total=1 for every entry; keys are simply `dispatch_<id>`.

Each entry's id matches the `dispatch_id` column emitted by our
profile_writer (single source of truth via extract_graph's
`_annotate_dispatches`), so XPU-RT can join the graph against the
results.csv directly.

Output path (XPU-RT convention, matching IREE):

    gen/vmfb/<model>/<target>/<hw>/<basename>/<basename>_dispatch_graph.json

with `<basename> = <model>.<quant>` (e.g. `mlp_control.fp32`).

Usage:
    python -m agents.pipeline.emit_dispatch_graph \\
        --ir agents/examples/mlp_control/fp32/generated/graph.json \\
        --out-root gen/vmfb \\
        --target generic_riscv64 \\
        --hw RVV
"""

from __future__ import annotations

import argparse
import json
import os


def _key(dispatch_id: int) -> str:
    return f"dispatch_{dispatch_id}"


def build_graph(ir: dict) -> dict:
    """Convert the IR's annotated dispatch records into XPU-RT format."""
    # Build dispatch_id -> set of producer dispatch_ids (for the
    # `dependencies` field). _annotate_dispatches already wrote
    # `depends_on` per non-view op; trust that.
    dispatches: dict[str, dict] = {}
    for op in ir.get("ops", []):
        if op["op"] == "view":
            continue  # zero-cost alias; not a runnable dispatch
        did = op["dispatch_id"]
        deps = [
            _key(d) for d in op.get("depends_on", [])
        ]
        dispatches[_key(did)] = {
            "id": int(did),
            "ordinal": 1,
            "total": 1,
            "dependencies": deps,
        }
    return {
        "dot_file": "",  # we don't emit a .dot today; field kept for compat
        "dispatches": dispatches,
    }


def output_path(out_root: str, model: str, target: str, hw: str,
                quant: str) -> str:
    basename = f"{model}.{quant}"
    return os.path.join(
        out_root, model, target, hw, basename,
        f"{basename}_dispatch_graph.json",
    )


def emit(ir_path: str, out_root: str, target: str, hw: str) -> str:
    with open(ir_path) as f:
        ir = json.load(f)
    if not ir.get("ops"):
        raise ValueError(f"{ir_path} has no ops")
    if any("dispatch_id" not in op for op in ir["ops"] if op.get("op") != "view"):
        raise ValueError(
            f"{ir_path}: ops missing dispatch_id; re-run extract_graph "
            "(the _annotate_dispatches pass adds it)"
        )

    graph = build_graph(ir)
    quant = ir.get("quant", "fp32")
    out_path = output_path(out_root, ir["name"], target, hw, quant)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="path to graph.json")
    ap.add_argument("--out-root", default="gen/vmfb",
                    help="output tree root (default: gen/vmfb)")
    ap.add_argument("--target", required=True,
                    help="target HW label, e.g. generic_riscv64, spacemit_x60")
    ap.add_argument("--hw", required=True,
                    help="HW backend, e.g. RVV, scalar")
    args = ap.parse_args()
    out_path = emit(args.ir, args.out_root, args.target, args.hw)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
