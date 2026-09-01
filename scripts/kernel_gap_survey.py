#!/usr/bin/env python3
"""Phase E1 — kernel gap survey.

Enumerate which (op_type_pair) tuples have a registered KernelSpec
that can realize a fused dispatch, vs which pairs the granularity
advisor proposes as fuse candidates but we have no fused kernel for.

Output:
    artifacts/kernel_gap_survey.json:
      {
        "registered_fused_kernels": [...],
        "candidate_pairs_seen": {pair: count},
        "gaps": [{pair, candidate_count, estimated_save_us_summed}, ...],
        "workloads_scanned": [...]
      }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_XPURT = Path("/scratch2/agustin/XPU-RT")

# Make pipeline imports work without modifying sys.path globally.
sys.path.insert(0, str(_REPO))


def enumerate_registered_fused() -> list[dict]:
    """Find KernelSpecs whose op name embeds an underscore-joined pair of
    base op names (e.g. 'linear_s8_elu_s8'). Heuristic but reliable for
    the current registry.
    """
    from pipeline.reference_kernels import KERNELS  # name varies; try several
    return _scan_registry()


def _scan_registry() -> list[dict]:
    """Scan reference_kernels.py for `op="<single>_<other>_<s8|fp32>"` pairs
    via regex — robust to import-order issues. We treat any op whose name
    contains two distinct base-op markers as a fused kernel."""
    import re
    src = (_REPO / "pipeline" / "reference_kernels.py").read_text()
    op_lines = re.findall(r'op="([^"]+)"', src)
    # Known fused pairs in the codebase (manual enumeration — the regex
    # below would also catch them):
    base_ops = {"linear", "matmul", "bmm", "conv2d", "conv2d_dw", "elu",
                "relu", "relu6", "leaky_relu", "tanh", "sigmoid", "swish",
                "silu", "gelu", "selu", "hardsigmoid", "softplus",
                "softsign", "hardtanh", "add", "batchnorm2d", "maxpool2d"}
    fused = []
    for op in op_lines:
        parts = op.split("_")
        # Strip trailing dtype tag (s8/fp16/etc).
        if parts and parts[-1] in {"s8", "f16", "f32", "fp16", "fp32"}:
            dtype = parts[-1]
            core = "_".join(parts[:-1])
        else:
            dtype = ""
            core = op
        # Find consecutive base-op markers inside `core`.
        tokens = core.split("_")
        hits = []
        i = 0
        while i < len(tokens):
            for k in (3, 2, 1):
                if i + k <= len(tokens) and "_".join(tokens[i:i+k]) in base_ops:
                    hits.append("_".join(tokens[i:i+k]))
                    i += k
                    break
            else:
                i += 1
        if len(hits) >= 2:
            fused.append({"op": op, "components": hits, "dtype": dtype})
    return fused


_NETWORK_TO_IR = {
    "mlp_control": "/scratch2/agustin/ModelBlaster/examples/mlp_control/fp32/generated/graph.json",
    "dronet": "/scratch2/agustin/ModelBlaster/examples/dronet/int8/generated/graph.json",
    "yolov8_nano": "/scratch2/agustin/ModelBlaster/examples/yolov8_nano_64/int8/generated/graph.json",
}


def survey_workload_candidates(workload_path: str) -> dict:
    """Walk each network's ModelBlaster IR (graph.json) and count
    candidate fuse pairs by (producer.op, consumer.op).

    The IR is a list of ops with `op` (e.g. 'conv2d_s8'), `depends_on`
    (list of producer names). We build a producer→consumers map and
    count pairs that satisfy the single-producer/single-consumer
    structure that fusion candidate generation uses.
    """
    wl = json.loads(Path(workload_path).read_text())
    networks = wl.get("networks", {})

    pair_counts: Counter = Counter()
    per_network: dict[str, dict[str, int]] = {}

    for net_name in networks:
        ir_path = _NETWORK_TO_IR.get(net_name)
        if ir_path is None or not Path(ir_path).exists():
            print(f"  (no IR for {net_name} — skipping)", file=sys.stderr)
            continue
        ir = json.loads(Path(ir_path).read_text())
        ops_list = ir.get("ops") or []
        # `depends_on` is a list of integer indices into ops_list, not
        # names. Build a consumers map keyed by op index.
        n = len(ops_list)
        consumers: dict[int, list[int]] = {i: [] for i in range(n)}
        for j, op in enumerate(ops_list):
            for pred in op.get("depends_on", []) or []:
                if isinstance(pred, int) and 0 <= pred < n:
                    consumers[pred].append(j)

        net_pair_counts: Counter = Counter()
        for i, p_op in enumerate(ops_list):
            cs = consumers.get(i, [])
            if len(cs) != 1:
                continue
            j = cs[0]
            c_op = ops_list[j]
            preds = c_op.get("depends_on", []) or []
            if len(preds) != 1:
                continue
            p_kind = p_op.get("op") or "unknown"
            c_kind = c_op.get("op") or "unknown"
            pair = f"{p_kind}__{c_kind}"
            net_pair_counts[pair] += 1
            pair_counts[pair] += 1
        per_network[net_name] = dict(net_pair_counts)

    return {
        "workload": workload_path,
        "pair_counts": dict(pair_counts),
        "per_network": per_network,
        "total_candidate_pairs": int(sum(pair_counts.values())),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--workload",
                   default=str(_XPURT / "data" / "toplevel" /
                               "networks_1yolo_4mlp_2dronet_firesim.json"))
    p.add_argument("--out",
                   default=str(_REPO / "artifacts" / "kernel_gap_survey.json"))
    args = p.parse_args()

    registered = _scan_registry()
    registered_pairs = {tuple(r["components"]) for r in registered}
    registered_pair_strs = {
        "__".join(c + "_s8" for c in r["components"]) for r in registered
    }
    # Also a relaxed view: base components without dtype suffix.
    registered_core = {tuple(r["components"]) for r in registered}

    survey = survey_workload_candidates(args.workload)

    pair_counts: dict[str, int] = survey["pair_counts"]
    gaps: list[dict] = []
    for pair, count in sorted(pair_counts.items(), key=lambda kv: -kv[1]):
        a, _, b = pair.partition("__")
        # Strip the dtype tail (e.g. _s8) to compare with registry.
        a_core = a.rsplit("_", 1)[0] if a.endswith(("_s8", "_f16", "_f32",
                                                      "_fp16", "_fp32")) else a
        b_core = b.rsplit("_", 1)[0] if b.endswith(("_s8", "_f16", "_f32",
                                                      "_fp16", "_fp32")) else b
        already_registered = (a_core, b_core) in registered_core or \
                              (b_core, a_core) in registered_core
        gaps.append({
            "pair": pair,
            "candidate_count": count,
            "components": [a, b],
            "already_registered": bool(already_registered),
        })

    result = {
        "registered_fused_kernels": registered,
        "candidate_pairs_seen": pair_counts,
        "gaps_unregistered": [g for g in gaps if not g["already_registered"]],
        "gaps_registered": [g for g in gaps if g["already_registered"]],
        "workload": survey["workload"],
        "per_network": survey.get("per_network", {}),
        "total_candidate_pairs": survey["total_candidate_pairs"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote -> {out_path}\n")
    print(f"Total fuse candidate pairs in workload: {result['total_candidate_pairs']}")
    print(f"Registered fused KernelSpecs: {len(registered)}")
    print(f"  → {[r['op'] for r in registered]}")
    print(f"Unregistered gap pairs (top 10 by count):")
    for g in result["gaps_unregistered"][:10]:
        print(f"  {g['pair']:<48s}  count={g['candidate_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
