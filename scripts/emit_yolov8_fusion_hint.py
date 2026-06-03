#!/usr/bin/env python3
"""Emit a modelblaster.fusion_hints/v1 hint that pairs up every
(conv2d_s8 → batchnorm2d_s8) and (batchnorm2d_s8 → silu_s8) chain in
the yolov8_nano IR. These pairs map to the registered fused
KernelSpecs (verified bit-exact on spike-hetero in Phase E2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REALIZABLE_PAIRS = {
    ("conv2d_s8", "batchnorm2d_s8"),
    ("batchnorm2d_s8", "silu_s8"),
}

# Greedy pair selection priority. When the same BN op can pair with
# either its conv producer or its silu consumer (the common YOLOv8
# conv→BN→SiLU triple), prefer the second option specified here. The
# CLI --prefer flag overrides this.
DEFAULT_PREFER = ("batchnorm2d_s8", "silu_s8")


def find_pairs(graph, prefer=DEFAULT_PREFER):
    """Walk the IR, return list of [(producer_idx, consumer_idx), ...] for
    each pair that is (a) producer's sole consumer is this consumer,
    (b) consumer's sole producer is this producer, (c) op-kind pair is
    realizable. When a BN op could pair with both its conv producer
    and its silu consumer, prefer the `prefer` pair."""
    ops = graph["ops"]
    n = len(ops)
    consumers = {i: [] for i in range(n)}
    for j, op in enumerate(ops):
        for pred in op.get("depends_on", []) or []:
            if isinstance(pred, int) and 0 <= pred < n:
                consumers.setdefault(pred, []).append(j)

    # Pre-pass 1: collect ALL feasible pairs (no consumption yet).
    all_pairs = []  # (i, j, (p_kind, c_kind))
    for i, p in enumerate(ops):
        cs = consumers.get(i, [])
        if len(cs) != 1:
            continue
        j = cs[0]
        c = ops[j]
        preds = c.get("depends_on", []) or []
        if len(preds) != 1:
            continue
        kp = (p.get("op"), c.get("op"))
        if kp not in REALIZABLE_PAIRS:
            continue
        all_pairs.append((i, j, kp))

    # Pre-pass 2: greedy selection with priority on `prefer` pairs.
    # Within the same priority tier, pick in IR order.
    all_pairs.sort(key=lambda t: (0 if t[2] == prefer else 1, t[0]))

    pairs = []
    consumed = set()
    for i, j, kp in all_pairs:
        if i in consumed or j in consumed:
            continue
        pairs.append((i, j))
        consumed.add(i); consumed.add(j)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolov8_nano")
    args = ap.parse_args()
    graph = json.loads(Path(args.ir).read_text())
    pairs = find_pairs(graph)
    # apply_fusion_hint expects integer dispatch_ids in fuse_groups.
    fuse_groups = [[graph["ops"][i]["dispatch_id"],
                    graph["ops"][j]["dispatch_id"]]
                   for i, j in pairs]
    hint = {
        "contract": "modelblaster.fusion_hints/v1",
        "networks": [
            {"network": args.model, "fuse_groups": fuse_groups}
        ],
        "_provenance": {
            "source": "scripts/emit_yolov8_fusion_hint.py",
            "pairs_realizable": [list(p) for p in REALIZABLE_PAIRS],
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(hint, indent=2))
    # Per-kind breakdown.
    by_kind = {}
    for i, j in pairs:
        kp = (graph["ops"][i]["op"], graph["ops"][j]["op"])
        by_kind[kp] = by_kind.get(kp, 0) + 1
    print(f"Wrote {args.out}: {len(fuse_groups)} fuse_groups")
    for k, n in sorted(by_kind.items()):
        print(f"  {k[0]:>20s} -> {k[1]:<20s}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
