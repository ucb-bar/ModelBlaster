"""Extractors for the per-cell graph summary.

``graph_summary.json`` is emitted by the arm driver from the pipeline's
``graph.json`` (which is per-model and lives outside the run dir). It
captures the graph-shape questions a dashboard column should answer
without re-parsing the full graph:

  * ``n_dispatches``      number of compute records the harness will
                          dispatch (== ``len(graph.ops)``).
  * ``n_distinct_op_kinds`` count of unique ``op`` strings -- a
                          proxy for "how many distinct kernel
                          implementations does this cell need."
  * ``n_distinct_shapes`` count of unique ``(op, shape_signature)``
                          pairs across the graph -- a proxy for
                          "how many shape-specialized kernel variants
                          would a perfect generator have to produce
                          to cover every dispatch with a tight fit."
  * ``by_op_kind``        per op kind: count + distinct shapes seen.

Shape signatures are the ordered tuple of shape values, so e.g.
``conv2d_s8`` with ``(N=1, IC=3, IH=112, IW=112, OC=32, ...)`` is
distinct from the same op kind at ``IC=32`` -- different kernel.

The aggregator's `n_dispatches_graph` / `n_distinct_op_kinds` /
`n_distinct_shapes` metrics all read from this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def n_dispatches_graph(path: Path) -> Optional[int]:
    """Total compute records in graph.json. Different from
    `n_ops_profiled` (which counts what the runner saw at runtime --
    aliases and noops drop out)."""
    v = _load(path).get("n_dispatches")
    return int(v) if v is not None else None


def n_distinct_op_kinds(path: Path) -> Optional[int]:
    """Unique op-name count. Tells you how many distinct kernel
    implementations (curated or LLM-synthesized) the cell needs."""
    v = _load(path).get("n_distinct_op_kinds")
    return int(v) if v is not None else None


def n_distinct_shapes(path: Path) -> Optional[int]:
    """Unique (op, shape) pairs. A high value relative to
    n_distinct_op_kinds means many shape-specialized kernels (or a
    high cost if every shape needs its own LLM-generated variant).
    A low value -- close to n_distinct_op_kinds -- means a few
    parametric kernels suffice."""
    v = _load(path).get("n_distinct_shapes")
    return int(v) if v is not None else None


def synthesize(graph_json_path: Path) -> dict[str, Any]:
    """Build the graph_summary.json payload by walking the ops in
    ``graph.json``. Shape signatures use a stable ordered tuple of the
    shape dict's values so identical shapes serialize identically.
    """
    data = _load(graph_json_path)
    ops = data.get("ops") or []
    op_kinds: dict[str, dict[str, Any]] = {}
    distinct_shapes: set[tuple[str, tuple]] = set()

    for op in ops:
        kind = str(op.get("op", "")) or "unknown"
        shape = op.get("shape") or {}
        # Stable signature: sort keys so two ops with the same dims
        # but different dict ordering hash identically.
        if isinstance(shape, dict):
            sig = tuple(sorted(shape.items()))
        else:
            sig = tuple(shape) if hasattr(shape, "__iter__") else (shape,)
        distinct_shapes.add((kind, sig))

        slot = op_kinds.setdefault(kind, {"count": 0,
                                          "distinct_shapes": set()})
        slot["count"] += 1
        slot["distinct_shapes"].add(sig)

    by_op_kind = {
        kind: {"count": slot["count"],
               "distinct_shapes": len(slot["distinct_shapes"])}
        for kind, slot in op_kinds.items()
    }

    return {
        "schema_version": 1,
        "n_dispatches": len(ops),
        "n_distinct_op_kinds": len(op_kinds),
        "n_distinct_shapes": len(distinct_shapes),
        "by_op_kind": by_op_kind,
    }
