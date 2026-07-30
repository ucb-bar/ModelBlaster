"""Runner-side helpers for hetero workload artifacts.

The aggregator's `cross_tile_bytes` extractor reads
``cross_tile_estimate.json`` per cell; this module produces that file
by joining the workload's IR (per-op output tensor sizes from
``graph.json``) with the XPU-RT schedule (per-dispatch slot
assignment from ``schedule.json``). Every dependency edge whose
producer-tile differs from its consumer-tile contributes the
producer's output bytes to the total -- a static upper bound on
inter-tile communication, not a measured number.

Static analysis is honest about its scope: it does NOT account for
scratch-buffer reuse, double-buffering, or any runtime placement
optimizations the harness might apply. Treat the number as
"max bytes that COULD have to cross the boundary," and pair it with
makespan + per-tile utilization to reason about whether a placement
heuristic is buying or losing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# Conventional dtype-to-bytes mapping for IR tensor entries. Covers
# the dtypes the extractor emits today; unknown strings default to
# 1 byte and the computation flags the cell with a note.
_DTYPE_BYTES = {
    "i8": 1, "u8": 1, "int8": 1, "uint8": 1, "bool": 1,
    "i16": 2, "u16": 2, "int16": 2, "uint16": 2,
    "f16": 2, "float16": 2, "bf16": 2, "bfloat16": 2,
    "i32": 4, "u32": 4, "int32": 4, "uint32": 4,
    "f32": 4, "float32": 4,
    "i64": 8, "u64": 8, "int64": 8, "uint64": 8,
    "f64": 8, "float64": 8,
}


def _tensor_bytes(shape: list[int], dtype: str) -> tuple[int, bool]:
    n = 1
    for d in shape:
        n *= int(d)
    dt_bytes = _DTYPE_BYTES.get(dtype)
    if dt_bytes is None:
        return n, True
    return n * dt_bytes, False


def compute_cross_tile_bytes(
    graph_path: Path, schedule_path: Path,
) -> dict[str, Any]:
    """Walk every dependency edge in graph.json; for each edge whose
    producer and consumer dispatches are scheduled to different
    abstract slot labels, add the producer's output-tensor bytes to
    the total. Returns the rollup ready to be written as
    ``cross_tile_estimate.json``.

    The schedule's ``hardware_target`` field is the abstract slot
    label (e.g. ``CPU_P#0`` / ``CPU_E#0``). Two consumers on the same
    slot don't move bytes between tiles regardless of how the registry
    resolves the slot, so comparing slot labels directly is the right
    granularity. (If a future config remaps slots after the schedule
    is generated, this estimate stays consistent with the schedule's
    intent rather than the runtime resolution.)
    """
    graph = json.loads(graph_path.read_text())
    schedule = json.loads(schedule_path.read_text())

    target_by_id: dict[int, str] = {}
    for entry in schedule.get("dispatches", {}).values():
        d_id = entry.get("id")
        target = entry.get("hardware_target")
        if d_id is None or not target:
            continue
        target_by_id[int(d_id)] = target

    tensors: dict[str, Any] = graph.get("tensors", {}) or {}
    ops = graph.get("ops", []) or []
    by_id = {op["dispatch_id"]: op for op in ops
             if op.get("dispatch_id") is not None}

    total_bytes = 0
    edges: list[dict[str, Any]] = []
    unknown_dtypes: set[str] = set()

    for consumer in ops:
        c_id = consumer.get("dispatch_id")
        if c_id is None:
            continue
        c_tile = target_by_id.get(int(c_id))
        if c_tile is None:
            continue
        for raw_dep in consumer.get("depends_on") or []:
            dep_id = int(raw_dep)
            p_tile = target_by_id.get(dep_id)
            if p_tile is None or p_tile == c_tile:
                continue
            producer = by_id.get(dep_id)
            if producer is None:
                continue
            for out_name in producer.get("outputs") or []:
                tinfo = tensors.get(out_name)
                if not tinfo:
                    continue
                shape = tinfo.get("shape") or []
                dtype = (tinfo.get("dtype") or "").strip().lower()
                size, was_unknown = _tensor_bytes(shape, dtype)
                if was_unknown:
                    unknown_dtypes.add(dtype or "<empty>")
                total_bytes += size
                edges.append({
                    "producer_id": dep_id,
                    "consumer_id": int(c_id),
                    "tensor": out_name,
                    "shape": shape,
                    "dtype": dtype,
                    "producer_tile": p_tile,
                    "consumer_tile": c_tile,
                    "bytes": size,
                })

    return {
        "schema_version": 1,
        "graph_source": str(graph_path),
        "schedule_source": str(schedule_path),
        "total_bytes": total_bytes,
        "n_cross_tile_edges": len(edges),
        "edges": edges,
        "unknown_dtypes": sorted(unknown_dtypes),
    }


def write_cross_tile_estimate(
    out_dir: Path, graph_path: Optional[Path],
    schedule_path: Optional[Path],
) -> Optional[dict[str, Any]]:
    """Resolve graph + schedule for the workload and emit
    ``cross_tile_estimate.json`` into ``out_dir``. Returns the rollup
    dict (or None when either input is unavailable -- common when
    extract_graph hasn't been run yet for the cell's quant level)."""
    if (graph_path is None or schedule_path is None
            or not graph_path.exists() or not schedule_path.exists()):
        return None
    rollup = compute_cross_tile_bytes(graph_path, schedule_path)
    (out_dir / "cross_tile_estimate.json").write_text(
        json.dumps(rollup, indent=2)
    )
    return rollup
