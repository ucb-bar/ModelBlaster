"""Apply XPU-RT axis-C split hints (modelblaster.split_hints/v1) to an IR.

The dual of `pipeline/apply_fusion_hint.py`. Where fusion collapses N
adjacent dispatches into one (good for too_fine workloads), splitting
decomposes one heavy dispatch into N smaller ones the scheduler can
place on separate cores in parallel (good for too_coarse workloads).

XPU-RT's `granularity_loop.py` emits a Contract-2 split hint like:

    {"contract": "modelblaster.split_hints/v1",
     "reason": "granularity 'too_coarse'; split conv2d_s8[4] across OC",
     "networks": [
        {"network": "yolov8_nano",
         "split_ops": [{"op": 4, "n_splits": 2}]}
     ]}

For `linear_s8` (Phase 1e initial scope), splitting along the output
feature dim (N) is the cleanest tile: each tile computes a disjoint
subset of output rows from the same input. No final concat needed —
each tile writes to its own output slice; downstream consumers
treat the original output tensor as the concat of the tiles by name
convention (`<orig>.tile_0`, `<orig>.tile_1`, ...).

For `conv2d_s8`, the tile dimension is OC (output channels) — same
shape, just on the conv output. Weight tensor surgery (slicing
`[OC, IC, KH, KW]` into N OC-slices) is required and is a follow-up.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


SPLIT_CONTRACT = "modelblaster.split_hints/v1"


class SplitHintError(ValueError):
    """Hint can't be applied — e.g. op kind not split-capable, N doesn't
    divide the tilable dim, op id missing from IR."""


def _ops_by_id(ops: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {op["dispatch_id"]: op for op in ops
            if op.get("dispatch_id") is not None}


def _split_linear_s8(op: dict[str, Any], n_splits: int,
                     network: str) -> list[dict[str, Any]]:
    """Split a linear_s8 op along the N (output features) dim.

    Generates `n_splits` new ops, each computing rows [t*N/n, (t+1)*N/n)
    of the original output. Weight and bias slices share the source
    tensor (no new buffers — the kernel call passes `weight + t*N/n*K`
    and `bias + t*N/n` as offsets) so we keep weight extraction simple.

    Output is one tensor `<orig.out>.tile_t` per tile; downstream
    consumers of the original tensor either rely on contiguous layout
    (the tiles are placed back-to-back in memory) or get a follow-up
    concat op.
    """
    shape = op["shape"]
    N = int(shape["N"])
    if N % n_splits != 0:
        raise SplitHintError(
            f"{network}: linear_s8 N={N} doesn't divide cleanly into "
            f"{n_splits} tiles (need N % n_splits == 0)")
    tile_n = N // n_splits
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    for t in range(n_splits):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["N"] = tile_n
        # New per-tile output name; the OG tensor is reassembled by the
        # codegen via offset aliasing (downstream consumers see the tile
        # output's contiguous storage).
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": n_splits, "tile_n": tile_n,
                              "tile_offset_N": t * tile_n}
        # `hardware_target` left as-is; the scheduler may place tiles
        # on different cores at scheduling time.
        tiles.append(tile)
    return tiles


_SPLITTABLE: dict[str, Any] = {
    "linear_s8": _split_linear_s8,
    # conv2d_s8 follow-up (needs weight slicing).
}


def apply_split_hint(graph: dict[str, Any],
                     split_ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Rewrite `graph` so each op listed in `split_ops` is decomposed
    into `n_splits` tile ops. Returns a new dict; input is not mutated.
    """
    out = copy.deepcopy(graph)
    if not split_ops:
        return out
    network = out.get("name", "<unknown>")
    ops = out["ops"]
    ops_by_id = _ops_by_id(ops)

    # Validate everything first; reject if any op is unsupported.
    for spec in split_ops:
        op_id = spec["op"]; n = int(spec.get("n_splits", 2))
        if op_id not in ops_by_id:
            raise SplitHintError(
                f"{network}: split_ops references unknown dispatch_id {op_id}")
        op = ops_by_id[op_id]
        if op["op"] not in _SPLITTABLE:
            raise SplitHintError(
                f"{network}: op kind {op['op']!r} (dispatch_id={op_id}) "
                f"not yet split-capable; supported kinds: "
                f"{sorted(_SPLITTABLE)}")
        if n < 2:
            raise SplitHintError(
                f"{network}: n_splits={n} must be >= 2 to split")

    # Single-pass rewrite: walk in original order, replacing split ops
    # with their tile lists. Re-assign dispatch_ids contiguously after.
    target_ids = {s["op"]: int(s.get("n_splits", 2)) for s in split_ops}
    new_ops: list[dict[str, Any]] = []
    id_remap: dict[int, list[int]] = {}  # original -> [new tile ids]
    next_new_id = 0
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            new_ops.append(copy.deepcopy(op))
            continue
        if did in target_ids:
            n = target_ids[did]
            splitter = _SPLITTABLE[op["op"]]
            tile_ops = splitter(op, n, network)
            new_tile_ids: list[int] = []
            for tile in tile_ops:
                tile["dispatch_id"] = next_new_id
                new_tile_ids.append(next_new_id)
                new_ops.append(tile)
                next_new_id += 1
            id_remap[did] = new_tile_ids
        else:
            nop = copy.deepcopy(op)
            id_remap[did] = [next_new_id]
            nop["dispatch_id"] = next_new_id
            new_ops.append(nop)
            next_new_id += 1

    # Rewire depends_on: a downstream op that depended on original
    # split op `K` now depends on ALL of K's tiles (the original
    # output is the concat of the tiles, so the consumer can't start
    # until every tile finishes).
    for nop in new_ops:
        if nop.get("dispatch_id") is None:
            continue
        new_deps: list[int] = []
        for d in nop.get("depends_on", []):
            if d in id_remap:
                new_deps.extend(id_remap[d])
        # Deduplicate while preserving order.
        seen: set[int] = set()
        nop["depends_on"] = [x for x in new_deps if not (x in seen or seen.add(x))]

    out["ops"] = new_ops
    return out


def _load_hint(path: Path) -> dict[str, Any]:
    hint = json.loads(path.read_text())
    if hint.get("contract") != SPLIT_CONTRACT:
        raise SplitHintError(
            f"hint contract is {hint.get('contract')!r}, expected "
            f"{SPLIT_CONTRACT!r}")
    return hint


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hint", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--ir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    hint = _load_hint(args.hint)
    split_ops: list[dict[str, Any]] = []
    for entry in hint.get("networks", []):
        if entry.get("network") == args.model:
            split_ops = entry.get("split_ops", [])
            break
    if not split_ops:
        print(f"apply_split_hint: no split_ops for {args.model!r} in "
              f"{args.hint} — copying IR through unchanged", file=sys.stderr)

    graph = json.loads(args.ir.read_text())
    if graph.get("name") != args.model:
        graph["name"] = args.model

    rewritten = apply_split_hint(graph, split_ops)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rewritten, indent=2))
    n_split = sum(1 for op in rewritten["ops"] if "split_from" in op)
    n_total = sum(1 for op in rewritten["ops"]
                  if op.get("dispatch_id") is not None)
    print(f"apply_split_hint: wrote {args.out} "
          f"({n_split} tile ops from splits, {n_total} total dispatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
