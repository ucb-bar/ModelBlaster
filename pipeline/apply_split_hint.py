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
shape, just on the conv output. Whether the weights need slicing is a
BACKEND question, and `generate_skeleton` answers it, not this module:
on a backend that keeps conv weights OIHW `[OC, IC, KH, KW]` an OC slice
is contiguous and a pointer offset suffices, while the rvv backends pack
them IHWOC `(IC, KH, KW, OC)` — OC innermost — where an OC slice is
strided and each tile is given its own re-packed array. See
`generate_skeleton.split_conv_tile_weights`.

Inserting tiles renumbers every op after the split point, so the output
graph carries an `id_remap` field mapping every input `dispatch_id` to
its id(s) in the rewritten graph (JSON object keys are strings)::

    "id_remap": {"0": [0, 1], "1": [2], "2": [3]}

Values are always lists: splitting is one-to-many, so op 0 above became
tiles 0 and 1, while untouched ops 1 and 2 (renumbered to 2 and 3) get
single-element lists. Consumers keyed on dispatch_id must translate
through this before joining a pre-rewrite profile / cost DB against a
post-rewrite graph — and must sum across the tiles when attributing a
split op's cost. (`apply_fusion_hint.py` emits the same field, but
scalar-valued — fusion is many-to-one.)
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


def _split_conv2d_s8(op: dict[str, Any], n_splits: int,
                     network: str) -> list[dict[str, Any]]:
    """Split a conv2d_s8 op along the OC (output channel) dim.

    Splitting along OC produces n tiles that each compute a disjoint
    slice of output channels. How the tile reaches its own weights is
    left to the codegen, because it depends on the layout the backend
    packs conv weights in: OIHW `[OC, IC, KH, KW]` makes an OC slice
    contiguous, so `weight + t*(OC/n)*IC*KH*KW` and `bias + t*(OC/n)`
    are enough; the rvv backends' IHWOC packing makes it strided, so the
    tile gets its own re-packed array instead. This function records
    only `tile_offset_OC` and `axis`, and
    `generate_skeleton.split_conv_tile_weights` decides.

    The output tensor `[N, OC, OH, OW]` splits along OC: tile_t
    writes channels `[t*OC/n, (t+1)*OC/n)`. Output is named
    `<orig>.tile_<t>`. Downstream consumers see the original tensor
    as the concat of tiles along OC (or read each tile slice
    explicitly by name).
    """
    shape = op["shape"]
    OC = int(shape["OC"])
    if OC % n_splits != 0:
        raise SplitHintError(
            f"{network}: conv2d_s8 OC={OC} doesn't divide cleanly into "
            f"{n_splits} tiles (need OC % n_splits == 0)")
    tile_oc = OC // n_splits
    out_tensor = op["outputs"][0]
    tiles: list[dict[str, Any]] = []
    for t in range(n_splits):
        tile = copy.deepcopy(op)
        tile["shape"] = dict(shape)
        tile["shape"]["OC"] = tile_oc
        tile["outputs"] = [f"{out_tensor}.tile_{t}"]
        tile["name"] = op["name"] + f".tile_{t}"
        tile["split_from"] = {"op_id": op["dispatch_id"], "tile": t,
                              "n_splits": n_splits, "tile_oc": tile_oc,
                              "tile_offset_OC": t * tile_oc,
                              "axis": "OC"}
        tiles.append(tile)
    return tiles


_SPLITTABLE: dict[str, Any] = {
    "linear_s8": _split_linear_s8,
    "conv2d_s8": _split_conv2d_s8,
}


def _register_tile_tensors(graph: dict[str, Any], op: dict[str, Any],
                           n_splits: int, tile_n: int) -> None:
    """When we split a linear_s8 op along N, the per-tile output names
    (`<orig>.tile_<i>`) need entries in the IR's tensors dict so the
    downstream skeleton emitter can allocate buffers for them. The
    tile shape is the parent op's output shape with the last dim
    (N) replaced by tile_n.

    Without this, generate_skeleton.py walks `graph["tensors"]` to
    emit `buf_<network>_<tensor>[size]` definitions, the tile tensors
    aren't found, and `model.c` references undeclared `buf_..._tile_0`
    at link time."""
    tensors = graph.setdefault("tensors", {})
    out_name = op["outputs"][0]
    parent = tensors.get(out_name)
    if not parent or "shape" not in parent:
        return  # nothing to do — IR has no shape info for this tensor
    parent_shape = list(parent["shape"])
    if not parent_shape:
        return
    tile_shape = list(parent_shape)
    # Split axis depends on the op kind: linear_s8 splits N (last dim),
    # conv2d_s8 splits OC. Both encode the per-tile size into tile_n
    # by the caller — for conv we splice the OC dim of [N,OC,OH,OW].
    if op.get("op") == "conv2d_s8" and len(tile_shape) >= 4:
        tile_shape[1] = tile_n   # NCHW: OC is dim 1
    else:
        tile_shape[-1] = tile_n  # linear_s8: N is last dim
    for t in range(n_splits):
        tile_name = f"{out_name}.tile_{t}"
        if tile_name not in tensors:
            entry = {
                "shape": tile_shape,
                "dtype": parent.get("dtype", "i8"),
                "split_from": out_name,
                "tile": t,
                "n_splits": n_splits,
            }
            # Carry the parent's quantization across. A tile is a slice of the
            # parent tensor -- same scale, same zero point, by construction --
            # and anything reading the tile's own metadata (the codegen's
            # inspect blocks, an accuracy comparison against a golden, a
            # downstream requantize) otherwise falls back to scale=1.0 and
            # silently reports raw int8 counts as physical values.
            if parent.get("quant") is not None:
                entry["quant"] = copy.deepcopy(parent["quant"])
            tensors[tile_name] = entry


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
            # Register the per-tile output tensors in the IR's tensors
            # dict so generate_skeleton can allocate per-tile buffers
            # (fixes "buf_<network>_<out>_tile_0 undeclared" build error).
            if tile_ops and "shape" in tile_ops[0]:
                # tile_n: for linear_s8 the per-tile N; for conv2d_s8 the per-tile OC
                shape = tile_ops[0]["shape"]
                tile_n = shape.get("N") if op.get("op") == "linear_s8" else shape.get("OC", 0)
                _register_tile_tensors(out, op, n, tile_n)
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
    # Publish the renumbering. Inserting tiles shifts every id after the
    # split point, so ops that were NOT split still change identity —
    # without this, anything keyed on dispatch_id (profile CSVs, cost
    # DBs, SchedulerReports, Gantt labels) silently re-attaches to the
    # wrong op after a rewrite.
    #
    # Values are LISTS because splitting is one-to-many: a split op maps
    # to all of its tile ids, in tile order, and collapsing that to the
    # first tile would misreport the op's cost as one tile's cost.
    # Untouched ops get a single-element list so consumers never have to
    # branch on the value type. (`apply_fusion_hint.py` emits the same
    # field scalar-valued — fusion is many-to-one.)
    out["id_remap"] = {str(old): list(new)
                       for old, new in sorted(id_remap.items())}
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
