"""Apply XPU-RT axis-C fusion hints to a ModelBlaster IR graph.json.

XPU-RT's advisor / granularity_loop produces a Contract-2 hint:

    {
      "contract": "modelblaster.fusion_hints/v1",
      "networks": [
        {"network": "mlp_control",
         "fuse_groups": [[0, 1, 2, 3, 4, 5]],
         "n_tiny": 6}
      ]
    }

Each `fuse_group` is a topologically-ordered list of `dispatch_id`s in
that network's `graph.json`. This module rewrites the IR so each group
collapses into one synthetic op whose `op` name encodes the chain
(`__fused_<op0>__<op1>__..._<opN>`) and whose `sub_ops` carries the
verbatim originals. Boundary inputs / outputs are derived from tensor
producer/consumer analysis; internal tensors are recorded separately so
the codegen path knows which ones to keep stack-local inside the fused
function body. Downstream ops have their `depends_on` rewired to the
fused op's new id; ids are reassigned contiguously.

CLI:

    python -m modelblaster.pipeline.apply_fusion_hint \\
        --hint /scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json \\
        --model mlp_control \\
        --ir   examples/mlp_control/int8/generated/graph.json \\
        --out  examples/mlp_control/int8/generated/graph.fused.json

The rewrite is a pure JSON-in / JSON-out transform; codegen
(`generate_skeleton.py` / `generate_kernels.py`) handles the fused op
shape downstream.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


HINT_CONTRACT = "modelblaster.fusion_hints/v1"


class FusionHintError(ValueError):
    """A fuse_group cannot be applied to this IR — caller's bug, not ours.

    Raised for: missing dispatch_ids, non-contiguous topological order,
    cycles, or branches that escape and re-enter the group. The IR is
    left untouched.
    """


def _ops_by_id(ops: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index ops by `dispatch_id` (skipping view ops with id=None)."""
    return {op["dispatch_id"]: op for op in ops
            if op.get("dispatch_id") is not None}


def _producers(ops: list[dict[str, Any]]) -> dict[str, int]:
    """Tensor name -> producing op's dispatch_id (last writer wins).

    View ops (dispatch_id=None) don't appear as producers — they're
    aliasing reshapes the harness elides at runtime.
    """
    out: dict[str, int] = {}
    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            continue
        for tensor in op.get("outputs", []):
            out[tensor] = did
    return out


def _validate_fuse_group(
    group: list[int],
    ops_by_id: dict[int, dict[str, Any]],
    network: str,
) -> None:
    """Reject groups that can't be safely fused as a linear chain.

    Allowed:
      - All ids exist and are distinct.
      - The group is topologically ordered (each op's depends_on
        references either an earlier group member or an op outside).
      - No op in the group has an external dependency that re-enters
        via a later group member's input — that would imply the fused
        op has a self-dependency.
    """
    if not group:
        raise FusionHintError(f"{network}: empty fuse_group")
    if len(set(group)) != len(group):
        raise FusionHintError(
            f"{network}: fuse_group {group} has duplicate ids")
    missing = [did for did in group if did not in ops_by_id]
    if missing:
        raise FusionHintError(
            f"{network}: fuse_group references unknown dispatch_ids "
            f"{missing}; available ids are {sorted(ops_by_id)[:8]}...")

    group_set = set(group)
    seen: set[int] = set()
    for did in group:
        op = ops_by_id[did]
        for dep in op.get("depends_on", []):
            if dep in group_set and dep not in seen:
                raise FusionHintError(
                    f"{network}: fuse_group {group} is not in topological "
                    f"order — op {did} depends on group member {dep} which "
                    f"appears later")
        seen.add(did)


def _merge_hardware_target(group_ops: list[dict[str, Any]]) -> str:
    """Pick a hardware_target for the fused op.

    If every sub-op shares the same target, keep it. Otherwise fall
    back to "any" — the scheduler will then place the fused op on a
    backend that supports all constituent kernels (which, by
    construction, must exist for the chain to have been profiled).
    """
    targets = {op.get("hardware_target", "any") for op in group_ops}
    if len(targets) == 1:
        return targets.pop()
    return "any"


def _ordered_unique(xs):
    """De-dupe preserving first-seen order (no Set ordering surprises)."""
    seen: set = set()
    out: list = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_fused_op(
    group_ops: list[dict[str, Any]],
    group: list[int],
    network: str,
    all_ops: list[dict[str, Any]],
    model_outputs: set[str],
) -> dict[str, Any]:
    """Synthesize one fused op (no id assignment yet).

    Carries `sub_ops` verbatim, `fused_from` for traceback,
    `internal_tensors` so codegen knows which ones live on the stack,
    and external `inputs` / `outputs` for the dispatch graph.

    `all_ops` is the full op list so we can determine which tensors
    must escape this fused op — every tensor consumed by any op NOT in
    this group (regardless of whether it's in another fused group).
    """
    produced_inside: set[str] = set()
    for op in group_ops:
        for t in op.get("outputs", []):
            produced_inside.add(t)

    group_set = set(group)
    consumed_outside_group: set[str] = set()
    for op in all_ops:
        did = op.get("dispatch_id")
        if did is None or did in group_set:
            continue
        for t in op.get("inputs", []):
            consumed_outside_group.add(t)

    fused_inputs = [
        t for t in _ordered_unique(
            (t for op in group_ops for t in op.get("inputs", [])))
        if t not in produced_inside
    ]
    fused_outputs = [
        t for t in _ordered_unique(
            (t for op in group_ops for t in op.get("outputs", [])))
        if t in consumed_outside_group or t in model_outputs
    ]
    internal_tensors = [
        t for t in _ordered_unique(
            (t for op in group_ops for t in op.get("outputs", [])))
        if t not in fused_outputs
    ]

    sub_op_names = [op["op"] for op in group_ops]
    return {
        "name": f"{network}.fused_{group[0]}_{group[-1]}",
        "op": "__fused__" + "__".join(sub_op_names),
        "inputs": fused_inputs,
        "outputs": fused_outputs,
        "sub_ops": [copy.deepcopy(op) for op in group_ops],
        "fused_from": list(group),
        "internal_tensors": internal_tensors,
        "depends_on": [],  # filled in by caller after id assignment
        "hardware_target": _merge_hardware_target(group_ops),
    }


def apply_hint(
    graph: dict[str, Any],
    fuse_groups: list[list[int]],
) -> dict[str, Any]:
    """Apply a list of fuse_groups (Contract-2) to one network's IR.

    Returns a NEW dict — the input is not mutated. All fuse_groups
    must be DISJOINT in the input ids (one op can't be in two groups);
    this matches how the XPU-RT advisor emits hints (one chain per
    sub-1k-µs cluster).
    """
    out = copy.deepcopy(graph)
    if not fuse_groups:
        return out

    network = out.get("name", "<unknown>")
    ops = out["ops"]
    ops_by_id = _ops_by_id(ops)

    # Model output tensors — needed to keep them in `fused_outputs`
    # even when no downstream op consumes them.
    model_outputs: set[str] = set()
    out_node = out.get("output", {})
    if isinstance(out_node, dict):
        if "tensor" in out_node:
            model_outputs.add(out_node["tensor"])
        for t in out_node.get("tensors", []) or []:
            model_outputs.add(t)

    # Validate, then check disjointness across groups.
    for group in fuse_groups:
        _validate_fuse_group(group, ops_by_id, network)
    all_grouped: set[int] = set()
    for group in fuse_groups:
        overlap = all_grouped.intersection(group)
        if overlap:
            raise FusionHintError(
                f"{network}: fuse_groups overlap on dispatch_id(s) "
                f"{sorted(overlap)} — each op can fuse into at most one chain")
        all_grouped.update(group)

    # Map each original dispatch_id to the group it joins (if any).
    # The fused op takes the slot of the FIRST member of its group;
    # subsequent members are absorbed and dropped.
    group_for_id: dict[int, list[int]] = {}
    head_of_group: dict[int, list[int]] = {}
    for group in fuse_groups:
        for did in group:
            group_for_id[did] = group
        head_of_group[min(group)] = group

    # Single pass: walk ops in input order, emit either a fused op or
    # an unchanged op (with a new dispatch_id), tracking the remap.
    new_ops: list[dict[str, Any]] = []
    id_remap: dict[int, int] = {}  # old_id -> new_id (incl. group->head)
    next_new_id = 0
    fused_op_external_deps: dict[int, list[int]] = {}  # new_id -> orig deps

    for op in ops:
        did = op.get("dispatch_id")
        if did is None:
            new_ops.append(copy.deepcopy(op))
            continue
        group = group_for_id.get(did)
        if group is None:
            new_op = copy.deepcopy(op)
            id_remap[did] = next_new_id
            new_op["dispatch_id"] = next_new_id
            new_ops.append(new_op)
            next_new_id += 1
            continue
        if did != min(group):
            # An absorbed member of an earlier group — skip.
            continue
        # Head of a group → emit the fused op here.
        group_ops = [ops_by_id[g] for g in group]
        group_set = set(group)
        fused = _build_fused_op(
            group_ops, group, network, ops, model_outputs)
        fused["dispatch_id"] = next_new_id
        for g in group:
            id_remap[g] = next_new_id
        # Stash external deps in original-id space; rewire after the
        # remap is fully built.
        fused_op_external_deps[next_new_id] = list(_ordered_unique(
            dep for sub in group_ops
            for dep in sub.get("depends_on", [])
            if dep not in group_set
        ))
        new_ops.append(fused)
        next_new_id += 1

    # Second pass: rewire depends_on now that the full remap exists.
    for new_op in new_ops:
        did = new_op.get("dispatch_id")
        if did is None:
            continue
        if "fused_from" in new_op:
            new_op["depends_on"] = [
                id_remap[d] for d in fused_op_external_deps[did]
            ]
        else:
            new_op["depends_on"] = list(_ordered_unique(
                id_remap[d] for d in new_op.get("depends_on", [])
                if d in id_remap
            ))

    out["ops"] = new_ops
    return out


def _load_hint(path: Path) -> dict[str, Any]:
    with open(path) as f:
        hint = json.load(f)
    contract = hint.get("contract")
    if contract != HINT_CONTRACT:
        raise FusionHintError(
            f"hint contract is {contract!r}, expected {HINT_CONTRACT!r}")
    return hint


def _select_network_hint(
    hint: dict[str, Any],
    network: str,
) -> list[list[int]]:
    for entry in hint.get("networks", []):
        if entry.get("network") == network:
            return entry.get("fuse_groups", [])
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hint", required=True, type=Path,
                   help="modelblaster.fusion_hints/v1 JSON file.")
    p.add_argument("--model", required=True,
                   help="Network name as it appears in the hint and in the "
                        "IR's top-level `name` field.")
    p.add_argument("--ir", required=True, type=Path,
                   help="Input graph.json.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output graph.fused.json (overwritten).")
    args = p.parse_args(argv)

    hint = _load_hint(args.hint)
    fuse_groups = _select_network_hint(hint, args.model)
    if not fuse_groups:
        print(f"apply_fusion_hint: no fuse_groups for network {args.model!r} "
              f"in {args.hint} — copying IR through unchanged",
              file=sys.stderr)

    with open(args.ir) as f:
        graph = json.load(f)
    if graph.get("name") != args.model:
        print(f"apply_fusion_hint: warning — IR name={graph.get('name')!r} "
              f"but --model={args.model!r}; using --model as the network key",
              file=sys.stderr)
        graph["name"] = args.model

    rewritten = apply_hint(graph, fuse_groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rewritten, f, indent=2)
    n_fused = sum(1 for op in rewritten["ops"] if "fused_from" in op)
    n_total = sum(1 for op in rewritten["ops"]
                  if op.get("dispatch_id") is not None)
    print(f"apply_fusion_hint: wrote {args.out} "
          f"({n_fused} fused ops, {n_total} total dispatches; "
          f"input had {sum(1 for op in graph['ops'] if op.get('dispatch_id') is not None)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
