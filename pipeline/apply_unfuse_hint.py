"""Undo a fusion: restore a fused op's constituents as separate dispatches.

The third rewriter, alongside `apply_fusion_hint` (many ops -> one) and
`apply_split_hint` (one op -> many tiles). Pure JSON-in / JSON-out; the
scheduler never edits C.

WHY IT EXISTS, and it is one measured failure rather than a general idea.
Curated kernels are looked up by EXACT op name, so a fused op like
`conv2d_batchnorm2d_silu_s8` matches no per-constituent kernel and silently
falls back to the scalar reference INSIDE a build labelled `rvv_x60`:

    yolov8_nano  rvv_x60   57 of 90 dispatches on reference, 99.8% of 4974.8 ms
                           -- 0.81x against the pure-scalar build

When each constituent does have a vector kernel, unfusing turns one scalar
dispatch into three vector ones. `compile_advice.unfuse_advice` gates on
exactly that and refuses everything else, because undoing a WORKING fusion is
a regression: the curated conv+BN+SiLU kernel applies both epilogues as a table
lookup inside the conv's OC-blocked register tile, so unfusing loses that,
doubles the dispatch count, and adds two full passes over the output tensor.

WHAT THIS KEYS ON, and why not the obvious thing. `apply_fusion_hint` records
`fused_from` and `internal_tensors` on the ops it creates. Almost no fused op
on disk came from it: 0 of yolov8_nano's 57 fused convs carry either field,
because they are built at export time by `extract_graph`'s conv->BN(->act)
recognizer. Keying on `fused_from` would refuse 100% of the ops that matter.
`sub_ops` is present on all 57 and on everything `apply_fusion_hint` emits, so
`sub_ops` is the only field this may rely on.

TWO CONSEQUENCES of that, both load-bearing:

* The internal tensors were never registered. `extract_graph` says so ("the
  conv/bn intermediates live inside the single kernel and need no global
  buffer"), and `l0_conv`/`l0_bn` are absent from `graph["tensors"]` while
  `l0_act` is present. They must be SYNTHESIZED here, not looked up. For ops
  that DID come from `apply_fusion_hint` the internal tensors are still
  registered, so both cases have to work.

* A `conv2d_s8` sub_op has `output_multiplier`/`output_shift` and no
  `scale_out`, so its output scale is recoverable only from the CONSUMER's
  `scale_in`. Verified: `l0_conv`'s scale is `sub_ops[1].quant.scale_in`
  = 0.04244. Where neither is available this refuses rather than defaulting to
  1.0 -- `apply_split_hint` documents what a wrong tensor scale does, which is
  that everything reading the metadata reports raw int8 counts as physical
  values.

`depends_on` is re-derived from tensor producers rather than translated through
the remap. That is the same rule `extract_graph._annotate_dispatches` uses, and
it is correct by construction for both halves of the problem at once -- the
restored chain, and the downstream consumers. Translating ids instead needs two
different maps (consumers attach to the TAIL piece only, while the remap covers
all pieces) and confusing them yields a graph that is structurally valid and
schedules a false dependency.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from typing import Any, Dict, List, Optional

HINT_CONTRACT = "modelblaster.unfuse_hints/v1"

#: Op-kind suffix -> IR dtype, for synthesizing an internal tensor. An
#: unrecognised suffix refuses rather than guessing: a wrong dtype is a silent
#: reinterpretation of every byte in the buffer.
_DTYPE_BY_SUFFIX = {"_s8": "i8", "_f16": "f16", "_f32": "f32"}

#: Ops that do not change tensor shape. Their output shape is the fused op's
#: output shape, which is registered and therefore authoritative -- far more
#: robust than reconstructing `[N,C,H,W]` from a per-kind shape dict.
_SHAPE_PRESERVING = {
    "batchnorm2d_s8", "silu_s8", "relu_s8", "elu_s8", "sigmoid_s8",
    "leaky_relu_s8", "gelu_s8",
}


class UnfuseHintError(ValueError):
    """The hint cannot be applied. The IR is left untouched."""


def _dtype_for(kind: str) -> str:
    for suf, dt in _DTYPE_BY_SUFFIX.items():
        if kind.endswith(suf):
            return dt
    raise UnfuseHintError(
        f"cannot infer a dtype for sub-op kind {kind!r}: no recognised suffix "
        f"({', '.join(sorted(_DTYPE_BY_SUFFIX))}). Refusing rather than "
        f"guessing -- a wrong dtype silently reinterprets the buffer.")


def _validate(op: dict, op_id: Any) -> List[dict]:
    """Return the sub_ops, or raise with the reason this op cannot be unfused."""
    subs = op.get("sub_ops")
    if not isinstance(subs, list) or len(subs) < 2:
        raise UnfuseHintError(
            f"op {op_id} ({op.get('op')!r}) is not a fused op: it has "
            f"{0 if not subs else len(subs)} sub_ops, need >= 2")

    # The chain must actually compose, or the "no renaming needed" property
    # below is false and downstream consumers would dangle.
    for i in range(len(subs) - 1):
        produced = (subs[i].get("outputs") or [None])[0]
        if produced not in (subs[i + 1].get("inputs") or []):
            raise UnfuseHintError(
                f"op {op_id}: sub_ops do not compose -- sub_ops[{i}] produces "
                f"{produced!r}, which sub_ops[{i+1}] does not consume "
                f"({subs[i+1].get('inputs')}). This sub_ops list does not "
                f"describe a chain and restoring it would dangle.")
    if (subs[-1].get("outputs") or []) != (op.get("outputs") or []):
        raise UnfuseHintError(
            f"op {op_id}: the last sub_op produces "
            f"{subs[-1].get('outputs')} but the fused op produces "
            f"{op.get('outputs')}. Restoring would silently rename the "
            f"output every downstream consumer reads.")

    for i, s in enumerate(subs):
        if not s.get("shape"):
            raise UnfuseHintError(
                f"op {op_id}: sub_ops[{i}] ({s.get('op')!r}) has no shape. "
                f"The codegen dereferences it per kind, so a shapeless "
                f"restored op is a KeyError in another file later.")
        if not s.get("quant"):
            raise UnfuseHintError(
                f"op {op_id}: sub_ops[{i}] ({s.get('op')!r}) has no quant "
                f"block; the restored dispatch would have no requantize.")
    return subs


def _internal_tensor(sub: dict, consumer: dict, fused_out_meta: dict,
                     op_id: Any) -> dict:
    """Metadata for a tensor that lived inside the fused kernel."""
    kind = sub.get("op", "")
    shape = (list(fused_out_meta.get("shape") or [])
             if kind in _SHAPE_PRESERVING else _shape_from_op(sub, op_id))
    # A conv sub_op carries output_multiplier/shift and no scale_out, so the
    # only place its output scale exists is the consumer's scale_in.
    scale = (consumer.get("quant") or {}).get("scale_in")
    if scale is None:
        scale = (sub.get("quant") or {}).get("scale_out")
    if scale is None:
        raise UnfuseHintError(
            f"op {op_id}: cannot recover the output scale for {kind!r}. It has "
            f"no `scale_out` and its consumer ({consumer.get('op')!r}) has no "
            f"`scale_in`. Defaulting to 1.0 would make everything reading this "
            f"tensor report raw int8 counts as physical values.")
    return {"shape": shape, "dtype": _dtype_for(kind),
            "quant": {"scale": scale, "zero_point": 0}}


def _shape_from_op(sub: dict, op_id: Any) -> List[int]:
    sh = sub.get("shape") or {}
    kind = sub.get("op", "")
    if kind.startswith("conv2d"):
        return [sh["N"], sh["OC"], sh["OH"], sh["OW"]]
    if kind.startswith("linear"):
        return [sh["M"], sh["N"]]
    raise UnfuseHintError(
        f"op {op_id}: {kind!r} is neither shape-preserving nor a kind whose "
        f"output shape this knows how to derive. Add it to _SHAPE_PRESERVING "
        f"or extend _shape_from_op.")


def apply_unfuse_hint(graph: dict, unfuse_ops: List[dict],
                      model: Optional[str] = None) -> dict:
    out = copy.deepcopy(graph)
    if not unfuse_ops:
        return out
    ops = out["ops"]
    by_id = {o["dispatch_id"]: o for o in ops if o.get("dispatch_id") is not None}

    targets: Dict[Any, List[dict]] = {}
    seen = set()
    for spec in unfuse_ops:                       # validate ALL before rewriting
        oid = spec.get("op")
        if oid not in by_id:
            raise UnfuseHintError(
                f"unfuse_ops references unknown dispatch_id {oid}")
        if oid in seen:
            raise UnfuseHintError(f"dispatch_id {oid} listed twice")
        seen.add(oid)
        targets[oid] = _validate(by_id[oid], oid)

    tensors = out.setdefault("tensors", {})
    new_ops: List[dict] = []
    id_remap: Dict[Any, List[int]] = {}
    next_id = 0

    for op in ops:
        did = op.get("dispatch_id")
        if did is None:                            # view op: no dispatch
            new_ops.append(op)
            continue
        if did not in targets:
            nid = next_id; next_id += 1
            op = dict(op, dispatch_id=nid)
            new_ops.append(op)
            id_remap[did] = [nid]
            continue

        subs = targets[did]
        fused_out = (op.get("outputs") or [None])[0]
        fused_meta = tensors.get(fused_out, {})
        produced_ids: List[int] = []
        for j, sub in enumerate(subs):
            r = copy.deepcopy(sub)
            # Stale metadata: sub_ops copied by apply_fusion_hint carry
            # PRE-rewrite dispatch_id/depends_on. Inheriting them reproduces
            # exactly the renumbering defect dispatch_lineage exists to catch.
            r.pop("depends_on", None)
            nid = next_id; next_id += 1
            r["dispatch_id"] = nid
            r.setdefault("hardware_target", op.get("hardware_target", "any"))
            r["unfused_from"] = {"op_id": did, "piece": j,
                                 "n_pieces": len(subs),
                                 "fused_op": op.get("op")}
            new_ops.append(r)
            produced_ids.append(nid)

            if j < len(subs) - 1:                  # an internal tensor
                name = (sub.get("outputs") or [None])[0]
                meta = _internal_tensor(sub, subs[j + 1], fused_meta, did)
                if name in tensors:
                    have = tensors[name].get("shape")
                    if have and list(have) != list(meta["shape"]):
                        raise UnfuseHintError(
                            f"op {did}: internal tensor {name!r} is already "
                            f"registered with shape {have}, but the sub_op "
                            f"says {meta['shape']}. Refusing to overwrite.")
                else:
                    meta["unfused_from"] = {"fused_op_id": did,
                                            "producer": sub.get("name")}
                    tensors[name] = meta
        id_remap[did] = produced_ids

    # depends_on from tensor producers, exactly as extract_graph does it. This
    # is correct for the restored chain AND for downstream consumers in one
    # pass, with no id translation to get wrong.
    producer_of: Dict[str, int] = {}
    for o in new_ops:
        if o.get("dispatch_id") is None:
            continue
        for t in (o.get("outputs") or []):
            producer_of[t] = o["dispatch_id"]
    for o in new_ops:
        if o.get("dispatch_id") is None:
            continue
        deps = {producer_of[t] for t in (o.get("inputs") or [])
                if t in producer_of and producer_of[t] != o["dispatch_id"]}
        o["depends_on"] = sorted(deps)

    out["ops"] = new_ops
    out["id_remap"] = {str(k): v for k, v in sorted(
        id_remap.items(), key=lambda kv: kv[0])}
    if isinstance(out.get("dispatches"), list):
        out["dispatches"] = [o["dispatch_id"] for o in new_ops
                             if o.get("dispatch_id") is not None]
    if model:
        out["name"] = model
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    hint = json.load(open(a.hint))
    if hint.get("contract") != HINT_CONTRACT:
        raise UnfuseHintError(
            f"hint contract is {hint.get('contract')!r}, expected "
            f"{HINT_CONTRACT!r}")
    entry = next((n for n in hint.get("networks", [])
                  if n.get("network") == a.model), None)
    graph = json.load(open(a.ir))
    if entry is None or not entry.get("unfuse_ops"):
        print(f"apply_unfuse_hint: no unfuse_ops for {a.model}; copying IR "
              f"through unchanged", file=sys.stderr)
        json.dump(graph, open(a.out, "w"), indent=1)
        return 0

    out = apply_unfuse_hint(graph, entry["unfuse_ops"], model=a.model)
    n_new = sum(1 for o in out["ops"] if o.get("unfused_from"))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"apply_unfuse_hint: wrote {a.out} "
          f"({n_new} restored ops from {len(entry['unfuse_ops'])} fused; "
          f"{len([o for o in out['ops'] if o.get('dispatch_id') is not None])} "
          f"total dispatches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
