"""Apply XPU-RT shard hints (modelblaster.shard_hints/v1) to an IR.

The fourth applier, and the only one that does NOT change the graph.

`apply_split_hint` cuts one dispatch into n and renumbers everything after it,
which is why it has to emit an `id_remap`. `apply_fusion_hint` collapses n into
one, same problem in the other direction. `apply_unfuse_hint` undoes a fusion.
All three rewrite the dispatch list.

A shard rewrites nothing. It records, on one op, that this dispatch is to be
COMPILED so its output channels run across n cores at once. Same dispatch
count, same ids, same edges, same dependencies -- one extra field:

    {"op": "conv2d_batchnorm2d_silu_s8", "dispatch_id": 3, ...,
     "shard_factor": 8}

**So there is no `id_remap`, and that is not an omission.** A consumer joining
a pre-rewrite profile against a post-rewrite graph needs no translation here,
because every dispatch id still means the same dispatch. What DOES change is
the cost of that dispatch, which is exactly what the reprofile is for.

WHY PER-OP AND NOT `MB_SHARD_FACTOR`
------------------------------------
`generate_skeleton.shard_factor()` reads one env var and applies it to every
shardable conv in the model. That was enough to answer "can this board shard
at all", but it cannot express what the advice actually says. The measured
per-dispatch scaling varies 4.8x WITHIN one model -- 4.02x on a wide-OC conv
down to 0.83x on a 1x1 -- so a single model-wide width is wrong for almost
every dispatch in it. `shard_advice` names dispatches individually; this
writes them down individually.

`MB_SHARD_FACTOR` still works, as the default for ops that carry no
`shard_factor` of their own. Nothing that ran before this file existed
behaves differently.

WHAT THIS REFUSES
-----------------
Everything `generate_skeleton.shard_conv_weights` would silently skip, because
a skip there is invisible: the build succeeds, the binary runs, the answer is
correct, and it is simply not sharded. In particular OC must be divisible by
the shard count, and a dispatch that is already a split tile (`split_from`) is
not eligible -- the planner passes over it.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

SHARD_CONTRACT = "modelblaster.shard_hints/v1"

#: Ops with a sharded emitter. `conv2d_batchnorm2d_s8` is here because its
#: shard wrapper exists; an op in this set with no wrapper produces an
#: undeclared-symbol build failure rather than a wrong answer, which is the
#: right failure but a slow one to diagnose.
SHARDABLE_OPS = {
    "conv2d_s8",
    "conv2d_batchnorm2d_s8",
    "conv2d_batchnorm2d_silu_s8",
    "conv2d_silu_s8",
    "linear_s8",
    "matmul_s8",
}


def _oc_of(op: dict[str, Any]) -> int:
    sh = op.get("shape") or {}
    for key in ("OC", "N"):
        if key in sh:
            try:
                return int(sh[key])
            except (TypeError, ValueError):
                return 0
    return 0


def apply_shard_hint(ir: dict[str, Any],
                     shard_ops: list[dict[str, int]]) -> dict[str, Any]:
    """Annotate `ir`'s ops with per-dispatch `shard_factor`.

    Returns a NEW ir; the input is not mutated. Refuses loudly rather than
    dropping an entry, because a dropped shard is indistinguishable from a
    shard that ran.
    """
    out = copy.deepcopy(ir)
    ops = out.get("ops", [])
    by_id = {op.get("dispatch_id"): op for op in ops}

    applied: list[dict[str, Any]] = []
    for entry in shard_ops:
        did = entry.get("op")
        n = int(entry.get("n_shards", 1))
        if n <= 1:
            raise ValueError(
                f"shard hint for dispatch {did} asks for n_shards={n}; "
                f"1 is 'not sharded' and belongs in the hint's absence, "
                f"not in the hint")
        op = by_id.get(did)
        if op is None:
            raise ValueError(
                f"shard hint names dispatch {did}, which is not in this "
                f"graph (has {sorted(x for x in by_id if x is not None)}). "
                f"The hint is about a different graph.")
        kind = op.get("op")
        if kind not in SHARDABLE_OPS:
            raise ValueError(
                f"dispatch {did} is {kind}, which has no sharded emitter "
                f"(shardable: {sorted(SHARDABLE_OPS)})")
        if op.get("split_from"):
            raise ValueError(
                f"dispatch {did} is already a split tile. "
                f"shard_conv_weights skips anything carrying `split_from`, "
                f"so this hint would build cleanly and do nothing.")
        oc = _oc_of(op)
        if oc <= 0 or oc % n != 0:
            raise ValueError(
                f"dispatch {did} has OC={oc}, not divisible by {n}. "
                f"shard_conv_weights would skip it and the build would be "
                f"silently unsharded -- correct answer, no speedup, no error.")
        op["shard_factor"] = n
        applied.append({"dispatch_id": did, "op": kind,
                        "oc": oc, "n_shards": n})

    meta = out.setdefault("_rewrite", {})
    meta.setdefault("shard", []).extend(applied)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ir", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    hint = json.loads(Path(a.hint).read_text())
    got = hint.get("contract")
    if got != SHARD_CONTRACT:
        print(f"ERROR: {a.hint} is contract {got!r}, expected "
              f"{SHARD_CONTRACT!r}. A split hint applied here would set a "
              f"shard factor from an n_splits field and quietly do the other "
              f"thing.", file=sys.stderr)
        return 2

    nets = {n["network"]: n for n in hint.get("networks", [])}
    if a.network not in nets:
        print(f"ERROR: hint has no entry for network {a.network!r} "
              f"(has {sorted(nets)})", file=sys.stderr)
        return 2

    ir = json.loads(Path(a.ir).read_text())
    try:
        out = apply_shard_hint(ir, nets[a.network].get("shard_ops", []))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")
    for rec in out.get("_rewrite", {}).get("shard", []):
        print(f"  dispatch {rec['dispatch_id']} {rec['op']}: OC={rec['oc']} "
              f"across {rec['n_shards']} shards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
