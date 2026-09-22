#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Common-subexpression elimination over a ModelBlaster IR graph.

    python3 -m modelblaster.pipeline.ir_cse --ir IN/graph.json --out OUT/graph.json
    python3 -m modelblaster.pipeline.ir_cse --selftest

WHAT THIS IS FOR.  `out/decint8/ir/graph.json` is the Moonshine int8 decoder unrolled over
24 decode steps.  The unroller replicated loop-INVARIANT work: each of the twelve
cross-attention inputs `kx0..kx5`/`vx0..vx5` is `permute4_s8`-d once per step, 24 times, to
the same 47,520-element result, because the encoder's K and V do not change while the decoder
decodes.  288 permutes compute twelve distinct values.  On board 0x5A5A0028 those 288
dispatches cost 89,109,728 of the decoder's 660,092,670 steady cycles; twelve of them cost
3,718,260.  This pass finds that class of redundancy structurally rather than by pattern, and
deletes it.

THE EQUIVALENCE RELATION, stated so it can be argued with.  Two ops are the same VALUE when a
recursive structural hash agrees:

    hash(op)     = H( kind | shape | quant | weight name | bias name | arity
                      | hash(input_0) | hash(input_1) | ... )
    hash(tensor) = "IN:<name>"         if it is a graph input  (an opaque, distinct value)
                 | (hash(producer), output index)              otherwise

Weights and graph inputs hash to their NAMES, not their contents: two ops reading different
weight tensors are different even if the arrays happen to be equal, which is the conservative
direction.  Every op kind in this IR is a pure function of its inputs and its attributes --
there is no in-place op, no RNG, no counter -- so equal hashes mean equal values and
substituting one for the other is exact.  Not approximately exact: bit-identical.  That is the
correctness bar the caller must hold this pass to, and `--out` plus a host-C golden diff is how.

WHAT IT REFUSES TO DO, rather than doing quietly:
  * it never deletes an op that produces a GRAPH OUTPUT tensor, even a provably redundant one,
    because the output contract names tensors and renaming them is a different change;
  * it dies if a tensor has two producers (the hash would be meaningless);
  * it dies if the rewritten graph does not re-verify -- every kept op's inputs produced
    earlier or a graph input, dispatch ids contiguous from zero, `depends_on` consistent.

`--kinds` restricts which op kinds may be collapsed (default: all).  `--dry-run` reports what
it would do and writes nothing.  The report JSON is the audit trail: per-kind counts, the
groups collapsed, and the tensor renames.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys


class CseError(Exception):
    """A refusal.  Raised rather than returning a silently degraded graph."""


# --------------------------------------------------------------------------------------------
# FUSED-OP SPECIFICATION RECORDS (patches/0112, MB_INT8_FUSE_ATTENTION=1)
# --------------------------------------------------------------------------------------------
# `_collapse_attention` rewrites each SDPA triple -- `x.qk`, `x.softmax`, `x.av` -- into one
# `x.attn` dispatch and DELIBERATELY KEEPS the three records, marked `fused_into`, as the
# SPECIFICATION of what the fused kernel must compute.  `_annotate_dispatches` gives them
# `dispatch_id = None` so nothing that walks dispatches sees them.
#
# This pass walked them anyway, and died: `.av` and `.attn` name the SAME output tensor, so the
# producer map saw two producers and refused.  The refusal was correct for its own premise and
# the premise was incomplete -- a record that is not a producer must not be read as one.
#
# THE MARKER IS `fused_into`, NOT `dispatch_id is None`.  `view` ops also carry
# `dispatch_id = None` and live ops DO read their outputs; excluding views from the producer
# map would make every view hash as `EXT:` and break the structural chain that finds the
# cross-attention 24x in the first place.  Only the absorbed records carry `fused_into`, and no
# live op reads their outputs -- `_collapse_attention` refuses a triple whose `scores`/`probs`
# have a second reader, precisely so that the fused unit need not materialise them.
#
# What "inert" means here, in the four places it has to mean it:
#   1. not a producer            -- the fused op owns the tensor
#   2. never collapsed           -- a specification is not a redundancy; dropping one would
#                                   delete the reference the fused kernel is checked against
#   3. not a re-definition       -- `verify` bars two LIVE producers of a tensor; a fused op
#                                   and the record it absorbed sharing one is the design
#   4. inputs STILL resolved     -- a collapsed producer must not leave a record naming a
#                                   tensor that no longer exists
def is_spec(op: dict) -> bool:
    """True for a record absorbed into a fused op: present as documentation, never run."""
    return bool(op.get("fused_into"))


# --------------------------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------------------------
def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def structural_hashes(ir: dict, producer: dict[str, int],
                      graph_inputs: set[str]) -> list[str]:
    """One hash per op, in `ir["ops"]` order.  See the module docstring for the relation."""
    ops = ir["ops"]
    memo: dict[int, str] = {}
    on_stack: set[int] = set()

    def h_tensor(t: str) -> str:
        if t in graph_inputs:
            return "IN:" + t
        if t not in producer:
            # A weight or an otherwise external tensor reaching an op's `inputs` list.
            # Hash it by name: conservative, and distinct from every computed value.
            return "EXT:" + t
        idx = producer[t]
        return "%s#%d" % (h_op(idx), ops[idx]["outputs"].index(t))

    def h_op(i: int) -> str:
        if i in memo:
            return memo[i]
        if i in on_stack:
            raise CseError("cycle in the IR through op %d (%s)" % (i, ops[i].get("name")))
        on_stack.add(i)
        o = ops[i]
        parts = [
            "k=" + o["op"],
            "s=" + _canon(o.get("shape")),
            "q=" + _canon(o.get("quant")),
            "w=" + _canon(o.get("weight")),
            "b=" + _canon(o.get("bias")),
            "n=%d" % len(o["outputs"]),
        ]
        parts += ["i=" + h_tensor(t) for t in o["inputs"]]
        hv = hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
        on_stack.discard(i)
        memo[i] = hv
        return hv

    return [h_op(i) for i in range(len(ops))]


def cse(ir: dict, kinds: set[str] | None = None) -> tuple[dict, dict]:
    """Return (rewritten IR, report).  `ir` is not mutated."""
    ops = ir["ops"]
    graph_inputs = set(ir.get("input", {}).get("tensors") or [])
    graph_outputs = set(ir.get("output", {}).get("tensors") or [])
    if ir.get("output", {}).get("tensor"):
        graph_outputs.add(ir["output"]["tensor"])

    producer: dict[str, int] = {}
    for i, o in enumerate(ops):
        if is_spec(o):
            continue            # absorbed into a fused op: documentation, not a producer
        for t in o["outputs"]:
            if t in producer:
                raise CseError("tensor %r has two producers: op %d (%s) and op %d (%s)"
                               % (t, producer[t], ops[producer[t]].get("name"),
                                  i, o.get("name")))
            producer[t] = i

    hashes = structural_hashes(ir, producer, graph_inputs)

    # Group by hash, keeping the FIRST op in IR order as the survivor.  IR order is
    # topological (extract_graph emits it that way and the verify below re-checks), so the
    # survivor is guaranteed to be computed before any consumer of a duplicate.
    groups: dict[str, list[int]] = {}
    for i, h in enumerate(hashes):
        if is_spec(ops[i]):
            continue            # never dropped, and never the survivor another op folds into
        groups.setdefault(h, []).append(i)

    rename: dict[str, str] = {}      # duplicate output tensor -> survivor's output tensor
    drop: set[int] = set()
    collapsed: list[dict] = []
    refused_output: list[str] = []
    for h, members in groups.items():
        if len(members) < 2:
            continue
        keep = members[0]
        kind = ops[keep]["op"]
        if kinds is not None and kind not in kinds:
            continue
        dups = []
        for j in members[1:]:
            if any(t in graph_outputs for t in ops[j]["outputs"]):
                # A redundant op whose result is named by the output contract.  Deleting it
                # would mean renaming a graph output; that is a contract change, not a CSE.
                refused_output.append(ops[j].get("name"))
                continue
            dups.append(j)
        if not dups:
            continue
        for j in dups:
            drop.add(j)
            for a, b in zip(ops[j]["outputs"], ops[keep]["outputs"]):
                rename[a] = b
        collapsed.append({"kind": kind, "hash": h, "keep": ops[keep].get("name"),
                          "keep_dispatch_id": ops[keep].get("dispatch_id"),
                          "dropped": len(dups),
                          "dropped_names": [ops[j].get("name") for j in dups]})

    # Renames can chain (a duplicate feeding a duplicate), so resolve to a fixed point.
    def resolve(t: str) -> str:
        seen = set()
        while t in rename:
            if t in seen:
                raise CseError("rename cycle through %r" % t)
            seen.add(t)
            t = rename[t]
        return t

    # ---- rewrite -------------------------------------------------------------------------
    out = copy.deepcopy(ir)
    new_ops = []
    did_remap: dict[int, int] = {}   # old dispatch id -> new dispatch id
    next_did = 0
    for i, o in enumerate(ops):
        if i in drop:
            continue
        n = copy.deepcopy(o)
        n["inputs"] = [resolve(t) for t in n["inputs"]]
        if n.get("dispatch_id") is not None:
            did_remap[n["dispatch_id"]] = next_did
            n["dispatch_id"] = next_did
            next_did += 1
        new_ops.append(n)
    # A dropped op's dispatch id must still resolve, for `depends_on` on a surviving consumer
    # that named it.  Point it at the survivor's NEW id.
    for j in drop:
        keep_t = resolve(ops[j]["outputs"][0])
        keep_i = producer[keep_t]
        old = ops[j].get("dispatch_id")
        keep_old = ops[keep_i].get("dispatch_id")
        if old is not None and keep_old is not None:
            did_remap[old] = did_remap[keep_old]

    for n in new_ops:
        deps = n.get("depends_on")
        if deps:
            seen, fixed = set(), []
            for d in deps:
                nd = did_remap.get(d)
                if nd is None or nd in seen:
                    continue
                seen.add(nd)
                fixed.append(nd)
            n["depends_on"] = fixed

    out["ops"] = new_ops
    out["dispatches"] = list(range(next_did))

    # Drop tensor records nothing refers to any more.  (The duplicates' outputs, and only
    # those: every other tensor is still named by some op, input or output.)
    live = set(graph_inputs) | set(graph_outputs)
    for n in new_ops:
        live.update(n["inputs"])
        live.update(n["outputs"])
    tensors = out.get("tensors") or {}
    pruned = [t for t in tensors if t not in live]
    for t in pruned:
        del tensors[t]

    report = {
        "pass": "ir_cse",
        "ops_before": len(ops), "ops_after": len(new_ops),
        "dispatches_before": sum(1 for o in ops if o.get("dispatch_id") is not None),
        "dispatches_after": next_did,
        "tensors_before": len(ir.get("tensors") or {}), "tensors_after": len(tensors),
        "tensors_pruned": len(pruned),
        "ops_removed_by_kind": _by_kind(ops, drop),
        "dispatches_removed_by_kind": _by_kind(
            ops, {j for j in drop if ops[j].get("dispatch_id") is not None}),
        "groups_collapsed": len(collapsed),
        "refused_graph_output": refused_output,
        "collapsed": collapsed,
        "renames": len(rename),
    }
    verify(out)
    return out, report


def _by_kind(ops: list[dict], idxs) -> dict[str, int]:
    d: dict[str, int] = {}
    for j in idxs:
        d[ops[j]["op"]] = d.get(ops[j]["op"], 0) + 1
    return dict(sorted(d.items()))


# --------------------------------------------------------------------------------------------
# the verifier -- run on every rewrite, not only under --selftest
# --------------------------------------------------------------------------------------------
def verify(ir: dict) -> None:
    ops = ir["ops"]
    graph_inputs = set(ir.get("input", {}).get("tensors") or [])
    tensors = set(ir.get("tensors") or {})
    seen: set[str] = set(graph_inputs)
    # The uniqueness bar belongs to LIVE producers only.  A fused op and the specification
    # record it absorbed name the same tensor BY DESIGN, so `.av` naming it does not make
    # `.attn` a re-definition -- while `seen` must still carry the records' outputs, because
    # `.softmax` legitimately reads `scores` from `.qk`.
    defined_live: set[str] = set(graph_inputs)
    weights: set[str] = set()
    for o in ops:
        for k in ("weight", "bias"):
            if o.get(k):
                weights.add(o[k])
    dids = []
    for i, o in enumerate(ops):
        spec = is_spec(o)
        for t in o["inputs"]:
            if t not in seen and t not in weights:
                raise CseError("op %d (%s) reads %r, which nothing before it produced"
                               % (i, o.get("name"), t))
        for t in o["outputs"]:
            if not spec:
                if t in defined_live:
                    raise CseError("op %d (%s) re-defines %r" % (i, o.get("name"), t))
                defined_live.add(t)
            seen.add(t)
            if tensors and t not in tensors:
                raise CseError("op %d (%s) produces %r with no `tensors` record"
                               % (i, o.get("name"), t))
        if o.get("dispatch_id") is not None:
            dids.append(o["dispatch_id"])
    if dids != list(range(len(dids))):
        raise CseError("dispatch ids are not 0..N-1 in IR order (first %s, last %s, n=%d)"
                       % (dids[:3], dids[-3:], len(dids)))
    if ir.get("dispatches") != list(range(len(dids))):
        raise CseError("`dispatches` does not match the %d dispatched ops" % len(dids))
    n = len(dids)
    for i, o in enumerate(ops):
        for d in (o.get("depends_on") or []):
            if not (0 <= d < n):
                raise CseError("op %d (%s) depends_on %r, out of range 0..%d"
                               % (i, o.get("name"), d, n - 1))
    for t in (ir.get("output", {}).get("tensors") or []):
        if t not in seen:
            raise CseError("graph output %r is produced by nothing" % t)


# --------------------------------------------------------------------------------------------
# selftest: a synthetic graph with a known answer, and the refusals
# --------------------------------------------------------------------------------------------
def _op(name, kind, ins, outs, did, shape=None, quant=None, weight=None, deps=None):
    return {"name": name, "op": kind, "inputs": list(ins), "outputs": list(outs),
            "shape": shape or {"n": 4}, "quant": quant, "weight": weight,
            "dispatch_id": did, "hardware_target": "any", "depends_on": list(deps or [])}


def _toy() -> dict:
    """x -> two identical permutes -> two adds; plus one permute that differs by quant."""
    ops = [
        _op("p_a", "permute4_s8", ["x"], ["pa"], 0, quant={"s": 1.0}),
        _op("p_b", "permute4_s8", ["x"], ["pb"], 1, quant={"s": 1.0}),      # dup of p_a
        _op("p_c", "permute4_s8", ["x"], ["pc"], 2, quant={"s": 2.0}),      # NOT a dup
        _op("a1", "add_s8", ["pa", "h"], ["y0"], 3, deps=[0]),
        _op("a2", "add_s8", ["pb", "h"], ["y1"], 4, deps=[1]),              # dup of a1? no:
        _op("a3", "add_s8", ["pc", "h"], ["y2"], 5, deps=[2]),
    ]
    tensors = {t: {"shape": [4], "dtype": "i8"}
               for t in ("x", "h", "pa", "pb", "pc", "y0", "y1", "y2")}
    return {"name": "toy", "version": 1, "quant": "int8",
            "input": {"tensors": ["x", "h"]},
            "output": {"tensors": ["y0", "y1", "y2"], "tensor": None},
            "tensors": tensors, "ops": ops, "dispatches": list(range(6))}


def _fused_toy() -> dict:
    """Two SDPA triples collapsed by `_collapse_attention`, sharing a duplicated K permute.

    This is Moonshine's decoder shape in miniature: the K tensor is loop-invariant and gets
    permuted once per step, so `p_ka` and `p_kb` are the redundancy CSE exists to find -- and
    the thing that must end up repointed is the LIVE `attention_s8` op's input, not only the
    specification record's.  `x.av` and `x.attn` name the same output tensor, which is the
    exact shape that used to make the producer map refuse.
    """
    def triple(base, q, k, sc, pr, out, did):
        return [
            dict(_op(base + ".qk", "matmul_b_s8", [q, k], [sc], None), fused_into=base + ".attn"),
            dict(_op(base + ".softmax", "softmax_s8", [sc], [pr], None), fused_into=base + ".attn"),
            dict(_op(base + ".av", "matmul_b_s8", [pr, "v"], [out], None), fused_into=base + ".attn"),
            _op(base + ".attn", "attention_s8", [q, k, "v"], [out], did),
        ]
    ops = [_op("p_ka", "permute4_s8", ["kx"], ["ka"], 0, quant={"s": 1.0})]
    ops += triple("s0", "q0", "ka", "sc0", "pr0", "o0", 1)
    ops += [_op("p_kb", "permute4_s8", ["kx"], ["kb"], 2, quant={"s": 1.0})]   # the duplicate
    ops += triple("s1", "q1", "kb", "sc1", "pr1", "o1", 3)
    tensors = {t: {"shape": [4], "dtype": "i8"}
               for t in ("kx", "v", "q0", "q1", "ka", "kb",
                         "sc0", "pr0", "o0", "sc1", "pr1", "o1")}
    return {"name": "fused_toy", "version": 1, "quant": "int8",
            "input": {"tensors": ["kx", "v", "q0", "q1"]},
            "output": {"tensors": ["o0", "o1"], "tensor": None},
            "tensors": tensors, "ops": ops, "dispatches": list(range(4))}


def selftest() -> int:
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print("    %-58s %s%s" % (label, "ok" if cond else "FAIL",
                                  ("  -- " + detail) if (detail and not cond) else ""))
        ok = ok and bool(cond)

    g = _toy()
    out, rep = cse(g)
    # p_b collapses into p_a.  a1 and a2 then become structurally identical -- but y1 is a
    # GRAPH OUTPUT, so a2 must be REFUSED, not deleted.
    check("collapses the duplicate permute", rep["dispatches_removed_by_kind"] ==
          {"permute4_s8": 1}, repr(rep["dispatches_removed_by_kind"]))
    check("refuses to delete an op that produces a graph output",
          rep["refused_graph_output"] == ["a2"], repr(rep["refused_graph_output"]))
    check("dispatch count 6 -> 5", rep["dispatches_after"] == 5, repr(rep["dispatches_after"]))
    check("a2 now reads the survivor's tensor",
          [o for o in out["ops"] if o["name"] == "a2"][0]["inputs"] == ["pa", "h"])
    check("the non-duplicate permute survives",
          any(o["name"] == "p_c" for o in out["ops"]))
    check("dead tensor record pruned", "pb" not in out["tensors"])
    check("dispatch ids renumbered contiguously",
          [o["dispatch_id"] for o in out["ops"]] == [0, 1, 2, 3, 4])
    check("depends_on of a2 repointed at the survivor",
          [o for o in out["ops"] if o["name"] == "a2"][0]["depends_on"] == [0],
          repr([o for o in out["ops"] if o["name"] == "a2"][0]["depends_on"]))
    check("the input graph was not mutated", g["ops"][1]["name"] == "p_b"
          and len(g["ops"]) == 6)

    # A graph with no redundancy must come out byte-identical, so an A/B against a
    # CSE'd build of an already-minimal graph is a no-op rather than a reshuffle.
    clean = _toy()
    del clean["ops"][1]          # remove the duplicate
    clean["ops"][3] = _op("a2", "add_s8", ["pa", "h"], ["y1"], 4, deps=[0])
    for k, o in enumerate(clean["ops"]):
        o["dispatch_id"] = k
    clean["dispatches"] = list(range(len(clean["ops"])))
    del clean["tensors"]["pb"]
    c2, rep2 = cse(clean)
    check("a graph with nothing to collapse is unchanged",
          json.dumps(c2, sort_keys=True) == json.dumps(clean, sort_keys=True))
    check("...and says so", rep2["groups_collapsed"] == 0 and rep2["dispatches_removed_by_kind"] == {})

    # --kinds gates the rewrite
    _, rep3 = cse(_toy(), kinds={"add_s8"})
    check("--kinds excludes the permute", rep3["dispatches_removed_by_kind"] == {},
          repr(rep3["dispatches_removed_by_kind"]))

    # two producers for one tensor is a refusal, not a silent pick
    bad = _toy()
    bad["ops"][1]["outputs"] = ["pa"]
    try:
        cse(bad)
        check("two producers for one tensor is refused", False, "no CseError raised")
    except CseError as e:
        check("two producers for one tensor is refused", "two producers" in str(e))

    # the verifier actually rejects a broken graph (a guard that cannot fire is not a guard)
    broken = _toy()
    broken["ops"][3]["inputs"] = ["never_produced", "h"]
    try:
        verify(broken)
        check("verify() rejects a dangling input", False, "verify() passed a broken graph")
    except CseError as e:
        check("verify() rejects a dangling input", "nothing before it produced" in str(e))

    broken2 = _toy()
    broken2["ops"][4]["dispatch_id"] = 9
    try:
        verify(broken2)
        check("verify() rejects non-contiguous dispatch ids", False, "verify() passed it")
    except CseError as e:
        check("verify() rejects non-contiguous dispatch ids", "not 0..N-1" in str(e))

    # ---- the fused-attention shape: the two passes must COMPOSE ---------------------------
    # Not "ir_cse stops throwing": the collapse the unfused graph finds must still be found,
    # and the LIVE fused op -- not just the specification record -- must be the thing whose
    # input gets repointed at the survivor.
    f = _fused_toy()
    fo, frep = cse(f)
    names = [o["name"] for o in fo["ops"]]
    attn = [o for o in fo["ops"] if o["op"] == "attention_s8"]
    spec = [o for o in fo["ops"] if is_spec(o)]
    check("fused graph: ir_cse does not refuse", True)
    check("the duplicate producer is still collapsed",
          frep["ops_removed_by_kind"] == {"permute4_s8": 1}, repr(frep["ops_removed_by_kind"]))
    check("all 6 specification records survive", len(spec) == 6, "%d" % len(spec))
    check("specification records stay inert",
          all(o.get("dispatch_id") is None and o.get("fused_into") for o in spec))
    check("BOTH fused ops now read the survivor's tensor",
          all(o["inputs"][1] == "ka" for o in attn),
          repr([o["inputs"] for o in attn]))
    check("a specification record's input was repointed too",
          all(o["inputs"][1] == "ka" for o in fo["ops"] if o["name"].endswith(".qk")),
          repr([o["inputs"] for o in fo["ops"] if o["name"].endswith(".qk")]))
    # 1 and 2, not 1 and 3: dropping `p_kb` renumbers, which is the contract verify() enforces.
    check("the fused ops keep their dispatches, renumbered, and the triple gets none",
          [o["dispatch_id"] for o in attn] == [1, 2]
          and all(o.get("dispatch_id") is None for o in spec),
          repr([o["dispatch_id"] for o in attn]))
    check("`x.av` and `x.attn` sharing a tensor is not a re-definition",
          frep["ops_after"] == len(fo["ops"]))
    # and the refusal it replaced must still fire for a REAL duplicate producer
    fbad = _fused_toy()
    fbad["ops"][-1] = copy.deepcopy(fbad["ops"][-1])
    fbad["ops"][-1]["fused_into"] = None          # a live op re-producing a live tensor
    fbad["ops"][-1]["outputs"] = ["o0"]
    try:
        cse(fbad)
        check("a genuine duplicate producer is still refused", False, "no CseError")
    except CseError as e:
        check("a genuine duplicate producer is still refused", "two producers" in str(e))

    print("    selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir", help="input graph.json")
    ap.add_argument("--out", help="output graph.json")
    ap.add_argument("--report", help="write the audit JSON here (default: <out>.cse.json)")
    ap.add_argument("--kinds", help="comma-separated op kinds eligible for collapsing "
                                    "(default: all)")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.ir:
        ap.error("--ir is required (or --selftest)")
    ir = json.load(open(a.ir))
    kinds = set(a.kinds.split(",")) if a.kinds else None
    out, rep = cse(ir, kinds)
    rep["ir"] = os.path.abspath(a.ir)
    rep["kinds"] = sorted(kinds) if kinds else "all"

    print("ir_cse: %d ops -> %d, %d dispatches -> %d, %d groups collapsed"
          % (rep["ops_before"], rep["ops_after"], rep["dispatches_before"],
             rep["dispatches_after"], rep["groups_collapsed"]))
    for k, v in rep["ops_removed_by_kind"].items():
        d = rep["dispatches_removed_by_kind"].get(k, 0)
        print("    %-16s -%d ops (%d of them dispatches)" % (k, v, d))
    if rep["refused_graph_output"]:
        print("    REFUSED (produces a graph output): %s"
              % ", ".join(rep["refused_graph_output"][:8]))
    if a.dry_run:
        print("    --dry-run: nothing written")
        if a.report:
            json.dump(rep, open(a.report, "w"), indent=1)
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    rp = a.report or (a.out + ".cse.json")
    rep["out"] = os.path.abspath(a.out)
    json.dump(rep, open(rp, "w"), indent=1)
    print("    wrote %s and %s" % (a.out, rp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
