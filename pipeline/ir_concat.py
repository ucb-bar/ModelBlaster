#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Concatenate two ModelBlaster IR graphs into one, wiring A's output to a packed input of B.

    python3 -m modelblaster.pipeline.ir_concat \
        --a out/attnfuse_on --b out/b112_pro/ir --wire enc --a-prefix e \
        --name moonshine_e2e --out out/b114_e2e/ir --report out/b114_e2e/ir_concat_report.json
    python3 -m modelblaster.pipeline.ir_concat --selftest

WHAT THIS IS FOR, AND WHY IT IS A GRAPH REWRITE AND NOT A LINKER SCRIPT.  `samples/
modelblaster_pext` builds ONE MODEL_DIR: one `model.c`, one `kernels.c`, one `weights.c`.
Two generated models in one image would mean two `kernels.c`, and every curated engine kernel
`#include`s `fpga/pynq-z2/sw/roccmoon/mbxr_rt.h`, which defines the engine runtime's
file-scope singletons -- `mbxr_rt_stats`, `mbxr_rt_job`, `mbxr_rt_go`, the hart-1 worker.  A
second translation unit including it is a second job structure and a second worker for one
RoCC, not merely a duplicate symbol.  So "one image" means ONE GRAPH, and putting two models
in one image is this pass.

    B112 lowered the cross-attention prologue into the decoder graph for the analogous
    reason and stated it the same way: "a prepended region of the decoder graph, not its own
    model -- samples/modelblaster_pext takes one MODEL_DIR".  The saving there was that
    kx/vx stop crossing the packed-input boundary.  Here the saving is smaller (A's output
    is 47,520 B against B's 577,152 B of kx/vx) and it is NOT the reason: the reason is the
    singleton above.  Both are recorded in the report so neither is inferred.

THE WIRE.  `--wire <field>` names a PACKED INPUT of B that A's graph output replaces.  After
the rewrite that tensor is an INTERMEDIATE: nothing packs it, nothing quantises it, and there
is no requantisation between the two models at all.  That is checked rather than assumed --
G1 refuses unless A's output tensor and B's wire tensor agree on dtype, shape AND quant scale
to the bit.  A concatenation across a scale boundary is a different program.

WHAT IT REFUSES TO DO:
  * wire across a shape, dtype or scale difference (G1);
  * merge when any name would collide after prefixing -- tensors, op names or weight arrays
    (G2).  The two Moonshine halves share 128 tensor names and 84 weight names, so this is
    the common case and not a corner;
  * emit a graph that `ir_cse.verify` rejects (G5): every op's inputs produced earlier,
    dispatch ids contiguous from zero, `depends_on` in range.

WHAT IT CHANGES IN REGION B, AND IT IS EXACTLY TWO THINGS.  B's ops are copied verbatim except
(a) every `dispatch_id` and every `depends_on` entry is shifted by A's dispatch count, and
(b) an op that reads the wire tensor gains A's last dispatch as a dependency, because the
tensor now has a producer.  G6 checks that and nothing else moved.  Region A is copied
verbatim under the prefix, with its output tensor renamed to the wire name; G7 checks that.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from . import ir_cse


class ConcatError(Exception):
    """A refusal.  Raised rather than returning a silently mis-wired graph."""


def _load(d: str) -> dict:
    p = d if d.endswith(".json") else os.path.join(d, "graph.json")
    return json.load(open(p))


def _rename_op(op: dict, tmap: dict, opre: str) -> dict:
    o = copy.deepcopy(op)
    o["name"] = opre + o["name"]
    o["inputs"] = [tmap.get(t, t) for t in o["inputs"]]
    o["outputs"] = [tmap.get(t, t) for t in o["outputs"]]
    for k in ("weight", "bias"):
        if o.get(k):
            o[k] = tmap[o[k]]
    for k in ("fused_into", "fused_from"):
        if o.get(k):
            v = o[k]
            o[k] = [opre + x for x in v] if isinstance(v, list) else opre + v
    return o


def concat(ga: dict, gb: dict, wire: str, prefix: str = "e",
           name: str | None = None) -> tuple[dict, dict, dict]:
    """(merged graph, {old A tensor/weight name: new name}, report).  Pure; no I/O."""
    rep: dict = {"gates": {}, "failures": []}
    fail = rep["failures"]

    # ---- G1  the wire: A's output IS B's packed field, to the bit -------------------------
    aout = ga["output"]["tensors"]
    if len(aout) != 1:
        raise ConcatError("graph A has %d outputs; the wire is a single tensor" % len(aout))
    aout = aout[0]
    pk = {p["name"]: p for p in gb["input"].get("packed_inputs") or []}
    if wire not in pk:
        raise ConcatError("%r is not a packed input of B (%s)" % (wire, ", ".join(sorted(pk))))
    ta, tb = ga["tensors"][aout], gb["tensors"][wire]
    g1 = {"a_output": aout, "b_field": wire,
          "a": {k: ta.get(k) for k in ("shape", "dtype")},
          "b": {k: tb.get(k) for k in ("shape", "dtype")},
          "a_scale": (ta.get("quant") or {}).get("scale"),
          "b_scale": (tb.get("quant") or {}).get("scale"),
          "b_field_bytes": pk[wire]["size"]}
    rep["gates"]["G1_wire"] = g1
    if [int(v) for v in ta["shape"]][-2:] != [int(v) for v in tb["shape"]][-2:] \
            or ta.get("dtype") != tb.get("dtype"):
        raise ConcatError("the wire does not match: A's %r is %s %s, B's %r is %s %s"
                          % (aout, ta.get("dtype"), ta["shape"], wire,
                             tb.get("dtype"), tb["shape"]))
    if g1["a_scale"] != g1["b_scale"]:
        raise ConcatError(
            "the wire crosses a QUANT SCALE boundary: A's %r is %r and B's %r is %r.  "
            "Concatenating here would insert a silent requantisation that neither graph "
            "describes." % (aout, g1["a_scale"], wire, g1["b_scale"]))

    # ---- G2  the rename map, and the collision check ---------------------------------------
    # WEIGHT ARRAYS ARE NOT IN `tensors` -- they live only in weights.npz and in the ops'
    # `weight`/`bias` fields -- so the map is over their union or `_rename_op` would KeyError
    # on the first convolution.
    tpre, wpre = prefix + "_", prefix + "."
    aw = {o[k] for o in ga["ops"] for k in ("weight", "bias") if o.get(k)}
    bw = {o[k] for o in gb["ops"] for k in ("weight", "bias") if o.get(k)}
    tmap: dict[str, str] = {}
    for t in set(ga["tensors"]) | aw:
        tmap[t] = wpre + t if t in aw else tpre + t
    tmap[aout] = wire
    ain = ga["input"]["tensor"]
    if ain not in gb["tensors"] and ain not in bw:
        tmap[ain] = ain                       # free: keep the readable name
    clash = sorted(set(tmap.values()) & ((set(gb["tensors"]) | bw) - {wire}))
    opre = wpre
    onames = {opre + o["name"] for o in ga["ops"]}
    oclash = sorted(onames & {o["name"] for o in gb["ops"]})
    rep["gates"]["G2_names"] = {"prefix": prefix, "a_tensors": len(ga["tensors"]),
                               "a_weights": len(aw), "a_input_kept_as": tmap[ain],
                               "tensor_collisions": clash[:8],
                               "op_name_collisions": oclash[:8],
                               "raw_tensor_overlap":
                                   len(set(ga["tensors"]) & set(gb["tensors"])),
                               "raw_weight_overlap": len(aw & bw)}
    if clash or oclash:
        raise ConcatError("names collide after prefixing with %r: tensors %s, ops %s"
                          % (prefix, clash[:5], oclash[:5]))

    # ---- the merge -------------------------------------------------------------------------
    nda = len(ga["dispatches"])
    ops = [_rename_op(o, tmap, opre) for o in ga["ops"]]
    wired = 0
    last_a = nda - 1
    for o in gb["ops"]:
        o = copy.deepcopy(o)
        if o.get("dispatch_id") is not None:
            o["dispatch_id"] += nda
        o["depends_on"] = [d + nda for d in (o.get("depends_on") or [])]
        if wire in o["inputs"] and last_a not in o["depends_on"]:
            o["depends_on"] = sorted(o["depends_on"] + [last_a])
            wired += 1
        ops.append(o)

    tensors = {tmap[t]: copy.deepcopy(v) for t, v in ga["tensors"].items()}
    tensors.update(copy.deepcopy(gb["tensors"]))      # B's `wire` record wins: same bytes

    fields = []
    a_in_size = 1
    for d in ga["tensors"][ain]["shape"]:
        a_in_size *= int(d)
    fields.append({"name": tmap[ain], "offset": 0, "size": a_in_size,
                   "dtype": ga["tensors"][ain].get("dtype", "i8"), "byte_offset": 0})
    off = a_in_size
    for p in gb["input"]["packed_inputs"]:
        if p["name"] == wire:
            continue
        q = dict(p)
        q["offset"] = off
        q["byte_offset"] = off
        fields.append(q)
        off += p["size"]

    g = {"name": name or (ga["name"] + "_" + gb["name"]),
         "version": gb.get("version", 1), "quant": gb.get("quant", "int8"),
         "input": {"tensor": fields[0]["name"], "tensors": [f["name"] for f in fields],
                   "packed_inputs": fields,
                   "packed_dtype": gb["input"].get("packed_dtype", "i8"),
                   "packed_bytes": off},
         "output": copy.deepcopy(gb["output"]),
         "tensors": tensors, "ops": ops,
         "dispatches": list(range(nda + len(gb["dispatches"])))}

    # ---- G3  the arithmetic ----------------------------------------------------------------
    rep["gates"]["G3_counts"] = {
        "ops": {"a": len(ga["ops"]), "b": len(gb["ops"]), "merged": len(g["ops"])},
        "dispatches": {"a": nda, "b": len(gb["dispatches"]),
                       "merged": len(g["dispatches"])},
        "a_dispatch_range": [0, nda - 1],
        "b_dispatch_range": [nda, nda + len(gb["dispatches"]) - 1],
        "wired_ops_gained_a_dependency": wired}
    if len(g["ops"]) != len(ga["ops"]) + len(gb["ops"]):
        fail.append("G3: merged op count is not A + B")
    if len(g["dispatches"]) != nda + len(gb["dispatches"]):
        fail.append("G3: merged dispatch count is not A + B")
    if wired == 0:
        fail.append("G3: no op in B reads %r -- the wire is dead and A's work is discarded"
                    % wire)

    # ---- G4  the packed input ----------------------------------------------------------
    rep["gates"]["G4_packed"] = {
        "b_bytes": gb["input"]["packed_bytes"], "merged_bytes": off,
        "removed_field": {"name": wire, "size": pk[wire]["size"]},
        "added_field": {"name": fields[0]["name"], "size": a_in_size},
        "fields": {"b": len(gb["input"]["packed_inputs"]), "merged": len(fields)},
        "delta_bytes": off - gb["input"]["packed_bytes"]}
    if off != gb["input"]["packed_bytes"] - pk[wire]["size"] + a_in_size:
        fail.append("G4: packed_bytes arithmetic does not close")

    # ---- G5  the graph verifies -----------------------------------------------------------
    try:
        ir_cse.verify(g)
        rep["gates"]["G5_verify"] = "PASS"
    except ir_cse.CseError as e:                                  # pragma: no cover - refusal
        rep["gates"]["G5_verify"] = "FAIL: %s" % e
        fail.append("G5: %s" % e)

    # ---- G6/G7  the two regions are their sources, and nothing else moved -------------------
    def strip(o, shift):
        o = copy.deepcopy(o)
        if o.get("dispatch_id") is not None:
            o["dispatch_id"] -= shift
        o["depends_on"] = sorted(d - shift for d in (o.get("depends_on") or [])
                                 if d - shift >= 0)
        return o
    bdiff = []
    for i, o in enumerate(gb["ops"]):
        want = copy.deepcopy(o)
        want["depends_on"] = sorted(want.get("depends_on") or [])
        got = strip(g["ops"][len(ga["ops"]) + i], nda)
        if got != want:
            bdiff.append(o["name"])
    adiff = []
    for i, o in enumerate(ga["ops"]):
        if g["ops"][i] != _rename_op(o, tmap, opre):
            adiff.append(o["name"])
    rep["gates"]["G6_region_b_is_b"] = {"ops": len(gb["ops"]), "differing": bdiff[:8],
                                        "n_differing": len(bdiff)}
    rep["gates"]["G7_region_a_is_a"] = {"ops": len(ga["ops"]), "differing": adiff[:8],
                                        "n_differing": len(adiff)}
    if bdiff:
        fail.append("G6: %d of B's ops are not B's after unshifting: %s"
                    % (len(bdiff), bdiff[:5]))
    if adiff:
        fail.append("G7: %d of A's ops are not A's under the prefix: %s"
                    % (len(adiff), adiff[:5]))

    rep["verdict"] = "PASS" if not fail else "FAIL"
    return g, tmap, rep


# ------------------------------------------------------------------------------------------
# selftest: two toy graphs with a known answer, and the three refusals
# ------------------------------------------------------------------------------------------
def _toy_a():
    return {"name": "a", "version": 1, "quant": "int8",
            "input": {"tensor": "x", "tensors": ["x"]},
            "output": {"tensors": ["y"], "tensor": "y"},
            "tensors": {"x": {"shape": [1, 4], "dtype": "i8", "quant": {"scale": 0.5}},
                        "mid": {"shape": [1, 4], "dtype": "i8", "quant": {"scale": 0.5}},
                        "y": {"shape": [1, 4], "dtype": "i8", "quant": {"scale": 0.25}}},
            "ops": [{"name": "r", "op": "relu_s8", "inputs": ["x"], "outputs": ["mid"],
                     "dispatch_id": 0, "depends_on": []},
                    {"name": "l", "op": "linear_s8", "inputs": ["mid"], "outputs": ["y"],
                     "weight": "w", "dispatch_id": 1, "depends_on": [0]}],
            "dispatches": [0, 1]}


def _toy_b(scale=0.25):
    return {"name": "b", "version": 1, "quant": "int8",
            "input": {"tensor": "in", "tensors": ["y", "h0"],
                      "packed_inputs": [
                          {"name": "y", "offset": 0, "size": 4, "dtype": "i8",
                           "byte_offset": 0},
                          {"name": "h0", "offset": 4, "size": 2, "dtype": "i8",
                           "byte_offset": 4}],
                      "packed_dtype": "i8", "packed_bytes": 6},
            "output": {"tensors": ["z"], "tensor": "z"},
            "tensors": {"y": {"shape": [1, 4], "dtype": "i8", "quant": {"scale": scale}},
                        "h0": {"shape": [1, 2], "dtype": "i8", "quant": {"scale": 1.0}},
                        "w2": {"shape": [4, 4], "dtype": "i8"},
                        "z": {"shape": [1, 4], "dtype": "i8", "quant": {"scale": 1.0}}},
            "ops": [{"name": "m", "op": "linear_s8", "inputs": ["y"], "outputs": ["z"],
                     "weight": "w2", "dispatch_id": 0, "depends_on": []}],
            "dispatches": [0]}


def selftest() -> int:
    bad = 0
    g, tmap, rep = concat(_toy_a(), _toy_b(), "y", prefix="e", name="ab")
    if rep["verdict"] != "PASS":
        print("    selftest: FAIL -- %s" % rep["failures"]); bad += 1
    if len(g["ops"]) != 3 or g["dispatches"] != [0, 1, 2]:
        print("    selftest: FAIL -- ops/dispatches %s %s" % (len(g["ops"]), g["dispatches"]))
        bad += 1
    if g["input"]["packed_bytes"] != 6:      # 4 (x) + 2 (h0); y leaves, x arrives, same size
        print("    selftest: FAIL -- packed_bytes %d" % g["input"]["packed_bytes"]); bad += 1
    if [f["name"] for f in g["input"]["packed_inputs"]] != ["x", "h0"]:
        print("    selftest: FAIL -- fields %s" % g["input"]["packed_inputs"]); bad += 1
    if g["ops"][2]["depends_on"] != [1]:
        print("    selftest: FAIL -- the wired op did not gain A's dispatch"); bad += 1
    if g["ops"][1]["outputs"] != ["y"] or g["ops"][0]["name"] != "e.r":
        print("    selftest: FAIL -- A's op was not renamed/rewired"); bad += 1

    for what, fn in (
            ("a scale mismatch", lambda: concat(_toy_a(), _toy_b(0.5), "y")),
            ("an unknown field", lambda: concat(_toy_a(), _toy_b(), "nope")),
    ):
        try:
            fn()
        except ConcatError:
            pass
        else:
            print("    selftest: FAIL -- %s was accepted" % what); bad += 1

    # a name collision: B already has a tensor called e_mid (A's `mid` under the prefix)
    b = _toy_b()
    b["tensors"]["e_mid"] = {"shape": [1, 1], "dtype": "i8"}
    try:
        concat(_toy_a(), b, "y")
    except ConcatError:
        pass
    else:
        print("    selftest: FAIL -- a tensor-name collision was accepted"); bad += 1

    print("    selftest: %s" % ("PASS" if not bad else "FAIL (%d)" % bad))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", help="graph A: the PREFIX region (a directory or a graph.json)")
    ap.add_argument("--b", help="graph B: the region A feeds")
    ap.add_argument("--wire", default="enc", help="the packed input of B that A's output is")
    ap.add_argument("--a-prefix", default="e")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", help="output directory (graph.json + weights.npz + io.npz)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for k in ("a", "b", "out"):
        if not getattr(a, k):
            ap.error("--%s is required" % k)

    ga, gb = _load(a.a), _load(a.b)
    g, tmap, rep = concat(ga, gb, a.wire, a.a_prefix, a.name)

    os.makedirs(a.out, exist_ok=True)
    json.dump(g, open(os.path.join(a.out, "graph.json"), "w"), indent=1)

    # ---- the weights, under the same map ---------------------------------------------------
    import numpy as np
    wa = np.load(os.path.join(os.path.dirname(_p(a.a)), "weights.npz"))
    wb = np.load(os.path.join(os.path.dirname(_p(a.b)), "weights.npz"))
    out = {tmap[k]: wa[k] for k in wa.files}
    dup = sorted(set(out) & set(wb.files))
    if dup:
        raise ConcatError("weight arrays collide after prefixing: %s" % dup[:5])
    out.update({k: wb[k] for k in wb.files})
    np.savez(os.path.join(a.out, "weights.npz"), **out)
    rep["gates"]["G8_weights"] = {"a": len(wa.files), "b": len(wb.files), "merged": len(out)}
    if len(out) != len(wa.files) + len(wb.files):
        rep["failures"].append("G8: merged weight count is not A + B")
        rep["verdict"] = "FAIL"

    rep["a"] = os.path.abspath(a.a)
    rep["b"] = os.path.abspath(a.b)
    rep["out"] = os.path.abspath(a.out)
    rep["wire"] = a.wire
    p = a.report or os.path.join(a.out, "ir_concat_report.json")
    json.dump(rep, open(p, "w"), indent=1)
    print("concat: %d + %d = %d ops, %d + %d = %d dispatches, packed %d -> %d B"
          % (len(ga["ops"]), len(gb["ops"]), len(g["ops"]),
             len(ga["dispatches"]), len(gb["dispatches"]), len(g["dispatches"]),
             gb["input"]["packed_bytes"], g["input"]["packed_bytes"]))
    for f in rep["failures"]:
        print("  FAILURE:", f)
    print(" ", rep["verdict"], "->", os.path.join(a.out, "graph.json"), "and", p)
    return 0 if rep["verdict"] == "PASS" else 1


def _p(d):
    return d if d.endswith(".json") else os.path.join(d, "graph.json")


if __name__ == "__main__":
    sys.exit(main())
