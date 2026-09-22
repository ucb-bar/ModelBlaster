#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Batch ONE REGION of the merged encoder+decoder graph: B utterances decoded at once.

    python3 -m modelblaster.pipeline.ir_regbatch --selftest
    python3 -m modelblaster.pipeline.ir_regbatch --ir in/graph.json --out out/graph.json \
        -B 8 --mode replicate --prefix e. --report r.json

WHY THIS PASS EXISTS.  `ir_batch.py` refuses the merged graph -- *"op 'e.stem.conv1' has kind
'conv2d_s8', which this pass has no rule for"* -- and B114 priced that refusal at **+0.3033 of
RTF_e2e**, the single largest term between the one-program 1.181612 and the goal.  ***The refusal
was protecting us.***  `roccmoon_conv2d_s8_roccmoon_engine.c:52` gates the accelerator on
`N == 1 && IH == 1 && KH == 1 && PH == 0`, so giving the encoder N = B would push conv2d OFF the
engine entirely -- B106 measured that failure mode on YOLOv8n, where the pick still read
`roccmoon_engine` while **0.00 %** of dispatches reached it.  Conv is 503 M of the encoder's
1,488 M MACs; hart 0 at 0.767 MAC/cyc costs roughly **+4 RTF**, not -0.30.

SO THIS PASS NEVER TOUCHES AN ENCODER OP'S SHAPE.  Not one.  In both modes every `conv2d_s8`
keeps `N = 1`, and the gate above is satisfied by construction rather than by a check after
the fact.

THE REGIONS ARE SEPARABLE TO ONE TENSOR, AND THAT IS MEASURED ON THE REAL GRAPH.  Of the 6,491
tensors, 154 are touched only by `e.` ops, 6,336 only by decoder ops, and ***exactly one
crosses***: `enc`, `[1,165,288]` i8, 47,520 B, produced by `e.layer_norm` and read by the twelve
prologue `linear_s8` ops at M = 165.  That one tensor is the whole boundary, which is what makes
a region-scoped rewrite possible at all.

TWO MODES, AND THEY DIFFER IN WHERE THE B ENCODER RUNS COME FROM.

  --mode replicate   B copies of the `e.` region in the GRAPH, each reading its own 64,000 B
                     slice of the `x` field, joined by view -> cat2_c1_s8 -> view into
                     `enc` [B,165,288].  ***A PURE IR PASS: no driver change, no codegen
                     change, `--batch-decode B` already does the rest.***  The join is exact
                     because `cat2_c1_s8` at N = H = W = 1 is a flat byte concatenation and the
                     B outputs share one quant scale by construction (same op, same weights).

  --mode once        ONE encoder in the graph and `enc` widened to [B,165,288].  ***NOT A PURE
                     PASS***: the AR driver walks `for (; d <= STEP_END[k]; d++)` from d = 0
                     exactly once per group, so something must run dispatches [0, N_ENC) B
                     times -- `st.input` moved to `ar_input + b*64,000` per pass and `enc`'s
                     buffer copied into slot b (descending b, so slot 0 is written in place
                     last).  This mode emits the graph that driver needs and nothing more.

*** THE CEILING IS NOT ram0, AND IT IS WHY THE TWO MODES ARE NOT INTERCHANGEABLE. ***
`mbxr_rt.h:129` puts `MBXR_RT_IMG_BASE` at **0x88000000**, so the guest image must end below
134,217,728 B from 0x80000000.  B114 measured the B = 1 merged image ending at `0x8518DFE8`:
***48,701,464 B clear.***  `generate_skeleton` gives every tensor its own file-scope buffer in
`buffers.c` and does NO liveness reuse, so B utterances in flight cost B times the activations.
Measured by running the codegen on this pass's own output:

    mode        B    buffers_bytes    delta over B = 1's 16,669,056     fits under 48,701,464 B
    replicate   2       33,433,152          +16,764,096                 yes
    replicate   4       67,103,904          +50,434,848                 ***NO, by 1.73 MB***
    replicate   8      135,015,648         +118,346,592                 ***NO, by 69.6 MB***
    once        4       34,145,760          +17,476,704                 yes
    once        8       57,448,032          +40,778,976                 yes, 7.9 MB to spare

***THE PURE-GRAPH ROUTE THEREFORE CAPS AT B = 2***, and the encoder's 10,843,488 B is spread
over 135 buffers of 47,520..287,712 B with no dominant one, so there is nothing to trim: only a
liveness-reusing allocator would change that arithmetic, and this tree does not have one.

THE PACKED INPUT IS LAID OUT TO MATCH THE BAKER, NOT THE OTHER WAY ROUND.  `scripts/74`'s B > 1
baker walks the ***unbatched*** field table and writes, per field, the group's B sequences
ADJACENT.  For the merged graph the fields are `x` (64,000) and `h0..h23` (288), so a group's
block is `[x_0 .. x_{B-1}][h0_0 .. h0_{B-1}]...` -- which is exactly `x0..x{B-1}` followed by
`h0..h23` at B rows each (replicate), and exactly `x` at B*64,000 followed by the same
(once).  ***Both modes reproduce that layout byte for byte, so the baker needs no change.***
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from . import ir_batch
from . import ir_cse

ENC_BYTES = 47520          # the wire: 165 x 288 int8


class RegBatchError(Exception):
    """A refusal.  Raised rather than returning a graph whose encoder left the engine."""


def _regions(ops, prefix):
    """(encoder ops, decoder ops, tensors owned by each, the crossing tensors)."""
    enc = [o for o in ops if o["name"].startswith(prefix)]
    dec = [o for o in ops if not o["name"].startswith(prefix)]
    et, dt = set(), set()
    for o in enc:
        et |= set(o["inputs"]) | set(o["outputs"])
    for o in dec:
        dt |= set(o["inputs"]) | set(o["outputs"])
    return enc, dec, et - dt, dt - et, et & dt


def _batch_decoder(g, B, is_enc):
    """ir_batch's rules, applied ONLY to ops the predicate calls decoder."""
    seen = set()
    for o in g["ops"]:
        if is_enc(o):
            continue
        kind = o.get("op") or "view"
        if kind not in ir_batch.ROW_KEY:
            raise RegBatchError(
                "decoder op %r has kind %r, which ir_batch has no rule for" % (o["name"], kind))
        key = ir_batch.ROW_KEY[kind]
        if key not in (o.get("shape") or {}):
            raise RegBatchError("decoder op %r (%s) has no %r in its shape"
                                % (o["name"], kind, key))
        o["shape"][key] *= B
        seen.add(kind)
        if kind == "rope_s8" and o["shape"].get("T") != 1:
            raise RegBatchError(
                "rope op %r has T=%s; the batch folds into H and that is only correct while "
                "every row shares one position" % (o["name"], o["shape"].get("T")))
    return seen


def _renumber(g):
    """Dispatch ids 0..N-1 in IR order, and depends_on rebuilt from the real producers."""
    d = 0
    for o in g["ops"]:
        if o.get("dispatch_id") is not None:
            o["dispatch_id"] = d
            d += 1
    prod = {t: o for o in g["ops"] for t in o["outputs"]}
    for o in g["ops"]:
        deps = []
        for t in o["inputs"]:
            p = prod.get(t)
            # a view is an alias with no dispatch of its own: walk through it to the op that
            # actually wrote the bytes, or the dependency would name nothing.
            while p is not None and p.get("dispatch_id") is None:
                p = prod.get(p["inputs"][0]) if p.get("inputs") else None
            if p is not None and p["dispatch_id"] not in deps:
                deps.append(p["dispatch_id"])
        o["depends_on"] = sorted(deps)
    g["dispatches"] = list(range(d))
    return d


def _repack(g, fields):
    off = 0
    for p in fields:
        p["offset"] = off
        p["byte_offset"] = off
        off += p["size"]
    g["input"]["packed_inputs"] = fields
    g["input"]["packed_bytes"] = off
    g["input"]["tensors"] = [p["name"] for p in fields]
    g["input"]["tensor"] = fields[0]["name"]
    return off


def regbatch(ir: dict, B: int, mode: str, prefix: str = "e.",
             wire: str = "enc") -> tuple[dict, dict]:
    if B < 1:
        raise RegBatchError("B must be >= 1")
    if mode not in ("replicate", "once"):
        raise RegBatchError("mode must be 'replicate' or 'once'")
    g = copy.deepcopy(ir)
    if B == 1:
        return g, {"B": 1, "mode": mode, "unchanged": True}

    enc_ops, dec_ops, enc_only, _dec_only, cross = _regions(g["ops"], prefix)
    if not enc_ops:
        raise RegBatchError("no op name starts with %r: this graph has no %s region"
                            % (prefix, prefix))
    if cross != {wire}:
        raise RegBatchError(
            "the regions touch at %s, not at the single wire %r -- a region-scoped batch is "
            "only defined while exactly one tensor crosses" % (sorted(cross) or "nothing", wire))
    inp = g["input"]["tensor"]
    old = {p["name"]: p for p in g["input"]["packed_inputs"]}
    if inp not in old:
        raise RegBatchError("the graph input %r is not a packed field" % inp)
    in_bytes = old[inp]["size"]
    scale = (g["tensors"][wire].get("quant") or {}).get("scale")
    rest = [p for p in g["input"]["packed_inputs"] if p["name"] != inp]

    if mode == "once":
        for t in _dec_only:
            g["tensors"][t]["shape"][0] *= B
        g["tensors"][wire]["shape"] = [B] + list(g["tensors"][wire]["shape"][1:])
        # The input field carries the group's B windows so the DECODER fields keep the offsets
        # the baker writes; the encoder op still reads only the first one.
        g["tensors"][inp]["shape"][0] *= B
        kinds = _batch_decoder(g, B, lambda o: o["name"].startswith(prefix))
        fields = [dict(old[inp], size=in_bytes * B)] + [dict(p, size=p["size"] * B) for p in rest]
        packed = _repack(g, fields)
        n = _renumber(g)
        ir_cse.verify(g)
        return g, {"B": B, "mode": mode, "kinds": sorted(kinds), "n_ops": len(g["ops"]),
                   "n_tensors": len(g["tensors"]), "packed_bytes": packed, "dispatches": n,
                   "encoder_dispatches": sum(1 for o in enc_ops
                                             if o.get("dispatch_id") is not None),
                   "wire_bytes": ENC_BYTES, "driver_required": True,
                   "driver_contract": "walk dispatches [0, encoder_dispatches) B times with "
                                      "st.input = ar_input + b*%d, copying the %r buffer's "
                                      "first %d B into slot b, b descending"
                                      % (in_bytes, wire, ENC_BYTES),
                   "replay_safe": False}

    # ---- replicate ---------------------------------------------------------------------
    keep = {t: v for t, v in g["tensors"].items()
            if t not in enc_only and t != wire and t != inp}
    ops, tensors = [], keep
    for r in range(B):
        tmap = {t: "e%d_%s" % (r, t) for t in enc_only}
        tmap[inp] = "%s%d" % (inp, r)
        tmap[wire] = "%s%d" % (wire, r)
        for t, nt in tmap.items():
            tensors[nt] = copy.deepcopy(g["tensors"][t])
        for o in enc_ops:
            o2 = copy.deepcopy(o)
            o2["name"] = "e%d.%s" % (r, o["name"][len(prefix):])
            o2["inputs"] = [tmap.get(t, t) for t in o["inputs"]]
            o2["outputs"] = [tmap.get(t, t) for t in o["outputs"]]
            for k in ("fused_into", "fused_from"):
                if o2.get(k):
                    v = o2[k]
                    o2[k] = ([("e%d.%s" % (r, x[len(prefix):])) for x in v]
                             if isinstance(v, list) else "e%d.%s" % (r, v[len(prefix):]))
            o2["_enc"] = True
            ops.append(o2)
    # THE JOIN.  view is a zero-cost alias (dispatch_id None), so the only real work is the
    # B-1 concatenations: 47,520*(B-1)*B/2 B of copy against a ~189 M-cycle utterance.
    q = {"scale": scale, "zero_point": 0}
    for r in range(B):
        tensors["%sf%d" % (wire, r)] = {"shape": [1, 1, ENC_BYTES], "dtype": "i8",
                                        "quant": dict(q)}
        ops.append({"name": "%sjoin.view%d" % (wire, r), "op": "view",
                    "inputs": ["%s%d" % (wire, r)], "outputs": ["%sf%d" % (wire, r)],
                    "shape": {"n": ENC_BYTES}, "quant": None, "dispatch_id": None,
                    "hardware_target": "any", "depends_on": [], "_enc": True})
    acc = "%sf0" % wire
    for r in range(1, B):
        out = "%sacc%d" % (wire, r)
        n = ENC_BYTES * (r + 1)
        tensors[out] = {"shape": [1, 1, n], "dtype": "i8", "quant": dict(q)}
        ops.append({"name": "%sjoin.cat%d" % (wire, r), "op": "cat2_c1_s8",
                    "inputs": [acc, "%sf%d" % (wire, r)], "outputs": [out],
                    "shape": {"N": 1, "H": 1, "W": 1,
                              "C_inputs": [ENC_BYTES * r, ENC_BYTES], "C_total": n},
                    "quant": {"scales_in": [scale, scale], "scale_out": scale,
                              "activation_min": -128, "activation_max": 127},
                    "dispatch_id": 0, "hardware_target": "any", "depends_on": [], "_enc": True})
        acc = out
    tensors[wire] = {"shape": [B] + list(g["tensors"][wire]["shape"][1:]), "dtype": "i8",
                     "quant": dict(q)}
    ops.append({"name": "%sjoin.reshape" % wire, "op": "view", "inputs": [acc],
                "outputs": [wire], "shape": {"n": ENC_BYTES * B}, "quant": None,
                "dispatch_id": None, "hardware_target": "any", "depends_on": [], "_enc": True})
    ops += [copy.deepcopy(o) for o in dec_ops]
    # EVERY DECODER TENSOR CARRIES B ROWS NOW.  Scaling the ops' row keys without scaling the
    # tensors leaves 6,336 buffers at one row: the graph still verifies, the codegen still
    # emits, and the image writes B rows into a 1-row buffer.  Caught by re-running the
    # codegen against this pass's own output -- the buffer COUNT was right and the byte total
    # was exactly one decoder region short.
    for t in _dec_only:
        tensors[t]["shape"][0] *= B
    g["ops"], g["tensors"] = ops, tensors
    fields = [dict(old[inp], name="%s%d" % (inp, r), size=in_bytes) for r in range(B)]
    fields += [dict(p, size=p["size"] * B) for p in rest]
    packed = _repack(g, fields)
    kinds = _batch_decoder(g, B, lambda o: o.get("_enc", False))
    for o in g["ops"]:
        o.pop("_enc", None)
    n = _renumber(g)
    ir_cse.verify(g)
    return g, {"B": B, "mode": mode, "kinds": sorted(kinds), "n_ops": len(g["ops"]),
               "n_tensors": len(g["tensors"]), "packed_bytes": packed, "dispatches": n,
               "encoder_replicas": B, "wire_bytes": ENC_BYTES, "driver_required": False,
               "replay_safe": False}


# --------------------------------------------------------------------------------------------
def _toy() -> dict:
    """A two-region toy with the same shape as the real graph: a conv stem whose N must never
    move, one crossing tensor, and a decoder whose every kind ir_batch has a rule for."""
    t = lambda *s: {"shape": list(s), "dtype": "i8", "quant": {"scale": 0.5, "zero_point": 0}}
    op = lambda n, k, i, o, sh, did, **kw: dict(
        {"name": n, "op": k, "inputs": list(i), "outputs": list(o), "shape": sh, "quant": {},
         "dispatch_id": did, "hardware_target": "any", "depends_on": []}, **kw)
    return {"name": "toy", "version": 1, "quant": "int8",
            "input": {"tensor": "x", "tensors": ["x", "h0"],
                      "packed_inputs": [{"name": "x", "offset": 0, "size": 800, "dtype": "i8",
                                         "byte_offset": 0},
                                        {"name": "h0", "offset": 800, "size": 288, "dtype": "i8",
                                         "byte_offset": 800}],
                      "packed_bytes": 1088},
            "output": {"tensors": ["y"], "tensor": None},
            "tensors": {"x": t(1, 1, 1, 800), "e_c": t(1, 4, 1, 10), "enc": t(1, 165, 288),
                        "h0": t(1, 1, 288), "p": t(1, 165, 288), "y": t(1, 1, 288)},
            "ops": [
                op("e.stem.conv1", "conv2d_s8", ["x"], ["e_c"],
                   {"N": 1, "IC": 1, "IH": 1, "IW": 800, "OC": 4, "OH": 1, "OW": 10,
                    "KH": 1, "KW": 127, "SH": 1, "SW": 64, "PH": 0, "PW": 0}, 0, weight="W0"),
                op("e.layer_norm", "layernorm_pc_s8", ["e_c"], ["enc"], {"M": 165, "K": 288}, 1),
                op("pk.0", "linear_s8", ["enc"], ["p"], {"M": 165, "K": 288, "N": 288}, 2,
                   weight="W1"),
                op("dec.0", "linear_s8", ["h0"], ["y"], {"M": 1, "K": 288, "N": 288}, 3,
                   weight="W2"),
            ],
            "dispatches": [0, 1, 2, 3]}


def selftest() -> int:
    bad = 0

    def check(label, cond, detail=""):
        nonlocal bad
        print("    %-64s %s%s" % (label, "PASS" if cond else "FAIL", "" if cond else "  " + detail))
        if not cond:
            bad += 1

    base = _toy()
    check("B=1 is a no-op", json.dumps(regbatch(base, 1, "replicate")[0], sort_keys=True)
          == json.dumps(base, sort_keys=True))

    for mode in ("replicate", "once"):
        out, rep = regbatch(base, 4, mode)
        ops = {o["name"]: o for o in out["ops"]}
        conv = [o for o in out["ops"] if o.get("op") == "conv2d_s8"]
        check("%s: *** every conv2d_s8 still has N=1 ***" % mode,
              conv and all(o["shape"]["N"] == 1 and o["shape"]["IH"] == 1
                           and o["shape"]["KH"] == 1 and o["shape"]["PH"] == 0 for o in conv),
              str([o["shape"] for o in conv][:1]))
        check("%s: the encoder's layernorm M is UNTOUCHED (165)" % mode,
              all(o["shape"]["M"] == 165 for o in out["ops"]
                  if o.get("op") == "layernorm_pc_s8"))
        check("%s: the prologue M went 165 -> 660" % mode,
              [o for o in out["ops"] if o["name"].endswith("pk.0")][0]["shape"]["M"] == 660)
        check("%s: the decoder M went 1 -> 4" % mode, ops["dec.0"]["shape"]["M"] == 4)
        check("%s: enc is [4,165,288]" % mode, out["tensors"]["enc"]["shape"] == [4, 165, 288],
              str(out["tensors"]["enc"]["shape"]))
        check("%s: *** the decoder TENSORS carry 4 rows, not just the op shapes ***" % mode,
              out["tensors"]["y"]["shape"] == [4, 1, 288]
              and out["tensors"]["p"]["shape"] == [4, 165, 288],
              "y=%s p=%s" % (out["tensors"]["y"]["shape"], out["tensors"]["p"]["shape"]))
        check("%s: an ENCODER-only tensor keeps its row count" % mode,
              all(v["shape"][0] == 1 for k, v in out["tensors"].items()
                  if k.endswith("e_c") or k == "e_c"),
              str([(k, v["shape"]) for k, v in out["tensors"].items() if "e_c" in k]))
        check("%s: packed layout matches the baker (x field 3200 B, then h0)" % mode,
              out["input"]["packed_bytes"] == 4352
              and [p["offset"] for p in out["input"]["packed_inputs"]][-1] == 3200,
              str([(p["name"], p["offset"], p["size"])
                   for p in out["input"]["packed_inputs"]]))
        check("%s: graph verifies (SSA, ids 0..N-1, outputs reachable)" % mode,
              ir_cse.verify(out) is None)
        check("%s: replay_safe is false" % mode, rep["replay_safe"] is False)

    out, rep = regbatch(base, 4, "replicate")
    check("replicate: 4 conv stems, one per utterance", len(
        [o for o in out["ops"] if o.get("op") == "conv2d_s8"]) == 4)
    check("replicate: the join is B-1 concatenations and B+1 free views",
          len([o for o in out["ops"] if o.get("op") == "cat2_c1_s8"]) == 3
          and len([o for o in out["ops"] if (o.get("op") or "view") == "view"]) == 5)
    check("replicate: needs no driver change", rep["driver_required"] is False)
    check("replicate: weights are SHARED, not copied",
          {o["weight"] for o in out["ops"] if o.get("weight")} == {"W0", "W1", "W2"})

    out, rep = regbatch(base, 4, "once")
    check("once: ONE conv stem", len(
        [o for o in out["ops"] if o.get("op") == "conv2d_s8"]) == 1)
    check("once: it declares the driver contract", rep["driver_required"] is True
          and rep["encoder_dispatches"] == 2, str(rep.get("encoder_dispatches")))

    # the refusals
    two = copy.deepcopy(base)
    two["ops"][3]["inputs"] = ["h0", "e_c"]          # a second crossing tensor
    try:
        regbatch(two, 2, "replicate"); check("refuses a second crossing tensor", False)
    except RegBatchError as e:
        check("refuses a second crossing tensor", "e_c" in str(e), str(e))
    none = copy.deepcopy(base)
    try:
        regbatch(none, 2, "replicate", prefix="zz."); check("refuses an absent region", False)
    except RegBatchError:
        check("refuses an absent region", True)
    odd = copy.deepcopy(base)
    odd["ops"][3]["op"] = "conv2d_s8"                 # a decoder op with no rule
    try:
        regbatch(odd, 2, "replicate"); check("refuses a DECODER op it has no rule for", False)
    except RegBatchError as e:
        check("refuses a DECODER op it has no rule for", "dec.0" in str(e), str(e))

    print("    %s" % ("selftest: PASS" if not bad else "selftest: %d FAILED" % bad))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir"); ap.add_argument("--out"); ap.add_argument("--report")
    ap.add_argument("-B", "--batch", type=int, default=2)
    ap.add_argument("--mode", choices=("replicate", "once"), default="replicate")
    ap.add_argument("--prefix", default="e.")
    ap.add_argument("--wire", default="enc")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.ir or not a.out:
        ap.error("--ir and --out are required")
    out, rep = regbatch(json.load(open(a.ir)), a.batch, a.mode, a.prefix, a.wire)
    json.dump(out, open(a.out, "w"))
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print("ir_regbatch: mode=%s B=%d, %d ops, %d tensors, packed %s B, driver_required=%s"
          % (a.mode, a.batch, rep.get("n_ops", 0), rep.get("n_tensors", 0),
             rep.get("packed_bytes"), rep.get("driver_required")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
