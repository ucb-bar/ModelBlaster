#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a permute's output already transposed for the matmul that reads it (Lab B62).

    python3 -m modelblaster.pipeline.ir_vlayout --ir IN/graph.json --out OUT/graph.json
    python3 -m modelblaster.pipeline.ir_vlayout --selftest

WHAT THIS IS FOR.  `matmul_b_s8` has two paths and they are 9.2x apart on the SAME op at the
SAME quantisation.  When `transpose_b = 1` the reduction runs along B's rows, which are
contiguous, and B59's M = 1 specialisation reads them in place as aligned 64-bit words.  When
`transpose_b = 0` the reduction runs DOWN A COLUMN and the kernel must pack the column eight
bytes at a time into a register before it can hand it to DOT8.  Board 0x5A5A0028, B59:

    cross QK^T   B=8 M=1 K=36  N=165  transpose_b=1   101.94 cycles/element
    cross PV     B=8 M=1 K=165 N=36   transpose_b=0   935.91 cycles/element

Moonshine's cross-attention V reaches that matmul through a `permute4_s8`, which ALREADY
copies every byte.  So the column layout is not forced by anything -- it is what the permute
happened to emit.  This pass changes what it emits: the last two axes of the permutation are
exchanged, the tensor's shape follows, and every consumer's `transpose_b` flips from 0 to 1.

VALUE-PRESERVING, and the bar is identity rather than a tolerance.  A permutation is a
bijection and `transpose_b` selects which of two index orders the kernel reads the same bytes
in; `out[b,i,j] = sum_k a[b,i,k] * B[k,j]` with B stored [K][N] and the same sum with B stored
[N][K] and `transpose_b = 1` accumulate the same products over the same k in the same order.
Nothing here changes an arithmetic expression.  The caller must hold this pass to a BYTE
IDENTICAL host-C golden against the un-rewritten graph (`58_...sh --golden-against`), the way
`ir_cse.py` is held.

WHY `--min-consumers 2` IS THE DEFAULT, and it is the whole economic argument.  Flipping the
layout does not delete work, it MOVES it: the transposing permute reads its input with a
stride instead of in runs, so the permute gets dearer while the matmul gets much cheaper.
That trade only pays when the permute is amortised over more than one matmul.  On this decoder
`ir_cse.py` has already collapsed the twelve cross-attention permutes to one dispatch each
serving 24 decode steps, so the six V permutes are paid ONCE and read 24 times; the 138
self-attention V permutes are paid once and read once, and are left alone by default.  Pass
`--min-consumers 1` to include them, deliberately.

WHAT IT REFUSES TO DO, rather than doing quietly:
  * it rewrites a permute only if EVERY consumer of its output is a `matmul_b_s8` reading it
    as operand B with `transpose_b = 0` -- one consumer that wants the old order is a
    fan-out problem and the honest answer is a second permute, not a silent miscompile;
  * it never rewrites a tensor that is a graph output or a graph input;
  * it dies if the permute's output does not end in the consumer's [K, N] -- the axes being
    exchanged must be exactly the ones the matmul indexes.

The report JSON is the audit trail: every permute considered, why it was taken or refused,
and the consumers each one carries.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys


class VLayoutError(Exception):
    """A refusal.  Raised rather than returning a silently degraded graph."""


def _out_shape(sh: dict) -> list[int]:
    d = [sh["d0"], sh["d1"], sh["d2"], sh["d3"]]
    p = [sh["p0"], sh["p1"], sh["p2"], sh["p3"]]
    return [d[p[k]] for k in range(4)]


def plan(ir: dict, min_consumers: int = 2) -> list[dict]:
    """One record per `permute4_s8`, saying whether it can be flipped and why not."""
    ops = ir["ops"]
    tensors = ir.get("tensors") or {}
    consumers: dict[str, list[dict]] = {}
    producers: dict[str, list[dict]] = {}
    for o in ops:
        for t in (o.get("inputs") or []):
            consumers.setdefault(t, []).append(o)
        for t in (o.get("outputs") or []):
            producers.setdefault(t, []).append(o)

    inp = ir.get("input") or {}
    in_names = set(inp.get("tensors") or ([inp["tensor"]] if inp.get("tensor") else []))
    of = ir.get("output") or {}
    out_names = set(of.get("tensors") or ([of["tensor"]] if of.get("tensor") else []))

    recs = []
    for o in ops:
        if o.get("op") != "permute4_s8":
            continue
        t = o["outputs"][0]
        cs = consumers.get(t, [])
        rec = {"permute": o.get("name"), "dispatch_id": o.get("dispatch_id"),
               "tensor": t, "consumers": len(cs),
               "shape_in": _out_shape(o["shape"]), "take": False, "why": ""}
        recs.append(rec)
        if t in in_names or t in out_names:
            rec["why"] = "the tensor is on the model's input or output surface"
            continue
        if not cs:
            rec["why"] = "no consumer"
            continue
        if len(cs) < min_consumers:
            rec["why"] = (f"{len(cs)} consumer(s) < --min-consumers {min_consumers}: "
                          f"the permute would get dearer and nothing amortises it")
            continue
        def _is_b_operand(c):
            return (c.get("op") == "matmul_b_s8"
                    and (c.get("inputs") or [None, None])[1] == t)
        if all(_is_b_operand(c) and int(c["shape"].get("transpose_b", 0)) == 1
               for c in cs):
            rec["why"] = "already on the transpose_b=1 path (this is a K, not a V)"
            continue
        bad = [c for c in cs
               if not _is_b_operand(c) or int(c["shape"].get("transpose_b", 0)) != 0]
        if bad:
            rec["why"] = ("fan-out: %d of %d consumers do not read it as matmul_b_s8's "
                          "operand B at transpose_b=0 (%s)"
                          % (len(bad), len(cs),
                             ", ".join(sorted({b.get("op", "?") for b in bad}))))
            continue
        osh = _out_shape(o["shape"])
        want = {(int(c["shape"]["K"]), int(c["shape"]["N"])) for c in cs}
        if want != {(osh[2], osh[3])}:
            rec["why"] = (f"the permute emits {osh} but its consumers index "
                          f"{sorted(want)} as [K, N]")
            continue
        nb = {int(c["shape"]["B"]) for c in cs}
        if nb != {osh[0] * osh[1]}:
            rec["why"] = (f"batch mismatch: permute leading axes {osh[:2]} against "
                          f"consumer B {sorted(nb)}")
            continue
        rec["take"] = True
        rec["why"] = "every consumer is a transpose_b=0 PV on this tensor"
        rec["shape_out"] = [osh[0], osh[1], osh[3], osh[2]]
        rec["consumer_names"] = [c.get("name") for c in cs]
    return recs


def rewrite(ir: dict, min_consumers: int = 2) -> tuple[dict, dict]:
    out = copy.deepcopy(ir)
    recs = plan(out, min_consumers)
    take = {r["permute"] for r in recs if r["take"]}
    if not take:
        return out, {"taken": 0, "permutes": recs}

    by_tensor = {}
    for o in out["ops"]:
        if o.get("op") == "permute4_s8" and o.get("name") in take:
            sh = o["shape"]
            sh["p2"], sh["p3"] = sh["p3"], sh["p2"]
            t = o["outputs"][0]
            meta = (out.get("tensors") or {}).get(t)
            if meta is None:
                raise VLayoutError(f"permute {o['name']} output {t} has no tensor record")
            s = list(meta["shape"])
            meta["shape"] = s[:-2] + [s[-1], s[-2]]
            by_tensor[t] = o["name"]
    flipped = 0
    for o in out["ops"]:
        if o.get("op") != "matmul_b_s8":
            continue
        b = (o.get("inputs") or [None, None])[1]
        if b not in by_tensor:
            continue
        if int(o["shape"].get("transpose_b", 0)) != 0:
            raise VLayoutError(f"{o.get('name')} reads a flipped tensor at transpose_b=1")
        o["shape"]["transpose_b"] = 1
        o["quant"]["transpose_b"] = 1
        flipped += 1
    rep = {"taken": len(take), "consumers_flipped": flipped,
           "min_consumers": min_consumers, "permutes": recs}
    out.setdefault("_rewrite", []).append(
        {"pass": "ir_vlayout", "permutes": sorted(take), "consumers_flipped": flipped})
    return out, rep


# --------------------------------------------------------------------------------------------
# selftest -- the pass is only worth what its refusals are worth, so they are what is tested
# --------------------------------------------------------------------------------------------
def _toy(fanout: bool = False, consumers: int = 2) -> dict:
    ops = [{"name": "vx", "op": "view", "inputs": ["in0"], "outputs": ["vx"],
            "shape": {"n": 6 * 2 * 3}},
           {"name": "p", "op": "permute4_s8", "inputs": ["vx"], "outputs": ["p"],
            "dispatch_id": 0,
            "shape": {"d0": 1, "d1": 6, "d2": 2, "d3": 3,
                      "p0": 0, "p1": 2, "p2": 1, "p3": 3},
            "quant": {"scale_in": 1.0, "scale_out": 1.0,
                      "activation_min": -128, "activation_max": 127}}]
    for i in range(consumers):
        ops.append({"name": f"m{i}", "op": "matmul_b_s8", "inputs": [f"a{i}", "p"],
                    "outputs": [f"o{i}"], "dispatch_id": 1 + i,
                    "shape": {"B": 2, "M": 1, "K": 6, "N": 3, "transpose_b": 0},
                    "quant": {"scale_a": 1.0, "scale_b": 1.0, "scale_out": 1.0,
                              "transpose_b": 0, "scale_div_sqrt_dk": 1.0,
                              "activation_min": -128, "activation_max": 127}})
    if fanout:
        ops.append({"name": "r", "op": "relu_s8", "inputs": ["p"], "outputs": ["r"],
                    "dispatch_id": 99, "shape": {"n": 36}, "quant": {}})
    return {"name": "toy",
            "input": {"tensor": "in0", "tensors": ["in0"] + [f"a{i}" for i in range(consumers)]},
            "output": {"tensor": "o0", "tensors": [f"o{i}" for i in range(consumers)]},
            "tensors": {"p": {"shape": [1, 2, 6, 3], "dtype": "i8",
                              "quant": {"scale": 1.0, "zero_point": 0}}},
            "ops": ops}


def selftest() -> int:
    import numpy as np
    bad = 0

    def check(label, cond, detail=""):
        nonlocal bad
        print(("  ok   " if cond else "  FAIL ") + label + ("  " + detail if detail else ""))
        if not cond:
            bad += 1

    g, rep = rewrite(_toy(), 2)
    p = [o for o in g["ops"] if o["name"] == "p"][0]
    check("taken", rep["taken"] == 1 and rep["consumers_flipped"] == 2, str(rep["taken"]))
    check("permutation exchanged", (p["shape"]["p2"], p["shape"]["p3"]) == (3, 1))
    check("tensor shape follows", g["tensors"]["p"]["shape"] == [1, 2, 3, 6])
    check("transpose_b flipped in both places",
          all(o["shape"]["transpose_b"] == 1 and o["quant"]["transpose_b"] == 1
              for o in g["ops"] if o["op"] == "matmul_b_s8"))

    _, rep = rewrite(_toy(fanout=True), 2)
    check("fan-out refused", rep["taken"] == 0,
          [r["why"] for r in rep["permutes"]][0][:48])
    _, rep = rewrite(_toy(consumers=1), 2)
    check("single consumer refused by default", rep["taken"] == 0)
    _, rep = rewrite(_toy(consumers=1), 1)
    check("single consumer taken at --min-consumers 1", rep["taken"] == 1)

    # THE VALUE ARGUMENT, executed rather than asserted: the same numbers through both graphs.
    rng = np.random.default_rng(7)
    x = rng.integers(-128, 128, size=(1, 6, 2, 3), dtype=np.int8)
    a = rng.integers(-128, 128, size=(2, 1, 6), dtype=np.int8)
    v_now = np.ascontiguousarray(x.transpose(0, 2, 1, 3)).reshape(2, 6, 3)
    v_new = np.ascontiguousarray(x.transpose(0, 2, 3, 1)).reshape(2, 3, 6)
    ref = np.stack([a[b].astype(np.int64) @ v_now[b].astype(np.int64) for b in range(2)])
    got = np.stack([a[b].astype(np.int64) @ v_new[b].astype(np.int64).T for b in range(2)])
    check("the two layouts compute the same integers", bool((ref == got).all()))

    print("SELFTEST " + ("PASS" if bad == 0 else "FAIL"))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir")
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--min-consumers", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.ir:
        ap.error("--ir is required (or --selftest)")
    ir = json.load(open(a.ir))
    new, rep = rewrite(ir, a.min_consumers)
    taken = [r for r in rep["permutes"] if r["take"]]
    print(f"[ir_vlayout] {len(rep['permutes'])} permute4_s8; "
          f"{len(taken)} flipped, {rep.get('consumers_flipped', 0)} matmul_b_s8 "
          f"consumers set transpose_b=1 (--min-consumers {a.min_consumers})")
    for r in taken:
        print(f"    {r['permute']:<14s} {r['shape_in']} -> {r['shape_out']}  "
              f"{r['consumers']} consumers")
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    if a.dry_run:
        return 0
    if not a.out:
        ap.error("--out is required unless --dry-run")
    json.dump(new, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
