#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Give a HOISTED attention V-permute the layout its matmul wants: (0,2,3,1) + transpose_b=1.

    python3 -m modelblaster.pipeline.ir_vperm --selftest
    python3 -m modelblaster.pipeline.ir_vperm --ir in/graph.json --out out/graph.json

WHY.  The cross-attention `av` matmul reads its V operand as [K][N] with transpose_b=0, so K is
STRIDED (36) and `pext_dot8_exact` falls to its scalar path.  The same producer permuted
(0,2,3,1) instead of (0,2,1,3) gives [N][K], K contiguous, and the 8-wide DOT8 applies.
Lab B101's decomposition, which B99 reproduced to the cycle:

    tb=0, no unroll   232,813 instr   270,065 cyc   CPI 1.160
    tb=1, M1_UNROLL8   58,645 instr   102,490 cyc   CPI 1.748
    3.970 x 0.664 = 2.635, measured 2.635

***THE CPI RATIO IS BELOW ONE.***  The slow side has the BETTER cache behaviour, which falsifies
the cache-residency hypothesis B99 carried for this op by the SIGN rather than by a margin: a
re-read penalty would show as a worse CPI on the re-reading side, and it shows the opposite.
The difference is WORK -- 3.970x the instructions -- not locality.

WHY ONLY HOISTED PERMUTES.  Transposing costs ~2.406x in the permute itself and is repaid once
per CONSUMER.  In this IR the cross-attention permutes are CSE'd to one per layer and feed 24
consumers each; the self-attention ones are per-step and feed exactly one.  Flipping a 1:1
permute pays the penalty with nothing to amortise it over, so this pass requires
`--min-consumers` (default 2) and leaves the per-step permutes alone.  Measured on the IR:
12 permutes with 24 consumers, 276 with 1, and NO permute feeding both a tb=0 and a tb=1
consumer -- so the flip is unambiguous.

BIT-EXACTNESS.  (0,2,3,1) with transpose_b=1 and (0,2,1,3) with transpose_b=0 address the SAME
elements in the same order; only the memory layout differs.  `pext_dot8_exact` is
accuracy_class bit_exact on both paths.  That is a claim to CHECK, not to assert -- run the
generated model both ways on one input and diff the output bytes.

CARRIED, NOT ASSUMED (B101, against its own fix): the 2.406x permute penalty was measured on a
REPLAY run.  This pass's economics depend on it, and a replay figure carried onto a real
autoregressive arm is exactly the transfer that cost this lab a board round.  Confirm it.
"""
from __future__ import annotations
import argparse, collections, copy, json, os, sys
from . import ir_cse

SRC = (0, 2, 1, 3)
DST = (0, 2, 3, 1)


class VPermError(Exception):
    pass


def flip(ir: dict, min_consumers: int = 2) -> tuple[dict, dict]:
    ir = copy.deepcopy(ir)
    ops, T = ir["ops"], ir["tensors"]
    cons = collections.defaultdict(list)
    for o in ops:
        for t in o["inputs"]:
            cons[t].append(o)
    flipped, skipped = [], []
    for o in ops:
        if o.get("op") != "permute4_s8":
            continue
        s = o["shape"]
        if (s["p0"], s["p1"], s["p2"], s["p3"]) != SRC:
            continue
        out = o["outputs"][0]
        c = [x for x in cons[out] if x.get("op") == "matmul_b_s8"]
        if not c or len(cons[out]) != len(c):
            continue                                  # a non-matmul reader: leave it alone
        tbs = {x["shape"].get("transpose_b") for x in c}
        if tbs != {0}:
            if 0 in tbs:
                raise VPermError("permute %r feeds both tb=0 and tb=1 consumers; the flip "
                                 "would be correct for one and wrong for the other"
                                 % o.get("name"))
            continue                                  # already tb=1: nothing to do
        if len(c) < min_consumers:
            skipped.append(o.get("name")); continue    # 1:1 -- the transpose would not repay
        s["p0"], s["p1"], s["p2"], s["p3"] = DST
        sh = T[out]["shape"]
        if len(sh) != 4:
            raise VPermError("permute %r output is rank %d, not 4" % (o.get("name"), len(sh)))
        sh[2], sh[3] = sh[3], sh[2]
        for x in c:
            x["shape"]["transpose_b"] = 1
            if "transpose_b" in (x.get("quant") or {}):
                x["quant"]["transpose_b"] = 1
        flipped.append({"permute": o.get("name"), "consumers": len(c), "shape": list(sh)})
    ir_cse.verify(ir)
    return ir, {"flipped": len(flipped), "consumers_repointed": sum(f["consumers"] for f in flipped),
                "skipped_1to1": len(skipped), "min_consumers": min_consumers,
                "detail": flipped[:4]}


def _toy(nc=24):
    t = lambda *s: {"shape": list(s), "dtype": "i8", "quant": {"scale": .5, "zero_point": 0}}
    T = {"x": t(1, 165, 8, 36), "p": t(1, 8, 165, 36), "q": t(1, 165, 8, 36), "r": t(1, 8, 165, 36)}
    ops = [{"name": "vperm", "op": "permute4_s8", "inputs": ["x"], "outputs": ["p"],
            "shape": dict(d0=1, d1=165, d2=8, d3=36, p0=0, p1=2, p2=1, p3=3), "quant": {},
            "dispatch_id": 0, "hardware_target": "any", "depends_on": []},
           {"name": "kperm", "op": "permute4_s8", "inputs": ["q"], "outputs": ["r"],
            "shape": dict(d0=1, d1=165, d2=8, d3=36, p0=0, p1=2, p2=1, p3=3), "quant": {},
            "dispatch_id": 1, "hardware_target": "any", "depends_on": []}]
    T["probs"] = t(1, 8, 1, 165)
    ops.append({"name": "mk", "op": "matmul_b_s8", "inputs": ["probs", "r"], "outputs": ["ok"],
                "shape": dict(B=8, M=1, K=36, N=165, transpose_b=1), "quant": {"transpose_b": 1},
                "dispatch_id": 2, "hardware_target": "any", "depends_on": []})
    T["ok"] = t(1, 8, 1, 165)
    for i in range(nc):
        T["o%d" % i] = t(1, 8, 1, 36)
        ops.append({"name": "av%d" % i, "op": "matmul_b_s8", "inputs": ["probs", "p"],
                    "outputs": ["o%d" % i], "shape": dict(B=8, M=1, K=165, N=36, transpose_b=0),
                    "quant": {"transpose_b": 0}, "dispatch_id": 3 + i,
                    "hardware_target": "any", "depends_on": []})
    return {"name": "toy", "version": 1, "quant": "int8",
            "input": {"tensor": "x", "tensors": ["x", "q", "probs"], "packed_inputs": []},
            "output": {"tensors": ["o0"], "tensor": None}, "tensors": T, "ops": ops,
            "dispatches": list(range(3 + nc))}


def selftest() -> int:
    bad = 0
    def ck(l, c, d=""):
        nonlocal bad
        print("    %-56s %s%s" % (l, "PASS" if c else "FAIL", "" if c else "  " + d))
        bad += (not c)
    ir, p = flip(_toy(24))
    o = {x["name"]: x for x in ir["ops"]}
    ck("the V permute flips to (0,2,3,1)",
       (o["vperm"]["shape"]["p2"], o["vperm"]["shape"]["p3"]) == (3, 1), str(o["vperm"]["shape"]))
    ck("its output tensor becomes [1,8,36,165]", ir["tensors"]["p"]["shape"] == [1, 8, 36, 165])
    ck("every av consumer becomes transpose_b=1",
       all(o["av%d" % i]["shape"]["transpose_b"] == 1 for i in range(24)))
    ck("quant.transpose_b tracks the shape", o["av0"]["quant"]["transpose_b"] == 1)
    ck("the K permute (already tb=1) is UNTOUCHED",
       (o["kperm"]["shape"]["p2"], o["kperm"]["shape"]["p3"]) == (1, 3)
       and ir["tensors"]["r"]["shape"] == [1, 8, 165, 36])
    ck("24 consumers repointed, 0 skipped", (p["flipped"], p["consumers_repointed"]) == (1, 24))
    ir1, p1 = flip(_toy(1))
    ck("a 1:1 permute is SKIPPED (transpose would not repay)",
       p1["flipped"] == 0 and p1["skipped_1to1"] == 1)
    ck("...and its graph is unchanged",
       json.dumps(ir1, sort_keys=True) == json.dumps(_toy(1), sort_keys=True))
    mixed = _toy(2); mixed["ops"][-1]["shape"]["transpose_b"] = 1
    try:
        flip(mixed); ck("refuses a permute feeding both tb=0 and tb=1", False)
    except VPermError:
        ck("refuses a permute feeding both tb=0 and tb=1", True)
    print("    %s" % ("selftest: PASS" if not bad else "selftest: %d FAILED" % bad))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir"); ap.add_argument("--out"); ap.add_argument("--report")
    ap.add_argument("--min-consumers", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest: return selftest()
    if not a.ir or not a.out: ap.error("--ir and --out are required")
    ir, plan = flip(json.load(open(a.ir)), a.min_consumers)
    json.dump(ir, open(a.out, "w"))
    if a.report: json.dump(plan, open(a.report, "w"), indent=1)
    print("ir_vperm: %d permute(s) flipped, %d consumers repointed, %d 1:1 skipped"
          % (plan["flipped"], plan["consumers_repointed"], plan["skipped_1to1"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
