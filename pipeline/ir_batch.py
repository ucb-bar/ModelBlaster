#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Give an unrolled decoder graph a BATCH dimension: decode B independent utterances at once.

    python3 -m modelblaster.pipeline.ir_batch --selftest
    python3 -m modelblaster.pipeline.ir_batch --ir in/graph.json --out out/graph.json -B 2

WHY. The decoder is fill-bound: `bytes_wgt / image_bytes` is 24.00 at M = 1 against 1.000 in the
encoder at M = 165, because at one row per step every weight byte is fetched and used exactly
once.  B independent utterances give M = B and divide the weight fill by B, and -- unlike beam
width or speculative decoding -- change no model and no output: each sequence decodes to exactly
the tokens it decodes alone (B99 step 1).

THE REWRITE IS ONE RULE PLUS ONE EXCEPTION.

  every tensor            shape[0] *= B
  linear/layernorm/softmax  M *= B          matmul_b   B *= B
  cat2_c1                   N *= B          permute4   d0 *= B
  add / mul / silu / view   n *= B
  rope_s8                   H *= B, T UNTOUCHED      <- the exception

`dim0 *= B` is uniform even though 576 tensors carry dim0 = 8 rather than 1: those are the
attention tensors with the heads already folded into the leading axis, and [B_seq][8][..]
flattens to 8*B_seq with the SEQUENCE OUTERMOST -- which is the order `matmul_b_s8` strides in
(`a += bi*M*K`, `b += bi*K*N`) and the order `permute4` with p0 = 0 produces.  That is why this
is a shape rewrite and not a re-layout.

ROPE IS THE EXCEPTION BECAUSE T IS NOT A BATCH AXIS, IT IS A POSITION.  `kernel_rope_s8` reads
`cos_tab + t*R2` and indexes `base = (t*H + h)*D`, so T indexes the POSITION TABLE -- a weight
whose shape must be exactly (T, rotary/2).  Scaling T would apply positions 0..B-1 to B rows
that are all at the SAME step, which is wrong, and the extractor refuses the shape anyway.
Scaling H instead is exact by that same indexing: every one of the 8*B head-rows gets position
t = 0's table row, which is what a batch at a shared step needs.  It also amortises the per-t
CK/SK table build over B times the rows.

*** A B > 1 GRAPH MUST NOT BE BUILT AS A REPLAY IMAGE. ***  `generate_skeleton` writes
`test_input.bin` and `test_golden.bin` straight from `io.npz`, which is B = 1 sized, while the
graph it just emitted declares B times those sizes in `model.h`.  Measured at B = 2:

    test_input   577,152 written   vs  MODEL_INPUT_SIZE  1,154,304 declared
    test_golden  786,432 written   vs  MODEL_OUTPUT_SIZE 1,572,864 declared

The replay harness `.incbin`s those files and hands the symbols to `run_model()`, so it would
read 577,152 bytes PAST THE END of the baked input and compare 1,572,864 bytes of output
against a 786,432-byte golden.  It would not crash; it would produce a number.  The
autoregressive path is unaffected -- it feeds `mb_ar_in` and never touches `test_io.S` -- and
that is the only path a batched image is for.  The report carries `replay_safe: false` so this
is refusable by a caller rather than remembered by a person.

WHAT THIS PASS DOES NOT DO.  The driver.  `MODEL_OUTPUT_SIZE` becomes B*786,432 with step k's
sequence b at `(k*B + b)*VOCAB`, and step k's input row b at `H_OFF[k] + b*DHID`; both are
driver arithmetic (samples/modelblaster_pext/src/main.c, -DMB_AR_BATCH=B).  `driver_meta.h` is
unchanged and does not carry B -- deliberately, so a B = 1 image built from a batched pass is
still exactly today's image.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from . import ir_cse

#: op kind -> the shape key that carries the row count.  `rope_s8` is H, not T; see the docstring.
ROW_KEY = {
    "linear_s8": "M", "layernorm_s8": "M", "softmax_s8": "M",
    "matmul_b_s8": "B", "cat2_c1_s8": "N", "permute4_s8": "d0",
    "add_s8": "n", "mul_s8": "n", "silu_s8": "n", "view": "n",
    "rope_s8": "H",
}


class BatchError(Exception):
    pass


def batch(ir: dict, B: int) -> tuple[dict, dict]:
    if B < 1:
        raise BatchError("B must be >= 1")
    ir = copy.deepcopy(ir)
    if B == 1:
        return ir, {"B": 1, "unchanged": True, "replay_safe": True}
    for name, t in ir["tensors"].items():
        sh = t["shape"]
        if not sh:
            raise BatchError("tensor %r has no shape" % name)
        sh[0] *= B
    seen = set()
    for o in ir["ops"]:
        kind = o.get("op") or "view"
        if kind not in ROW_KEY:
            raise BatchError("op %r has kind %r, which this pass has no rule for -- refusing "
                             "rather than leaving it at one row" % (o.get("name"), kind))
        key = ROW_KEY[kind]
        sh = o.get("shape") or {}
        if key not in sh:
            raise BatchError("op %r (%s) has no %r in its shape %s"
                             % (o.get("name"), kind, key, sorted(sh)))
        sh[key] *= B
        seen.add(kind)
        if kind == "rope_s8" and sh.get("T") != 1:
            raise BatchError("rope op %r has T=%s; this pass folds the batch into H and that is "
                             "only correct while every row shares one position (T == 1)"
                             % (o.get("name"), sh.get("T")))
    # the input surface: every packed input carries B rows now, so every offset moves
    pk = ir.get("input", {}).get("packed_inputs")
    if pk:
        off = 0
        for p in pk:
            p["size"] *= B
            p["offset"] = off
            p["byte_offset"] = off
            off += p["size"]
        ir["input"]["packed_bytes"] = off
    ir_cse.verify(ir)
    return ir, {"B": B, "kinds": sorted(seen), "n_tensors": len(ir["tensors"]),
                "n_ops": len(ir["ops"]), "packed_bytes": ir.get("input", {}).get("packed_bytes"),
                # generate_skeleton bakes test_input/test_golden at io.npz's B=1 size into a
                # graph declaring B times that -- see the docstring.  AR images are unaffected.
                "replay_safe": B == 1}


# --------------------------------------------------------------------------------------------
def _toy() -> dict:
    t = lambda *s: {"shape": list(s), "dtype": "i8", "quant": {"scale": 0.5, "zero_point": 0}}
    return {"name": "toy", "version": 1, "quant": "int8",
            "input": {"tensor": "h", "tensors": ["h", "kx"],
                      "packed_inputs": [{"name": "h", "offset": 0, "size": 288,
                                         "dtype": "i8", "byte_offset": 0},
                                        {"name": "kx", "offset": 288, "size": 47520,
                                         "dtype": "i8", "byte_offset": 288}],
                      "packed_bytes": 47808},
            "output": {"tensors": ["y"], "tensor": None},
            "tensors": {"h": t(1, 1, 288), "kx": t(1, 165, 288), "q": t(1, 1, 288),
                        "r": t(1, 1, 8, 36), "s": t(8, 1, 3), "y": t(1, 1, 288)},
            "ops": [
                {"name": "q", "op": "linear_s8", "inputs": ["h"], "outputs": ["q"],
                 "weight": "W", "shape": {"M": 1, "K": 288, "N": 288}, "quant": {},
                 "dispatch_id": 0, "hardware_target": "any", "depends_on": []},
                {"name": "rope", "op": "rope_s8", "inputs": ["q"], "outputs": ["r"],
                 "shape": {"T": 1, "H": 8, "D": 36, "R": 32}, "quant": {},
                 "dispatch_id": 1, "hardware_target": "any", "depends_on": [0]},
                {"name": "qk", "op": "matmul_b_s8", "inputs": ["r", "kx"], "outputs": ["s"],
                 "shape": {"B": 8, "M": 1, "K": 36, "N": 3, "transpose_b": 1}, "quant": {},
                 "dispatch_id": 2, "hardware_target": "any", "depends_on": [1]},
                {"name": "o", "op": "linear_s8", "inputs": ["s"], "outputs": ["y"],
                 "weight": "W2", "shape": {"M": 1, "K": 288, "N": 288}, "quant": {},
                 "dispatch_id": 3, "hardware_target": "any", "depends_on": [2]},
            ],
            "dispatches": [0, 1, 2, 3]}


def selftest() -> int:
    bad = 0

    def check(label, cond, detail=""):
        nonlocal bad
        print("    %-58s %s%s" % (label, "PASS" if cond else "FAIL", "" if cond else "  " + detail))
        if not cond:
            bad += 1

    base = _toy()
    out, plan = batch(base, 2)
    T, ops = out["tensors"], {o["name"]: o for o in out["ops"]}
    check("B=1 is a no-op", json.dumps(batch(base, 1)[0], sort_keys=True)
          == json.dumps(base, sort_keys=True))
    check("every tensor dim0 doubled (1 -> 2)", T["h"]["shape"] == [2, 1, 288]
          and T["q"]["shape"] == [2, 1, 288], str(T["h"]["shape"]))
    check("a dim0=8 attention tensor becomes 16, not 8x2 elsewhere",
          T["s"]["shape"] == [16, 1, 3], str(T["s"]["shape"]))
    check("inner dims untouched (K, N, 165, 36 unchanged)",
          T["kx"]["shape"] == [2, 165, 288] and T["r"]["shape"] == [2, 1, 8, 36])
    check("linear M doubled, K and N untouched",
          ops["q"]["shape"] == {"M": 2, "K": 288, "N": 288}, str(ops["q"]["shape"]))
    check("matmul_b leading axis 8 -> 16",
          ops["qk"]["shape"]["B"] == 16 and ops["qk"]["shape"]["M"] == 1
          and ops["qk"]["shape"]["N"] == 3, str(ops["qk"]["shape"]))
    check("*** rope H doubled and T LEFT AT 1 ***",
          ops["rope"]["shape"] == {"T": 1, "H": 16, "D": 36, "R": 32},
          str(ops["rope"]["shape"]))
    check("packed inputs resized and re-offset",
          [(p["name"], p["offset"], p["size"]) for p in out["input"]["packed_inputs"]]
          == [("h", 0, 576), ("kx", 576, 95040)]
          and out["input"]["packed_bytes"] == 95616,
          str(out["input"]["packed_inputs"]))
    check("report flags a B>1 graph as replay-unsafe", plan["replay_safe"] is False)
    check("B=1 report is replay-safe", batch(_toy(), 1)[1].get("replay_safe", True) is not False)
    check("dispatch ids untouched (no op added or removed)",
          [o["dispatch_id"] for o in out["ops"]] == [0, 1, 2, 3]
          and out["dispatches"] == [0, 1, 2, 3])

    # the refusals
    weird = _toy(); weird["ops"][0]["op"] = "conv2d_s8"
    try:
        batch(weird, 2); check("refuses an op kind it has no rule for", False)
    except BatchError:
        check("refuses an op kind it has no rule for", True)
    weird2 = _toy(); weird2["ops"][1]["shape"]["T"] = 3
    try:
        batch(weird2, 2); check("refuses a rope whose T is not 1", False)
    except BatchError:
        check("refuses a rope whose T is not 1", True)

    print("    %s" % ("selftest: PASS" if not bad else "selftest: %d FAILED" % bad))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir"); ap.add_argument("--out")
    ap.add_argument("-B", "--batch", type=int, default=2)
    ap.add_argument("--report"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.ir or not a.out:
        ap.error("--ir and --out are required")
    out, plan = batch(json.load(open(a.ir)), a.batch)
    json.dump(out, open(a.out, "w"))
    if a.report:
        json.dump(plan, open(a.report, "w"), indent=1)
    print("ir_batch: B=%d, %d ops over %s, packed input %s B"
          % (a.batch, plan.get("n_ops", 0), plan.get("kinds"), plan.get("packed_bytes")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
