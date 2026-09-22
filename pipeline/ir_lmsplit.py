#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Split a `linear_s8` that is too wide to image in one piece into N-shards, IN THE GRAPH.

    python3 -m modelblaster.pipeline.ir_lmsplit --selftest
    python3 -m modelblaster.pipeline.ir_lmsplit --ir in/graph.json --weights in/weights.npz \
        --out out/graph.json --out-weights out/weights.npz --report out/ir_lmsplit.json

WHY THIS EXISTS, AND WHY IT IS A GRAPH PASS AND NOT A KERNEL FIX.

`kernel_linear_s8`'s engine path already shards a wide layer by N -- Moonshine's `lm_head` is
N = 32,768, K = 288, and `N * Kp = 9,437,184` is over `mbxr_rt.h`'s 8 MiB staging guard, so the
kernel builds two images of N = 16,384 and calls the engine twice.  But that path is gated:

    if (Nc > 0 && Nc < N && M == 1) {        /* kernels.c */

and the guard is CORRECT, not an oversight: the kernel takes no output stride, so an N-shard
writing `out + n0` addresses the right elements only while there is one row.  At M > 1 the
shard path is skipped, the whole layer is asked for at once, `mbxr_rt_image` refuses it on the
same 8 MiB guard, and **the largest layer in the decoder silently falls back to the CPU
reference kernel** -- 51.8 % of the weight image leaving the engine, reported only as
`calls_fallback`.  That is Lab B28's defect, and a batched decoder (B99) walks straight into it.

Sharding in the GRAPH instead gives each shard its own output tensor, so there is no stride to
get wrong and the M == 1 gate is never reached -- `Nc == N` for every shard.  It needs no change
under `sw/roccmoon/` (frozen for B98) and none to either driver.

WHAT IS PRESERVED, BY CONSTRUCTION AND BY CHECK.

  * THE SHARD BOUNDARIES ARE THE KERNEL'S OWN.  `chunk_n` below is `mbxr_lin_chunk_n`
    transliterated, so the shards are the images the kernel builds for itself today and
    `image_bytes` does not move by a byte.
  * THE OUTPUT IS BIT-IDENTICAL.  Same weights (row-sliced), same bias (sliced), same
    multiplier and shift, same input.  The shards are re-joined by a `cat2_c1_s8` whose three
    scales are all the original tensor's, so the kernel takes its identity path and the join is
    a memcpy, not a requantise.
  * THE GRAPH OUTPUT TENSOR IS UNCHANGED -- same name, same shape, same scale.  That is what
    keeps `emit_driver_meta.py` (N_STEPS = len(output.tensors), VOCAB = its last dim,
    STEP_END[k] = its producer's dispatch id), `dec_driver_tok.c` and the board driver all
    untouched.  The alternative -- promoting the shards to graph outputs -- silently gives
    N_STEPS = 48 and VOCAB = 16,384, which is wrong tokens with nothing raising.

THE COST, STATED: one `cat2_c1_s8` per split op, which is `N` bytes of memcpy.  For Moonshine's
decoder that is 32,768 B per step against a ~8.4 M-cycle step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import ir_cse            # the verifier, not re-implemented here

#: mbxr_rt.h's staging guard, and kernels.c's MBXR_LIN_STAGE_LIMIT.  One number, two files.
STAGE_LIMIT = 8 << 20
#: mbxr.h's MBXR_NCH -- shards are whole quads, as the engine wants.
NCH = 4


class SplitError(Exception):
    pass


def chunk_n(N: int, Kp: int) -> int:
    """`mbxr_lin_chunk_n` (kernels.c), transliterated so the shards ARE the kernel's."""
    need = N * Kp
    if need <= STAGE_LIMIT:
        return N
    chunks = (need + STAGE_LIMIT - 1) // STAGE_LIMIT
    nc = (N + chunks - 1) // chunks
    nc = (nc + NCH - 1) & ~(NCH - 1)
    if nc * Kp > STAGE_LIMIT:
        nc -= NCH
    return nc if nc > 0 else 0


def shards_for(N: int, K: int) -> list[tuple[int, int]]:
    """[(n0, width)] -- empty when the op needs no split."""
    Kp = (K + 7) & ~7
    nc = chunk_n(N, Kp)
    if nc <= 0 or nc >= N:
        return []
    out, n0 = [], 0
    while n0 < N:
        out.append((n0, min(nc, N - n0)))
        n0 += nc
    return out


def split(ir: dict) -> tuple[dict, dict]:
    """Returns (new_ir, plan). `plan` names every weight slice the caller must materialise."""
    ops = ir["ops"]
    tensors = ir["tensors"]
    new_ops: list[dict] = []
    slices: list[dict] = []
    n_split = 0

    for o in ops:
        if o.get("op") != "linear_s8" or o.get("dispatch_id") is None:
            new_ops.append(o)
            continue
        sh = o["shape"]
        parts = shards_for(int(sh["N"]), int(sh["K"]))
        if not parts:
            new_ops.append(o)
            continue
        out_t = o["outputs"][0]
        rec = tensors.get(out_t)
        if rec is None:
            raise SplitError("op %r produces %r with no `tensors` record"
                             % (o.get("name"), out_t))
        if rec["shape"][-1] != int(sh["N"]):
            raise SplitError("op %r: output %r last dim %d != N %d -- this pass only splits a "
                             "layer whose output tensor is [.., N]"
                             % (o.get("name"), out_t, rec["shape"][-1], sh["N"]))
        n_split += 1
        old_id = o["dispatch_id"]
        piece = []
        for i, (n0, w) in enumerate(parts):
            pn = "%s__s%d" % (out_t, i)
            piece.append(pn)
            # THE SHARD CARRIES THE PARENT'S SCALE.  Same multiplier and shift, so these are
            # the parent's codes; a different scale would make the join a requantise and the
            # argmax over the joined row meaningless.
            tensors[pn] = {"shape": rec["shape"][:-1] + [w], "dtype": rec["dtype"],
                           "quant": dict(rec["quant"])}
            wn = "%s__s%d" % (o["weight"], i)
            bn = ("%s__s%d" % (o["bias"], i)) if o.get("bias") else None
            slices.append({"src_weight": o["weight"], "dst_weight": wn,
                           "src_bias": o.get("bias"), "dst_bias": bn,
                           "row0": n0, "rows": w})
            q = dict(o)
            q["name"] = "%s__s%d" % (o["name"], i)
            q["outputs"] = [pn]
            q["weight"] = wn
            if bn:
                q["bias"] = bn
            q["shape"] = dict(sh, N=w)
            q["dispatch_id"] = old_id           # provisional; _renumber fixes every id
            q["_shard_of"] = old_id
            new_ops.append(q)
        # the join: left-folded cat2 pairs, all three scales the parent's
        acc, acc_w = piece[0], parts[0][1]
        for i in range(1, len(piece)):
            last = i == len(piece) - 1
            dst = out_t if last else "%s__j%d" % (out_t, i)
            w = parts[i][1]
            if not last:
                tensors[dst] = {"shape": rec["shape"][:-1] + [acc_w + w],
                                "dtype": rec["dtype"], "quant": dict(rec["quant"])}
            new_ops.append({
                "name": "%s__join%d" % (o["name"], i), "op": "cat2_c1_s8",
                "inputs": [acc, piece[i]], "outputs": [dst],
                # NHWC with the logits on C: stride H*W = 1, so cat2's identity path is a
                # straight memcpy of each side.
                # THE SCHEMA IS THE IR's OWN, not one invented here: `C_inputs`/`C_total` and
                # `scales_in`/`scale_out` are what generate_skeleton reads (it raised
                # KeyError: 'C_inputs' on the first spelling).  With H = W = 1 the kernel's
                # stride is 1 and the two sides land at out[0] and out[acc_w] -- two memcpys.
                "shape": {"N": 1, "H": 1, "W": 1,
                          "C_inputs": [acc_w, w], "C_total": acc_w + w},
                "quant": {"scales_in": [rec["quant"]["scale"], rec["quant"]["scale"]],
                          "scale_out": rec["quant"]["scale"],
                          "activation_min": -128, "activation_max": 127},
                "dispatch_id": old_id, "hardware_target": "any",
                "depends_on": [], "_shard_of": old_id,
            })
            acc, acc_w = dst, acc_w + w

    new_ir = dict(ir)
    new_ir["ops"] = new_ops
    new_ir["tensors"] = tensors
    _renumber(new_ir)
    ir_cse.verify(new_ir)
    plan = {"ops_split": n_split, "weight_slices": slices,
            "dispatches_before": len(ir.get("dispatches") or []),
            "dispatches_after": len(new_ir["dispatches"]),
            "stage_limit": STAGE_LIMIT, "nch": NCH}
    return new_ir, plan


def _renumber(ir: dict) -> None:
    """dispatch ids 0..N-1 in IR order, `depends_on` remapped, `dispatches` rebuilt.

    A consumer that depended on a split op must now depend on its LAST piece -- the join that
    produces the parent's tensor -- so the old id maps to the HIGHEST new id that carried it.
    That is the whole reason this is not a simple enumerate()."""
    old_ids = [o.get("dispatch_id") for o in ir["ops"]]
    old_to_last: dict[int, int] = {}
    nxt = 0
    for o, old in zip(ir["ops"], old_ids):
        if old is None:
            continue
        o["dispatch_id"] = nxt
        old_to_last[old] = nxt          # later pieces overwrite: the last one wins
        nxt += 1
    for o in ir["ops"]:
        deps = o.get("depends_on")
        if deps:
            o["depends_on"] = sorted({old_to_last[d] for d in deps if d in old_to_last})
    # the pieces' own edges: shard i depends on what the parent depended on; the join on the
    # shards it reads.  Done after the remap so the ids are the new ones.
    produced_by = {}
    for o in ir["ops"]:
        if o.get("dispatch_id") is not None:
            for t in o["outputs"]:
                produced_by[t] = o["dispatch_id"]
    for o in ir["ops"]:
        if o.pop("_shard_of", None) is None:
            continue
        if o["op"] == "cat2_c1_s8":
            o["depends_on"] = sorted(produced_by[t] for t in o["inputs"] if t in produced_by)
    ir["dispatches"] = list(range(nxt))


# --------------------------------------------------------------------------------------------
# selftest: a toy graph with a known answer, and the refusals
# --------------------------------------------------------------------------------------------
def _toy(N: int = 32768, K: int = 288) -> dict:
    t = {"x": {"shape": [1, 1, K], "dtype": "i8", "quant": {"scale": 0.5, "zero_point": 0}},
         "y": {"shape": [1, 1, N], "dtype": "i8", "quant": {"scale": 0.25, "zero_point": 0}},
         "z": {"shape": [1, 1, N], "dtype": "i8", "quant": {"scale": 0.25, "zero_point": 0}}}
    ops = [
        {"name": "lin", "op": "linear_s8", "inputs": ["x"], "outputs": ["y"],
         "weight": "W", "bias": "B", "shape": {"M": 1, "K": K, "N": N},
         "quant": {"output_multiplier": 3, "output_shift": 1,
                   "activation_min": -128, "activation_max": 127},
         "dispatch_id": 0, "hardware_target": "any", "depends_on": []},
        {"name": "act", "op": "silu_s8", "inputs": ["y"], "outputs": ["z"],
         "shape": {"n": N}, "quant": {}, "dispatch_id": 1, "hardware_target": "any",
         "depends_on": [0]},
    ]
    return {"name": "toy", "version": 1, "quant": "int8",
            "input": {"tensor": "x", "tensors": ["x"],
                      "packed_inputs": [{"name": "x", "offset": 0, "size": K,
                                         "dtype": "i8", "byte_offset": 0}]},
            "output": {"tensors": ["z"], "tensor": None},
            "tensors": t, "ops": ops, "dispatches": [0, 1]}


def selftest() -> int:
    import copy
    bad = 0

    def check(label, cond, detail=""):
        nonlocal bad
        print("    %-58s %s%s" % (label, "PASS" if cond else "FAIL",
                                  "" if cond else "  " + detail))
        if not cond:
            bad += 1

    # the shard boundaries are the kernel's
    check("lm_head 32768x288 shards 2 x 16384", shards_for(32768, 288) == [(0, 16384), (16384, 16384)],
          str(shards_for(32768, 288)))
    check("16384x288 needs no split", shards_for(16384, 288) == [])
    check("288x288 needs no split", shards_for(288, 288) == [])
    check("1152x288 needs no split", shards_for(1152, 288) == [])
    check("288x1152 needs no split", shards_for(288, 1152) == [])
    check("every shard is inside the staging limit",
          all(w * 288 <= STAGE_LIMIT for _, w in shards_for(32768, 288)))
    check("every shard is a whole number of quads",
          all(w % NCH == 0 for _, w in shards_for(32768, 288)))
    check("shards tile N exactly",
          sum(w for _, w in shards_for(32768, 288)) == 32768)

    ir, plan = split(copy.deepcopy(_toy()))
    ops = ir["ops"]
    check("one op split", plan["ops_split"] == 1, str(plan["ops_split"]))
    check("dispatches 2 -> 4 (2 shards + 1 join + silu)",
          (plan["dispatches_before"], plan["dispatches_after"]) == (2, 4),
          str((plan["dispatches_before"], plan["dispatches_after"])))
    check("dispatch ids contiguous 0..3",
          [o["dispatch_id"] for o in ops] == [0, 1, 2, 3])
    check("graph output tensor `z` untouched", ir["output"]["tensors"] == ["z"])
    check("`y` still [1,1,32768] with its own scale",
          ir["tensors"]["y"]["shape"] == [1, 1, 32768]
          and ir["tensors"]["y"]["quant"]["scale"] == 0.25)
    check("shards carry the parent's scale",
          all(ir["tensors"]["y__s%d" % i]["quant"]["scale"] == 0.25 for i in (0, 1)))
    check("shards carry the parent's requantise",
          all(o["quant"]["output_multiplier"] == 3 and o["quant"]["output_shift"] == 1
              for o in ops if o["name"].startswith("lin__s")))
    join = [o for o in ops if o["op"] == "cat2_c1_s8"]
    check("one join, producing `y`", len(join) == 1 and join[0]["outputs"] == ["y"])
    check("join is a memcpy (all three scales equal)",
          join[0]["quant"]["scales_in"] == [0.25, 0.25]
          and join[0]["quant"]["scale_out"] == 0.25)
    check("join carries the IR's own cat2 schema",
          set(join[0]["shape"]) == {"N", "H", "W", "C_inputs", "C_total"}
          and join[0]["shape"]["C_inputs"] == [16384, 16384]
          and join[0]["shape"]["C_total"] == 32768,
          json.dumps(join[0]["shape"]))
    silu = [o for o in ops if o["op"] == "silu_s8"][0]
    check("the consumer now depends on the JOIN, not on a shard",
          silu["depends_on"] == [join[0]["dispatch_id"]],
          "%s vs join %d" % (silu["depends_on"], join[0]["dispatch_id"]))
    check("weight slices tile the rows exactly",
          [(s["row0"], s["rows"]) for s in plan["weight_slices"]]
          == [(0, 16384), (16384, 16384)])
    import numpy as _np
    wmap = {"W": _np.zeros((32768, 288), dtype=_np.int8),
            "B": _np.zeros((32768,), dtype=_np.int32),
            "other": _np.ones((3,), dtype=_np.int8)}
    mat = materialise(wmap, plan)
    check("shards materialise, parent DROPPED (no duplicate in the image)",
          set(mat) == {"W__s0", "W__s1", "B__s0", "B__s1", "other"}, str(sorted(mat)))
    check("unrelated weights survive untouched",
          _np.array_equal(mat["other"], wmap["other"]))
    check("bias is sliced beside the weight",
          all(s["dst_bias"] == s["dst_weight"].replace("W", "B")
              for s in plan["weight_slices"]))

    # a graph that needs no split comes back byte-identical
    narrow = _toy(N=1152)
    before = json.dumps(narrow, sort_keys=True)
    out2, plan2 = split(copy.deepcopy(narrow))
    check("a graph needing no split is unchanged",
          json.dumps(out2, sort_keys=True) == before and plan2["ops_split"] == 0)

    # the refusal
    weird = _toy()
    weird["tensors"]["y"]["shape"] = [1, 1, 999]
    try:
        split(weird)
        check("refuses an output whose last dim is not N", False)
    except SplitError:
        check("refuses an output whose last dim is not N", True)

    print("    %s" % ("selftest: PASS" if not bad else "selftest: %d FAILED" % bad))
    return 1 if bad else 0


def materialise(weights: dict, plan: dict) -> dict:
    """Apply `plan`'s row slices to a {name: ndarray} weight map, returning the new map.

    THE SLICES DEDUPE, AND THAT IS THE POINT.  Every unrolled step's `lm_head` names the SAME
    weight tensor -- the weights are shared across the 24 steps -- so all 24 split ops ask for
    the same two slices under the same two names.  Materialising them once is what keeps the
    runtime's image cache at one image per shard and `image_bytes` where it was."""
    out = dict(weights)
    want: dict[str, tuple[str, int, int]] = {}
    cover: dict[str, set] = {}
    for s in plan["weight_slices"]:
        for src, dst in ((s["src_weight"], s["dst_weight"]), (s["src_bias"], s["dst_bias"])):
            if not src:
                continue
            key = (src, s["row0"], s["rows"])
            prev = want.get(dst)
            if prev is not None and prev != key:
                raise SplitError("%r is asked for as %s rows [%d,%d) and also as %s rows "
                                 "[%d,%d)" % (dst, prev[0], prev[1], prev[1] + prev[2],
                                              src, s["row0"], s["row0"] + s["rows"]))
            want[dst] = key
            cover.setdefault(src, set()).add((s["row0"], s["rows"]))
    for src, rs in cover.items():
        if src not in weights:
            raise SplitError("no weight %r in the npz" % src)
        rows = sorted(rs)
        n = weights[src].shape[0]
        at = 0
        for r0, k in rows:
            if r0 != at:
                raise SplitError("%s: shards do not tile rows -- gap or overlap at %d"
                                 % (src, at))
            at += k
        if at != n:
            raise SplitError("%s has %d rows; its shards cover %d" % (src, n, at))
    for dst, (src, r0, k) in want.items():
        out[dst] = weights[src][r0:r0 + k]
    # DROP THE PARENT ONCE ITS SHARDS COVER IT.  Nothing references it after the split -- every
    # split op names a shard -- and generate_skeleton emits EVERY npz key as C, so leaving it
    # in puts lm_head's 9.4 MB of weights in the image TWICE.  Measured: weights.c 77.8 MB ->
    # 117.7 MB before this, which is ~16 baked utterances of the image budget under the
    # 0x8800_0000 arena ceiling (B99 step 4) spent on a duplicate.
    for src in cover:
        del out[src]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ir")
    ap.add_argument("--weights")
    ap.add_argument("--out")
    ap.add_argument("--out-weights")
    ap.add_argument("--report")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    for need in ("ir", "out"):
        if not getattr(a, need):
            ap.error("--%s is required" % need)
    ir = json.load(open(a.ir))
    new_ir, plan = split(ir)
    json.dump(new_ir, open(a.out, "w"))
    if a.weights:
        import numpy as np
        src = np.load(a.weights)
        w = {k: src[k] for k in src.files}
        nw = materialise(w, plan)
        if not a.out_weights:
            ap.error("--weights given without --out-weights")
        np.savez(a.out_weights, **nw)
    if a.report:
        json.dump(plan, open(a.report, "w"), indent=1)
    print("ir_lmsplit: %d op(s) split, dispatches %d -> %d, %d weight slice(s)"
          % (plan["ops_split"], plan["dispatches_before"], plan["dispatches_after"],
             len(plan["weight_slices"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
