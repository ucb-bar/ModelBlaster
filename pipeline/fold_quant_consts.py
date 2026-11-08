#!/usr/bin/env python3
"""Fold per-frame float requantization constants into build-time integers.

  fold_quant_consts.py GRAPH.json WEIGHTS.npz [--ops batchnorm2d_s8,add_s8]
                       [--out-graph G] [--out-weights W] [--require-exact]

Two int8 ops recompute their requantization constants in float on every call:

  batchnorm2d_s8  per channel: cs = scale[c]*scale_in/scale_out, cb = bias[c]/scale_out,
                  then Mc = lrintf(cs*2^S), Bc = lrintf(cb*2^S) for a per-channel S
                  (kernels/rvv/rvv_batchnorm2d_s8_direct.c).
  add_s8          ma = lrintf(scale_a/scale_out*2^S), mb = lrintf(scale_b/scale_out*2^S)
                  (kernels/rvv/rvv_add_s8_rvv_frm_rmm.c, gemmini_q31_rvv_add_s8_rvv.c).

On a core built with CONFIG_FPU=n every one of those float operations is a soft-float
call, and the BN prologue alone is ~2.2k cycles per channel per frame. The values are
compile-time constants, so this pass computes them once, here, and rewrites the ops:

  batchnorm2d_s8 -> batchnorm2d_q_s8   weights <bn>.fold_m (int32), <bn>.fold_bp (int32,
                                       Bc + 2^(S-1)) and <bn>.fold_s (int16)
  add_s8         -> add_q_s8           quant {ma, mb, shift}

The integer semantics of the new ops (reference_kernels.BATCHNORM2D_Q_S8 / ADD_Q_S8) are
exactly what the curated integer kernels above compute today: same constants, same
round-half-away, same clamp. That is what those kernels run, so a folded graph is
bit-identical to the unfolded one on the integer backends by construction.

The constants are evaluated with the SAME float32 expressions, in the same order, as the
C kernels: every intermediate is a numpy float32 (IEEE single, round to nearest even,
which is what both the hardware FPU and libgcc's soft-float give), lrintf is np.rint
(ties to even) and the float literals are the ones generate_skeleton._f32 prints (9
significant digits, which round-trips float32 exactly).

Checks, per folded op:
  * int32 headroom: 128*|Mc| + |Bc| + 2^(S-1) < 2^31 (BN), 128*(|ma|+|mb|) + 2^(S-1) < 2^31
    (add). A constant that fails keeps the op unfolded (and is reported).
  * exhaustive comparison over the whole int8 input domain (256 values per BN channel,
    65,536 operand pairs per add) of the folded integer result against the op's FLOAT
    reference (reference_kernels BATCHNORM2D_S8 / ADD_S8). Mismatches are reported;
    --require-exact makes any mismatch fatal. The integer kernels this replaces carry
    the same drift, so the default is to report.

The pass is opt-in: a build that does not run it is unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

FOLDABLE = ("batchnorm2d_s8", "add_s8")
F32 = np.float32


def _f32(x) -> np.float32:
    """The float32 the C compiler sees for generate_skeleton._f32(x)."""
    return F32(x)


def _lrintf(x: np.float32) -> int:
    return int(np.rint(F32(x)))


def _round_away(acc: np.ndarray, S: int) -> np.ndarray:
    """sign(acc) * ((|acc| + 2^(S-1)) >> S), exact in int64."""
    rnd = 1 << (S - 1)
    a = np.abs(acc)
    return np.where(acc >= 0, (a + rnd) >> S, -((a + rnd) >> S))


def bn_constants(scale, bias, scale_in, scale_out):
    """(M, B, S) per channel, mirroring rvv_batchnorm2d_s8_direct.c line for line."""
    scale = np.asarray(scale, F32)
    bias = np.asarray(bias, F32)
    si, so = _f32(scale_in), _f32(scale_out)
    inv_so = F32(1.0) / so
    M, B, S = [], [], []
    for c in range(scale.size):
        cs = (scale[c] * si) * inv_so
        cb = bias[c] * inv_so
        cs_abs, cb_abs = abs(cs), abs(cb)
        s = 24
        while s > 1 and (cs_abs * F32(1 << s) >= F32(4194304.0)
                         or cb_abs * F32(1 << s) >= F32(1073741824.0)):
            s -= 1
        M.append(_lrintf(cs * F32(1 << s)))
        B.append(_lrintf(cb * F32(1 << s)))
        S.append(s)
    return np.array(M, np.int64), np.array(B, np.int64), np.array(S, np.int64)


def add_constants(scale_a, scale_b, scale_out):
    """(ma, mb, S), mirroring rvv_add_s8_rvv_frm_rmm.c line for line."""
    sa, sb, so = _f32(scale_a), _f32(scale_b), _f32(scale_out)
    a_ratio = sa / so
    b_ratio = sb / so
    mx = max(abs(a_ratio), abs(b_ratio))
    s = 24
    while s > 1 and mx * F32(1 << s) >= F32(8388608.0):
        s -= 1
    return _lrintf(a_ratio * F32(1 << s)), _lrintf(b_ratio * F32(1 << s)), s


def _clamp(v, lo, hi):
    lo = max(-128, min(127, lo))
    hi = max(-128, min(127, hi))
    return np.minimum(np.maximum(v, lo), hi)


def _float_round(y: np.ndarray) -> np.ndarray:
    """C roundf on float32 values: ties away from zero."""
    y = y.astype(np.float64)          # exact widening; roundf rounds the float32 value
    return np.where(y >= 0, np.floor(y + 0.5), -np.floor(-y + 0.5)).astype(np.int64)


def fold_bn(op, weights, require_exact, log):
    q = op["quant"]
    scale = weights[op["weight"]]
    bias = weights[op["bias"]]
    M, B, S = bn_constants(scale, bias, q["scale_in"], q["scale_out"])
    lo, hi = q["activation_min"], q["activation_max"]
    x = np.arange(-128, 128, dtype=np.int64)
    mism = 0
    for c in range(M.size):
        worst = 128 * abs(int(M[c])) + abs(int(B[c])) + (1 << (int(S[c]) - 1))
        if worst >= (1 << 31):
            log(f"  {op['name']}: channel {c} would overflow int32 ({worst}); not folded")
            return None
        got = _clamp(_round_away(x * M[c] + B[c], int(S[c])), lo, hi)
        # BATCHNORM2D_S8.reference_impl: fv = x*scale_in; y = s*fv + b; roundf(y/scale_out)
        fv = x.astype(F32) * _f32(q["scale_in"])
        y = (F32(scale[c]) * fv + F32(bias[c])) / _f32(q["scale_out"])
        ref = _clamp(_float_round(y.astype(F32)), lo, hi)
        mism += int((got != ref).sum())
    n = 256 * M.size
    log(f"  {op['name']}: folded {M.size} channels, S in [{S.min()}, {S.max()}]; "
        f"vs float reference: {n - mism}/{n} input values identical")
    if mism and require_exact:
        sys.exit(f"[fold_quant_consts] {op['name']}: {mism} values differ from the float "
                 f"reference and --require-exact was given")
    base = op["name"]
    keys = {"m": f"{base}.fold_m", "bp": f"{base}.fold_bp", "s": f"{base}.fold_s"}
    rnd = np.left_shift(np.int64(1), S - 1)
    new_w = {keys["m"]: M.astype(np.int32),
             keys["bp"]: (B + rnd).astype(np.int32),
             keys["s"]: S.astype(np.int16)}
    new_op = dict(op)
    new_op["op"] = "batchnorm2d_q_s8"
    new_op.pop("weight", None)
    new_op.pop("bias", None)
    new_op["fold_weights"] = keys
    new_op["quant"] = {"activation_min": lo, "activation_max": hi,
                       "folded_from": {"op": "batchnorm2d_s8", "weight": op["weight"],
                                       "bias": op["bias"], "scale_in": q["scale_in"],
                                       "scale_out": q["scale_out"],
                                       "float_reference_mismatches": mism}}
    return new_op, new_w, (op["weight"], op["bias"])


def fold_add(op, require_exact, log):
    q = op["quant"]
    ma, mb, S = add_constants(q["scale_a"], q["scale_b"], q["scale_out"])
    worst = 128 * (abs(ma) + abs(mb)) + (1 << (S - 1))
    if worst >= (1 << 31):
        log(f"  {op['name']}: would overflow int32 ({worst}); not folded")
        return None
    lo, hi = q["activation_min"], q["activation_max"]
    a = np.repeat(np.arange(-128, 128, dtype=np.int64), 256)
    b = np.tile(np.arange(-128, 128, dtype=np.int64), 256)
    got = _clamp(_round_away(a * ma + b * mb, S), lo, hi)
    # ADD_S8.reference_impl: fa = a*sa; fb = b*sb; roundf((fa + fb)/so)
    fa = a.astype(F32) * _f32(q["scale_a"])
    fb = b.astype(F32) * _f32(q["scale_b"])
    ref = _clamp(_float_round(((fa + fb) / _f32(q["scale_out"])).astype(F32)), lo, hi)
    mism = int((got != ref).sum())
    log(f"  {op['name']}: ma={ma} mb={mb} S={S}; vs float reference: "
        f"{65536 - mism}/65536 operand pairs identical")
    if mism and require_exact:
        sys.exit(f"[fold_quant_consts] {op['name']}: {mism} pairs differ from the float "
                 f"reference and --require-exact was given")
    new_op = dict(op)
    new_op["op"] = "add_q_s8"
    new_op["quant"] = {"ma": ma, "mb": mb, "shift": S,
                       "activation_min": lo, "activation_max": hi,
                       "folded_from": {"op": "add_s8", "scale_a": q["scale_a"],
                                       "scale_b": q["scale_b"], "scale_out": q["scale_out"],
                                       "float_reference_mismatches": mism}}
    return new_op


def fold(ir, weights, ops=FOLDABLE, require_exact=False, log=print):
    weights = dict(weights)
    out_ops, dropped, n = [], set(), {"batchnorm2d_s8": 0, "add_s8": 0}
    for op in ir["ops"]:
        kind = op.get("op")
        if kind == "batchnorm2d_s8" and kind in ops:
            r = fold_bn(op, weights, require_exact, log)
            if r is not None:
                new_op, new_w, old = r
                weights.update(new_w)
                dropped.update(old)
                out_ops.append(new_op)
                n[kind] += 1
                continue
        elif kind == "add_s8" and kind in ops:
            r = fold_add(op, require_exact, log)
            if r is not None:
                out_ops.append(r)
                n[kind] += 1
                continue
        out_ops.append(op)
    # Drop float arrays nothing reads any more, so weights.c holds only live data.
    still = set()
    for op in out_ops:
        for k in ("weight", "bias"):
            if op.get(k):
                still.add(op[k])
        for s in op.get("sub_ops") or []:
            for k in ("weight", "bias"):
                if s.get(k):
                    still.add(s[k])
    for k in dropped - still:
        weights.pop(k, None)
    ir = dict(ir)
    ir["ops"] = out_ops
    ir.setdefault("_rewrite", []).append(
        {"pass": "fold_quant_consts", "folded": n, "require_exact": require_exact})
    return ir, weights, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("--ops", default=",".join(FOLDABLE),
                    help=f"comma list of op kinds to fold (default {','.join(FOLDABLE)})")
    ap.add_argument("--out-graph", default=None, help="default: overwrite GRAPH")
    ap.add_argument("--out-weights", default=None, help="default: overwrite WEIGHTS")
    ap.add_argument("--require-exact", action="store_true",
                    help="fail if a folded op differs from its float reference anywhere "
                         "in the int8 input domain")
    a = ap.parse_args()
    ops = tuple(o.strip() for o in a.ops.split(",") if o.strip())
    bad = set(ops) - set(FOLDABLE)
    if bad:
        sys.exit(f"[fold_quant_consts] cannot fold {sorted(bad)}; foldable: {FOLDABLE}")
    ir = json.load(open(a.graph))
    w = dict(np.load(a.weights))
    print(f"[fold_quant_consts] {a.graph}")
    ir, w, n = fold(ir, w, ops, a.require_exact)
    print(f"[fold_quant_consts] folded {n}")
    json.dump(ir, open(a.out_graph or a.graph, "w"), indent=1)
    np.savez(a.out_weights or a.weights, **w)


if __name__ == "__main__":
    main()
