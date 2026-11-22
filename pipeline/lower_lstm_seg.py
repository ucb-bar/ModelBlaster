#!/usr/bin/env python3
"""Per-segment quantization of one graph input that feeds an LSTM's first layer (int8 IR pass).

  lower_lstm_seg.py GRAPH.json WEIGHTS.npz IO.npz --model NAME [--lstm lstm.l0] [--input lowdim]
                    [--segments flow:0:2,rest:2:21] [--lo-digit flow] [--num-calibration 32] [--R 16]
                    [--cal-percentile 100]

Why. extract_graph quantizes a concat [a | b | x] that feeds an nn.LSTM with ONE per-tensor scale, and the LSTM's
W_ih with one per-tensor scale. When x is a raw state vector whose features span very different ranges (e.g. an
optical flow ~300 next to a quaternion ~1) that scale crushes a and b to zero, and the int8 weights of x's
columns are far too coarse for inputs that large. This pass takes x out of the concat and turns its contribution to
the gates into a per-tick int32 bias:

  * x is split into segments (contiguous feature ranges), each a separate int8 graph input with its own scale
    (max |x_seg| over the calibration samples / 127; quantized as extract_graph does: round half-even, clamp 127);
  * W_ih[:, x] is re-quantized from the float model per segment (s = max|W_seg| / 127); a segment named in
    --lo-digit also gets a low digit, the weight residual at s/254, run as a second int8 pass over the same input;
  * a new op `linear_seg_s32` computes, every tick,
        out[j] = (sum_p (W_p . x_p)[j] * M_p + 2^(R-1)) >> R                      (int64 products, int32 out)
    with M_p = rha(s_xp * s_wp / (s_in * s_w) * 2^R), i.e. in the unit of the LSTM's A_x accumulator;
  * the concat keeps the other inputs at scale_out = max of their scales; the LSTM's W_ih keeps the other columns,
    re-quantized per tensor from the float model; b_ih = rha((b_ih + b_hh) / (s_in * s_w)), b_hh = 0, and the
    LSTM op reads the per-tick bias as its second input (`bias_hh_input`: b_hh comes from that int32 tensor).

Everything is integer and exact. The LSTM cell (lstm_s8 or, after lower_lstm_int, lstm_q_s8) is unchanged: the
bias simply adds into A_x. io.npz's input is re-quantized from the model's float sample (the new input scales);
run lower_lstm_int afterwards to re-bake the golden. The float weights and calibration samples come from the model
module (`modelblaster.models.NAME`: get_model, get_calibration_samples, get_sample_input), as extract_graph uses.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys

import numpy as np


def _rha(v: float) -> int:
    return math.floor(v + 0.5) if v >= 0 else -math.floor(-v + 0.5)


def _qsym(w, s):
    return np.clip(np.rint(np.asarray(w, np.float64) / s), -127, 127)


def _lstm_module(model, op_name):
    """(nn.LSTM, layer) for an IR lstm op named '<module>.l<k>'."""
    import torch.nn as nn
    mod_name, _, layer = op_name.rpartition(".l")
    m = dict(model.named_modules())[mod_name]
    if not isinstance(m, nn.LSTM):
        raise SystemExit(f"lower_lstm_seg: {mod_name} is not an nn.LSTM")
    return m, int(layer)


def _annotate(ops):
    try:
        from modelblaster.pipeline.extract_graph import _annotate_dispatches
    except ImportError:
        sys.path.insert(0, "pipeline")
        from extract_graph import _annotate_dispatches  # type: ignore
    return _annotate_dispatches(ops)


def lower(ir, w, io, model_mod, lstm_name="lstm.l0", input_name="lowdim", segments=None, lo_digit=(),
          n_cal=32, R=16, pct=100.0, log=print):
    import torch
    model = model_mod.get_model()
    lstm, layer = _lstm_module(model, lstm_name)
    W = getattr(lstm, f"weight_ih_l{layer}").detach().double().numpy()
    has_bias = lstm.bias
    b = (getattr(lstm, f"bias_ih_l{layer}") + getattr(lstm, f"bias_hh_l{layer}")).detach().double().numpy() \
        if has_bias else np.zeros(W.shape[0])

    ops = ir["ops"]
    li = next(i for i, o in enumerate(ops) if o["name"] == lstm_name)
    lop = ops[li]
    cat_t = lop["inputs"][0]
    ci = next(i for i, o in enumerate(ops) if cat_t in o["outputs"])
    cop = ops[ci]
    if not cop["op"].startswith("cat") or input_name not in cop["inputs"]:
        raise SystemExit(f"lower_lstm_seg: {lstm_name}'s input is not a concat of {input_name}")
    in_names = [e["name"] for e in ir["input"]["packed_inputs"]]
    xi_graph = in_names.index(input_name)
    pos = cop["inputs"].index(input_name)
    if pos != len(cop["inputs"]) - 1:
        raise SystemExit("lower_lstm_seg: the split input must be the concat's last input")
    sizes = [int(np.prod(ir["tensors"][t]["shape"])) for t in cop["inputs"]]
    col0 = sum(sizes[:pos])
    nx = sizes[pos]
    segs = segments or [(input_name, 0, nx)]

    # calibration: per-segment max |x| over the model's calibration samples (the extract_graph source)
    cal = model_mod.get_calibration_samples(n_cal)
    X = np.array([np.asarray(s[xi_graph], np.float64).reshape(-1) for s in cal])
    # range = max |x| over the segment's features and samples (pct < 100: that percentile of |x| per feature, the
    # largest over the segment's features; values beyond it saturate at +-127)
    seg_scale = {n: float(np.percentile(np.abs(X[:, a:bb]), pct, axis=0).max()) / 127.0 for n, a, bb in segs}

    # concat of the remaining inputs
    keep = cop["inputs"][:pos]
    q = cop["quant"]
    s_in = max(q["scales_in"][:pos])
    W_keep = W[:, :col0]
    s_w = float(np.abs(W_keep).max()) / 127.0
    p_in = s_in * s_w
    cop["inputs"] = keep
    cop["op"] = f"cat{len(keep)}_c1_s8"
    q["scales_in"] = q["scales_in"][:pos]
    q["scale_out"] = s_in
    for key in ("n_inputs", "C_inputs"):
        if key in cop.get("shape", {}):
            cop["shape"][key] = cop["shape"][key][:pos]
    if "C_total" in cop.get("shape", {}):
        cop["shape"]["C_total"] = col0
    ir["tensors"][cat_t]["shape"] = [1, col0]
    ir["tensors"][cat_t]["quant"]["scale"] = s_in

    # the LSTM: other columns, folded bias, per-tick bias input
    lq = lop["quant"]
    w[lop["weight"]] = _qsym(W_keep, s_w).astype(np.int8)
    w[lop["bias"]] = np.rint(b / p_in).astype(np.int32)
    w[lop["bias_hh"]] = np.zeros(W.shape[0], np.int32)
    lq.update(scale_in=s_in, scale_w_ih=s_w, scale_b=p_in, has_bias=1)
    lop["shape"]["input_size"] = col0
    bias_t = f"{lstm_name}.bias_dyn"
    lop["inputs"] = [cat_t, bias_t]
    lop["bias_hh_input"] = 1

    # graph inputs: replace input_name by its segments (same bytes, same offsets)
    ent = ir["input"]["packed_inputs"]
    base = next(e for e in ent if e["name"] == input_name)
    new_ent = []
    for e in ent:
        if e["name"] != input_name:
            new_ent.append(e)
            continue
        for n, a, bb in segs:
            new_ent.append(dict(e, name=n, offset=e["offset"] + a, size=bb - a, byte_offset=e["byte_offset"] + a))
    ir["input"]["packed_inputs"] = new_ent
    ir["input"]["tensors"] = [e["name"] for e in new_ent]
    del ir["tensors"][input_name]
    for n, a, bb in segs:
        ir["tensors"][n] = {"shape": [1, bb - a], "dtype": "i8", "quant": {"scale": seg_scale[n], "zero_point": 0}}

    # linear_seg_s32: one pass per segment (+ a low-digit pass)
    passes = []
    for n, a, bb in segs:
        Wseg = W[:, col0 + a:col0 + bb]
        sw = float(np.abs(Wseg).max()) / 127.0
        hi = _qsym(Wseg, sw)
        key = f"{lstm_name}.seg_{n}.w"
        w[key] = hi.astype(np.int8)
        passes.append({"input": n, "weight": key, "M": _rha(seg_scale[n] * sw / p_in * 2.0 ** R),
                       "scale_w": sw, "K": bb - a})
        if n in lo_digit:
            slo = sw / 254.0
            key = f"{lstm_name}.seg_{n}.w_lo"
            w[key] = _qsym(Wseg - hi * sw, slo).astype(np.int8)
            passes.append({"input": n, "weight": key, "M": _rha(seg_scale[n] * slo / p_in * 2.0 ** R),
                           "scale_w": slo, "K": bb - a})
    N = W.shape[0]
    for p in passes:
        if not (0 < abs(p["M"]) < (1 << 31)):
            raise SystemExit(f"lower_lstm_seg: multiplier {p['M']} out of int32 range (lower --R)")
    seg_op = {"name": f"{lstm_name}.seg", "op": "linear_seg_s32", "inputs": [n for n, _, _ in segs],
              "outputs": [bias_t], "passes": passes, "shape": {"N": N, "n_passes": len(passes)},
              "quant": {"R": R, "unit": p_in}}
    ir["tensors"][bias_t] = {"shape": [1, N], "dtype": "i32", "quant": {"scale": p_in, "zero_point": 0}}
    ops.insert(li, seg_op)
    ir["dispatches"] = _annotate(ops)

    # bound check: |A_x| of the LSTM incl. the per-tick bias over all int8 inputs (fixed-function datapaths
    # such as 24-bit post units need it)
    bound = np.abs(w[lop["weight"]].astype(np.int64)).sum(1) * 128 + np.abs(w[lop["bias"]].astype(np.int64))
    for p in passes:
        bound = bound + (np.abs(w[p["weight"]].astype(np.int64)).sum(1) * 127 * abs(p["M"]) >> R) + 1
    seg_op["quant"]["ax_bound"] = int(bound.max())

    # io: re-quantize the float sample's segments (tick-0 input)
    smp = model_mod.get_sample_input()
    xs = np.asarray(smp[xi_graph], np.float64).reshape(-1)
    packed = io["input"].reshape(-1).astype(np.int8).copy()
    for n, a, bb in segs:
        t = torch.as_tensor(xs[a:bb] / seg_scale[n], dtype=torch.float64)
        packed[base["byte_offset"] + a:base["byte_offset"] + bb] = torch.clamp(torch.round(t), -127, 127).to(torch.int8).numpy()
    io["input"] = packed.reshape(io["input"].shape).astype(io["input"].dtype)
    log(f"[lower_lstm_seg] {lstm_name}: {input_name}[{nx}] -> segments " +
        ", ".join(f"{n}[{bb - a}] s={seg_scale[n]:.6g}" for n, a, bb in segs) +
        f"; concat {keep} at s={s_in:.6g}; W_ih[:, :{col0}] s={s_w:.6g}; linear_seg_s32 passes " +
        ", ".join(f"{p['input']}/{p['weight'].rsplit('.', 1)[-1]} M={p['M']}" for p in passes) +
        f" R={R}; |A_x| <= {seg_op['quant']['ax_bound']}")
    return ir, w, io


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("io")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lstm", default="lstm.l0")
    ap.add_argument("--input", default="lowdim")
    ap.add_argument("--segments", default="", help="name:start:end,... (feature ranges of --input)")
    ap.add_argument("--lo-digit", default="", help="comma list of segment names that get a low-digit pass")
    ap.add_argument("--num-calibration", type=int, default=32)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--cal-percentile", type=float, default=100.0)
    ap.add_argument("--mode", default="bias", choices=["bias", "digits"],
                    help="bias: linear_seg_s32 per-tick bias (default); digits: fold into W_ih (no new op)")
    a = ap.parse_args()
    mm = importlib.import_module(f"modelblaster.models.{a.model}")
    segs = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in a.segments.split(",") if s]
    ir = json.load(open(a.graph))
    w = dict(np.load(a.weights))
    io = dict(np.load(a.io))
    if a.mode == "digits":
        ir, w, io = lower_digits(ir, w, io, mm, a.lstm, a.input, a.num_calibration, a.cal_percentile)
    else:
        ir, w, io = lower(ir, w, io, mm, a.lstm, a.input, segs, [x for x in a.lo_digit.split(",") if x],
                          a.num_calibration, a.R, a.cal_percentile)
    json.dump(ir, open(a.graph, "w"), indent=2)
    np.savez(a.weights, **w)
    np.savez(a.io, **io)



def lower_digits(ir, w, io, model_mod, lstm_name="lstm.l0", input_name="lowdim", n_cal=32, pct=100.0, log=print):
    """--mode digits: the state's gate term folded into the LSTM's own int8 W_ih (no new op, no per-tick bias).

    Each state feature k is quantized by the host with its own scale s_k (max |x_k| over the calibration samples /
    127, or the pct percentile) to an int8 code q_k. Its exact contribution to A_x, in A_x's unit p = s_in * s_w, is
    G[j,k] * q_k with G = rha(W_ih[j,k] * s_k / p), an integer that may exceed int8; it is split into n_k digits
    g_1..g_n (|g| <= 127, same sign, sum = G) and the host feeds q_k n_k times. The graph input becomes `<input>_rep`
    (sum n_k bytes) and joins the concat as an identity input (its declared scale is the concat's output scale), so
    the LSTM input is [other inputs | q copies] and W_ih = [W_other (int8 at s_w) | digit columns]. Everything stays
    lstm_s8 / lstm_q_s8: any backend, and any engine that runs a concat plus an int8 matvec, runs it unchanged.
    The host encoding (feature, scale per byte) is recorded in the input tensor's `host_encode`."""
    import torch
    model = model_mod.get_model()
    lstm, layer = _lstm_module(model, lstm_name)
    W = getattr(lstm, f"weight_ih_l{layer}").detach().double().numpy()
    b = (getattr(lstm, f"bias_ih_l{layer}") + getattr(lstm, f"bias_hh_l{layer}")).detach().double().numpy() \
        if lstm.bias else np.zeros(W.shape[0])
    ops = ir["ops"]
    li = next(i for i, o in enumerate(ops) if o["name"] == lstm_name)
    lop = ops[li]
    cat_t = lop["inputs"][0]
    cop = next(o for o in ops if cat_t in o["outputs"])
    pos = cop["inputs"].index(input_name)
    if pos != len(cop["inputs"]) - 1:
        raise SystemExit("lower_lstm_seg: the state input must be the concat's last input")
    sizes = [int(np.prod(ir["tensors"][t]["shape"])) for t in cop["inputs"]]
    col0, nx = sum(sizes[:pos]), sizes[pos]
    in_names = [e["name"] for e in ir["input"]["packed_inputs"]]
    xi_graph = in_names.index(input_name)
    cal = model_mod.get_calibration_samples(n_cal)
    X = np.abs(np.array([np.asarray(s[xi_graph], np.float64).reshape(-1) for s in cal]))
    rng = np.percentile(X, pct, axis=0)
    s_k = np.where(rng > 0, rng / 127.0, 0.0)          # constant-zero features get no columns

    q = cop["quant"]
    s_in = max(q["scales_in"][:pos])
    s_w = float(np.abs(W[:, :col0]).max()) / 127.0
    p = s_in * s_w
    cols, enc = [], []
    for k in range(nx):
        if s_k[k] == 0:
            continue
        G = np.rint(W[:, col0 + k] * s_k[k] / p).astype(np.int64)
        n = max(1, int(-(-np.abs(G).max() // 127)))
        base, rem = np.divmod(np.abs(G), n)            # |G| = n*base + rem: rem digits of base+1, the rest base
        for d in range(n):
            cols.append(np.sign(G) * (base + (d < rem)))
            enc.append([k, float(s_k[k])])
    D = np.stack(cols, 1)
    assert np.abs(D).max() <= 127
    rep = f"{input_name}_rep"
    n_rep = D.shape[1]
    w[lop["weight"]] = np.concatenate([np.clip(np.rint(W[:, :col0] / s_w), -127, 127), D], 1).astype(np.int8)
    w[lop["bias"]] = np.rint(b / p).astype(np.int32)
    w[lop["bias_hh"]] = np.zeros(W.shape[0], np.int32)
    lop["quant"].update(scale_in=s_in, scale_w_ih=s_w, scale_b=p, has_bias=1)
    lop["shape"]["input_size"] = col0 + n_rep
    cop["inputs"][pos] = rep
    q["scales_in"][pos] = s_in
    q["scale_out"] = s_in
    for key in ("n_inputs", "C_inputs"):
        if key in cop.get("shape", {}):
            cop["shape"][key][pos] = n_rep
    if "C_total" in cop.get("shape", {}):
        cop["shape"]["C_total"] = col0 + n_rep
    ir["tensors"][cat_t]["shape"] = [1, col0 + n_rep]
    ir["tensors"][cat_t]["quant"]["scale"] = s_in
    ent = ir["input"]["packed_inputs"]
    e0 = next(e for e in ent if e["name"] == input_name)
    delta = n_rep - e0["size"]
    for e in ent:
        if e["byte_offset"] > e0["byte_offset"]:
            e["byte_offset"] += delta; e["offset"] += delta
    e0.update(name=rep, size=n_rep)
    ir["input"]["tensors"] = [e["name"] for e in ent]
    ir["input"]["packed_bytes"] = ir["input"].get("packed_bytes", 0) + delta
    del ir["tensors"][input_name]
    ir["tensors"][rep] = {"shape": [1, n_rep], "dtype": "i8", "quant": {"scale": s_in, "zero_point": 0},
                          "host_encode": {"source": input_name, "bytes": enc,
                                          "rule": "byte i = clamp(round_half_even(x[feature_i] / scale_i), -127, 127)"}}
    bound = np.abs(w[lop["weight"]].astype(np.int64)).sum(1) * 128 + np.abs(w[lop["bias"]].astype(np.int64))
    lop["quant"]["ax_bound"] = int(bound.max())
    # io: re-encode the float sample
    smp = model_mod.get_sample_input()
    xs = np.asarray(smp[xi_graph], np.float64).reshape(-1)
    old = io["input"].reshape(-1).astype(np.int8)
    off = e0["byte_offset"]
    codes = np.array([np.clip(torch.round(torch.tensor(xs[k] / s)).item(), -127, 127) for k, s in enc], np.int8)
    io["input"] = np.concatenate([old[:off], codes, old[off + nx:]]).reshape(1, -1).astype(io["input"].dtype)
    log(f"[lower_lstm_seg] {lstm_name} (digits): {input_name}[{nx}] -> {rep}[{n_rep}] (copies per feature "
        f"{np.bincount([k for k, _ in enc], minlength=nx).tolist()}); W_ih IS {col0} -> {col0 + n_rep}; "
        f"concat scale {s_in:.6g}, W_ih s={s_w:.6g}; |A_x| <= {lop['quant']['ax_bound']}")
    return ir, w, io


if __name__ == "__main__":
    main()
