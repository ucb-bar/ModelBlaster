#!/usr/bin/env python3
"""Lower lstm_s8 to the all-integer cell lstm_q_s8, and re-bake the golden.

  lower_lstm_int.py GRAPH.json WEIGHTS.npz IO.npz [--conv-numerics reference|gemmini]
                    [--seq-inputs SEQ_IN.npz --seq-out SEQ.npz]

lstm_s8's reference cell (kernel_lstm_s8) dequantizes and evaluates the gates, exp and tanh in double. That is
exact against the golden only because both sides use IEEE double, it costs ~350 M cycles per FusedSensorNet tick
on a soft-float core, and no fixed-function datapath can reproduce it. This pass replaces it with the integer
cell of pipeline/lstm_q.py (same int8 weights, biases and h/c state buffers):

  * every lstm_s8 op becomes lstm_q_s8 with quant {Mx, Mh, S} added (the float scales stay in the record);
  * IO.npz's `output` is recomputed from its input with the new numerics (the old one is kept as
    `output_lstm_s8_reference`), so the harness's baked golden is the golden of what the image runs;
  * with --seq-inputs (int8 packed inputs [T, packed_bytes]), SEQ.npz gets the same inputs and the golden output
    of every tick, run from reset with the state carried (the multi-tick check of a stateful model).

The rest of the graph is simulated with the numerics of ModelBlaster's C kernels:
  conv2d_s8   reference: two-stage Q0.31 requantize (scalar reference, Gemmini im2col_full_C); or
              gemmini:   Gemmini tiled_conv ACC_SCALE fold, sat8(((acc+1)*round(M/2^s) + 2^30) >> 31)
  linear_s8   two-stage (every kernel); cat/BN/leaky_relu in float32 with roundf (half away from zero), as the
  C reference kernels do; maxpool, avgpool (round half away, C integer division), view.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

try:
    from modelblaster.pipeline import lstm_q
except ImportError:  # run as a script from the repository root
    sys.path.insert(0, "pipeline")
    import lstm_q  # type: ignore


def _rha(v):
    return np.where(v >= 0, np.floor(v + 0.5), -np.floor(-v + 0.5))


def _req2(acc, m, s):
    p = (acc.astype(np.int64) * m + (1 << 30)) >> 31
    return (p + (1 << (s - 1))) >> s if s > 0 else (p << (-s) if s < 0 else p)


def _reqg(acc, m, s):
    sq = (m + (1 << (s - 1))) >> s if s > 0 else m
    return ((acc.astype(np.int64) + 1) * sq + (1 << 30)) >> 31


def _conv(x, w, b, sh, q, numerics):
    IC, IH, IW = sh["IC"], sh["IH"], sh["IW"]
    x = x.reshape(IC, IH, IW).astype(np.int64)
    KH, KW, SH, SW, PH, PW, OH, OW = (sh[k] for k in ("KH", "KW", "SH", "SW", "PH", "PW", "OH", "OW"))
    xp = np.zeros((IC, IH + 2 * PH, IW + 2 * PW), np.int64)
    xp[:, PH:PH + IH, PW:PW + IW] = x
    acc = np.zeros((w.shape[0], OH, OW), np.int64) + b.astype(np.int64)[:, None, None]
    for kh in range(KH):
        for kw in range(KW):
            win = xp[:, kh:kh + SH * (OH - 1) + 1:SH, kw:kw + SW * (OW - 1) + 1:SW]
            acc += np.einsum("oi,ihw->ohw", w[:, :, kh, kw].astype(np.int64), win)
    v = (_reqg if numerics == "gemmini" else _req2)(acc, q["output_multiplier"], q["output_shift"])
    return np.clip(np.clip(v, -128, 127), q["activation_min"], q["activation_max"]).astype(np.int8)


def simulate(ir, w, inputs, state, conv_numerics="reference"):
    """One tick. inputs: name -> int8 array; state: dict carried across ticks (mutated). Returns tensors."""
    T = dict(inputs)
    for op in ir["ops"]:
        k, q, sh = op["op"], op.get("quant", {}), op.get("shape", {})
        x = T[op["inputs"][0]]
        o = op["outputs"][0]
        if k == "conv2d_s8":
            T[o] = _conv(x, w[op["weight"]], w[op["bias"]], sh, q, conv_numerics)
        elif k == "linear_s8":
            acc = w[op["weight"]].reshape(sh["N"], sh["K"]).astype(np.int64) @ x.reshape(-1).astype(np.int64)
            v = _req2(acc + w[op["bias"]].astype(np.int64), q["output_multiplier"], q["output_shift"])
            T[o] = np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif k == "view":
            T[o] = x.reshape(-1)
        elif k in ("cat2_c1_s8", "cat3_c1_s8", "cat4_c1_s8"):
            parts = [np.clip(_rha((T[n].reshape(-1).astype(np.float32) * np.float32(si)).astype(np.float32)
                                  / np.float32(q["scale_out"])), q["activation_min"], q["activation_max"])
                     for n, si in zip(op["inputs"], q["scales_in"])]
            T[o] = np.concatenate(parts).astype(np.int8)
        elif k == "batchnorm2d_s8":
            fv = x.reshape(sh["C"], -1).astype(np.float32) * np.float32(q["scale_in"])
            y = (w[op["weight"]].astype(np.float32)[:, None] * fv + w[op["bias"]].astype(np.float32)[:, None])
            v = _rha((y.astype(np.float32) / np.float32(q["scale_out"])).astype(np.float32))
            T[o] = np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif k == "leaky_relu_s8":
            fv = x.astype(np.float32) * np.float32(q["scale_in"])
            y = np.where(fv > 0, fv, (np.float32(q.get("negative_slope", 0.01)) * fv).astype(np.float32))
            v = _rha((y.astype(np.float32) / np.float32(q["scale_out"])).astype(np.float32))
            T[o] = np.clip(v, q["activation_min"], q["activation_max"]).astype(np.int8)
        elif k in ("maxpool2d_s8", "avgpool2d_s8"):
            if sh.get("PH", 0) or sh.get("PW", 0):
                raise NotImplementedError(f"{op['name']}: padded pooling")
            xi = x.reshape(sh["C"], sh["IH"], sh["IW"]).astype(np.int64)
            wins = [xi[:, kh:kh + sh["SH"] * (sh["OH"] - 1) + 1:sh["SH"], kw:kw + sh["SW"] * (sh["OW"] - 1) + 1:sh["SW"]]
                    for kh in range(sh["KH"]) for kw in range(sh["KW"])]
            if k == "maxpool2d_s8":
                T[o] = np.max(wins, axis=0).astype(np.int8)
            else:
                s_, d = np.sum(wins, axis=0), sh["KH"] * sh["KW"]
                T[o] = np.clip(np.where(s_ >= 0, (s_ + d // 2) // d, -((-s_ + d // 2) // d)), -128, 127).astype(np.int8)
        elif k in ("lstm_s8", "lstm_q_s8"):
            H, IS = sh["hidden_size"], sh["input_size"]
            hn, cn = op["state"]
            h = state.get(hn, np.zeros(H, np.int8))
            c = state.get(cn, np.zeros(H, np.int8))
            wi, wh, bi, bh = w[op["weight"]], w[op["weight_hh"]], w[op["bias"]], w[op["bias_hh"]]
            xv = x.reshape(-1)[:IS]
            if k == "lstm_q_s8":
                h2, c2 = lstm_q.step(xv, wi, wh, bi, bh, h, c, q["Mx"], q["Mh"], q["S"], q["has_bias"])
            else:
                h2, c2 = _lstm_s8_ref(xv, wi, wh, bi, bh, h, c, q)
            state[hn], state[cn] = h2, c2
            T[o] = h2
        else:
            raise NotImplementedError(f"lower_lstm_int simulator: op {k}")
    return T


def _lstm_s8_ref(x, wi, wh, bi, bh, h, c, q):
    """kernel_lstm_s8 (float64 cell, float32 scale arguments)."""
    f = lambda k: float(np.float32(q[k]))  # noqa: E731
    H = h.size
    g = ((x * f("scale_in"))[None, :] * (wi * f("scale_w_ih"))).sum(1) \
        + ((h * f("scale_h"))[None, :] * (wh * f("scale_w_hh"))).sum(1)
    if q["has_bias"]:
        g = g + bi * f("scale_b") + bh * f("scale_b")
    sg = lambda v: 1.0 / (1.0 + np.exp(-v))  # noqa: E731
    cn = sg(g[H:2 * H]) * (c * f("scale_c")) + sg(g[:H]) * np.tanh(g[2 * H:3 * H])
    hn = sg(g[3 * H:]) * np.tanh(cn)
    return (np.clip(_rha(hn / f("scale_h")), -128, 127).astype(np.int8),
            np.clip(_rha(cn / f("scale_c")), -128, 127).astype(np.int8))


def unpack_inputs(ir, packed):
    ent = ir["input"].get("packed_inputs")
    if not ent:
        return {ir["input"]["tensor"]: packed.astype(np.int8)}
    return {e["name"]: packed[e["byte_offset"]:e["byte_offset"] + e["size"]].astype(np.int8) for e in ent}


def run_sequence(ir, w, packed_seq, conv_numerics):
    state, outs, finals = {}, [], []
    out_names = ir["output"]["tensors"]
    for p in packed_seq:
        T = simulate(ir, w, unpack_inputs(ir, p), state, conv_numerics)
        outs.append(np.concatenate([T[n].reshape(-1) for n in out_names]).astype(np.int8))
    return np.array(outs), state


def lower(ir, log=print):
    n = 0
    for op in ir["ops"]:
        if op["op"] != "lstm_s8":
            continue
        mx, mh, S = lstm_q.layer_consts(op["quant"])
        op["op"] = "lstm_q_s8"
        op["quant"].update(Mx=mx, Mh=mh, S=S)
        log(f"[lower_lstm_int] {op['name']}: lstm_s8 -> lstm_q_s8  Mx={mx} Mh={mh} S={S}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("io")
    ap.add_argument("--conv-numerics", default="reference", choices=["reference", "gemmini"])
    ap.add_argument("--seq-inputs", default=None, help="npz with `inputs` int8 [T, packed_bytes]")
    ap.add_argument("--seq-out", default=None)
    ap.add_argument("--no-lower", action="store_true",
                    help="keep lstm_s8 (re-bake the golden / write the sequence for the float cell)")
    a = ap.parse_args()
    ir = json.load(open(a.graph))
    w = dict(np.load(a.weights))
    io = dict(np.load(a.io))
    if not a.no_lower:
        if lower(ir) == 0:
            sys.exit("lower_lstm_int: no lstm_s8 op in the graph")
        json.dump(ir, open(a.graph, "w"), indent=2)
    packed0 = io["input"].reshape(-1).astype(np.int8)
    out0, _ = run_sequence(ir, w, [packed0], a.conv_numerics)
    if "output_lstm_s8_reference" not in io:
        io["output_lstm_s8_reference"] = io["output"]
    print(f"[lower_lstm_int] baked golden: {io['output'].astype(np.int8).tolist()} (before) -> "
          f"{out0[0].tolist()} ({'lstm_q_s8' if not a.no_lower else 'lstm_s8'}, conv {a.conv_numerics})")
    io["output"] = out0[0].astype(io["output"].dtype)
    np.savez(a.io, **io)
    if a.seq_inputs:
        seq = np.load(a.seq_inputs)["inputs"].astype(np.int8)
        if not np.array_equal(seq[0], packed0):
            print("[lower_lstm_int] note: sequence tick 0 is not the baked input")
        outs, state = run_sequence(ir, w, list(seq), a.conv_numerics)
        np.savez(a.seq_out, inputs=seq, outputs=outs,
                 **{f"final_{k}": v for k, v in state.items()})
        print(f"[lower_lstm_int] sequence: {seq.shape[0]} ticks -> {a.seq_out}")


if __name__ == "__main__":
    main()
