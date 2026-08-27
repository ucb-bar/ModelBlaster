#!/usr/bin/env python3
"""Measure int8 *model* quality against the fp32 model, per op family.

This answers a different question from the harness's bit-exact check, and the
two are routinely confused. The harness asks: does the device compute what
ModelBlaster's int8 reference says it should? That is kernel correctness, and it
is either exact or a bug. This asks: does the int8 *model* still agree with the
fp32 model it was quantized from? That is quantization quality, and it is never
exact -- the question is whether the error is acceptable.

Conflating them means either shipping a broken kernel because "int8 is lossy
anyway", or rejecting a correct kernel because PTQ moved the answer.

The immediate motivation is RoPE: rotary embedding quantized to int8 is a
reasonable thing to be suspicious of, because the sin/cos argument is a
position-derived angle and an int8 grid over a wide angular range is coarse.
Rather than assert either way, measure it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-dir", required=True,
                    help="directory holding io.npz from extract_graph --quant int8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mod = __import__(f"models.{a.model}", fromlist=["get_model"])
    net = mod.get_model()

    io_np = np.load(os.path.join(a.gen_dir, "io.npz"))
    x_q = io_np["input"]
    y_q = io_np["output"]
    meta = json.load(open(os.path.join(a.gen_dir, "graph.json")))
    # graph.json names the input/output tensors; scales live in the tensors
    # table keyed by those names.
    tensors = meta.get("tensors", {})
    in_name = (meta.get("input") or {}).get("tensor")
    out_name = (meta.get("output") or {}).get("tensor")
    in_scale = (tensors.get(in_name, {}).get("quant", {}) or {}).get("scale")
    out_scale = (tensors.get(out_name, {}).get("quant", {}) or {}).get("scale")
    if in_scale is None or out_scale is None:
        print("could not read input/output scales from graph.json",
              file=sys.stderr)
        return 1

    # Feed the fp32 model the DEQUANTIZED int8 input, not the original float:
    # otherwise the comparison also folds in input-quantization error, which
    # belongs to the input pipeline rather than to the model.
    with torch.no_grad():
        x = torch.from_numpy(
            (x_q.astype(np.float32) * np.float32(in_scale))
        ).reshape(mod.get_sample_input().shape)
        y_ref = net(x).detach().cpu().numpy().reshape(-1).astype(np.float64)

    y_int8 = y_q.astype(np.float64).reshape(-1) * float(out_scale)
    n = min(len(y_ref), len(y_int8))
    y_ref, y_int8 = y_ref[:n], y_int8[:n]

    err = y_int8 - y_ref
    denom = max(float(np.abs(y_ref).max()), 1e-12)
    rmse = float(np.sqrt((err ** 2).mean()))
    res = {
        "model": a.model, "n": n,
        "max_abs_err": float(np.abs(err).max()),
        "rmse": rmse,
        "range_of_reference": denom,
        "max_err_as_pct_of_range": 100.0 * float(np.abs(err).max()) / denom,
        "rmse_as_pct_of_range": 100.0 * rmse / denom,
        "cosine_similarity": float(
            np.dot(y_ref, y_int8)
            / max(np.linalg.norm(y_ref) * np.linalg.norm(y_int8), 1e-12)),
        "output_lsb": float(out_scale),
        "max_err_in_output_lsb": float(np.abs(err).max()) / float(out_scale),
    }
    print(f"int8 model vs fp32 model -- {a.model} ({n} outputs)")
    print(f"  output LSB                 : {res['output_lsb']:.6g}")
    print(f"  max |err|                  : {res['max_abs_err']:.6g}"
          f"  ({res['max_err_in_output_lsb']:.2f} LSB)")
    print(f"  RMSE                       : {res['rmse']:.6g}")
    print(f"  max |err| as % of range    : {res['max_err_as_pct_of_range']:.2f}%")
    print(f"  RMSE as % of range         : {res['rmse_as_pct_of_range']:.2f}%")
    print(f"  cosine similarity          : {res['cosine_similarity']:.6f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
