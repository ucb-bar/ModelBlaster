#!/usr/bin/env python3
"""Replace an integer-closed IR's constants with externally supplied ones, verbatim.

  apply_qparams.py GRAPH.json WEIGHTS.npz PATCH.json PATCH.npz --out-dir DIR

ModelBlaster derives every multiplier, shift and table from its own calibration (integer_close.py). When an external
reference fixes the quantization instead (an integer golden), its constants must be used exactly, never recomputed
from a float scale. This pass imports them and refuses anything that does not line up.

PATCH.json:
  {"ops": {"<ir op name>": {"op": "<expected kind>",             # refuse if the op's kind differs
                            "arrays": {"<field>": "<npz key>"},   # weight / bias / quant *_key fields: replace
                            "quant": {"<field>": <int>}}},        #   the array they name; set scalar quant fields
   "tensors": {"<ir tensor>": {"scale": <float>}}}                # metadata (no kernel reads it)

<field> in "arrays" is "weight", "bias", or a quant field holding a weight key ("table_key", "ma_key",
"output_multiplier_per_oc_key", ...). The new array must have the old array's shape (a reshape of the same size is
refused too) and an integer dtype; it is stored with the old array's dtype after a range check. A scalar quant field
must already exist. Unknown op or tensor names are an error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


class QparamsError(ValueError):
    pass


def apply(ir: dict, weights: dict, patch: dict, arrays: dict) -> dict:
    ops = {op["name"]: op for op in ir["ops"]}
    w = dict(weights)
    n_arr = n_q = 0
    for name, p in patch.get("ops", {}).items():
        op = ops.get(name)
        if op is None:
            raise QparamsError(f"patch names op {name!r}, which the graph does not have")
        if "op" in p and p["op"] != op["op"]:
            raise QparamsError(f"{name}: patch expects kind {p['op']}, graph has {op['op']}")
        for field, akey in p.get("arrays", {}).items():
            if field in ("weight", "bias"):
                wkey = op.get(field)
            else:
                wkey = (op.get("quant") or {}).get(field)
            if not isinstance(wkey, str) or wkey not in w:
                raise QparamsError(f"{name}: field {field!r} names no weight array")
            new = np.asarray(arrays[akey])
            old = np.asarray(w[wkey])
            if new.shape != old.shape:
                raise QparamsError(f"{name}.{field}: shape {new.shape} != graph's {old.shape}")
            if not np.issubdtype(new.dtype, np.integer):
                raise QparamsError(f"{name}.{field}: {new.dtype} is not an integer type")
            info = np.iinfo(old.dtype)
            if new.size and (new.min() < info.min or new.max() > info.max):
                raise QparamsError(f"{name}.{field}: values outside {old.dtype}")
            w[wkey] = new.astype(old.dtype)
            n_arr += 1
        for field, val in p.get("quant", {}).items():
            q = op.get("quant")
            if q is None or field not in q:
                raise QparamsError(f"{name}: quant field {field!r} does not exist")
            if isinstance(q[field], bool) or not isinstance(q[field], int):
                raise QparamsError(f"{name}: quant field {field!r} is not an integer field")
            q[field] = int(val)
            n_q += 1
    for t, p in patch.get("tensors", {}).items():
        meta = ir["tensors"].get(t)
        if meta is None:
            raise QparamsError(f"patch names tensor {t!r}, which the graph does not have")
        if "scale" in p:
            meta.setdefault("quant", {})["scale"] = float(p["scale"])
    return {"weights": w, "arrays": n_arr, "quant": n_q, "tensors": len(patch.get("tensors", {}))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graph")
    ap.add_argument("weights")
    ap.add_argument("patch_json")
    ap.add_argument("patch_npz")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    ir = json.load(open(a.graph))
    res = apply(ir, dict(np.load(a.weights)), json.load(open(a.patch_json)), dict(np.load(a.patch_npz)))
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    json.dump(ir, open(od / "graph.json", "w"), indent=2)
    np.savez(od / "weights.npz", **res["weights"])
    print(f"[apply_qparams] {res['arrays']} arrays, {res['quant']} quant fields, {res['tensors']} tensor scales -> {od}")


if __name__ == "__main__":
    main()
