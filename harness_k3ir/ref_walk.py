"""Host side of the K3 IR runtime's numerics ladder.

  feed    IR_DIR NAMES_JSON INPUTS.npz FEED.bin [--keep-ref REF.npz]
          walk the IR on the CPU (pipeline/int8_mixed.walk: fp32 ops by walk_torch's rules, int8 kinds
          bit-exactly as their reference C) and write every tensor an op reads (k3ir feed format)
  compare IR_DIR NAMES_JSON FEED.bin DUMP.bin [--json OUT]
          the isolation gate (walk_vulkan.evaluate_int8_mixed's rule): every int8 kind's outputs equal
          to the reference walk byte for byte, every fp32 op within the fp32 per-slot bound
  smolvla-inputs OBS.pt OUT_DIR            per-graph inputs of one observation (vision / prefix / step)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_eval" / "harness"))
from mbir import cuda_dtypes, load_ir  # noqa: E402

from modelblaster.pipeline import int8_mixed as M  # noqa: E402
import f16_ir  # noqa: E402  (per-tensor storage: fp16 outputs rounded once; fp32 IRs walk as before)

S8 = {"quantize_rows_s8", "linear_dq_s8", "linear3_dq_s8", "linear_dq_add_s8"}
NP = {"f32": np.float32, "i8": np.int8, "i32": np.int32, "i64": np.int32, "u8": np.int8, "b8": np.int8,
      "f16": np.float16}


def temb(t, dim=720, lo=4e-3, hi=4.0):
    f = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    x = (1.0 / (lo * (hi / lo) ** f) * 2 * math.pi) * t
    return np.concatenate([np.sin(x), np.cos(x)])[None, :].astype(np.float32)


def walk_all(ir, inputs: dict) -> dict:
    vals = {k: torch.as_tensor(v) for k, v in inputs.items()}
    rec = lambda n, v: vals.__setitem__(n, v.detach().clone())
    f16_ir.walk(ir, M.inputs_of(ir, [inputs[n] for n in ir_input_names(ir)]), record=rec)
    return vals


def ir_input_names(ir):
    inp = ir.graph["input"]
    return [p["name"] for p in inp.get("packed_inputs") or [{"name": inp["tensor"]}]]


def write_feed(path, names, vals, needed):
    tid = {n: i for i, n in enumerate(names["tensors"])}
    items = []
    for n in needed:
        v = vals[n]
        a = np.ascontiguousarray(v.numpy().astype(NP[names["dtypes"][n]]))
        items.append((tid[n], a.tobytes()))
    with open(path, "wb") as f:                 # every field 64-byte aligned (k3ir feed)
        f.write(np.int64(len(items)).tobytes() + b"\0" * 56)
        for i, b in items:
            f.write(np.array([i, len(b)], np.int64).tobytes() + b"\0" * 48)
            f.write(b + b"\0" * ((-len(b)) % 64))


def read_dump(path):
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw[:8], np.int64)[0])
    p, out = 8, {}
    for _ in range(n):
        i, nb = np.frombuffer(raw[p:p + 16], np.int64)
        p += 16
        out[int(i)] = raw[p:p + nb]
        p += int(nb)
    return out


def cmd_feed(a):
    ir = load_ir(a.ir)
    names = json.loads(Path(a.names).read_text())
    z = np.load(a.inputs)
    inputs = {n: torch.from_numpy(z[n]).reshape(ir.tshape(n)) for n in ir_input_names(ir)}
    vals = walk_all(ir, inputs)
    T = ir.graph["tensors"]
    needed = sorted({t for op in ir.ops if op["op"] != "view" for t in op["inputs"] if not T[t].get("const")},
                    key=names["tensors"].index)
    write_feed(a.out, names, vals, needed)
    if a.keep_ref:
        np.savez(a.keep_ref, **{n: vals[n].numpy() for n in vals if n in T and not T[n].get("const")})
    outs = ir.output_names()
    print(json.dumps({"fed_tensors": len(needed), "out_absmax": float(vals[outs[0]].abs().max())}))


def cmd_compare(a):
    ir = load_ir(a.ir)
    names = json.loads(Path(a.names).read_text())
    T = ir.graph["tensors"]
    z = np.load(a.ref)
    dump = read_dump(a.dump)
    tid = {n: i for i, n in enumerate(names["tensors"])}
    tols = {"f32": cuda_dtypes.TOLERANCE["fp32"], "f16": cuda_dtypes.TOLERANCE["fp16"]}
    s8_bad, f32_bad, worst, first = {}, 0, 0.0, None
    by_kind, exact = {}, {}
    for op in ir.ops:
        if op["op"] == "view":
            continue
        for t in op["outputs"]:
            ref = z[t]
            got = np.frombuffer(dump[tid[t]], NP[names["dtypes"][t]]).reshape(ref.shape)
            if op["op"] in S8:
                nb = int((got.view(np.uint8) != np.ascontiguousarray(ref.astype(got.dtype)).view(np.uint8)).sum())
                s8_bad[op["op"]] = s8_bad.get(op["op"], 0) + nb
                if nb and first is None:
                    first = f"{op['name']} ({op['op']}): {nb} bytes differ"
                continue
            tol = tols["f16" if names["dtypes"][t] == "f16" else "f32"]
            if names["dtypes"][t] == "f16":
                ex = exact.setdefault(op["op"], [0, 0])
                ex[0] += int((got.view(np.uint16) != ref.astype(np.float16).view(np.uint16)).sum())
                ex[1] += got.size
            g, r = got.astype(np.float64), ref.astype(np.float64)
            same = (g == r) | (np.isnan(g) & np.isnan(r))
            d = np.where(same, 0.0, np.abs(g - r))
            scale = 0.0
            for v in [r] + [z[i].astype(np.float64) for i in op["inputs"] if i in z.files]:
                fin = v[np.isfinite(v)]
                scale = max(scale, float(np.abs(fin).max()) if fin.size else 0.0)
            e = float(d.max()) / max(scale, 1e-12) if d.size else 0.0
            worst = max(worst, e)
            k = by_kind.setdefault(op["op"], [0, 0.0])
            k[0] += 1
            k[1] = max(k[1], e)
            bad = (d.size and float(d.max()) > tol["slot_rel"] * scale + tol["slot_abs"]) or \
                not (np.isfinite(g) == np.isfinite(r)).all()
            if bad:
                f32_bad += 1
                if first is None:
                    first = f"{op['name']} ({op['op']}) rel_err={e:.2e}"
    res = {"s8_mismatch_bytes_by_kind": s8_bad, "n_fp32_ops_over_slot_bound": f32_bad,
           "worst_fp32_rel_err": worst, "fp32_worst_by_kind": {k: v[1] for k, v in by_kind.items()},
           "first_divergent": first, "slot_bound": tols,
           "f16_elements_differing_from_walk_by_kind": {k: f"{v[0]}/{v[1]}" for k, v in exact.items()},
           "ok": f32_bad == 0 and not any(s8_bad.values())}
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def cmd_smolvla_inputs(a):
    """vision / prefix / step inputs of one observation (step at t = 1, the first flow step)."""
    o = torch.load(a.obs, weights_only=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "vision_in.npz", pixels=o["pixels"].float().numpy())
    np.savez(out / "prefix_lang_state.npz", lang_emb=o["lang"].float().numpy(), state=o["state"].float().numpy())
    np.savez(out / "step_noise_temb.npz", x_t=o["noise"].float().numpy(), time_emb=temb(1.0))


def cmd_smolvla_chain(a):
    """feeds + reference values of vision, prefix and the first flow step on one observation, chained:
    the prefix reads the reference walk's vision output, the step the prefix's."""
    o = torch.load(a.obs, weights_only=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prev = None
    for part in ("vision", "prefix", "step"):
        ir = load_ir(f"{a.irs}/{part}")
        names = json.loads(Path(f"{a.packed}/{part}.names.json").read_text())
        ins = ir_input_names(ir)
        src = {"vision": [o["pixels"]], "prefix": [prev, o["lang"], o["state"]],
               "step": [prev, o["noise"], torch.from_numpy(temb(1.0))]}[part]
        inputs = {n: torch.as_tensor(v).float().reshape(ir.tshape(n)) for n, v in zip(ins, src)}
        vals = walk_all(ir, inputs)
        T = ir.graph["tensors"]
        needed = sorted({t for op in ir.ops if op["op"] != "view" for t in op["inputs"] if not T[t].get("const")},
                        key=names["tensors"].index)
        write_feed(out / f"{part}.feed", names, vals, needed)
        np.savez(out / f"{part}.ref.npz", **{n: vals[n].numpy() for n in vals if n in T and not T[n].get("const")})
        np.concatenate([inputs[n].numpy().reshape(-1) for n in ins]).astype(np.float32).tofile(out / f"{part}.in.bin")
        prev = vals[ir.output_names()[0]]
        prev.numpy().astype(np.float32).tofile(out / f"{part}.out.bin")
        print(part, "fed", len(needed), "out absmax", float(prev.abs().max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("feed")
    f.add_argument("ir"); f.add_argument("names"); f.add_argument("inputs"); f.add_argument("out")
    f.add_argument("--keep-ref")
    c = sub.add_parser("compare")
    c.add_argument("ir"); c.add_argument("names"); c.add_argument("ref"); c.add_argument("dump")
    c.add_argument("--json")
    s = sub.add_parser("smolvla-inputs")
    s.add_argument("obs"); s.add_argument("out")
    ch = sub.add_parser("smolvla-chain")
    ch.add_argument("obs"); ch.add_argument("irs"); ch.add_argument("packed"); ch.add_argument("out")
    a = ap.parse_args()
    {"smolvla-chain": cmd_smolvla_chain, "feed": cmd_feed, "compare": cmd_compare, "smolvla-inputs": cmd_smolvla_inputs}[a.cmd](a)
