"""fp16 builds for the K3 IR runtime.

ModelBlaster's fp16 contract (docs/fp16_cuda.md, pipeline/cuda_dtypes.py): an ``_f16`` op stores its float
tensors (activations, weights, folded constants) as IEEE binary16 and does its math in fp32: every element is
widened on load, accumulated in fp32 (or better) and rounded once, round-to-nearest-even, on store. This module
turns an IR the runtime already executes into one with that storage:

  fused   fp32 FX extract -> the int8_mixed pass with every linear kept fp32 (keep_fp32 ["*"], no outliers):
          the same fusions as the int8 builds (rms_norm, rope_half, silu_mul, the attention scale fold, strided
          bmm operands, sibling merge), so the runtime has a kernel for every op; numerically the fp32 model.
  f16     any IR (a fused fp32 IR -> the pure fp16 build; an int8_mixed IR -> the mixed build: int8 linears,
          fp16 instead of fp32 everywhere else) -> per-op storage dtypes:
            * every float op becomes its ``_f16`` kind: its float inputs, outputs, weights and constants fp16;
            * the int8 kinds keep their contract (int8 operands, int32 accumulation, the double-precision
              dequantisation, fp32 row / weight scales, bias and SmoothQuant factors); their float activation
              operands (quantize_rows_s8's input, linear_dq*'s outputs and addend) are fp16: the input widens
              exactly, the fp32 result is rounded once on store;
            * fp32 islands (cuda_dtypes / precision_policy): every op that reads or writes a tensor whose
              calibrated max |value| exceeds ``limit`` (default 32752, half the fp16 max) stays fp32;
            * graph inputs and outputs stay fp32 (casts ``cast_f32_to_f16`` / ``cast_f16_to_f32`` at the
              interface and at island boundaries), so the runtime's drivers (chunk, ddpm) are unchanged.
  walk    the CPU reference of such an IR: each op's fp32 rule (int8_mixed.TORCH_RULES, then walk_torch), every
          output stored in its tensor's dtype (fp16 outputs rounded once, nearest-even).

    python f16_ir.py fused FP32_IR OUT [--policy JSON] [--model NAME]
    python f16_ir.py f16 IR OUT [--calib INPUTS.npz ...] [--limit 32752] [--keep-fp32 PATTERN ...]
    python f16_ir.py cnn FP32_IR OUT                (classic CNN extract -> runtime kinds, BN folded)
    python f16_ir.py check IR                       (walk vs the IR's golden)
    python f16_ir.py golden-inputs IR OUT.npz       (the golden inputs, for ref_walk.py feed)
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_eval" / "harness"))
from mbir import load_ir  # noqa: E402

from modelblaster.pipeline import int8_mixed as M  # noqa: E402

S8 = ("quantize_rows_s8", "linear_dq_s8", "linear3_dq_s8", "linear_dq_add_s8")
NOKERNEL = ("view",)
CASTS = ("cast_f32_to_f16", "cast_f16_to_f32")
WKEYS = ("weight", "bias", "gamma_key", "beta_key")
# inputs that are not float activations of the op (left alone): gather indices
NONFLOAT_INPUTS = {"index_select": (1,), "gather": (1,)}
LIMIT = 65504.0 / 2


def base_kind(k: str) -> str:
    return k[:-4] if k.endswith("_f16") else k


# ------------------------------------------------------------------ fused fp32 IR
def fused(fp32_dir: str, out: str, policy: dict | None = None) -> dict:
    """The int8_mixed pass with nothing quantized: the runtime's fused fp32 graph of the model."""
    ir = load_ir(fp32_dir)
    pol = dict(policy or {"base": "linear"})
    pol.update(keep_fp32=["*"], outliers={}, conv_as_linear=False)
    golden = dict(ir.input_tensors("cpu"))
    g, w, rep = M.quantize_graph(ir, [golden], pol)
    g["quant"] = "fp32"
    g["k3_fp16"] = {"stage": "fused", "from": str(fp32_dir), "policy": pol}
    fir = M.as_ir(g, w)
    o = M.walk(fir, golden)
    y = torch.cat([o[t].reshape(-1).float() for t in fir.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    g["k3_fp16"]["vs_fp32_golden_max_abs"] = float((y - r).abs().max())
    _save(g, w, out, np.load(Path(fp32_dir, "io.npz"))["input"], y.numpy())
    return {"ops": len(g["ops"]), "max_abs_vs_golden": g["k3_fp16"]["vs_fp32_golden_max_abs"],
            "op_kinds": rep["op_kinds"]}


def int8(fused_dir: str, out: str, calib: list[str], policy: dict) -> dict:
    """int8_mixed (dynamic per-row int8 activations, SmoothQuant, per-channel int8 weights) on a fused IR, e.g.
    YOLOv8x with conv_as_linear: every conv an im2col + int8 linear. Calibration: INPUTS.npz files."""
    ir = load_ir(fused_dir)
    samples = load_samples(ir, calib) if calib else [dict(ir.input_tensors("cpu"))]
    g, w, rep = M.quantize_graph(ir, samples, policy)
    q = M.as_ir(g, w)
    golden = dict(ir.input_tensors("cpu"))
    o = walk(q, golden)
    y = torch.cat([o[t].reshape(-1).float() for t in q.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    g["k3_fp16"] = {"stage": "int8", "from": str(fused_dir), "policy": policy, "n_calib": len(samples),
                    "vs_fused_golden": {"max_abs": float((y - r).abs().max()), "ref_absmax": float(r.abs().max())}}
    _save(g, w, out, np.load(Path(fused_dir, "io.npz"))["input"], y.numpy())
    return {"n_int8_linears": rep["n_int8_linears"], "op_kinds": rep["op_kinds"], **g["k3_fp16"]["vs_fused_golden"]}


def lower_cnn(fp32_dir: str, out: str) -> dict:
    """Classic-extract CNN IR (YOLOv8: conv2d + batchnorm2d + silu, chunk2_c1, cat{2,3,4}_c1, maxpool2d,
    upsample_nearest; ViNT's EfficientNet: conv2d_grouped + batchnorm2d, adaptive_avg_pool2d) -> the runtime's
    kinds: inference BN folded into the conv before it (W' = W * scale[oc], b' = b * scale + shift, in float64,
    stored fp32), grouped convs with G = 1 -> conv2d, chunk2_c1 -> two views, channel cats (N = 1) -> catn,
    upsample_nearest with its output size, global average pools -> mean_dim. Like `fused`, the IR's golden is
    its own fp32 walk; a graph with bmm etc. then goes through `fused` for the transformer fusions."""
    ir = load_ir(fp32_dir)
    g = copy.deepcopy(ir.graph)
    W = dict(ir.weights)
    readers: dict[str, int] = {}
    for op in g["ops"]:
        for t in op["inputs"]:
            readers[t] = readers.get(t, 0) + 1
    ops, skip, nfold = [], set(), 0
    for i, op in enumerate(g["ops"]):
        if i in skip:
            continue
        op = copy.deepcopy(op)
        k, sh = op["op"], op.get("shape", {})
        nxt = g["ops"][i + 1] if i + 1 < len(g["ops"]) else None
        if k == "conv2d_grouped" and sh.get("G", 1) == 1:      # ViNT's EfficientNet: plain convs
            op["op"] = k = "conv2d"
        if (k in ("conv2d", "conv2d_grouped") and nxt is not None and nxt["op"] == "batchnorm2d"
                and nxt["inputs"] == op["outputs"] and readers.get(op["outputs"][0], 0) == 1):
            sc = np.asarray(W[nxt["weight"]], np.float64).reshape(-1)
            sf = np.asarray(W[nxt["bias"]], np.float64).reshape(-1)
            wt = np.asarray(W[op["weight"]], np.float64)
            b = np.asarray(W[op["bias"]], np.float64) if op.get("bias") else np.zeros(wt.shape[0])
            wk, bk = op["weight"] + ".bnfold", op["weight"] + ".bnfold_bias"
            W[wk] = (wt * sc.reshape(-1, *([1] * (wt.ndim - 1)))).astype(np.float32)
            W[bk] = (b * sc + sf).astype(np.float32)
            op.update(weight=wk, bias=bk, outputs=list(nxt["outputs"]))
            ops.append(op)
            skip.add(i + 1)
            nfold += 1
            continue
        if k == "chunk2_c1":
            assert sh["N"] == 1, op["name"]
            hw = sh["H"] * sh["W"]
            for j, t in enumerate(op["outputs"]):
                ops.append({"name": f"{op['name']}__{j}", "op": "view", "inputs": list(op["inputs"]), "outputs": [t],
                            "offset": j * sh["c_each"] * hw, "shape": {"n": sh["c_each"] * hw}})
            continue
        if k in ("cat2_c1", "cat3_c1", "cat4_c1"):
            assert sh["N"] == 1, op["name"]
            cs = list(sh["C_inputs"])
            nsh = {"n_in": len(cs), "outer": 1, "inner": sh["H"] * sh["W"]}
            nsh.update({f"c{j}": (cs[j] if j < len(cs) else 0) for j in range(8)})
            op.update(op="catn", shape=nsh)
        elif k == "upsample_nearest":
            op["shape"] = dict(sh, OH=sh["IH"] * sh["scale"], OW=sh["IW"] * sh["scale"])
        elif k == "adaptive_avg_pool2d":                       # global average pool (to 1 x 1) -> mean_dim
            assert int(np.prod(g["tensors"][op["outputs"][0]]["shape"])) == sh["N"] * sh["C"], op["name"]
            op.update(op="mean_dim", shape={"outer": sh["N"] * sh["C"], "R": sh["IH"] * sh["IW"], "inner": 1})
        ops.append(op)
    # unique op names: pipeline/layout_fold.py removes folded ops by name, and ViNT's decomposed
    # nn.MultiheadAttention gives its linear, views, scale and softmax one shared name (the fold then drops the
    # in-projection linear too)
    seen: dict[str, int] = {}
    for o in ops:
        n = seen.get(o["name"], 0)
        seen[o["name"]] = n + 1
        if n:
            o["name"] = f"{o['name']}__{n}"
    g["ops"] = ops
    used = {o.get(r) for o in ops for r in WKEYS} | {m.get("const") for m in g["tensors"].values()}
    W = {k: v for k, v in W.items() if k in used}
    live = {t for o in ops for t in o["inputs"] + o["outputs"]}
    live |= set(g["output"].get("tensors") or [g["output"]["tensor"]]) | set(input_names(ir))
    g["tensors"] = {t: m for t, m in g["tensors"].items() if t in live}
    g["k3_fp16"] = {"stage": "fused", "from": str(fp32_dir), "bn_folded": nfold}
    fir = M.as_ir(g, W)
    golden = dict(ir.input_tensors("cpu"))
    o = walk(fir, golden)
    y = torch.cat([o[t].reshape(-1).float() for t in fir.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    g["k3_fp16"]["vs_fp32_golden_max_abs"] = float((y - r).abs().max())
    _save(g, W, out, np.load(Path(fp32_dir, "io.npz"))["input"], y.numpy())
    import collections
    return {"ops": len(ops), "bn_folded": nfold, "max_abs_vs_golden": g["k3_fp16"]["vs_fp32_golden_max_abs"],
            "ref_absmax": float(r.abs().max()), "op_kinds": dict(collections.Counter(o["op"] for o in ops))}


def _save(g, w, out, gin, gout):
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "graph.json").write_text(json.dumps(g, indent=1))
    np.savez(o / "weights.npz", **w)
    np.savez(o / "io.npz", input=np.asarray(gin, np.float32), output=np.asarray(gout, np.float32))


# ------------------------------------------------------------------ calibration ranges
def absmax(ir, samples: list[dict]) -> dict:
    """max |finite value| of every produced float tensor over the samples (the source IR's own walk)."""
    rng = {}

    def rec(name, v):
        if v.is_floating_point():
            a = v.detach().abs()
            a = a[torch.isfinite(a)]
            m = float(a.max()) if a.numel() else 0.0
            rng[name] = max(rng.get(name, 0.0), m)
    for s in samples:
        walk(ir, s, record=rec)
    return rng


# ------------------------------------------------------------------ the transform
def f16ify(g: dict, w: dict, rng: dict | None = None, limit: float = LIMIT, keep_fp32: list | None = None):
    """(graph, weights) -> (graph, weights, report) with fp16 storage as described in the module docstring."""
    g = copy.deepcopy(g)
    T = g["tensors"]
    W = dict(w)
    rng = rng or {}
    keep_fp32 = list(keep_fp32 or [])
    outs = list(g["output"].get("tensors") or [g["output"]["tensor"]])
    big = {t for t, m in rng.items() if m > limit}
    const_big = set()
    for t, m in T.items():
        if m.get("const") and np.asarray(W[m["const"]]).dtype.kind == "f":
            a = np.abs(np.asarray(W[m["const"]], np.float64))
            a = a[np.isfinite(a)]
            if a.size and a.max() > limit:
                const_big.add(t)
    dt = {t: m.get("dtype", "f32") for t, m in T.items()}
    cur: dict[str, str] = {}          # tensor -> the name that holds its value now
    view_src: dict[str, tuple[str, int]] = {}   # view output -> (const base, element offset)
    cache: dict[tuple[str, str], str] = {}
    new_ops: list = []
    rep = {"fp32_ops": [], "casts": 0, "f16_consts": 0, "islands_tensors": sorted(big | const_big)}

    def nel(t):
        return int(np.prod(T[t]["shape"])) if T[t]["shape"] else 1

    def isfloat(t):
        return dt.get(t, "f32") in ("f32", "f16")

    def const_values(t):
        m = T[t]
        if m.get("const"):
            return np.asarray(W[m["const"]])
        base, off = view_src[t]
        return np.asarray(W[T[base]["const"]]).reshape(-1)[off:off + nel(t)].reshape(T[t]["shape"])

    def need(t, want):
        n = cur.get(t, t)
        if not isfloat(n) or dt[n] == want:
            return n
        if (n, want) in cache:
            return cache[(n, want)]
        if T[n].get("const") or n in view_src:            # a constant: stored at the wanted dtype
            v = const_values(n)
            new = f"{n}__{want}"
            key = f"const.{new}"
            W[key] = v.astype(np.float16) if want == "f16" else v.astype(np.float32)
            T[new] = {"shape": list(T[n]["shape"]), "dtype": want, "quant": None, "const": key}
            dt[new] = want
            rep["f16_consts"] += want == "f16"
        else:
            new = f"{n}__{want}"
            T[new] = {"shape": list(T[n]["shape"]), "dtype": want, "quant": None}
            dt[new] = want
            new_ops.append({"name": new, "op": f"cast_{dt[n]}_to_{want}", "inputs": [n], "outputs": [new],
                            "shape": {"n": nel(n)}})
            rep["casts"] += 1
        cache[(n, want)] = new
        return new

    def set_out(t, d):
        T[t]["dtype"] = d
        dt[t] = d

    def wcopy(op):
        for r in WKEYS:
            k = op.get(r)
            if k is None:
                continue
            a = np.asarray(W[k])
            if a.dtype.kind != "f":
                continue
            nk = k + ".h"
            if nk not in W:
                W[nk] = a.astype(np.float16)
            op[r] = nk

    for op in g["ops"]:
        op = copy.deepcopy(op)
        k = op["op"]
        if k == "view":
            src = cur.get(op["inputs"][0], op["inputs"][0])
            op["inputs"] = [src]
            o = op["outputs"][0]
            set_out(o, dt[src])
            if T[src].get("const"):
                view_src[o] = (src, int(op.get("offset", 0)))
            elif src in view_src:
                b, off0 = view_src[src]
                view_src[o] = (b, off0 + int(op.get("offset", 0)))
            new_ops.append(op)
            continue
        if k in S8:
            ins = list(op["inputs"])
            if k == "quantize_rows_s8":
                ins[0] = cur.get(ins[0], ins[0])
            od = "f32" if any(t in big for t in op["outputs"] if isfloat(t)) else "f16"
            if k != "quantize_rows_s8":
                for t in op["outputs"]:
                    set_out(t, od)
            if k == "linear_dq_add_s8":
                ins[2] = need(ins[2], od)
            op["inputs"] = ins
            new_ops.append(op)
        else:
            skip = NONFLOAT_INPUTS.get(base_kind(k), ())
            fl = [t for i, t in enumerate(op["inputs"]) if i not in skip]
            isl = (any(t in big or cur.get(t, t) in const_big or t in const_big for t in fl)
                   or any(t in big for t in op["outputs"])
                   or any(fnmatch.fnmatchcase(op["name"], p) for p in keep_fp32))
            want = "f32" if isl else "f16"
            op["inputs"] = [cur.get(t, t) if i in skip else need(t, want) for i, t in enumerate(op["inputs"])]
            for t in op["outputs"]:
                if isfloat(t):
                    set_out(t, want)
            if want == "f16":
                op["op"] = k + "_f16"
                wcopy(op)
            else:
                rep["fp32_ops"].append(op["name"])
            new_ops.append(op)
        # graph outputs stay fp32
        for i, t in enumerate(op["outputs"]):
            if t in outs and dt[t] == "f16":
                h = f"{t}__h"
                T[h] = {"shape": list(T[t]["shape"]), "dtype": "f16", "quant": None}
                dt[h] = "f16"
                op["outputs"][i] = h
                cur[t] = h
                new_ops.append({"name": t, "op": "cast_f16_to_f32", "inputs": [h], "outputs": [t],
                                "shape": {"n": nel(t)}})
                set_out(t, "f32")
                rep["casts"] += 1
    g["ops"] = new_ops
    g["quant"] = "native"
    used = {op.get(r) for op in new_ops for r in WKEYS + ("w_scale",)} | {m.get("const") for m in T.values()}
    W = {k: v for k, v in W.items() if k in used}
    live = {t for op in new_ops for t in op["inputs"] + op["outputs"]} | set(outs)
    live |= {p["name"] for p in g["input"].get("packed_inputs") or [{"name": g["input"]["tensor"]}]}
    g["tensors"] = {t: m for t, m in T.items() if t in live}
    rep["n_f16_ops"] = sum(op["op"].endswith("_f16") for op in new_ops)
    rep["n_fp32_float_ops"] = len(rep["fp32_ops"])
    rep["limit"] = limit
    g["k3_fp16"] = dict(g.get("k3_fp16", {}), stage="f16", **{k: v for k, v in rep.items() if k != "fp32_ops"},
                        fp32_ops=rep["fp32_ops"][:200])
    return g, W, rep


# ------------------------------------------------------------------ reference walk
def walk(ir, inputs: dict, record=None, keep=None) -> dict:
    """int8_mixed.walk with per-tensor storage: every float output rounded to its tensor's dtype."""
    wt = M._walk_torch()
    T = ir.graph["tensors"]
    env = {k: (v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v))).float().reshape(ir.tshape(k))
           for k, v in inputs.items()}
    env.update(ir.const_tensors("cpu"))
    keep = set(keep or ()) | set(ir.output_names())
    last = {}
    for k, op in enumerate(ir.ops):
        for t in op["inputs"]:
            last[t] = k
    with torch.no_grad():
        for k, op in enumerate(ir.ops):
            ins = [env[t] for t in op["inputs"]]
            kind = op["op"]
            rule = M.TORCH_RULES.get(base_kind(kind))
            if rule is not None:
                outs = rule(ir, op, ins)
            else:
                fn = wt._rule(kind)
                if fn is None:
                    raise NotImplementedError(f"f16 walk: no CPU rule for {kind}")
                outs = fn(ir, op, ins, "cpu")
            for name, val in zip(op["outputs"], outs):
                val = val.reshape(ir.tshape(name))
                if T[name].get("dtype") == "f16" and val.is_floating_point():
                    val = val.half().float()
                env[name] = val
                if record is not None:
                    record(name, val)
            for t in op["inputs"]:
                if t not in keep and last.get(t, -1) <= k and not T.get(t, {}).get("const"):
                    env.pop(t, None)
    return {k: v for k, v in env.items() if k in keep}


def input_names(ir):
    inp = ir.graph["input"]
    return [p["name"] for p in inp.get("packed_inputs") or [{"name": inp["tensor"]}]]


def load_samples(ir, paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        z = np.load(p)
        out.append({n: torch.from_numpy(np.asarray(z[n], np.float32)).reshape(ir.tshape(n)) for n in input_names(ir)})
    return out


def build_f16(src_dir: str, out: str, calib: list[str] | None = None, limit: float = LIMIT,
              keep_fp32: list | None = None) -> dict:
    ir = load_ir(src_dir)
    golden = dict(ir.input_tensors("cpu"))
    samples = load_samples(ir, calib) if calib else []
    samples = [golden] + samples
    rng = absmax(ir, samples)
    g, w, rep = f16ify(ir.graph, ir.weights, rng, limit, keep_fp32)
    hir = M.as_ir(g, w)
    o = walk(hir, golden)
    y = torch.cat([o[t].reshape(-1).float() for t in hir.output_names()])
    r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
    rep["vs_source_golden"] = {"max_abs": float((y - r).abs().max()), "ref_absmax": float(r.abs().max()),
                               "cos": float(torch.nn.functional.cosine_similarity(y.double(), r.double(), dim=0))}
    rep["calib_samples"] = len(samples)
    g["k3_fp16"]["vs_source_golden"] = rep["vs_source_golden"]
    g["k3_fp16"]["from"] = str(src_dir)
    _save(g, w, out, np.load(Path(src_dir, "io.npz"))["input"], y.numpy())
    rep.pop("fp32_ops", None)
    return rep


def golden_inputs(ir_dir: str, out: str):
    """the IR's golden inputs as INPUTS.npz ({name: array}) for ref_walk.py feed"""
    ir = load_ir(ir_dir)
    np.savez(out, **{k: v.numpy() for k, v in ir.input_tensors("cpu").items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fused")
    a.add_argument("src"); a.add_argument("out"); a.add_argument("--policy", default=None)
    b = sub.add_parser("f16")
    b.add_argument("src"); b.add_argument("out"); b.add_argument("--calib", nargs="*", default=[])
    b.add_argument("--limit", type=float, default=LIMIT); b.add_argument("--keep-fp32", nargs="*", default=[])
    c = sub.add_parser("check")
    c.add_argument("src")
    q8 = sub.add_parser("int8")
    q8.add_argument("src"); q8.add_argument("out"); q8.add_argument("--calib", nargs="*", default=[])
    q8.add_argument("--policy", required=True)
    e = sub.add_parser("cnn")
    e.add_argument("src"); e.add_argument("out")
    d = sub.add_parser("golden-inputs")
    d.add_argument("src"); d.add_argument("out")
    x = ap.parse_args()
    if x.cmd == "int8":
        print(json.dumps(int8(x.src, x.out, x.calib, json.loads(x.policy))))
        sys.exit(0)
    if x.cmd == "cnn":
        print(json.dumps(lower_cnn(x.src, x.out)))
        sys.exit(0)
    if x.cmd == "golden-inputs":
        golden_inputs(x.src, x.out)
        sys.exit(0)
    if x.cmd == "fused":
        print(json.dumps(fused(x.src, x.out, json.loads(x.policy) if x.policy else None)))
    elif x.cmd == "f16":
        print(json.dumps(build_f16(x.src, x.out, x.calib, x.limit, x.keep_fp32)))
    else:
        ir = load_ir(x.src)
        o = walk(ir, dict(ir.input_tensors("cpu")))
        y = torch.cat([o[t].reshape(-1).float() for t in ir.output_names()])
        r = torch.from_numpy(ir.golden_out.reshape(-1).astype(np.float32))
        print(json.dumps({"max_abs": float((y - r).abs().max()), "ref_absmax": float(r.abs().max())}))
