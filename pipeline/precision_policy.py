"""Precision policy: fp32 islands in fp16 IRs for overflow-prone ops (A2 capabilities, item 3).

Problem. The flow stores every float tensor of an fp16 extract as binary16 (``cuda_dtypes``: fp16 storage, fp32
math). Some models produce values beyond the fp16 range (65504): 48_Mamba2ReturnY folds ``exp(segsum(A))``,
``exp(A_cumsum)`` and their products into constants and outputs values up to ~1e22 in fp32, so the fp16 extract
stores ``inf`` in four folded constants (~1.1M elements) and its golden output has ~21M non-finite elements of 134M;
the reference walk propagates inf - inf = nan and the model fails its first profile ("fails on fp16 overflow" in the
Level 3 sweep). PyTorch itself in fp16 storage gives the same inf/nan.

Policy (opt-in, ``--precision-policy auto``; default off = nothing changes):

1. **detector** -- an fp16 extract is flagged when its golden output has non-finite values that the fp32 model output
   does not have (cheap: one extraction, the fp32 golden is captured before the storage cast through
   ``extract_graph.GOLDEN_SINK``), or -- when a constant was stored as inf/nan -- when the fp16-storage walk of the IR
   is non-finite where the fp32 model output is finite (overflow the output hides). Only then:
2. a second, identically seeded extraction in fp32 gives the exact fp32 values of every constant (same graph, same
   tensor names; anything else and the policy refuses, leaving the fp16 IR unchanged);
3. the fp32 IR is walked once in fp32 (fp16-rounded inputs, as the golden): every tensor whose finite max |value|
   exceeds ``limit`` (default 32752 = half the fp16 max, margin for re-rounding) is *flagged*;
4. **islands**: every non-alias op that reads or writes a flagged tensor runs its fp32 kind (``bmm_f16`` -> ``bmm``)
   and stores its outputs in fp32 (views follow their base); flagged constants get their fp32 values;
   ``cast_f16_to_f32`` / ``cast_f32_to_f16`` ops (``precision_casts``) are inserted where an island reads an fp16
   tensor or an fp16 op reads an island tensor; a graph output in an island is fp32 and the golden output is the
   fp32 model output (``io.npz`` output stored as float32);
5. the result is recorded in ``graph.json`` ``precision_policy`` (limit, fp32 ops / constants, casts, the deviation).

Reference semantics / deviation: the walk (and so every partition's data.bin and the driver's checks) computes fp32
islands in fp32 storage and the rest as before (fp32 math, fp16 storage). It therefore does NOT reproduce PyTorch's
fp16 behaviour on these models (which is inf/nan); it matches the fp32 PyTorch model on fp16-rounded inputs up to the
fp16 rounding of the tensors outside the islands. That deviation is stated in ``precision_policy.deviation``.

Export: a partition mixing fp16 ops and island ops takes slots from both hand libraries and the fp16 tolerances
(``kb_export._MixedLibrary``); IRs without a policy contain no mixed partition (checked on all sweep IRs), so their
exports are unchanged.

CLI::

    python -m modelblaster.pipeline.precision_policy extract --bench-file P.py --out-dir IR [--policy auto|off]
        [--limit 32752] [--seed 0] [-- extra extract_graph args]
    python -m modelblaster.pipeline.precision_policy check IR      # walk the IR, report non-finite / max error
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

FP16_MAX = 65504.0
DEFAULT_LIMIT = FP16_MAX / 2
ALIAS_OPS = ("view", "chunk2_c1")
FORMAT = "mb-precision-policy/1"
DEVIATION = ("fp32 islands: ops reading or writing tensors beyond the fp16 range run in fp32 storage (fp32 kinds, "
             "fp32 constants, casts at the island boundary). PyTorch with fp16 storage produces inf/nan on these "
             "tensors; this IR instead matches the fp32 model on fp16-rounded inputs, up to the fp16 rounding of "
             "the tensors outside the islands.")


def enabled(value: str | None = None) -> str:
    v = (value if value is not None else os.environ.get("MB_PRECISION_POLICY", "off")).strip().lower()
    if v not in ("off", "auto"):
        raise ValueError(f"precision policy {v!r}: expected off|auto")
    return v


# ------------------------------------------------------------------ extraction

def _seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _extract(bench_file: Path, out_dir: Path, quant: str, extra: list[str], seed: int, sink: list | None):
    from . import extract_graph as eg
    argv = sys.argv
    sys.argv = ["extract_graph", "--bench-file", str(bench_file), "--out-dir", str(out_dir), "--quant", quant, *extra]
    eg.GOLDEN_SINK = sink
    try:
        _seed(seed)
        eg.main()
    finally:
        eg.GOLDEN_SINK = None
        sys.argv = argv


def _load(d: Path):
    g = json.loads((d / "graph.json").read_text())
    w = np.load(d / "weights.npz")
    io = np.load(d / "io.npz")
    return g, {k: w[k] for k in w.files}, {k: io[k] for k in io.files}


def extract(bench_file, out_dir, *, extra: list[str] | None = None, policy: str = "auto", limit: float = DEFAULT_LIMIT,
            seed: int = 0, log=print) -> dict:
    """fp16 extraction (seeded) into ``out_dir``; with ``policy="auto"`` and a detected overflow, the IR there is
    replaced by its fp32-island version. Returns the policy report (also ``out_dir/precision_policy.json``)."""
    bench_file, out = Path(bench_file).resolve(), Path(out_dir).resolve()
    extra = list(extra or [])
    sink: list = []
    _extract(bench_file, out, "fp16", extra, seed, sink)
    g16, w16, io16 = _load(out)
    rep = {"format": FORMAT, "policy": policy, "limit": limit, "seed": seed, "applied": False}
    gold32 = sink[0] if sink else None
    det = detect(io16["output"], gold32)
    # constants stored as inf/nan in fp16 (folded exp chains) and intermediate overflow that the output hides
    det["nonfinite_consts"] = sorted(t for t, m in g16["tensors"].items() if m.get("const")
                                     and np.asarray(w16[m["const"]]).dtype.kind == "f"
                                     and not np.isfinite(np.asarray(w16[m["const"]], np.float32)).all())
    if gold32 is not None and not det["overflow"] and det["nonfinite_consts"]:
        outs = g16["output"].get("tensors") or [g16["output"]["tensor"]]
        _, kept = walk_stats(g16, w16, io16["input"], rounding=True, keep=tuple(outs))
        wo = np.concatenate([kept[t].reshape(-1).double().numpy() for t in outs])
        det["nonfinite_walk_fp16"] = int((~np.isfinite(wo)).sum())
        det["overflow"] = bool((~np.isfinite(wo) & np.isfinite(np.asarray(gold32, np.float64))).any())
    rep["detector"] = det
    if policy == "off" or not det["overflow"]:
        rep["reason"] = "policy off" if policy == "off" else "no fp16 overflow in the model output"
        (out / "precision_policy.json").write_text(json.dumps(rep, indent=1))
        return rep
    tmp = Path(tempfile.mkdtemp(prefix="mb_prec32_"))
    try:
        _extract(bench_file, tmp, "fp32", extra, seed, None)
        g32, w32, _ = _load(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res = apply(g16, w16, io16, g32, w32, gold32, limit=limit, log=log)
    rep.update({k: v for k, v in res.items() if k not in ("graph", "weights", "io")})
    if res.get("graph") is None:
        (out / "precision_policy.json").write_text(json.dumps(rep, indent=1))
        return rep
    (out / "graph.json").write_text(json.dumps(res["graph"], indent=2))
    np.savez(out / "weights.npz", **res["weights"])
    np.savez(out / "io.npz", **res["io"])
    rep["applied"] = True
    rep["check"] = check(out, log=log)
    (out / "precision_policy.json").write_text(json.dumps(rep, indent=1))
    return rep


def detect(golden16: np.ndarray, golden32: np.ndarray | None) -> dict:
    """Overflow detector on the model output: fp16 golden non-finite where the fp32 model output is finite."""
    g16 = np.asarray(golden16, dtype=np.float64)
    bad16 = ~np.isfinite(g16)
    rec = {"n": int(g16.size), "nonfinite_fp16": int(bad16.sum())}
    if golden32 is None:
        rec["overflow"] = False
        rec["note"] = "no fp32 golden captured"
        return rec
    g32 = np.asarray(golden32, dtype=np.float64)
    fin32 = np.isfinite(g32)
    rec["nonfinite_fp32"] = int((~fin32).sum())
    rec["max_abs_fp32"] = float(np.abs(g32[fin32]).max()) if fin32.any() else None
    rec["overflow"] = bool((bad16 & fin32).any())
    return rec


# ------------------------------------------------------------------ walking

def _ir(graph, weights, golden_in, golden_out=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu_eval" / "harness"))
    from mbir import IR
    return IR(graph.get("name", "ir"), graph, weights, golden_in,
              golden_out if golden_out is not None else np.zeros(1, np.float32))


def walk_stats(graph, weights, golden_in, *, rounding: bool, keep: tuple = ()) -> tuple[dict, dict]:
    """Walk the IR on CPU, freeing tensors after their last use: ({tensor: (finite max |v|, non-finite count)},
    {kept tensor: value}). ``rounding`` False runs every op as fp32 (no fp16 rounding of outputs)."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu_eval" / "harness"))
    import walk_torch as wt
    ir = _ir(graph, weights, golden_in)
    ops = graph["ops"]
    last = {}
    for i, o in enumerate(ops):
        for t in o["inputs"]:
            last[t] = i
    outs = set(ir.output_names()) | set(keep)
    stats, kept = {}, {}

    def stat(name, v):
        if not torch.is_tensor(v) or not v.is_floating_point():
            return
        f = v.detach().reshape(-1).float()
        fin = torch.isfinite(f)
        stats[name] = (float(f[fin].abs().max()) if bool(fin.any()) else 0.0, int((~fin).sum()))

    with torch.no_grad():
        env = ir.input_tensors("cpu")
        consts = {n: m["const"] for n, m in graph["tensors"].items() if m.get("const")}
        for n, v in env.items():
            stat(n, v)
        for i, op in enumerate(ops):
            for t in op["inputs"]:
                if t not in env and t in consts:
                    arr = weights[consts[t]]
                    tt = torch.from_numpy(np.ascontiguousarray(arr))
                    env[t] = tt.to(torch.int64) if arr.dtype.kind == "i" else tt.float()
                    stat(t, env[t])
            fn = wt._rule(op["op"])
            if fn is None:
                raise NotImplementedError(op["op"])
            res = fn(ir, op, [env[t] for t in op["inputs"]], "cpu")
            if rounding and op["op"].endswith("_f16"):
                res = [wt._round_f16(v) for v in res]
            for name, val in zip(op["outputs"], res):
                env[name] = val.reshape(ir.tshape(name)) if name in graph["tensors"] else val
                stat(name, env[name])
            for t in set(op["inputs"]):
                if last.get(t) == i and t not in outs:
                    env.pop(t, None)
            for t in op["outputs"]:
                if t not in last and t not in outs:
                    env.pop(t, None)
        for t in outs:
            if t in env:
                kept[t] = env[t]
    return stats, kept


# ------------------------------------------------------------------ the rewrite

def _strip(kind: str) -> str:
    return kind[:-4] if kind.endswith("_f16") else kind


def _structure(g):
    return [(o["name"], _strip(o["op"]), list(o["inputs"]), list(o["outputs"])) for o in g["ops"]]


def apply(g16: dict, w16: dict, io16: dict, g32: dict, w32: dict, golden32, *, limit: float = DEFAULT_LIMIT,
          log=print) -> dict:
    """The fp32-island IR of an fp16 IR (see the module doc). Returns {"graph", "weights", "io", ...report}; graph
    None (with ``refused``) when the two extractions differ in structure or an island op has no fp32 kind."""
    from .reference_kernels import KERNEL_SPECS
    if _structure(g16) != _structure(g32) or set(g16["tensors"]) != set(g32["tensors"]):
        return {"graph": None, "refused": "fp16 and fp32 extractions differ in structure"}
    stats, _ = walk_stats(g32, w32, io16["input"], rounding=False)
    flagged = {t for t, (mx, nf) in stats.items() if mx > limit}
    if not flagged:
        return {"graph": None, "refused": f"no tensor above {limit:g} in the fp32 walk"}
    g = json.loads(json.dumps(g16))
    tens = g["tensors"]
    fp32_ops = [i for i, o in enumerate(g["ops"]) if o["op"] not in ALIAS_OPS
                and (set(o["inputs"]) | set(o["outputs"])) & flagged]
    for i in fp32_ops:
        k = _strip(g["ops"][i]["op"])
        if k not in KERNEL_SPECS:
            return {"graph": None, "refused": f"{g['ops'][i]['name']}: no fp32 kind {k}"}
    fp32_set = set(fp32_ops)
    s32 = {t for t in flagged if tens.get(t, {}).get("const")}
    for i, o in enumerate(g["ops"]):
        if i in fp32_set or (o["op"] in ALIAS_OPS and any(t in s32 for t in o["inputs"])):
            s32.update(o["outputs"])

    def is_float(t):
        return tens.get(t, {}).get("dtype") in ("f16", "f32") and not (
            tens[t].get("const") and np.asarray(w16[tens[t]["const"]]).dtype.kind in "iub")

    new_ops, casts, made = [], [], {}
    for i, o in enumerate(g["ops"]):
        o = dict(o)
        if o["op"] not in ALIAS_OPS:
            need = "f32" if i in fp32_set else "f16"
            ins = []
            for t in o["inputs"]:
                have = "f32" if t in s32 else "f16"
                if not is_float(t) or have == need:
                    ins.append(t)
                    continue
                key = (t, need)
                if key not in made:
                    ct = f"{t}__as_{need}"
                    kind = "cast_f16_to_f32" if need == "f32" else "cast_f32_to_f16"
                    n = int(np.prod(tens[t]["shape"]))
                    tens[ct] = {"shape": list(tens[t]["shape"]), "dtype": need, "quant": None}
                    new_ops.append({"name": f"{ct}__cast", "op": kind, "inputs": [t], "outputs": [ct],
                                    "shape": {"n": n}, "dispatch_id": None, "hardware_target": "any",
                                    "depends_on": []})
                    casts.append({"tensor": t, "to": need, "op": f"{ct}__cast", "n": n})
                    made[key] = ct
                ins.append(made[key])
            o["inputs"] = ins
            if i in fp32_set:
                o["op"] = _strip(o["op"])
        new_ops.append(o)
    g["ops"] = new_ops
    for t in s32:
        if t in tens:
            tens[t]["dtype"] = "f32"
    w = dict(w16)
    fp32_consts = []
    for t in sorted(s32):
        key = tens.get(t, {}).get("const")
        if key:
            w[key] = np.asarray(w32[key], dtype=np.float32)
            fp32_consts.append(t)
    outs = g["output"].get("tensors") or [g["output"]["tensor"]]
    io = dict(io16)
    out32 = [t for t in outs if t in s32]
    if out32:
        if golden32 is None or np.asarray(golden32).size != np.asarray(io16["output"]).size:
            return {"graph": None, "refused": "island output but no matching fp32 golden"}
        io["output"] = np.asarray(golden32, dtype=np.float32)
    from .graph_hints import unique_names
    u = unique_names(json.loads(json.dumps(g16)))
    rep = {"format": FORMAT, "limit": limit,
           "flagged": sorted(flagged), "fp32_ops": [f"{u['ops'][i]['name']} ({_strip(u['ops'][i]['op'])})" for i in fp32_ops],
           "fp32_consts": fp32_consts, "casts": casts, "fp32_outputs": out32,
           "n_ops_fp16": len(g16["ops"]) - len(fp32_ops) - sum(o["op"] in ALIAS_OPS for o in g16["ops"]),
           "max_abs_fp32": max(stats[t][0] for t in flagged), "deviation": DEVIATION}
    g["precision_policy"] = rep
    log(f"[precision_policy] {len(flagged)} tensors above {limit:g}: {len(fp32_ops)} ops -> fp32, "
        f"{len(fp32_consts)} constants fp32, {len(casts)} casts, outputs fp32: {out32}")
    return {"graph": g, "weights": w, "io": io, **rep}


def check(ir_dir, log=print) -> dict:
    """Walk an IR with its storage semantics; compare the output with the golden (non-finite counts, max error
    relative to max |golden|)."""
    g, w, io = _load(Path(ir_dir))
    outs = g["output"].get("tensors") or [g["output"]["tensor"]]
    _, kept = walk_stats(g, w, io["input"], rounding=True, keep=tuple(outs))
    out = np.concatenate([kept[t].reshape(-1).double().numpy() for t in outs])
    gold = np.asarray(io["output"], dtype=np.float64)
    fin = np.isfinite(out) & np.isfinite(gold)
    scale = float(np.abs(gold[np.isfinite(gold)]).max()) if np.isfinite(gold).any() else 0.0
    err = float(np.abs(out[fin] - gold[fin]).max()) if fin.any() else math.inf
    rec = {"n": int(out.size), "nonfinite_walk": int((~np.isfinite(out)).sum()),
           "nonfinite_golden": int((~np.isfinite(gold)).sum()), "max_abs_golden": scale,
           "max_abs_err": err, "max_rel_err": (err / scale) if scale else None,
           "golden_dtype": str(io["output"].dtype)}
    log(f"[precision_policy] check {ir_dir}: {json.dumps(rec)}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--bench-file", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--policy", default="auto", choices=["auto", "off"])
    e.add_argument("--limit", type=float, default=DEFAULT_LIMIT)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("extra", nargs="*", help="extract_graph arguments after --")
    c = sub.add_parser("check")
    c.add_argument("ir")
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        rep = extract(a.bench_file, a.out_dir, extra=a.extra, policy=a.policy, limit=a.limit, seed=a.seed)
        print(json.dumps({k: rep.get(k) for k in ("applied", "reason", "refused", "detector", "fp32_ops", "check")},
                         indent=1))
    else:
        print(json.dumps(check(a.ir)))


if __name__ == "__main__":
    main()
