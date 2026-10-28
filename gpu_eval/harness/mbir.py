"""Minimal ModelBlaster IR loader shared by the GPU walkers.

Reads the three artifacts `modelblaster.pipeline.extract_graph` writes
(graph.json, weights.npz, io.npz) without importing ModelBlaster, so the
harnesses run on a GPU box that only has torch + numpy.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ops ModelBlaster lowers to aliases (no kernel call in model.c).
ZERO_COST_OPS = {"view", "chunk2_c1"}

# storage dtypes + tolerances (pipeline/cuda_dtypes.py, loaded by path like mbabi)
_spec = importlib.util.spec_from_file_location(
    "cuda_dtypes", Path(__file__).resolve().parents[2] / "pipeline" / "cuda_dtypes.py")
cuda_dtypes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cuda_dtypes)


def _weight_derivations():
    """pipeline/weight_derivations.py, loaded by path like cuda_dtypes (only IRs with derived weights need it)."""
    spec = importlib.util.spec_from_file_location(
        "weight_derivations", Path(__file__).resolve().parents[2] / "pipeline" / "weight_derivations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class IR:
    name: str
    graph: dict
    weights: dict[str, np.ndarray]
    golden_in: np.ndarray
    golden_out: np.ndarray
    # a mixed IR's io.npz (--quant native, docs/tensor_dtypes.md): every graph
    # input / output at its own dtype, as `in.<name>` / `out.<name>`
    io_tensors: dict | None = None
    # the directory it was loaded from (sol_tolerance finds SOL's reference outputs beside it)
    dir: str | None = None

    @property
    def ops(self) -> list[dict]:
        return self.graph["ops"]

    @property
    def dtype(self) -> str:
        """fp16 / bf16 for an `--quant fp16` / `--quant bf16` extract (every float
        tensor stored as binary16 / bfloat16), else fp32. A mixed IR (`--quant
        native`) reads as its narrowest float storage (cuda_dtypes.ir_dtype);
        `tdtype` is each tensor's own."""
        return cuda_dtypes.ir_dtype(self.graph)

    def tdtype(self, t: str) -> str:
        """IR dtype of tensor t (f32 / f16 / bf16 / i64 / i32 / b8 / u8)."""
        return cuda_dtypes.tensor_dtype(self.graph, t)

    def tshape(self, t: str) -> list[int]:
        return self.graph["tensors"][t]["shape"]

    def input_tensors(self, device) -> dict[str, torch.Tensor]:
        """Graph inputs by name. Multi-input graphs (matmul A/B, sdpa Q/K/V)
        pack every input into io.npz['input'] at `packed_inputs` offsets. A
        mixed IR's come from its per-tensor arrays: floats as float32 (bf16
        values exact), integer / bool inputs at their own dtype."""
        packed = self.graph["input"].get("packed_inputs") or [
            {"name": self.graph["input"]["tensor"], "offset": 0, "size": self.golden_in.size}]
        per = self.io_tensors or {}
        if packed and all(f"in.{p['name']}" in per for p in packed):
            out = {}
            for p in packed:
                a = np.ascontiguousarray(per[f"in.{p['name']}"])
                t = torch.from_numpy(a.astype(np.float32) if a.dtype.kind == "f" else a)
                out[p["name"]] = t.reshape(self.tshape(p["name"])).to(device).clone()
            return out
        flat = torch.from_numpy(self.golden_in.astype(np.float32))
        return {p["name"]: flat[p["offset"]:p["offset"] + p["size"]]
                .reshape(self.tshape(p["name"])).to(device).clone()
                for p in packed}

    def const_tensors(self, device, int_dtype=torch.int64) -> dict[str, torch.Tensor]:
        """IR tensors backed by weights.npz (folded constants, gather indices)."""
        out = {}
        for name, meta in self.graph["tensors"].items():
            key = meta.get("const")
            if key:
                arr = self.weights[key]
                t = torch.from_numpy(np.ascontiguousarray(arr))
                out[name] = (t.to(int_dtype) if arr.dtype.kind == "i" else t.float()).to(device)
        return out

    def output_names(self) -> list[str]:
        o = self.graph["output"]
        return o.get("tensors") or [o["tensor"]]

    def weight(self, key: str | None, device) -> torch.Tensor | None:
        if key is None:
            return None
        return torch.from_numpy(np.ascontiguousarray(self.weights[key], dtype=np.float32)).to(device)


def load_ir(d: str | Path) -> IR:
    d = Path(d)
    g = json.loads((d / "graph.json").read_text())
    w = np.load(d / "weights.npz")
    io = np.load(d / "io.npz")
    per = {k: io[k] for k in io.files if k.startswith(("in.", "out."))}
    weights = {k: w[k] for k in w.files}
    if g.get("weight_derivations"):
        # derived weights (pipeline/weight_derivations.py): rebuild any entry weights.npz lacks from its base weights
        weights = _weight_derivations().fill_missing(g, weights)
    return IR(g["name"], g, weights, io["input"], io["output"], per or None, str(d))


def _sol_tolerance():
    """pipeline/sol_tolerance.py, loaded by path like cuda_dtypes (numpy only at import)."""
    spec = importlib.util.spec_from_file_location(
        "sol_tolerance", Path(__file__).resolve().parents[2] / "pipeline" / "sol_tolerance.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_sol(out, ref, tol: dict | None) -> dict:
    """SOL-ExecBench's per-element check of ``out`` against ``ref`` (sol_tolerance.sol_compare): matched ratio on
    ``|out - ref| <= max_atol + max_rtol * |ref|``, inf / NaN, all-zero vs non-zero, max_error_cap."""
    return _sol_tolerance().sol_compare(out, ref, tol)


def sol_for(ir: "IR") -> dict | None:
    """The SOL tolerance the golden checks add (docs/sol_gate.md): the first output's recorded ``sol_tolerance``, only when the
    SOL gate is on (``MB_SOL_GATE`` final | accept) -- off (the default): None, the checks are what they were."""
    import os
    if os.environ.get("MB_SOL_GATE", "off").strip().lower() in ("", "off", "0", "no", "false"):
        return None
    outs = (ir.graph.get("sol_tolerance") or {}).get("outputs")
    return dict(outs[0]) if outs else None


def compare(out: torch.Tensor, golden: np.ndarray, dtype: str = "fp32", sol: dict | None = None) -> dict:
    """max |out - golden| <= golden_rel * max|finite golden| + golden_abs (per dtype). With ``sol`` (a SOL_TOLERANCE
    dict) the result also carries ``sol`` = SOL's check of ``out`` against the golden, and ``ok`` is both."""
    res = _compare_bound(out, golden, dtype)
    if sol is not None and "why" not in res:
        res["sol"] = compare_sol(out, golden, sol)
        res["ok"] = bool(res["ok"] and res["sol"]["ok"])
    return res


def _compare_bound(out: torch.Tensor, golden: np.ndarray, dtype: str = "fp32") -> dict:
    tol = cuda_dtypes.TOLERANCE[dtype]
    o = out.detach().float().cpu().reshape(-1).numpy()
    g = golden.astype(np.float32).reshape(-1)
    if o.shape != g.shape:
        return {"ok": False, "why": f"shape {o.shape} != golden {g.shape}"}
    # matching infinities / NaNs (masked_fill(-inf) outputs) are exact
    same = (o == g) | (np.isnan(o) & np.isnan(g))
    with np.errstate(invalid="ignore"):
        err = np.where(same, 0.0, np.abs(o - g))
    fin = g[np.isfinite(g)]
    denom = max(float(np.abs(fin).max()) if fin.size else 0.0, 1e-12)
    max_abs = float(err.max()) if err.size else 0.0
    # fp32 golden comes from PyTorch on CPU; GPU reduction order differs, so
    # use a relative-to-range bound rather than bit-exactness.
    ok = bool((np.isfinite(o) | same).all() and max_abs <= tol["golden_rel"] * denom + tol["golden_abs"])
    return {"ok": ok, "max_abs_err": max_abs, "rel_to_range": max_abs / denom}


def strict_fp32():
    """Disable TF32 so GPU fp32 conv/matmul are comparable to the CPU golden."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
