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


@dataclass
class IR:
    name: str
    graph: dict
    weights: dict[str, np.ndarray]
    golden_in: np.ndarray
    golden_out: np.ndarray

    @property
    def ops(self) -> list[dict]:
        return self.graph["ops"]

    @property
    def dtype(self) -> str:
        """fp16 / bf16 for an `--quant fp16` / `--quant bf16` extract (every float
        tensor stored as binary16 / bfloat16), else fp32."""
        return cuda_dtypes.ir_dtype(self.graph)

    def tshape(self, t: str) -> list[int]:
        return self.graph["tensors"][t]["shape"]

    def input_tensors(self, device) -> dict[str, torch.Tensor]:
        """Graph inputs by name. Multi-input graphs (matmul A/B, sdpa Q/K/V)
        pack every input into io.npz['input'] at `packed_inputs` offsets."""
        flat = torch.from_numpy(self.golden_in.astype(np.float32))
        packed = self.graph["input"].get("packed_inputs") or [
            {"name": self.graph["input"]["tensor"], "offset": 0, "size": flat.numel()}]
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
    return IR(g["name"], g, {k: w[k] for k in w.files}, io["input"], io["output"])


def compare(out: torch.Tensor, golden: np.ndarray, dtype: str = "fp32") -> dict:
    """max |out - golden| <= golden_rel * max|finite golden| + golden_abs (per dtype)."""
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
