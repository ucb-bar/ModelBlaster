"""SmolVLA end to end on the Vulkan backend: vision -> prefix (KV cache) -> N flow steps.

Runs the three IRs extracted from models/smolvla_split.py (``extract_graph
--model smolvla_split --quant fp32`` with MB_SMOLVLA_PART=vision|prefix|step)
as replayable Vulkan graphs (walk_vulkan.VkGraph) and the flow loop on the
host, exactly as lerobot's VLAFlowMatching.sample_actions does, then compares
the action chunk with a reference (lerobot's own sample_actions output for the
same observation and noise) and times each component.

    python smolvla_vulkan.py --irs DIR --obs obs.pt [--steps 5] [--reps 10] [--json out.json]

DIR holds vision/ prefix/ step/; obs.pt is a torch.save'd dict with pixels
[1,3,H,W] in [-1,1], lang [1,L,960] (embedded prompt), state [1,32], noise
[1,50,32] and, optionally, ref [1,50,32] (the PyTorch actions).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from mbir import load_ir
from mbvk import VkBackend
from walk_vulkan import VkGraph, VkKernelSet, Weights, build_plan


def time_embedding(t: float, dim: int, min_period: float = 4e-3, max_period: float = 4.0) -> np.ndarray:
    """lerobot create_sinusoidal_pos_embedding (float64 math, fp32 result) -> [1, dim]."""
    fraction = np.linspace(0.0, 1.0, dim // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction
    x = (1.0 / period * 2 * math.pi) * t
    return np.concatenate([np.sin(x), np.cos(x)])[None, :].astype(np.float32)


class SmolVLAVulkan:
    def __init__(self, ir_root: str | Path, vk: VkBackend, specs: dict, f16_weights: bool = False):
        self.vk = vk
        ks = VkKernelSet(specs, vk)
        self.graphs, self.load_s = {}, {}
        # chained in place: prefix reads vision's output buffer, step reads prefix's KV cache
        feeds = {"vision": None, "prefix": ("vision", 0), "step": ("prefix", 0)}
        for part in ("vision", "prefix", "step"):
            t0 = time.perf_counter()
            ir = load_ir(Path(ir_root) / part)
            plan, missing = build_plan(ir, ks)
            if missing:
                raise RuntimeError(f"{part}: no Vulkan kernel for {missing}")
            w = Weights(ir, vk, f16=f16_weights)
            w.preload(plan)
            bound = None
            if feeds[part]:
                src = self.graphs[feeds[part][0]]
                bound = {list(ir.input_tensors("cpu"))[feeds[part][1]]: src.env[src.outputs[0]]}
            self.graphs[part] = VkGraph(ir, plan, vk, w, bound=bound)
            ir.weights.clear()                  # the GPU copy is the one that is used
            self.load_s[part] = time.perf_counter() - t0
        self.e_hidden = self.graphs["step"].env[self.graphs["step"].inputs[-1]].size

    def sample_actions(self, pixels, lang, state, noise, num_steps: int, timings: dict | None = None):
        t = time.perf_counter
        t0 = t()
        self.graphs["vision"](pixels)                 # -> image tokens, read in place by prefix
        t1 = t()
        self.graphs["prefix"](lang, state)            # -> KV cache, read in place by every step
        t2 = t()
        dt = -1.0 / num_steps
        x = np.asarray(noise, dtype=np.float32).copy()
        step_s = []
        for s in range(num_steps):
            ts = t()
            (v,) = self.graphs["step"](x, time_embedding(1.0 + s * dt, self.e_hidden))
            x = x + dt * v.reshape(x.shape)
            step_s.append(t() - ts)
        if timings is not None:
            timings.update(vision=t1 - t0, prefix=t2 - t1, denoise=sum(step_s), step_mean=sum(step_s) / num_steps,
                           total=t() - t0)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irs", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--lib", default=None)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--f16-weights", action="store_true",
                    help="the prefix / expert linears read fp16 weights (fp32 activations and accumulation)")
    ap.add_argument("--json")
    a = ap.parse_args()
    vk = VkBackend(a.lib)
    print(f"# {vk.info}", flush=True)
    obs = torch.load(a.obs)
    f = lambda k: obs[k].detach().float().numpy()
    model = SmolVLAVulkan(a.irs, vk, json.load(open(a.specs)), f16_weights=a.f16_weights)
    print(f"# load (IR read + weight upload + capture): " +
          ", ".join(f"{k} {v:.1f}s" for k, v in model.load_s.items()), flush=True)
    tim = {}
    first = model.sample_actions(f("pixels"), f("lang"), f("state"), f("noise"), a.steps, tim)
    runs = []
    for _ in range(a.reps):
        r = {}
        out = model.sample_actions(f("pixels"), f("lang"), f("state"), f("noise"), a.steps, r)
        runs.append(r)
    med = {k: 1e3 * statistics.median(r[k] for r in runs) for k in runs[0]}
    rec = {"device": vk.device, "steps": a.steps, "f16_weights": a.f16_weights,
           "packed_linears": int(vk.lib.mb_vk_packed_count()) if hasattr(vk.lib, "mb_vk_packed_count") else 0, "first_ms": round(1e3 * tim["total"], 1),
           "median_ms": {k: round(v, 2) for k, v in med.items()},
           "deterministic": bool(np.array_equal(first, out)), "finite": bool(np.isfinite(out).all())}
    if "ref" in obs:
        ref = obs["ref"].detach().double().numpy().reshape(-1)
        o = out.astype(np.float64).reshape(-1)
        rec["vs_pytorch"] = {
            "cosine": float(o @ ref / (np.linalg.norm(o) * np.linalg.norm(ref))),
            "max_abs_err": float(np.abs(o - ref).max()),
            "ref_abs_max": float(np.abs(ref).max()),
            "rel_to_range": float(np.abs(o - ref).max() / np.abs(ref).max())}
        ad = obs.get("action_dim", 6)
        o6, r6 = o.reshape(-1, out.shape[-1])[:, :ad], ref.reshape(-1, out.shape[-1])[:, :ad]
        rec["vs_pytorch"][f"max_abs_err_first{ad}"] = float(np.abs(o6 - r6).max())
    print(json.dumps(rec), flush=True)
    if a.json:
        json.dump(rec, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
