"""SmolVLA on the Vulkan backend over a set of observations: action-space accuracy against each
observation's reference (lerobot's sample_actions), run-to-run determinism, and, optionally, the
distance to a CPU golden of the same IRs (pipeline/int8_mixed.walk: int8 bit-exact, fp32 ops as
walk_torch) -- the device's own fp32 rounding, separated from the quantization error.

    python smolvla_eval_set.py --irs DIR --specs specs.json --obs 'glob' [--obs ...] [--golden DIR --golden-tag T]
                               [--repeat 3] [--json out.jsonl] [--save DIR]

The model is loaded once; every observation runs `--repeat` times (replayed graphs), and the action
chunks must be bit-identical (deterministic). --golden DIR holds T__<obs stem>.npy (validate.py --save)."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from mbvk import VkBackend
from smolvla_vulkan import SmolVLAVulkan


def metrics(o: np.ndarray, r: np.ndarray, ad: int = 6) -> dict:
    o, r = o.astype(np.float64).reshape(-1, 32), r.astype(np.float64).reshape(-1, 32)
    d = np.abs(o - r)
    of, rf = o.reshape(-1), r.reshape(-1)
    return {"cos": float(of @ rf / (np.linalg.norm(of) * np.linalg.norm(rf))), "maxerr": float(d.max()),
            "rmse": float(np.sqrt((d ** 2).mean())), "joint_maxerr": [round(float(x), 5) for x in d[:, :ad].max(0)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--irs", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--obs", action="append", required=True)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--golden")
    ap.add_argument("--golden-tag", default="")
    ap.add_argument("--json")
    ap.add_argument("--save")
    a = ap.parse_args()
    files = sorted({f for g in a.obs for f in glob.glob(g)})
    vk = VkBackend()
    model = SmolVLAVulkan(a.irs, vk, json.load(open(a.specs)))
    fo = open(a.json, "a") if a.json else None
    n_nondet = 0
    for f in files:
        obs = torch.load(f)
        g = lambda k: obs[k].detach().float().numpy()
        outs = [model.sample_actions(g("pixels"), g("lang"), g("state"), g("noise"), a.steps) for _ in range(a.repeat)]
        det = all(np.array_equal(outs[0].view(np.uint32), o.view(np.uint32)) for o in outs[1:])
        n_nondet += not det
        rec = {"obs": Path(f).name, "image": obs.get("image"), "deterministic": det,
               **metrics(outs[0], obs["ref"].detach().numpy())}
        if a.golden:
            gp = Path(a.golden) / f"{a.golden_tag}__{Path(f).stem}.npy"
            if gp.exists():
                gm = metrics(outs[0], np.load(gp))
                rec["vs_golden"] = {"cos": gm["cos"], "maxerr": gm["maxerr"]}
        if a.save:
            np.save(Path(a.save) / f"{Path(f).stem}.npy", outs[0])
        line = json.dumps(rec)
        print(line, flush=True)
        if fo:
            fo.write(line + "\n")
            fo.flush()
    print(json.dumps({"n": len(files), "nondeterministic": n_nondet}), flush=True)


if __name__ == "__main__":
    main()
