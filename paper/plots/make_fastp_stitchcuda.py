#!/usr/bin/env python3
"""fast_p(r) on L40S, KernelBench Level 1 and 2: ModelBlaster vs StitchCUDA and AI CUDA Engineer (Sakana).

fast_p(r) is the share of valid (correct) problems whose speedup is at least r.
  All three systems are shown as speedup over PyTorch eager.
  Ours:       final model time from <farm>/results_l12/6NNNN_*/work/summary.json against the X-168 eager baseline
              (<farm>/analysis/l12_sweep/torch_compile/summary.csv), wall-clock.
  Sakana:     AI CUDA Engineer kernels rerun end to end on the L40S (CUDA-event wall-clock, robust-kbench harness; --sakana-rerun
              summary.csv, column speedup_native = speedup over torch native). Correct tasks only; tasks the old pipeline marked dirty
              (library-dependent) were not run. Older NCU-cycle variant: --sakana-ncu.
  StitchCUDA: per-problem CSVs (--stitch-l1: problem_id, correct, best_speedup_vs_pytorch;
              --stitch-l2: problem, correct, speedup); wall-clock speedup over PyTorch in its own harness.
Usage: python3 make_fastp_stitchcuda.py --stitch-l1 L1.csv --stitch-l2 L2.csv [--farm DIR] [--out-dir DIR]
"""
import argparse, csv, glob, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stitch-l1", required=True)
ap.add_argument("--stitch-l2", required=True)
ap.add_argument("--sakana-rerun", default="/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/sakana_rerun/summary.csv")
ap.add_argument("--sakana-ncu", action="store_true", help="use the old NCU-cycle Sakana data instead of the wall-clock rerun")
ap.add_argument("--sakana-dir", default="/Users/krisd/Documents/ModelBlaster-clone/Blaster/plot/fastp_with_l3/inputs/baselines")
ap.add_argument("--xscale", choices=("log", "linear"), default="log", help="log shows ModelBlaster's long Level 2 tail (up to ~300x)")
ap.add_argument("--farm", default="/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1]))
a = ap.parse_args()
FARM = Path(a.farm)
R_MAX = 25.0


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def sakana(lvl):
    if not a.sakana_ncu:
        with open(a.sakana_rerun) as fh:
            return [num(r["speedup_native"]) for r in csv.DictReader(fh)
                    if r["level"] == f"level_{lvl}" and r["correct"] == "True" and r["old_csv_status"] != "dirty" and num(r["speedup_native"])]
    d = Path(a.sakana_dir)
    suf = "_level2" if lvl == 2 else ""

    def load(name):
        with open(d / name, encoding="utf-8-sig") as fh:
            return {r["problem"].strip(): num(r["avg_Elapsed_Cycles"]) for r in csv.DictReader(fh)}

    pt, sk = load(f"baseline_pytorch{suf}_l40s.csv"), load(f"baseline_sakana_l40s{suf}_clean.csv")
    return [pt[k] / v for k, v in sk.items() if v and pt.get(k)]


def ours():
    base = {}
    with open(FARM / "analysis/l12_sweep/torch_compile/summary.csv") as fh:
        for r in csv.DictReader(fh):
            base[r["name"]] = (num(r["eager_us"]), num(r["compile_us"]))
    out = {1: [], 2: []}
    for f in sorted(glob.glob(str(FARM / "results_l12/6*/work/summary.json"))):
        n = f.split("/")[-3]
        try:
            r = list(json.load(open(f))["irs"].values())[0]
        except Exception:
            continue
        m = num(r.get("model_e2e_us"))
        e, c = base.get(n, (None, None))
        if not (r.get("model_passed") and m and e):
            continue
        lvl = 1 if n.startswith("61") else 2
        out[lvl].append(e / m)
    return out


def stitch(path, col):
    with open(path) as fh:
        return [num(r[col]) for r in csv.DictReader(fh) if r["correct"] == "True" and num(r[col])]


def curve(sp, grid):
    sp = np.asarray(sp)
    return (sp.reshape(1, -1) >= grid.reshape(-1, 1)).mean(axis=1) * 100.0


mine = ours()
data = {
    1: dict(sakana=sakana(1), stitch=stitch(a.stitch_l1, "best_speedup_vs_pytorch"), ours=mine[1]),
    2: dict(sakana=sakana(2), stitch=stitch(a.stitch_l2, "speedup"), ours=mine[2]),
}
grid = np.logspace(np.log10(0.2), np.log10(400), 700) if a.xscale == "log" else np.linspace(0, R_MAX, 501)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 10.5})
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
for ax, lvl in zip(axes, (1, 2)):
    d = data[lvl]
    ax.plot(grid, curve(d["sakana"], grid), color="orange", lw=2.6, label=f"AI CUDA Engineer (Sakana), n={len(d['sakana'])}")
    ax.plot(grid, curve(d["stitch"], grid), color="tab:green", lw=2.6, label=f"StitchCUDA, n={len(d['stitch'])}")
    ax.plot(grid, curve(d["ours"], grid), color="steelblue", lw=2.8, label=f"ModelBlaster (ours), n={len(d['ours'])}")
    ax.set_title(f"L40S Level {lvl}")
    ax.set_xlabel(r"r ($\times$ speedup over PyTorch eager)")
    if a.xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(0.2, 400)
        ax.axvline(1.0, color="gray", lw=0.8, ls=":")
    else:
        ax.set_xlim(0, R_MAX)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    if a.xscale == "log":   # curves fill the panel, so the legend goes below it
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    else:
        ax.legend(loc="upper right", framealpha=0.95)
    for k, v in d.items():
        print(f"L{lvl} {k:7s} n={len(v):3d} geomean={np.exp(np.log(v).mean()):.3f} median={np.median(v):.3f} "
              f"fast_1={curve(v, np.array([1.0]))[0]:.1f}% fast_2={curve(v, np.array([2.0]))[0]:.1f}%")
axes[0].set_ylabel(r"% of correct kernels with speedup $\geq$ r")
fig.tight_layout()
out = Path(a.out_dir)
for ext in ("png", "pdf"):
    fig.savefig(out / f"fastp_stitchcuda_l40s.{ext}", dpi=300, bbox_inches="tight")
print("wrote", out / "fastp_stitchcuda_l40s.png")
