#!/usr/bin/env python3
"""fast_p(r) for KernelBench Level 1, 2 and 3, rebuilt from the raw run directories.

Same method as tachyon-rl-fresh/results/plot_speedup_over_pytorch_combined.py:
  * Level 1, 2 (H100, A6000, L40S): ours = min over Best_1..Best_10 from raw/<run>/rl_ncu files
    (checker-passed problems only); speedup = torch.compile cycles / ours; fast_p is the share of
    VALID problems with speedup >= r. Level 1 also has the OpenCL series (Adreno).
  * Level 3 (L40S): X-152 headline sweep, speedup = torch eager / our final model (46 problems).
Run:  python3 make_fastp.py   (needs numpy, pandas, matplotlib)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import reproduce_from_raw as R

HERE = Path(__file__).resolve().parent
R_MAX = 25.0
COLORS = {"H100": "#1f77b4", "A6000": "#ff7f0e", "L40S": "#2ca02c", "OpenCL": "#d62728"}


MERGE_CUDNN = "--cudnn-fill" in sys.argv   # mirrors run_plots.sh --cudnn-csv-level1/2 / *_cudnnfill.csv


def _cudnn_cycles(gpu: str, level: int) -> dict:
    name = f"baseline_cudnn{'_level2' if level == 2 else ''}_{gpu}.csv"
    p = R.BASE / name
    if not p.is_file():
        return {}
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    return {str(r.iloc[0]): float(r.iloc[1]) for _, r in df.iterrows() if pd.notna(r.iloc[1]) and float(r.iloc[1]) > 0}


def level_speedups(label: str, level: int) -> np.ndarray:
    run, gpu = R.RUNS[(label, level)]
    ours = R.ours_table(run, level)
    if not MERGE_CUDNN or gpu == "a6000":
        return R.speedups(R.load_base(R.base_csv(gpu, level, "compile")), ours)
    # merge with the library-enabled flow: baseline = better of eager/compile; ours = faster of this run and the
    # earlier KernelAgent-flow kernels (cuDNN/CUTLASS allowed) profiled by run_cudnn_ncu.sh -> baseline_cudnn_*.csv
    comp, eag = R.load_base(R.base_csv(gpu, level, "compile")), R.load_base(R.base_csv(gpu, level, "eager"))
    m = pd.merge(eag, comp, on="problem", how="outer", suffixes=("_e", "_c"))
    pt = pd.DataFrame({"problem": m["problem"].astype(str), "cycles": np.fmin(m["cycles_e"], m["cycles_c"])})
    mm = pd.merge(pt, ours, on="problem", how="left")
    b, o = mm["cycles"].to_numpy(float), mm["ours"].to_numpy(float)
    sp = np.where((b > 0) & (o > 0), b / o, np.nan)
    cd = _cudnn_cycles(gpu, level)
    for i, p in enumerate(mm["problem"]):
        c = cd.get(p)
        if b[i] > 0 and c:
            s = b[i] / c
            if not np.isfinite(sp[i]) or s > sp[i]:
                sp[i] = s
    return sp[np.isfinite(sp) & (sp > 0)]


def opencl_l1() -> np.ndarray:
    b = pd.read_csv(R.BASE / "baseline_opencl_pws_level1.csv")
    t = pd.read_csv(R.BASE / "tachyon_opencl_pws_level1.csv")
    m = pd.merge(b, t, left_on="problem", right_on="Kernel ID")
    base, ours = m["avg_Elapsed_Cycles"].to_numpy(float), m["Best_1"].to_numpy(float)
    ok = (base > 0) & (ours > 0)
    return base[ok] / ours[ok]


def level3_l40s() -> np.ndarray:
    df = pd.read_csv(HERE / "inputs" / "l3_flow_results.csv")
    df = df[df["dataset"] == "l3_l40s_sweep"]
    m = pd.to_numeric(df["model_us"], errors="coerce").to_numpy(float)
    e = pd.to_numeric(df["torch_eager_us"], errors="coerce").to_numpy(float)
    ok = (m > 0) & (e > 0)
    return e[ok] / m[ok]


def curve(sp: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return (sp.reshape(1, -1) >= grid.reshape(-1, 1)).mean(axis=1) * 100.0


def gmean(sp: np.ndarray) -> float:
    return float(np.exp(np.log(sp).mean()))


def main() -> None:
    grid = np.linspace(0, R_MAX, 501)
    l1 = {k: level_speedups(k, 1) for k in ("H100", "A6000", "L40S")}
    l1["OpenCL"] = opencl_l1()
    l2 = {k: level_speedups(k, 2) for k in ("A6000", "H100", "L40S")}
    l3 = {"L40S": level3_l40s()}
    tag = "best of eager/compile; merged with library-enabled runs" if MERGE_CUDNN else "vs torch.compile"
    panels = [("Level 1", l1, tag), ("Level 2", l2, tag), ("Level 3", l3, "vs torch eager")]
    suffix = "_cudnnfill" if MERGE_CUDNN else ""

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for ax, (title, series, base) in zip(axes, panels):
        for key, sp in series.items():
            ax.plot(grid, curve(sp, grid), color=COLORS[key], linewidth=2.6, label=f"{key}, n={sp.size}")
        ax.set_title(f"{title} ({base})")
        ax.set_xlabel("r (times faster than baseline)")
        ax.set_xlim(0, R_MAX)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.95)
    axes[0].set_ylabel(r"% valid kernels with speedup $\geq$ r")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fastp_l1_l2_l3{suffix}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for title, series, base in panels:
        for key, sp in series.items():
            print(f"{title} {key:6s} n={sp.size:3d} geomean={gmean(sp):.3f} median={np.median(sp):.3f} "
                  f"fast_1={curve(sp, np.array([1.0]))[0]:.1f}% fast_2={curve(sp, np.array([2.0]))[0]:.1f}%  ({base})")


if __name__ == "__main__":
    main()
