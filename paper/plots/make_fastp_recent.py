#!/usr/bin/env python3
"""Figure B: recent-model results. Only finished experiments are plotted; panels or series appear
automatically once their data exist.
  L40S (CUDA): speedup over torch eager. L3 = X-152 (done). L1+L2 = X-162 (running; add rows to
  l3_farm flow_results.csv with datasets 'l40s_l1' / 'l40s_l2' and rerun).
  Qualcomm QRB5165 (Adreno 650, OpenCL): speedup over the flow's own seed (no torch on the board).
  L2 = X-159 (done, inputs/qrb_l2_results.json), L3 = X-161 (done), L1 = X-165 (running).
Baselines differ (eager vs seed), so they are stated in the legend. Log x axis. Output: fastp_recent_models.{png,pdf}"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FR = pd.read_csv(HERE / "inputs" / "l3_flow_results.csv")


def col(ds, base):
    d = FR[FR["dataset"] == ds]
    if d.empty:
        return np.array([])
    m = pd.to_numeric(d["model_us"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(d[base], errors="coerce").to_numpy(float)
    ok = (m > 0) & (b > 0)
    return b[ok] / m[ok]


import json
qrb_l2 = np.array([v["speedup"] for v in json.load(open(HERE / "inputs" / "qrb_l2_results.json")).values() if v.get("speedup", 0) > 0], float)
qrb_l3 = pd.read_csv(HERE / "inputs" / "qrb_l3_speedups.csv")["speedup_over_seed"].to_numpy(float)
series = {  # level -> list of (label, speedups, color)
    1: [("L40S (vs eager)", col("l40s_l1", "torch_eager_us"), "#2ca02c"), ("Qualcomm (vs seed)", col("qrb_l1", "seed_us"), "#d62728")],
    2: [("L40S (vs eager)", col("l40s_l2", "torch_eager_us"), "#2ca02c"), ("Qualcomm (vs seed)", qrb_l2, "#d62728")],
    3: [("L40S (vs eager)", col("l3_l40s_sweep", "torch_eager_us"), "#2ca02c"), ("Qualcomm (vs seed)", qrb_l3, "#d62728")],
}
panels = [(lv, [s for s in ser if s[1].size]) for lv, ser in series.items()]
panels = [p for p in panels if p[1]]
grid = np.logspace(0, np.log10(600), 400)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11})
fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.8), sharey=True, squeeze=False)
for ax, (lv, ser) in zip(axes[0], panels):
    for label, sp, c in ser:
        y = (sp.reshape(1, -1) >= grid.reshape(-1, 1)).mean(axis=1) * 100
        ax.plot(grid, y, color=c, linewidth=2.6, label=f"{label}, n={sp.size}")
    ax.set_xscale("log"); ax.set_xlim(1, 600); ax.set_ylim(0, 100)
    ax.set_title(f"Level {lv}"); ax.set_xlabel("r (times faster than baseline)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper right")
axes[0][0].set_ylabel(r"% kernels with speedup $\geq$ r")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fastp_recent_models.{ext}", dpi=300, bbox_inches="tight")
for lv, ser in panels:
    for label, sp, _ in ser:
        print(f"Level {lv} {label}: n={sp.size} geomean={np.exp(np.log(sp).mean()):.2f} median={np.median(sp):.2f}")
