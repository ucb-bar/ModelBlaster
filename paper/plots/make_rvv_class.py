#!/usr/bin/env python3
"""Per-class speedup of ModelBlaster over ExecuTorch with XNNPACK on RVV (Saturn V256D128, FireSim cycles).

Reads THREE_WAY_COMPARISON.csv (columns bucket, KB_vs_ET; KB is our system, ET is ExecuTorch + XNNPACK; >1 means ours is faster)
from the firesim2 rebuttal bundle (rebuttal-result_08-03-2026/rvv). The CSV stays local.
Usage: python3 make_rvv_class.py THREE_WAY_COMPARISON.csv [OUT_DIR]
"""
import csv, math, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1]
out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1])
ORDER = [("matmul", "Matmul"), ("reduction", "Reduction"), ("norm", "Norm"), ("activation", "Activation"),
         ("conv1d", "Conv1d"), ("conv2d", "Conv2d"), ("conv3d", "Conv3d")]
by = {}
for r in csv.DictReader(open(src)):
    by.setdefault(r["bucket"], []).append(float(r["KB_vs_ET"]))
gm = lambda a: math.exp(sum(map(math.log, a)) / len(a))
rows = [(lab, gm(by[k]), sum(x > 1 for x in by[k]), len(by[k])) for k, lab in ORDER]
for lab, g, w, n in rows:
    print(f"{lab:11s} n={n:2d} wins={w:2d} geomean={g:.3f}")

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(figsize=(6.6, 3.4))
x = range(len(rows))
colors = ["#2ca02c" if g >= 1 else "#d62728" for _, g, _, _ in rows]
ax.bar(x, [g - 1.0 for _, g, _, _ in rows], bottom=1.0, color=colors, width=0.62)   # bars start at the 1x line
ax.axhline(1.0, color="black", lw=1)
ax.set_yscale("log")
ax.set_ylim(0.012, 1500)
ax.set_xticks(list(x))
ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel(r"Speedup over ExecuTorch ($\times$, log)")
for i, (_, g, w, n) in enumerate(rows):
    ax.text(i, g * (1.2 if g >= 1 else 0.8), f"{g:.2f}$\\times$ ({w}/{n})", ha="center", va="bottom" if g >= 1 else "top", fontsize=8.5)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(out / f"rvv_class_speedup.{ext}", dpi=300, bbox_inches="tight")
print("wrote", out / "rvv_class_speedup.png")
