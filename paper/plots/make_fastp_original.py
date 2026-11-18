#!/usr/bin/env python3
"""Figure A: original same-flow, same-LLM comparison across GPUs (H100, A6000, L40S), Levels 1 and 2.
Built from the raw run directories (raw/), checker-passed problems only, speedup over torch.compile,
fast_p = share of valid problems with speedup >= r. Output: fastp_original_l1_l2.{png,pdf}"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import make_fastp as M

HERE = Path(__file__).resolve().parent
grid = np.linspace(0, M.R_MAX, 501)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, lvl in zip(axes, (1, 2)):
    for g in ("H100", "A6000", "L40S"):
        sp = M.level_speedups(g, lvl)
        ax.plot(grid, M.curve(sp, grid), color=M.COLORS[g], linewidth=2.6, label=f"{g}, n={sp.size}")
    ax.set_title(f"Level {lvl}"); ax.set_xlabel("r (times faster than PyTorch)")
    ax.set_xlim(0, M.R_MAX); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3); ax.legend(loc="upper right")
axes[0].set_ylabel(r"% valid kernels with speedup $\geq$ r")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"fastp_original_l1_l2.{ext}", dpi=300, bbox_inches="tight")
