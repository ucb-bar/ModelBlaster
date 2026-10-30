"""Phase D composite Gantt — assemble the 12 per-cell PNGs into one 3x4 figure.

Reads the sweep grid CSV and the per-cell directories, lays out the
Gantts as a single matplotlib figure with policies as rows and
frequency configs as columns.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


POLICY_ORDER = ["yolo_anchor", "periodic_anchor", "critical_path_first",
                "cpsat_unconstrained", "hybrid_periodic_mosek_yolo"]


def render(sweep_dir: Path, out_png: Path) -> int:
    grid_csv = sweep_dir / "grid.csv"
    if not grid_csv.is_file():
        print(f"missing {grid_csv}")
        return 1
    cells_by_freq_policy = {}
    freqs_seen = []
    policies_seen = []
    rows = list(csv.DictReader(open(grid_csv)))
    for r in rows:
        freq = r["freq_label"]
        policy = r["policy"]
        if freq not in freqs_seen:
            freqs_seen.append(freq)
        if policy not in policies_seen:
            policies_seen.append(policy)
        cells_by_freq_policy[(freq, policy)] = r

    # Order policies by canonical order, falling back to seen order.
    policies = [p for p in POLICY_ORDER if p in policies_seen]
    for p in policies_seen:
        if p not in policies:
            policies.append(p)
    freqs = freqs_seen

    n_rows, n_cols = len(policies), len(freqs)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 3 * n_rows),
                              squeeze=False)

    for i, policy in enumerate(policies):
        for j, freq in enumerate(freqs):
            ax = axes[i][j]
            ax.set_axis_off()
            r = cells_by_freq_policy.get((freq, policy))
            if r is None:
                ax.set_title(f"(missing) {policy}\n{freq}")
                continue
            cell_dir = sweep_dir / "cells" / r["cell"]
            gantt = cell_dir / "gantt.png"
            mksp = r.get("makespan_us", "?")
            dlmiss = r.get("n_deadline_miss", "?")
            status = r.get("status", "?")
            if gantt.is_file():
                img = mpimg.imread(str(gantt))
                ax.imshow(img)
            try:
                mksp_fmt = f"{float(mksp):.1f} ms"
            except (TypeError, ValueError):
                mksp_fmt = str(mksp)
            title = (f"{policy}\n{freq}\n"
                     f"makespan={mksp_fmt} miss={dlmiss}")
            if status not in ("ok", "reused"):
                title += f"\n(status={status})"
            ax.set_title(title, fontsize=9)

    fig.suptitle(
        f"Phase D sweep — {sweep_dir.name}\n"
        f"canonical 4 MLP + 2 Dronet + 1 Yolo on hetero bitstream",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"wrote {out_png}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    return render(args.sweep_dir, args.out)


if __name__ == "__main__":
    import sys
    sys.exit(main())
