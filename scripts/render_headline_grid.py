#!/usr/bin/env python3
"""D4 — assemble the 12-cell sweep into a single 3×4 figure.

Rows = frequency configurations (canon / tight_mlp / slack_dronet).
Cols = policies (yolo_anchor / periodic_anchor / critical_path_first /
       cpsat_unconstrained).

Each cell shows the band-aware Gantt with period bands + red overruns.
Output: artifacts/sweeps/<run>/headline_3x4.png + .pdf.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


CONFIGS = ["canon", "tight_mlp", "slack_dronet"]
POLICIES = ["yolo_anchor", "periodic_anchor", "critical_path_first",
            "cpsat_unconstrained"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    gantts = sweep / "gantts"
    if not gantts.exists():
        raise SystemExit(f"no gantts dir at {gantts}")

    out_path = Path(args.out) if args.out else sweep / "headline_3x4.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    nrows, ncols = 3, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(28, 12))

    # Load CSV to get per-cell summary metrics for annotation.
    summary = {}
    csv_path = sweep / "grid_headline.csv"
    if csv_path.exists():
        import csv
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                summary[row["cell_id"]] = row

    for r, cfg in enumerate(CONFIGS):
        for c, pol in enumerate(POLICIES):
            ax = axes[r, c]
            cell_id = f"{cfg}__{pol}"
            png_path = gantts / f"{cell_id}.png"
            if png_path.exists():
                img = imread(str(png_path))
                ax.imshow(img, aspect="auto")
            else:
                ax.text(0.5, 0.5, f"(missing\n{cell_id})",
                         ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            row = summary.get(cell_id, {})
            mksp = row.get("makespan_ms", "?")
            dl = row.get("n_deadline_miss", "?")
            ax.set_title(f"{cfg}  /  {pol}\nmksp={float(mksp):.1f}ms  dl_miss={dl}"
                          if mksp and mksp != "?" else f"{cfg}/{pol}",
                          fontsize=9)
            if c == 0:
                ax.set_ylabel(cfg, fontsize=11)

    fig.suptitle(
        "Phase D headline: 4 MLP + 2 Dronet + 1 Yolo on hetero — "
        "rows = frequency config, cols = policy",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
