"""Aggregated bar chart: predicted vs actual makespan across the 3-way
frequency-sweep cells.

The 5 cells we capture (different (yolo, dronet, mlp) instance counts
modeling different mission profiles) come from the original
`scripts/gen_3way_schedule.py` family — naive topological layouts, not
MOSEK. They use the same hetero bitstream and are useful as a
"how-bad-is-naive" reference against the MOSEK headline.

Output: notes/figures/frequency_sweep_makespan.png — one cell per group,
two bars per group (predicted vs actual median makespan), with the
cell's instance counts annotated.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO / "benchmarks" / "results" / "A"
FIXTURES = REPO / "schedule_fixtures"

# Order chosen by ascending dispatch count so the X axis reads
# small-to-large workload (makes the makespan curve monotonic-ish).
CELLS = [
    ("baseline",     "1y160+1d+4m",   "3way_baseline.json",     "3way_baseline"),
    ("conservative", "1y160+2d+9m",   "3way_conservative.json", "3way_conservative"),
    ("imu-only-hi",  "1y160+1d+90m",  "3way_imu-only-hi.json",  "3way_imu-only-hi"),
    ("camera-30hz",  "1y160+14d+45m", "3way_camera-30hz.json",  "3way_camera-30hz"),
    ("camera-60hz",  "1y160+28d+90m", "3way_camera-60hz.json",  "3way_camera-60hz"),
]


def _predicted_ms(fixture_path: pathlib.Path) -> float | None:
    if not fixture_path.exists():
        return None
    fx = json.loads(fixture_path.read_text())
    p = fx.get("_provenance", {})
    if "makespan_ms" in p:
        return float(p["makespan_ms"])
    return max(
        d["start_time"] + d["duration"] for d in fx.get("dispatches", {}).values()
    ) if fx.get("dispatches") else None


def _actual_ms(cell_dir: pathlib.Path, expected_n: int | None = None) -> tuple[float | None, int]:
    """Return (median actual makespan in ms across reps, n_reps_used)."""
    if not cell_dir.exists():
        return None, 0
    vals: list[float] = []
    for run in sorted(d for d in cell_dir.iterdir() if d.is_dir() and d.name != "latest"):
        trace = run / "xpurt_trace.csv"
        if not trace.exists():
            continue
        rows = [r for r in csv.DictReader(trace.open()) if r and r.get("actual_end_cycles", "").strip()]
        if not rows:
            continue
        if expected_n is not None and len(rows) != expected_n:
            continue
        vals.append(max(int(r["actual_end_cycles"]) for r in rows) / 1000.0)
    if not vals:
        return None, 0
    vals.sort()
    return vals[len(vals) // 2], len(vals)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels: list[str] = []
    preds: list[float] = []
    actuals: list[float] = []
    annotations: list[str] = []

    for cell_id, mix, fx_name, dir_name in CELLS:
        fx_path = FIXTURES / fx_name
        fx = json.loads(fx_path.read_text()) if fx_path.exists() else {}
        n_disp = len(fx.get("dispatches", {})) if fx else None
        pred = _predicted_ms(fx_path)
        act, n_reps = _actual_ms(RESULTS / dir_name, expected_n=n_disp)

        labels.append(f"{cell_id}\n({mix})")
        preds.append(pred or 0.0)
        actuals.append(act or 0.0)
        annotations.append(
            f"{n_disp}d\n{n_reps}rep" if n_disp else "?"
        )

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars_p = ax.bar(x - width/2, preds, width, label="Predicted (naive)", color="#3b82f6", edgecolor="black", linewidth=0.4)
    bars_a = ax.bar(x + width/2, actuals, width, label="Actual FireSim (median)", color="#ef4444", edgecolor="black", linewidth=0.4)

    # Annotate bars with ms values.
    for bp in bars_p:
        h = bp.get_height()
        if h:
            ax.annotate(f"{h:.0f}", xy=(bp.get_x() + bp.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color="#1e3a8a")
    for ba in bars_a:
        h = ba.get_height()
        if h:
            ax.annotate(f"{h:.0f}", xy=(ba.get_x() + ba.get_width()/2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color="#7f1d1d")

    # n_dispatches / n_reps annotations under x labels.
    for i, ann in enumerate(annotations):
        ax.annotate(ann, xy=(i, 0), xytext=(0, -45), textcoords="offset points",
                    ha="center", fontsize=7, color="#475569")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Makespan (ms)")
    ax.set_title(
        "3-way frequency sweep — predicted (naive layout) vs actual FireSim\n"
        "GemminiAndOPUShuttleConfig hetero bitstream; cells use pre-MOSEK gen_3way_schedule layouts"
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")

    # Add the qrb image reference horizontal line + MOSEK headline reference.
    ax.axhline(75.71, color="#16a34a", linestyle="--", linewidth=1, alpha=0.8)
    ax.annotate("qrb image 75.71 ms (1y+2d+4m, hetero ref)", xy=(len(labels)-0.5, 75.71),
                xytext=(5, 8), textcoords="offset points", color="#15803d", fontsize=8)
    ax.axhline(25.24, color="#7c3aed", linestyle="--", linewidth=1, alpha=0.8)
    ax.annotate("MOSEK no-yolo 25.24 ms (2d+4m, our headline)", xy=(len(labels)-0.5, 25.24),
                xytext=(5, 4), textcoords="offset points", color="#6d28d9", fontsize=8)

    out = REPO / "notes" / "figures" / "frequency_sweep_makespan.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"wrote {out}")
    for label, p, a, ann in zip(labels, preds, actuals, annotations):
        cell_id = label.split("\n")[0]
        print(f"  {cell_id:<13s} pred={p:>7.1f} ms   actual={a:>7.1f} ms   ({ann.replace(chr(10), ' ')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
