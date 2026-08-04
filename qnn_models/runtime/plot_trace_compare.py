"""Stacked Gantt comparison across multiple runtime traces.

Each input log carries one `AGENTS_QNN_TRACE` block from generate_runtime's
runtime_main. This renders N panes (one per log), sharing the x-axis, so
the eye reads "where did the extra wall time go" across runs that
executed the same schedule on different platforms / runtime variants.

Usage:
    python3 plot_trace_compare.py \\
        --log runs/v3_bundles_dsp9/run.log:Pre-refactor (physical, budget=9) \\
        --log runs/v3_bundles_multigraph_dense_phys/run.log:Multi-graph dense (physical) \\
        --log runs/v3_bundles_multigraph_dense_cloud/run.log:Multi-graph dense (cloud) \\
        --out plots/v3_bundles_trace_compare.png
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys

BEGIN = "=== AGENTS_QNN_TRACE_BEGIN ==="
END   = "=== AGENTS_QNN_TRACE_END ==="


def extract_rows(log_path: str) -> list[dict]:
    with open(log_path) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        sys.exit(f"no AGENTS_QNN_TRACE block in {log_path}")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    rows = list(csv.DictReader(io.StringIO(block)))
    for r in rows:
        for k in ("predicted_start_ms", "predicted_duration_ms",
                  "actual_start_ms", "actual_end_ms",
                  "dep_wait_done_ms", "gate_done_ms"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
            else:
                r[k] = 0.0
        for k in ("seg_id",):
            if k in r and r[k] != "":
                r[k] = int(r[k])
    return rows


def wall_from_rows(rows: list[dict]) -> float:
    return max(r["actual_end_ms"] for r in rows)


# Color by backend lane: CPU=blue, DSP=orange, HTA=green
LANE_COLOR = {
    "CPU": (0.40, 0.55, 0.85),
    "DSP": (0.95, 0.55, 0.20),
    "HTA": (0.30, 0.70, 0.40),
    "GPU": (0.70, 0.40, 0.70),
}


def lane_name(r) -> str:
    return r.get("actual_backend") or r.get("backend_label") or "?"


def render(panes: list[tuple[str, list[dict]]], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Unified lane set across all panes for a shared y-axis.
    lane_keys: list[str] = []
    for _, rows in panes:
        for r in rows:
            n = lane_name(r)
            if n not in lane_keys:
                lane_keys.append(n)
    lane_y = {k: i for i, k in enumerate(lane_keys)}

    # Max x across all panes for a common time axis.
    x_max = 0.0
    for _, rows in panes:
        x_max = max(x_max, max(r["actual_end_ms"] for r in rows))
        x_max = max(x_max, max(r["predicted_start_ms"] + r["predicted_duration_ms"]
                               for r in rows))

    n = len(panes)
    fig, axes = plt.subplots(n, 1, figsize=(16, 1.2 + 1.6 * n * (1 + len(lane_keys) * 0.15)),
                              sharex=True,
                              gridspec_kw={"hspace": 0.45})
    if n == 1:
        axes = [axes]

    for ax, (title, rows) in zip(axes, panes):
        wall = wall_from_rows(rows)
        pred = max(r["predicted_start_ms"] + r["predicted_duration_ms"] for r in rows)

        # Faint predicted-end vertical line for reference.
        ax.axvline(pred, color="black", linestyle=":", alpha=0.4,
                   linewidth=0.8, zorder=0)

        for r in rows:
            ly = lane_y[lane_name(r)]
            color = LANE_COLOR.get(lane_name(r), (0.5, 0.5, 0.5))

            # Bar = actual execute window
            if r["actual_end_ms"] > 0 and r["actual_start_ms"] > 0:
                ax.barh(y=ly,
                         width=r["actual_end_ms"] - r["actual_start_ms"],
                         left=r["actual_start_ms"],
                         height=0.7, color=color,
                         edgecolor="black", linewidth=0.4, zorder=2)

            # Tinted dep-wait region (gating before kickoff)
            if r["dep_wait_done_ms"] > 0 and r["actual_start_ms"] > r["dep_wait_done_ms"]:
                ax.barh(y=ly,
                         width=r["actual_start_ms"] - r["dep_wait_done_ms"],
                         left=r["dep_wait_done_ms"],
                         height=0.7, color=color, alpha=0.18,
                         edgecolor="none", zorder=1)

        ax.set_yticks(list(lane_y.values()))
        ax.set_yticklabels(lane_keys)
        ax.set_ylabel("backend")
        # Title shows wall + drift vs predicted.
        drift_pct = (wall - pred) / pred * 100
        ax.set_title(
            f"{title}   —   wall {wall:.0f} ms, predicted {pred:.0f} ms ({drift_pct:+.1f}%)",
            loc="left", fontsize=10,
        )
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    axes[-1].set_xlabel("time (ms)")
    axes[0].set_xlim(0, x_max * 1.02)

    # Shared legend at the bottom: lane colors + faint dep-wait + predicted line.
    handles = [mpatches.Patch(color=LANE_COLOR[k], label=k)
               for k in lane_keys if k in LANE_COLOR]
    handles.append(mpatches.Patch(facecolor=(0.5, 0.5, 0.5), alpha=0.18,
                                  label="dep wait (gating)"))
    handles.append(plt.Line2D([0], [0], color="black", linestyle=":",
                              label="predicted makespan"))
    axes[-1].legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.4), ncol=len(handles),
                    frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", action="append", required=True,
                    help="<path>:<title> (repeatable)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    panes = []
    for spec in args.log:
        if ":" in spec:
            path, title = spec.split(":", 1)
        else:
            path, title = spec, os.path.basename(spec)
        panes.append((title, extract_rows(path)))
    render(panes, args.out)


if __name__ == "__main__":
    main()
