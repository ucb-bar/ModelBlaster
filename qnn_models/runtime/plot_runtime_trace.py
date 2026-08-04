"""Reconstruct a Gantt chart from the runtime's AGENTS_QNN_TRACE block.

The QNN runtime (built via build_and_run.sh from generate_runtime.py
output) prints a CSV trace between `=== AGENTS_QNN_TRACE_BEGIN ===` and
`=== AGENTS_QNN_TRACE_END ===` markers after pthread_join. Each row has
both the schedule-side prediction (`predicted_start_ms`,
`predicted_duration_ms`) and the runtime-side measurement
(`actual_start_ms`, `actual_end_ms`, plus `dep_wait_done_ms` and
`gate_done_ms` for diagnosing where the slack went).

This script renders two stacked Gantt panes — top = predicted, bottom =
measured — sharing the same x-axis and using the same colour per
(network, instance) so the diff between the two is obvious. Mirrors the
zephyr xpurt trace plot.

Usage:
    bash qnn_models/runtime/build_and_run.sh <gen_dir> | tee run.log
    python3 qnn_models/runtime/plot_runtime_trace.py \\
        --log run.log --out plots/qrb5165_qnn_runtime_gantt.png

Or read directly from the captured stdout file:
    python3 plot_runtime_trace.py --log /tmp/qnn_run.log --out plot.png
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys


BEGIN = "=== AGENTS_QNN_TRACE_BEGIN ==="
END   = "=== AGENTS_QNN_TRACE_END ==="


def extract_trace_csv(log_path: str) -> list[dict]:
    """Pull the CSV block between the markers and parse it."""
    with open(log_path) as f:
        text = f.read()
    if BEGIN not in text or END not in text:
        sys.exit(f"no AGENTS_QNN_TRACE block in {log_path}")
    block = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    rows = list(csv.DictReader(io.StringIO(block)))
    # Cast numeric columns.
    for r in rows:
        for k in ("predicted_start_ms", "predicted_duration_ms",
                  "actual_start_ms", "actual_end_ms",
                  "dep_wait_done_ms", "gate_done_ms"):
            if k in r:
                r[k] = float(r[k]) if r[k] != "" else 0.0
        for k in ("seg_id", "ctx_seg_id", "instance", "n_ops", "kind_idx"):
            if k in r:
                r[k] = int(r[k]) if r[k] != "" else -1
    return rows


# Per-model-kind colormap assignments. Mirrors xpu-rt/plot.py — a single
# "all blues are dronet, all reds are resnet50" reading across the
# scheduler-side gantt and the runtime-side gantt. Periodic instances of
# the same model (mlp_control#0 .. #10) draw distinct shades from the
# same colormap, so 11 mlp bars don't get 11 unrelated colors.
KIND_TO_CMAP = {
    "dronet":      "Blues",
    "yolov8":      "Oranges",
    "yolov8_nano": "Oranges",
    "yolov8n":     "Oranges",
    "mlp":         "Greens",
    "mlp_control": "Greens",
    "mobilenet":   "Purples",
    "mobilenet_v2": "Purples",
    "resnet50":    "Reds",
    "tinyyolo":    "YlOrBr",
}
# Fallback cmaps for unknown networks. Skip Greys (collides with the
# faint dep-wait tint).
_FALLBACK_CMAPS = ["Greens", "Purples", "Reds", "YlOrBr", "PuRd",
                   "BuGn", "OrRd", "GnBu"]


def _build_family_palette(rows: list[dict]) -> dict:
    """Return {(network, instance): rgb_triple} where same-network
    instances draw shades from one colormap. Built once up-front from
    the full row set so the shade range is stable across re-renders."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return {}

    # Collect unique (network, instance) tuples grouped by network.
    by_kind: dict[str, set[int]] = {}
    for r in rows:
        net = r.get("network") or "?"
        inst = int(r.get("instance", 0)) if r.get("instance") not in (None, "") else 0
        by_kind.setdefault(net, set()).add(inst)

    # Stable cmap assignment: known kinds use their declared cmap;
    # unknown kinds drain the fallback list (sorted for reproducibility).
    used = {KIND_TO_CMAP[k] for k in by_kind if k in KIND_TO_CMAP}
    fb = iter(c for c in _FALLBACK_CMAPS if c not in used)
    kind_to_cmap = {}
    for kind in sorted(by_kind):
        kind_to_cmap[kind] = KIND_TO_CMAP.get(kind) or next(fb, "Greys")

    # Within each kind, assign shades. Avoid the very-light end (<0.4)
    # so bars don't blend with the white grid; cap at 0.95 so the
    # white seg-id text overlay stays readable.
    palette: dict[tuple[str, int], tuple] = {}
    for kind, instances in by_kind.items():
        cmap = plt.get_cmap(kind_to_cmap[kind])
        instances_sorted = sorted(instances)
        if len(instances_sorted) == 1:
            shades = [0.7]
        else:
            shades = np.linspace(0.4, 0.95, len(instances_sorted))
        for shade, inst in zip(shades, instances_sorted):
            palette[(kind, inst)] = tuple(cmap(shade)[:3])
    return palette


def _color_for(network: str, instance: int, palette: dict) -> tuple:
    """Look up the precomputed family-shade for (network, instance).
    Falls back to mid-tone gray if the trace contains a (network,
    instance) the up-front pass missed."""
    return palette.get((network, instance), (0.5, 0.5, 0.5))


def render_gantt(rows: list[dict], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as e:
        sys.exit(f"matplotlib needed: {e}")

    # Y-axis lanes are the actual physical backends (HTA / DSP / CPU /
    # GPU) rather than the scheduler's abstract slot names (CPU_P /
    # CPU_E). Multiple scheduler kinds or networks that resolve to the
    # same physical backend (e.g. CPU_X + CPU_X2 from a 2-CPU-lane MILP
    # schedule, or one CPU_X kind split per-network at runtime via
    # XPURT_SPLIT_BY_NETWORK) all collapse onto the same row — the
    # diagram reads as a hardware utilisation view, and family colors
    # already tell which network is running. Bars from concurrent
    # workers may visually overlap if they execute simultaneously; in
    # practice the MILP serialises CPU work so this is rare.
    def lane_name(r, prefer_actual=False):
        if prefer_actual and r.get("actual_backend") and r["actual_backend"] != "?":
            return r["actual_backend"]
        return r.get("backend_label") or r.get("kind", "?")

    # Build the lane set from the actual_backend column (post-run
    # ground truth). If the trace is from an old build without that
    # column, fall back to backend_label.
    lane_keys = []
    for r in rows:
        n = lane_name(r, prefer_actual=True)
        if n not in lane_keys:
            lane_keys.append(n)
    kind_y = {k: i for i, k in enumerate(lane_keys)}

    # 2 panes — top: predicted; bottom: measured. Share x.
    fig, axes = plt.subplots(2, 1, figsize=(14, 4 + 0.5 * len(lane_keys)),
                              sharex=True, gridspec_kw={"hspace": 0.18})
    ax_pred, ax_meas = axes

    palette = _build_family_palette(rows)
    label_seen: set[tuple[str, int]] = set()

    pred_max = 0.0
    meas_max = 0.0
    for r in rows:
        net = r["network"]; inst = r["instance"]
        color = _color_for(net, inst, palette)
        label = f"{net}#{inst}"

        # Top pane: predicted. Use the same lane as the measured row
        # (post-runtime resolution) so predicted vs measured land on
        # the same y-row even when backend_label and actual_backend
        # differ (e.g. yolov8n's HTA_split → DSP).
        py = kind_y[lane_name(r, prefer_actual=True)]
        ax_pred.barh(
            y=py, width=r["predicted_duration_ms"],
            left=r["predicted_start_ms"],
            height=0.7, color=color,
            edgecolor="black", linewidth=0.5,
            label=label if (net, inst) not in label_seen else None,
        )
        ax_pred.text(
            r["predicted_start_ms"] + r["predicted_duration_ms"] / 2,
            py, str(r["seg_id"]),
            ha="center", va="center", fontsize=6, color="white",
        )
        pred_max = max(pred_max, r["predicted_start_ms"] + r["predicted_duration_ms"])
        label_seen.add((net, inst))

        # Bottom pane: measured.
        if r["actual_end_ms"] >= 0 and r["actual_start_ms"] >= 0:
            ax_meas.barh(
                y=py, width=r["actual_end_ms"] - r["actual_start_ms"],
                left=r["actual_start_ms"], height=0.7, color=color,
                edgecolor="black", linewidth=0.5,
            )
            ax_meas.text(
                (r["actual_start_ms"] + r["actual_end_ms"]) / 2, py,
                str(r["seg_id"]),
                ha="center", va="center", fontsize=6, color="white",
            )
            meas_max = max(meas_max, r["actual_end_ms"])
            # Faint span for dep-wait → start (the "slack" the runtime
            # paid before kicking off the dispatch). Helps diagnose
            # whether late=actual-predicted is from dep wait or compute.
            if r["dep_wait_done_ms"] >= 0:
                ax_meas.barh(
                    y=py, width=r["actual_start_ms"] - r["dep_wait_done_ms"],
                    left=r["dep_wait_done_ms"], height=0.7,
                    color=color, alpha=0.18, edgecolor="none",
                )

    ax_pred.set_yticks(list(kind_y.values()))
    ax_pred.set_yticklabels(lane_keys)
    ax_meas.set_yticks(list(kind_y.values()))
    ax_meas.set_yticklabels(lane_keys)
    ax_pred.set_ylabel("backend")
    ax_meas.set_ylabel("backend")
    ax_pred.set_title("Predicted (schedule-side)", loc="left", fontsize=10)
    ax_meas.set_title("Measured (runtime-side, light tint = pre-start dep wait)",
                       loc="left", fontsize=10)
    ax_meas.set_xlabel("time (ms)")
    ax_pred.set_xlim(0, max(pred_max, meas_max) * 1.02)
    ax_meas.set_xlim(0, max(pred_max, meas_max) * 1.02)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Legend (one per network-instance), placed outside.
    handles = []
    for (net, inst), color in palette.items():
        handles.append(mpatches.Patch(color=color, label=f"{net}#{inst}"))
    fig.legend(handles=handles, loc="upper center",
                ncol=min(len(handles), 8), bbox_to_anchor=(0.5, 1.02),
                frameon=False, fontsize=9)

    pred_makespan = pred_max
    meas_makespan = meas_max
    fig.suptitle(
        f"QNN runtime walk — predicted={pred_makespan:.2f} ms, "
        f"measured={meas_makespan:.2f} ms  ({meas_makespan/pred_makespan:.2f}× longer)",
        y=1.06, fontsize=11,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    print(f"wrote {out_path}  (predicted={pred_makespan:.2f} ms, "
          f"measured={meas_makespan:.2f} ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True,
                    help="captured stdout from the runtime (must contain the "
                         "AGENTS_QNN_TRACE_BEGIN/END block)")
    ap.add_argument("--out", required=True, help="output .png path")
    args = ap.parse_args()
    rows = extract_trace_csv(args.log)
    print(f"parsed {len(rows)} trace rows from {args.log}")
    render_gantt(rows, args.out)


if __name__ == "__main__":
    main()
