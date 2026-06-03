#!/usr/bin/env python3
"""Render a CLEAR side-by-side comparison of the 4 + F2g policies on the
headline workload.

Improvements over the existing band Gantts:
  - One row PER NETWORK (not per core). MLP, Dronet, Yolo each get a
    lane — instance index becomes the y-offset within the lane. This
    makes each periodic instance immediately visible.
  - Period boundary lines are labeled (10ms tick: "mlp window k", etc).
  - Below each Gantt, a 3-row summary table: per-network ops, completion
    time, deadline-miss count.
  - Big legend, big titles, no tiny bars.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

REPO = Path(__file__).resolve().parents[1]
XPURT = Path("/scratch2/agustin/XPU-RT")
sys.path.insert(0, str(XPURT / "xpu-rt"))
from diagnostics.band_invariant import check_band_invariant, _periodic_metadata


PALETTE = {
    "mlp_control": "#1f77b4",   # blue
    "dronet":      "#2ca02c",   # green
    "yolov8_nano": "#ff7f0e",   # orange
}


def _parse_inst(name, periodic_bases):
    """Return (network_base, instance_idx) for a dispatch name."""
    for base in sorted(periodic_bases, key=lambda s: -len(s)):
        if name.startswith(base):
            tail = name[len(base):]
            m = re.match(r"^(\d+)_", tail)
            if m:
                return base, int(m.group(1))
    for base in ("yolov8_nano",):
        if name.startswith(base):
            return base, 0
    return "unknown", 0


def render_clear_policy(fixture_path, workload, out_path, *,
                          policy_name, makespan_ms, dl_miss):
    fixture = json.loads(Path(fixture_path).read_text())
    dispatches = fixture["dispatches"]
    periodic, nonperiodic = _periodic_metadata(workload)
    report = check_band_invariant(fixture, workload, solver=policy_name)
    violations = {v.dispatch for v in report.violations}

    # Group dispatches by (network_base, instance_idx).
    grouped = {}
    for name, entry in dispatches.items():
        base, inst = _parse_inst(name, periodic)
        grouped.setdefault((base, inst), []).append((name, entry))

    # Build lane order: each (network, instance) is one lane.
    lane_order = []
    for net in ("mlp_control", "dronet", "yolov8_nano"):
        if net in periodic:
            n_inst = int(periodic[net][0])
            for i in range(n_inst):
                if (net, i) in grouped:
                    lane_order.append((net, i))
        elif net in nonperiodic or (net, 0) in grouped:
            lane_order.append((net, 0))

    fig, (ax, ax_summary) = plt.subplots(
        2, 1, figsize=(18, 6 + 0.4 * len(lane_order)),
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Plot lanes.
    BAR_H = 0.6
    for lane_idx, (net, inst) in enumerate(lane_order):
        color = PALETTE.get(net, "#94a3b8")

        # Draw the instance band as a backdrop.
        if net in periodic:
            n_inst_total, period, window, start_t = periodic[net]
            R_k = start_t + inst * period
            D_k = R_k + window
            ax.axvspan(R_k, D_k, ymin=(lane_idx + 0.5 - 0.45) / len(lane_order),
                        ymax=(lane_idx + 0.5 + 0.45) / len(lane_order),
                        color=color, alpha=0.10, zorder=0)
            # Vertical dashed for the window edges.
            ax.plot([R_k, R_k], [lane_idx - 0.4, lane_idx + 0.4],
                     color=color, linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
            ax.plot([D_k, D_k], [lane_idx - 0.4, lane_idx + 0.4],
                     color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

        # Draw ops on this lane.
        for name, entry in grouped[(net, inst)]:
            s = float(entry["start_time"])
            w = float(entry["duration"])
            is_miss = name in violations
            edge = "#dc2626" if is_miss else "black"
            ew = 1.5 if is_miss else 0.1
            face = "#fee2e2" if is_miss else color
            ax.add_patch(mpatches.Rectangle(
                (s, lane_idx - BAR_H/2), max(w, 0.05), BAR_H,
                facecolor=face, edgecolor=edge, linewidth=ew, zorder=2,
            ))

    # Axis labels.
    labels = [f"{net}[{inst}]" for net, inst in lane_order]
    ax.set_yticks(range(len(lane_order)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.7, len(lane_order) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_xlim(-1, max(makespan_ms * 1.05, 80))
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Period boundary annotations along the top.
    if "mlp_control" in periodic:
        _, period, window, st = periodic["mlp_control"]
        for k in range(int(periodic["mlp_control"][0]) + 1):
            x = st + k * period
            ax.axvline(x, color=PALETTE["mlp_control"], linestyle="--",
                        linewidth=0.5, alpha=0.3, zorder=0)
            ax.text(x, -0.6, f"{int(x)}", ha="center", va="bottom",
                     fontsize=7, color=PALETTE["mlp_control"])

    title = (f"{policy_name}   |   makespan = {makespan_ms:.1f} ms   |   "
              f"deadline misses = {dl_miss}\n"
              f"Each row is one network INSTANCE. Light band = its "
              f"[release, deadline] window. Red box = overran the deadline.")
    ax.set_title(title, fontsize=11, loc="left")

    # Summary table below: per-network stats.
    ax_summary.axis("off")
    rows = []
    for net in ("mlp_control", "dronet", "yolov8_nano"):
        net_ops = sum(len(grouped.get((net, i), []))
                      for i in range(20))
        net_finish = 0.0
        net_misses = 0
        for (n, i), items in grouped.items():
            if n != net:
                continue
            for name, entry in items:
                f = float(entry["start_time"]) + float(entry["duration"])
                if f > net_finish:
                    net_finish = f
                if name in violations:
                    net_misses += 1
        rows.append([net, str(net_ops), f"{net_finish:.1f}", str(net_misses)])
    table = ax_summary.table(
        cellText=rows,
        colLabels=["network", "n_ops", "last_finish (ms)", "deadline misses"],
        cellLoc="center", loc="center", colColours=["#e5e7eb"]*4,
    )
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1.0, 1.4)
    for net_idx, net in enumerate(["mlp_control", "dronet", "yolov8_nano"]):
        cell = table.get_celld()[(net_idx+1, 0)]
        cell.set_facecolor(PALETTE[net])
        cell.set_text_props(color="white", weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    workload_path = XPURT / "data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json"
    workload = json.loads(workload_path.read_text())

    policies = [
        ("periodic_anchor",
         "schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_decomposed_profiled.json",
         75.57, 0),
        ("critical_path_first (heft)",
         "schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_heft_profiled.json",
         54.43, 88),
        ("yolo_anchor (greedy_periodic)",
         "schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_greedy_periodic_profiled.json",
         61.20, 67),
        ("MOSEK F2g decomposed",
         "schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_mosek_decomposed.json",
         51.10, 25),
    ]

    out_dir = REPO / "artifacts" / "policies" / "clear"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, rel_path, mksp, miss in policies:
        full = XPURT / rel_path
        if not full.exists():
            print(f"missing: {full}")
            continue
        out = out_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        render_clear_policy(str(full), workload, str(out),
                              policy_name=name, makespan_ms=mksp, dl_miss=miss)


if __name__ == "__main__":
    sys.exit(main())
