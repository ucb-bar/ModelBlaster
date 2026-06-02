"""Render a stacked comparison Gantt of multiple measured FireSim traces.

Each row of the figure is one (scheduler, measured trace) pair on the
same x-axis; period boundaries (10 ms mlp_control, 20 ms dronet) are
overlaid as shaded windows so frequency-respect is visible at a
glance. Use this to compare schedulers side by side after all their
measured traces have landed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


NETWORK_COLORS = {
    "yolov8_nano": "#4C78A8",
    "mlp_control": "#59A14F",
    "dronet":      "#F58518",
}
NETWORK_PERIOD_MS = {
    "mlp_control": 10.0,
    "dronet":      20.0,
}


def load_trace(path: Path, clock_mhz: float = 1000.0) -> list[dict]:
    rows = []
    with open(path) as f:
        # Skip blank-leading lines.
        content = [ln for ln in f if ln.strip()]
    reader = csv.DictReader(content)
    for r in reader:
        try:
            a_s = int(r["actual_start_cycles"])
            a_e = int(r["actual_end_cycles"])
        except (KeyError, ValueError):
            continue
        if a_s == 0 and a_e == 0:
            continue
        cycles_per_ms = clock_mhz * 1000.0
        rows.append({
            "network": r["network"].strip(),
            "instance": int(r.get("instance", 0)),
            "lane": r["core_kind"].strip(),
            "start_ms": a_s / cycles_per_ms,
            "duration_ms": max(0.0, (a_e - a_s) / cycles_per_ms),
            "is_fused": "__fused__" in r.get("op", ""),
        })
    return rows


def _lane_y(lane: str) -> int:
    if "gemmini" in lane.lower(): return 0
    return 1


def render_stack(panels: list[dict], out_png: Path, title: str,
                 deadline_ms: float | None = None,
                 x_max_ms: float | None = None) -> None:
    """
    panels: list of {label: str, trace: Path, subtitle: str, annotation: str}.
    Each panel renders 2 lanes (CPU_P/Gemmini, CPU_E/OPU+V) — so the
    figure has 2 * len(panels) lanes stacked vertically.
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels),
                              sharex=True)
    if n_panels == 1:
        axes = [axes]

    networks_present = set()
    max_x = 0.0

    for idx, (panel, ax) in enumerate(zip(panels, axes)):
        rows = load_trace(panel["trace"])
        for r in rows:
            y = _lane_y(r["lane"]) + 1
            color = NETWORK_COLORS.get(r["network"], "#888")
            hatch = "////" if r["is_fused"] else None
            edgecolor = "black" if r["is_fused"] else "none"
            ax.broken_barh(
                [(r["start_ms"], max(r["duration_ms"], 0.01))],
                (y - 0.35, 0.7),
                facecolors=color, edgecolors=edgecolor, hatch=hatch,
                linewidth=1.2 if r["is_fused"] else 0)
            networks_present.add(r["network"])
            max_x = max(max_x, r["start_ms"] + r["duration_ms"])

        # Panel labels.
        meas_makespan = max((r["start_ms"] + r["duration_ms"]) for r in rows) if rows else 0.0
        n_dispatches = len(rows)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(["CPU_P (Gemmini)", "CPU_E (OPU+V)"], fontsize=9)
        ax.set_ylim(0.4, 2.6)
        ax.grid(axis="x", linestyle=":", alpha=0.3)
        label = (f"{panel['label']}  ·  "
                 f"measured makespan {meas_makespan:.2f} ms  ·  "
                 f"{n_dispatches} dispatches")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold")
        if panel.get("subtitle"):
            ax.text(0.99, 0.93, panel["subtitle"],
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, color="#666", style="italic")

    if x_max_ms is None:
        x_max_ms = max_x * 1.05
    for ax in axes:
        ax.set_xlim(0, x_max_ms)

    # Period-window shading on EVERY panel.
    for ax in axes:
        if "mlp_control" in networks_present:
            t = 0.0
            for k in range(int(x_max_ms / 10.0) + 2):
                w_start = t
                w_end = min(t + 10.0, x_max_ms)
                ax.axvspan(w_start, w_end, ymin=0.0, ymax=1.0,
                           color=NETWORK_COLORS["mlp_control"], alpha=0.04,
                           zorder=0)
                if k > 0:
                    ax.axvline(t, color=NETWORK_COLORS["mlp_control"],
                               linestyle="--", alpha=0.5, linewidth=0.8)
                t += 10.0
        if "dronet" in networks_present:
            t = 0.0
            while t <= x_max_ms:
                if t > 0:
                    ax.axvline(t, color=NETWORK_COLORS["dronet"],
                               linestyle="--", alpha=0.4, linewidth=1.2)
                t += 20.0
        if deadline_ms is not None and deadline_ms <= x_max_ms:
            ax.axvline(deadline_ms, color="red", linewidth=2, alpha=0.6)

    axes[-1].set_xlabel("Time (ms) — same scale across all panels",
                        fontsize=10)

    # Legend at top.
    handles = [mpatches.Patch(color=NETWORK_COLORS[n], label=n)
               for n in sorted(networks_present)]
    handles.append(mpatches.Patch(facecolor=NETWORK_COLORS["mlp_control"],
                                   alpha=0.3,
                                   label="mlp_control period (10 ms)"))
    handles.append(mpatches.Patch(facecolor=NETWORK_COLORS["dronet"],
                                   alpha=0.3,
                                   label="dronet period (20 ms)"))
    if any(load_trace(p["trace"])[0].get("is_fused")
           for p in panels if load_trace(p["trace"])):
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                       hatch="////", label="fused dispatch"))
    fig.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 1.0 - 0.005),
                ncol=len(handles), fontsize=9, frameon=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    plt.subplots_adjust(top=0.94 - 0.01 * n_panels, hspace=0.35)
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="Schedulers compared on FireSim (MEASURED)")
    p.add_argument("--deadline-ms", type=float, default=None)
    p.add_argument("--x-max-ms", type=float, default=None)
    p.add_argument("--panel", action="append", nargs=3,
                   metavar=("LABEL", "TRACE_CSV", "SUBTITLE"),
                   help="Add a panel: LABEL TRACE_CSV SUBTITLE. Repeatable.")
    args = p.parse_args(argv)

    if not args.panel:
        raise SystemExit("supply at least one --panel LABEL TRACE_CSV SUBTITLE")
    panels = [{"label": l, "trace": Path(t), "subtitle": s}
              for (l, t, s) in args.panel]
    render_stack(panels, args.out, args.title,
                 deadline_ms=args.deadline_ms, x_max_ms=args.x_max_ms)


if __name__ == "__main__":
    raise SystemExit(main())
