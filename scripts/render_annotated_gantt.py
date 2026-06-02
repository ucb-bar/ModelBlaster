"""Render an annotated Gantt chart highlighting the agentic loop's reasoning.

Each step of the iterative scheduling loop gets one PNG with:
  - Per-backend lanes (CPU_P#0 / Gemmini, CPU_E#0 / OPU+V)
  - Colored bars per network (yolov8_nano / mlp_control / dronet)
  - Periodic-frequency boundaries as dashed vertical lines
    (mlp_control's 10ms period, dronet's 20ms period)
  - Deadline marker
  - Annotation box explaining: deadline verdict, bottleneck backend,
    granularity verdict, and what changed vs the previous step
  - Fused dispatches drawn with hatched fill so the
    "merged from N sub-ops" structure is visible

Two input shapes are supported:
  --fixture FILE        XPU-RT schedule fixture (predicted-only)
  --trace   FILE        ModelBlaster xpurt_trace.csv (predicted + measured)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


NETWORK_COLORS = {
    "yolov8_nano": "#4C78A8",   # blue
    "mlp_control": "#59A14F",   # green
    "dronet":      "#F58518",   # orange
}
NETWORK_PERIOD_MS = {
    "mlp_control": 10.0,
    "dronet":      20.0,
}
LANE_NAMES = {
    "gemmini":    "CPU_P#0 (Gemmini)",
    "rvv_opu":    "CPU_E#0 (OPU + V)",
    "CPU_P#0":    "CPU_P#0 (Gemmini)",
    "CPU_E#0":    "CPU_E#0 (OPU + V)",
}


def _color_for(network: str, instance: int) -> str:
    """Base color per network; instance darkens slightly so periodic
    instances are distinguishable within a network."""
    base = NETWORK_COLORS.get(network, "#888888")
    if instance == 0:
        return base
    # Darken or lighten by instance for visual diff.
    import colorsys
    r, g, b = int(base[1:3], 16) / 255, int(base[3:5], 16) / 255, int(base[5:7], 16) / 255
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.25, l - 0.12 * (instance % 4))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def load_from_fixture(path: Path) -> list[dict]:
    """Read XPU-RT schedule fixture and return per-dispatch rows in
    a uniform shape used by the renderer."""
    j = json.loads(path.read_text())
    rows = []
    d = j["dispatches"]
    for k, entry in d.items():
        job = entry.get("job_name", "")
        if not job:
            continue
        # job_name is e.g. "mlp_control3" → network=mlp_control, instance=3
        network = job
        instance = 0
        for n in NETWORK_COLORS:
            if job.startswith(n):
                network = n
                rest = job[len(n):]
                if rest.isdigit():
                    instance = int(rest)
                break
        ht = entry.get("hardware_target", "")
        # "CPU_P#0" or "CPU_E#0"
        lane = ht
        rows.append({
            "network": network,
            "instance": instance,
            "lane": lane,
            "start_ms": float(entry["start_time"]),
            "duration_ms": float(entry["duration"]),
            "is_fused": False,
            "sub_count": 1,
            "op": entry.get("module_name", "?"),
        })
    return rows


def load_from_trace(path: Path, clock_mhz: float = 1000.0) -> list[dict]:
    """Read xpurt_trace.csv (predicted + measured) and return rows
    keyed for the renderer. Measured times use actual_*_cycles
    converted to ms at the given clock."""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                a_s = int(r["actual_start_cycles"])
                a_e = int(r["actual_end_cycles"])
            except (KeyError, ValueError):
                continue
            if a_s == 0 and a_e == 0:
                # view ops / zero-cost — skip in the measured render.
                continue
            cycles_per_ms = clock_mhz * 1000.0
            rows.append({
                "network": r["network"].strip(),
                "instance": int(r.get("instance", 0)),
                "lane": r["core_kind"].strip(),
                "start_ms": a_s / cycles_per_ms,
                "duration_ms": max(0.0, (a_e - a_s) / cycles_per_ms),
                "is_fused": "__fused__" in r.get("op", ""),
                "sub_count": 1,
                "op": r.get("op", "?"),
            })
    return rows


def render(rows: list[dict], out_png: Path, title: str,
           subtitle: str, annotation: str,
           deadline_ms: float | None = None,
           show_periods: bool = True,
           highlight_fused: bool = True,
           x_max_ms: float | None = None,
           changed_dispatch_ids: set[int] | None = None) -> None:
    if not rows:
        raise SystemExit(f"no rows to render for {out_png}")

    # Lane ordering: CPU_P#0 / Gemmini on top, CPU_E#0 / OPU on bottom.
    lanes_seen = sorted({r["lane"] for r in rows})
    # Normalize: prefer CPU_P first, then CPU_E.
    def _lane_sort_key(l):
        if "P#" in l or "gemmini" in l.lower(): return 0
        if "E#" in l or "rvv_opu" in l.lower() or "opu" in l.lower(): return 1
        return 2
    lanes_seen = sorted(lanes_seen, key=_lane_sort_key)
    lane_to_y = {l: i for i, l in enumerate(lanes_seen)}

    fig, ax = plt.subplots(figsize=(15, 5.5))

    networks_present = set()
    for r in rows:
        y = lane_to_y[r["lane"]] + 1
        color = _color_for(r["network"], r["instance"])
        hatch = "////" if (highlight_fused and r["is_fused"]) else None
        edgecolor = "black" if r["is_fused"] else "none"
        linewidth = 1.5 if r["is_fused"] else 0
        ax.broken_barh(
            [(r["start_ms"], max(r["duration_ms"], 0.01))],
            (y - 0.35, 0.7),
            facecolors=color, edgecolors=edgecolor, hatch=hatch,
            linewidth=linewidth)
        networks_present.add(r["network"])

    # Periodic-frequency boundaries.
    x_max = x_max_ms if x_max_ms is not None else max(
        r["start_ms"] + r["duration_ms"] for r in rows)
    if show_periods:
        for net, period in NETWORK_PERIOD_MS.items():
            if net not in networks_present:
                continue
            t = period
            while t < x_max + 0.5:
                ax.axvline(t, color=NETWORK_COLORS[net], linestyle="--",
                           alpha=0.35, linewidth=1)
                ax.text(t, len(lanes_seen) + 0.85,
                        f"{net} t={int(t)}ms",
                        rotation=0, fontsize=7,
                        color=NETWORK_COLORS[net], ha="center", va="bottom",
                        alpha=0.65)
                t += period

    # Deadline marker. Skip drawing if it falls outside the x-axis range
    # — the chart's title/annotation states the relationship explicitly,
    # and an off-screen vertical line just confuses the layout.
    if deadline_ms is not None and deadline_ms <= x_max * 1.02:
        ax.axvline(deadline_ms, color="red", linestyle="-",
                   linewidth=2, alpha=0.7)
        ax.text(deadline_ms, len(lanes_seen) + 0.55,
                f"deadline\n{deadline_ms:.1f} ms",
                color="red", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Y-axis labels.
    ax.set_yticks(range(1, len(lanes_seen) + 1))
    ax.set_yticklabels([LANE_NAMES.get(l, l) for l in lanes_seen],
                       fontsize=10)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylim(0.4, len(lanes_seen) + 1.6)
    ax.set_xlim(0, x_max * 1.02)
    ax.grid(axis="x", linestyle=":", alpha=0.25)

    # Title + subtitle.
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=10, color="#444")

    # Legend: one swatch per network present, plus a hatched swatch
    # for the fused class if any.
    legend_handles = []
    for net in networks_present:
        legend_handles.append(mpatches.Patch(
            color=NETWORK_COLORS[net], label=net))
    if any(r["is_fused"] for r in rows):
        legend_handles.append(mpatches.Patch(
            facecolor="white", edgecolor="black", hatch="////",
            label="fused (N sub-ops in one dispatch)"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              framealpha=0.9)

    # Annotation box at the bottom (verdict + what changed).
    if annotation:
        fig.text(0.02, 0.02, annotation, fontsize=9.5, family="monospace",
                 bbox=dict(facecolor="#FFF8DC", edgecolor="#888",
                          boxstyle="round,pad=0.6"))
        plt.subplots_adjust(bottom=0.20 + 0.025 * annotation.count("\n"))

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, help="XPU-RT schedule fixture (predicted)")
    p.add_argument("--trace", type=Path, help="xpurt_trace.csv (measured)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", required=True)
    p.add_argument("--subtitle", default="")
    p.add_argument("--annotation", default="")
    p.add_argument("--deadline-ms", type=float, default=None)
    p.add_argument("--clock-mhz", type=float, default=1000.0)
    p.add_argument("--x-max-ms", type=float, default=None)
    args = p.parse_args(argv)

    if args.fixture:
        rows = load_from_fixture(args.fixture)
    elif args.trace:
        rows = load_from_trace(args.trace, clock_mhz=args.clock_mhz)
    else:
        raise SystemExit("supply --fixture or --trace")

    render(rows, args.out, args.title, args.subtitle, args.annotation,
           deadline_ms=args.deadline_ms, x_max_ms=args.x_max_ms)


if __name__ == "__main__":
    raise SystemExit(main())
