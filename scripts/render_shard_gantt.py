#!/usr/bin/env python3
"""Render a side-by-side 2-hart measured Gantt comparing sharded vs unsharded
yolov8 runs from xpurt_trace CSV files.

Schema of the input CSV (as written by ModelBlaster's xpurt_main with the
-DMODELBLASTER_XPURT_TRACE build flag) follows what XPU-RT's plot_gantt.py
already consumes:

    entry_id,network,instance,dispatch_id,op,name,core_kind,hart,
    predicted_start_ms,predicted_duration_ms,
    actual_start_cycles,actual_end_cycles

For this renderer we only need:
    network, instance, op, name, core_kind, hart,
    actual_start_cycles (alias: start_cycles),
    actual_end_cycles   (alias: end_cycles)

We accept either the long XPU-RT names or the short "start_cycles/end_cycles"
names so hand-fabricated fixtures stay readable.

The output is a two-row figure:

    [ Unsharded baseline ]
        gemmini (hart 0):  ...op rectangles...
        rvv_opu (hart 1):  ...op rectangles...

    [ Sharded scenario   ]
        gemmini (hart 0):  ...tile_0 in parallel with...
        rvv_opu (hart 1):  ...tile_1...

Both subplots share the x-axis so the wall difference reads at a glance.
Highlighted ops (matched by --highlight-name as a substring) are drawn with
a bold red outline and a slightly taller height, with the op name printed
inside the rectangle.

Usage:
    python scripts/render_shard_gantt.py \\
        --unsharded-trace artifacts/.../v26_unsharded_baseline/xpurt_trace.csv \\
        --sharded-trace   artifacts/.../v26_shard_l0/xpurt_trace.csv \\
        --out             artifacts/.../v26_shard_l0/shard_comparison.png \\
        --highlight-name  l0.conv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Optional

# Matplotlib forced to Agg before pyplot import so this runs headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402


# ---- palette (kept in lockstep with XPU-RT/xpu-rt/plot_gantt.py) ----

def _network_palette() -> dict:
    return {
        "yolov8_nano":     "#3b82f6",  # blue
        "yolov8_nano_64":  "#1e40af",  # darker blue
        "yolov8":          "#3b82f6",
        "dronet":          "#ef4444",  # red
        "mlp_control":     "#10b981",  # green
    }


def _instance_shade(base_hex: str, inst: int, n_inst: int = 4) -> str:
    base = base_hex.lstrip("#")
    r = int(base[0:2], 16)
    g = int(base[2:4], 16)
    b = int(base[4:6], 16)
    frac = min(0.55, 0.0 if n_inst <= 1 else 0.55 * inst / (n_inst - 1))
    rr = int(r + (255 - r) * frac)
    gg = int(g + (255 - g) * frac)
    bb = int(b + (255 - b) * frac)
    return f"#{rr:02x}{gg:02x}{bb:02x}"


def _network_root(name: str) -> str:
    for prefix in ("yolov8_nano_64", "yolov8_nano", "yolov8", "dronet", "mlp_control"):
        if name.startswith(prefix):
            return prefix
    return name


# ---- CSV reader (tolerates both XPU-RT and short fabricated schemas) ----

def _read_trace(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            # blank lines
            if not any((v or "").strip() for v in r.values() if isinstance(v, str)):
                continue
            try:
                start_key = "actual_start_cycles" if "actual_start_cycles" in r else "start_cycles"
                end_key = "actual_end_cycles" if "actual_end_cycles" in r else "end_cycles"
                rows.append({
                    "network": r.get("network", ""),
                    "instance": int(r.get("instance", 0) or 0),
                    "op": r.get("op", ""),
                    "name": r.get("name", ""),
                    "core_kind": r.get("core_kind", ""),
                    "hart": int(r["hart"]),
                    "start": float(r[start_key]),
                    "end": float(r[end_key]),
                })
            except (KeyError, ValueError):
                continue
    return rows


# ---- main rendering ----

LANES = [
    ("gemmini",  0, "gemmini (hart 0)"),
    ("rvv_opu",  1, "rvv_opu (hart 1)"),
]


def _lane_index(core_kind: str, hart: int) -> Optional[int]:
    for idx, (ck, h, _label) in enumerate(LANES):
        if ck == core_kind and h == hart:
            return idx
    # fall back: any rvv-family on hart 1, any gemmini/scalar on hart 0
    if hart == 0:
        return 0
    if hart == 1:
        return 1
    return None


def _draw_panel(ax, rows: list[dict], title_prefix: str, highlight: Optional[str],
                n_inst_per_net: dict[str, int]) -> tuple[float, dict]:
    """Draw one panel. Returns (wall_cycles, highlight_summary dict)."""
    palette = _network_palette()
    BAR_H = 0.55
    HL_H = 0.75

    wall = max((r["end"] for r in rows), default=0.0)
    highlight_spans: list[tuple[int, float, float, str]] = []  # (lane, start, end, name)

    for r in rows:
        lane = _lane_index(r["core_kind"], r["hart"])
        if lane is None:
            continue
        net = _network_root(r["network"])
        base = palette.get(net, "#94a3b8")
        color = _instance_shade(base, r["instance"], n_inst_per_net.get(net, 1))
        dur = r["end"] - r["start"]
        is_hl = bool(highlight) and (highlight in r["name"] or highlight in r["op"])
        h = HL_H if is_hl else BAR_H
        ax.broken_barh(
            [(r["start"], dur)],
            (lane - h / 2, h),
            facecolors=color,
            edgecolors=("#dc2626" if is_hl else "black"),
            linewidth=(2.2 if is_hl else 0.3),
            zorder=(3 if is_hl else 2),
        )
        if is_hl:
            highlight_spans.append((lane, r["start"], r["end"], r["name"]))
            # Label inside the rectangle
            ax.text(
                r["start"] + dur / 2,
                lane,
                r["name"],
                ha="center", va="center",
                fontsize=7, color="black", weight="bold", zorder=4,
            )

    # Y axis
    ax.set_yticks(range(len(LANES)))
    ax.set_yticklabels([lbl for _ck, _h, lbl in LANES])
    ax.set_ylim(len(LANES) - 0.5, -0.5)  # lane 0 on top
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Wall-end marker
    ax.axvline(wall, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    # Annotate wall on top edge
    ax.annotate(
        f"wall={wall:,.0f} cyc",
        xy=(wall, -0.5), xytext=(-5, 6),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.85),
    )

    ms = wall / 1e6
    ax.set_title(f"{title_prefix} — wall = {wall:,.0f} cycles ({ms:.3f} ms)", fontsize=10)

    return wall, {"spans": highlight_spans}


def _compute_overlap(spans: list[tuple[int, float, float, str]]) -> float:
    """Given highlight spans, compute overlap between lane-0 and lane-1 spans."""
    lane0 = [(s, e) for (lane, s, e, _n) in spans if lane == 0]
    lane1 = [(s, e) for (lane, s, e, _n) in spans if lane == 1]
    overlap = 0.0
    for s0, e0 in lane0:
        for s1, e1 in lane1:
            o = min(e0, e1) - max(s0, s1)
            if o > 0:
                overlap += o
    return overlap


def render(unsharded_csv: str, sharded_csv: str, out_path: str,
           highlight: Optional[str] = None) -> dict:
    unsharded = _read_trace(unsharded_csv)
    sharded = _read_trace(sharded_csv)
    if not unsharded:
        raise RuntimeError(f"no rows in {unsharded_csv}")
    if not sharded:
        raise RuntimeError(f"no rows in {sharded_csv}")

    # Per-network instance count over the union (so shading is consistent)
    n_inst_per_net: dict[str, int] = {}
    for r in unsharded + sharded:
        net = _network_root(r["network"])
        n_inst_per_net[net] = max(n_inst_per_net.get(net, 0), r["instance"] + 1)

    fig, (ax_u, ax_s) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    wall_u, _ = _draw_panel(ax_u, unsharded, "Unsharded baseline", highlight, n_inst_per_net)
    wall_s, hl_s = _draw_panel(ax_s, sharded,
                               f"Sharded {highlight or ''}".rstrip(),
                               highlight, n_inst_per_net)

    # x-axis shared. Bottom label in cycles; secondary axis on top in ms@1GHz.
    ax_s.set_xlabel("cycles since run start")
    xmax = max(wall_u, wall_s) * 1.02
    ax_u.set_xlim(0, xmax)
    ax_s.set_xlim(0, xmax)

    # Secondary "ms (1 GHz)" axis on top of the upper panel.
    secax = ax_u.secondary_xaxis(
        "top",
        functions=(lambda x: x / 1e6, lambda x: x * 1e6),
    )
    secax.set_xlabel("ms (assuming 1 GHz)")

    # Legend at bottom — one entry per (network, instance) seen
    palette = _network_palette()
    seen = sorted({(_network_root(r["network"]), r["instance"]) for r in unsharded + sharded})
    handles = []
    for net, inst in seen:
        base = palette.get(net, "#94a3b8")
        shade = _instance_shade(base, inst, n_inst_per_net.get(net, 1))
        label = f"{net} (inst {inst})" if n_inst_per_net.get(net, 1) > 1 else net
        handles.append(mpatches.Patch(color=shade, label=label))
    if highlight:
        handles.append(mpatches.Patch(facecolor="white", edgecolor="#dc2626",
                                      linewidth=2.0, label=f"highlight: {highlight}"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
               fontsize=8, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("2-hart measured Gantt: sharded vs unsharded", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Summary line
    delta = wall_s - wall_u
    pct = (100.0 * delta / wall_u) if wall_u > 0 else 0.0

    tile_durs: dict[str, float] = {}
    for (lane, s, e, name) in hl_s["spans"]:
        tile_durs[name] = tile_durs.get(name, 0.0) + (e - s)
    overlap = _compute_overlap(hl_s["spans"])

    parts = [
        f"unsharded wall={wall_u:,.0f} cyc",
        f"sharded wall={wall_s:,.0f} cyc",
        f"delta={delta:+,.0f} cyc ({pct:+.1f}%)",
    ]
    if tile_durs:
        tiles_str = " ".join(f"{n}={d/1e6:.2f}M" for n, d in sorted(tile_durs.items()))
        parts.append(f"-- {highlight} {tiles_str} overlap={overlap/1e6:.2f}M")
    summary = "  ".join(parts)

    return {
        "wall_unsharded": wall_u,
        "wall_sharded": wall_s,
        "delta": delta,
        "pct": pct,
        "summary": summary,
        "out_path": out_path,
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--unsharded-trace", required=True, help="xpurt_trace.csv for baseline")
    ap.add_argument("--sharded-trace",   required=True, help="xpurt_trace.csv for sharded run")
    ap.add_argument("--out",             required=True, help="output PNG path")
    ap.add_argument("--highlight-name",  default=None,
                    help="substring match against name/op to highlight in red")
    args = ap.parse_args(argv)

    result = render(
        unsharded_csv=args.unsharded_trace,
        sharded_csv=args.sharded_trace,
        out_path=args.out,
        highlight=args.highlight_name,
    )
    print(result["summary"])
    print(f"wrote {result['out_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
