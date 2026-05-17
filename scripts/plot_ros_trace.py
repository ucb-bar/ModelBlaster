"""Reconstruct a Gantt timeline from a harness_microros execution trace.

The fixed-HW ROS baseline (`harness_microros`) emits a CSV block between
``=== MODELBLASTER_ROS_TRACE_BEGIN ===`` and ``=== MODELBLASTER_ROS_TRACE_END ===``
once net_b (the long-running, one-shot network) completes. Each row carries:

  entry_id, network, instance, dispatch_id, op, name, kind, hart,
  start_cycles, end_cycles

Cycles are wall-relative to a single global t0 captured in main() right
before the worker threads start. We render one Gantt panel: bars per
(kind, hart) lane, colored by network. Multiple instances of a periodic
network appear as repeated bar runs along the timeline.

The op + name fields are intentionally left blank by the on-target
emitter — the host plotter optionally fills them in by looking up
dispatch_id in each model's graph.json (passed via --graph
network=path).

Usage:
    python -m modelblaster.scripts.plot_ros_trace baseline.uartlog \\
        --clock-mhz 1 --source firesim \\
        --out plots/ros_baseline.png --csv plots/ros_baseline.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass


_BEGIN = "=== MODELBLASTER_ROS_TRACE_BEGIN ==="
_END = "=== MODELBLASTER_ROS_TRACE_END ==="


@dataclass
class TraceEntry:
    entry_id: int
    network: str
    instance: int
    dispatch_id: int
    op: str
    name: str
    kind: str
    hart: int
    actual_start_ms: float
    actual_end_ms: float

    @property
    def actual_duration_ms(self) -> float:
        return max(self.actual_end_ms - self.actual_start_ms, 0.0)


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path) as f:
        return f.read()


def _extract_block(text: str) -> str:
    if _BEGIN not in text or _END not in text:
        raise SystemExit(
            "MODELBLASTER_ROS_TRACE markers not found — was the binary built "
            "from harness_microros and was the full uartlog captured?"
        )
    start = text.index(_BEGIN) + len(_BEGIN)
    end = text.index(_END, start)
    return text[start:end].strip()


def _load_graph_op_map(graph_path: str) -> dict[int, tuple[str, str]]:
    """Map dispatch_id -> (op, name) from a model's graph.json."""
    with open(graph_path) as f:
        graph = json.load(f)
    out: dict[int, tuple[str, str]] = {}
    # graph.json shape: list of dispatches with {dispatch_id, op, name}
    # under either top-level "ops", "dispatches", or "nodes" — accept all.
    # Prefer "ops" since "dispatches" may be a flat list of int IDs.
    items = []
    if isinstance(graph, dict):
        for k in ("ops", "nodes", "dispatches"):
            if k in graph and isinstance(graph[k], list) and graph[k] and isinstance(graph[k][0], dict):
                items = graph[k]
                break
    elif isinstance(graph, list):
        items = [x for x in graph if isinstance(x, dict)]
    for it in items:
        did = it.get("dispatch_id")
        if did is None:
            continue
        op = it.get("op", "")
        name = it.get("name", "")
        out[int(did)] = (str(op), str(name))
    return out


def parse_trace(text: str, clock_mhz: float,
                graph_for: dict[str, dict[int, tuple[str, str]]] | None = None
                ) -> list[TraceEntry]:
    block = _extract_block(text)
    rows: list[TraceEntry] = []
    reader = csv.DictReader(io.StringIO(block))
    cycles_per_ms = clock_mhz * 1000.0  # cycles in 1 ms
    graph_for = graph_for or {}
    for r in reader:
        net = r["network"]
        did = int(r["dispatch_id"])
        op = r["op"]
        name = r["name"]
        if (not op or not name) and net in graph_for and did in graph_for[net]:
            g_op, g_name = graph_for[net][did]
            op = op or g_op
            name = name or g_name
        actual_start_cyc = int(r["start_cycles"])
        actual_end_cyc = int(r["end_cycles"])
        rows.append(TraceEntry(
            entry_id=int(r["entry_id"]),
            network=net,
            instance=int(r["instance"]),
            dispatch_id=did,
            op=op,
            name=name,
            kind=r["kind"],
            hart=int(r["hart"]),
            actual_start_ms=actual_start_cyc / cycles_per_ms,
            actual_end_ms=actual_end_cyc / cycles_per_ms,
        ))
    return rows


def write_csv(rows: list[TraceEntry], path: str) -> None:
    cols = [
        "entry_id", "network", "instance", "dispatch_id", "op", "name",
        "kind", "hart",
        "actual_start_ms", "actual_end_ms", "actual_duration_ms",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.entry_id, r.network, r.instance, r.dispatch_id, r.op, r.name,
                r.kind, r.hart,
                f"{r.actual_start_ms:.6f}", f"{r.actual_end_ms:.6f}",
                f"{r.actual_duration_ms:.6f}",
            ])


def _summary(rows: list[TraceEntry]) -> str:
    if not rows:
        return "(no trace entries)"
    actual_makespan = max(r.actual_end_ms for r in rows)
    by_net: dict[tuple[str, int], list[TraceEntry]] = {}
    for r in rows:
        by_net.setdefault((r.network, r.instance), []).append(r)
    parts = [
        f"entries: {len(rows)}",
        f"  total makespan: {actual_makespan:.3f} ms",
    ]
    # Per-network-instance wall durations (entry_id-monotonic — first to
    # last entry for that (network, instance)).
    for (net, inst), grp in sorted(by_net.items()):
        s = min(g.actual_start_ms for g in grp)
        e = max(g.actual_end_ms for g in grp)
        parts.append(
            f"  {net}#{inst}: {len(grp)} ops, "
            f"start={s:.3f} ms, end={e:.3f} ms, dur={e - s:.3f} ms")
    return "\n".join(parts)


def render_plot(rows: list[TraceEntry], out_path: str,
                source: str = "firesim",
                title: str | None = None,
                init_bar: bool = True,
                init_bar_min_ms: float = 1.0) -> None:
    """Single-panel Gantt: actual ROS execution per (kind, hart) lane.

    `source` is the simulator that produced the trace — used in the
    plot title (e.g. "ROS execution on FireSim").

    When `init_bar=True` and the first dispatch on any (kind, hart) lane
    starts more than `init_bar_min_ms` after t=0, prepend a hatched
    "ROS/runtime init" bar covering the gap. That makes the cost of
    micro-ROS handshake / executor-init visible (vs the bare-metal
    xpurt path which starts at t≈0).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib not installed — `pip install matplotlib` "
            "or omit --out and use the printed CSV / summary instead.")

    nets = sorted({r.network for r in rows})
    palette = plt.get_cmap("tab10")
    color_for = {n: palette(i % 10) for i, n in enumerate(nets)}

    # Lanes: one per (kind, hart) — labels show both for clarity.
    lane_keys = sorted({(r.kind, r.hart) for r in rows},
                       key=lambda x: (x[1], x[0]))  # sort by hart then kind
    lane_idx = {k: i for i, k in enumerate(lane_keys)}

    actual_makespan = max(r.actual_end_ms for r in rows)
    xmax = actual_makespan * 1.02

    fig, ax = plt.subplots(1, 1, figsize=(14, 3.5 + 0.4 * len(lane_keys)))

    bar_h = 0.6

    # Per-lane "init" bar (gap from t=0 to first dispatch on that lane).
    # The initial gap is mostly micro-ROS session-establishment + node /
    # timer / executor setup. xpurt has near-zero gap so this is a no-op
    # for those traces.
    init_drawn = False
    if init_bar:
        first_start_per_lane: dict[tuple[str, int], float] = {}
        first_net_per_lane: dict[tuple[str, int], str] = {}
        for r in rows:
            key = (r.kind, r.hart)
            if (key not in first_start_per_lane
                    or r.actual_start_ms < first_start_per_lane[key]):
                first_start_per_lane[key] = r.actual_start_ms
                first_net_per_lane[key] = r.network
        for key, t0 in first_start_per_lane.items():
            if t0 < init_bar_min_ms:
                continue
            ax.barh(lane_idx[key], t0, left=0.0, height=bar_h,
                    color="#e0e0e0", edgecolor="#666666",
                    linewidth=0.5, hatch="///",
                    label="_init")
            init_drawn = True
            # Annotate the init bar with its width if it's wide enough
            # to fit text inside.
            if t0 >= 30:
                ax.text(t0 / 2.0, lane_idx[key],
                        f"ROS init\n{t0:.0f} ms",
                        ha="center", va="center", fontsize=8,
                        color="#333333")

    for r in rows:
        c = color_for[r.network]
        ax.barh(lane_idx[(r.kind, r.hart)],
                r.actual_duration_ms,
                left=r.actual_start_ms, height=bar_h,
                color=c, edgecolor="black", linewidth=0.3)

    ax.set_yticks(list(lane_idx.values()))
    ax.set_yticklabels([f"{kind} (hart {hart})"
                        for kind, hart in lane_keys])
    if title is None:
        title = (f"ROS baseline execution on {source} "
                 f"— makespan {actual_makespan:.2f} ms")
    ax.set_title(title)
    ax.set_xlabel("time (ms)")
    ax.set_xlim(0, xmax)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.3)

    # Mark dronet timer ticks at every 50 ms (or whatever the period was)
    # for visual sanity.  Inferred from min instance gap if periodic.
    instance_starts: dict[str, list[float]] = {}
    for r in rows:
        instance_starts.setdefault(r.network, []).append(
            (r.instance, r.actual_start_ms))
    for net, ss in instance_starts.items():
        first_per_inst: dict[int, float] = {}
        for inst, t in ss:
            if inst not in first_per_inst or t < first_per_inst[inst]:
                first_per_inst[inst] = t
        if len(first_per_inst) > 1:
            for t in sorted(first_per_inst.values()):
                ax.axvline(t, color=color_for[net], linestyle=":",
                           linewidth=0.6, alpha=0.4)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[n], label=n)
                      for n in nets]
    if init_drawn:
        legend_handles.append(plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#e0e0e0", edgecolor="#666666", hatch="///",
            label="ROS / runtime init"))
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), ncol=1, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="microros uartlog capture (- for stdin)")
    ap.add_argument("--clock-mhz", type=float, default=1.0,
                    help="cycle counter rate in MHz (default 1; "
                         "k_cycle_get_64 on this Zephyr config returns "
                         "microsecond-resolution ticks).")
    ap.add_argument("--out", default=None,
                    help="output PNG path (omit to skip plot rendering).")
    ap.add_argument("--csv", default=None,
                    help="output CSV path for downstream analysis.")
    ap.add_argument("--source", default="firesim",
                    help="label for the simulator that produced the trace.")
    ap.add_argument("--title", default=None,
                    help="override plot title (default auto-generated).")
    ap.add_argument("--graph", action="append", default=[],
                    help="lookup op/name from per-network graph.json: "
                         "--graph dronet=/path/to/dronet/graph.json")
    ap.add_argument("--no-init-bar", action="store_true",
                    help="suppress the hatched 'ROS init' bar that "
                         "covers the gap from t=0 to the first dispatch "
                         "on each lane (default: show).")
    ap.add_argument("--init-bar-min-ms", type=float, default=1.0,
                    help="minimum gap (in ms) before the init bar is "
                         "drawn (default 1 ms — lanes that start within "
                         "1 ms of t=0, e.g. xpurt, get no init bar).")
    args = ap.parse_args()

    text = _read_text(args.input)
    graph_for: dict[str, dict[int, tuple[str, str]]] = {}
    for spec in args.graph:
        if "=" not in spec:
            raise SystemExit(f"--graph expects net=path, got {spec!r}")
        net, path = spec.split("=", 1)
        if not os.path.exists(path):
            raise SystemExit(f"--graph {net}: {path} not found")
        graph_for[net] = _load_graph_op_map(path)
    rows = parse_trace(text, clock_mhz=args.clock_mhz,
                       graph_for=graph_for)
    if not rows:
        raise SystemExit("trace block empty")

    print(_summary(rows))

    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    if args.out:
        render_plot(rows, args.out, source=args.source, title=args.title,
                    init_bar=not args.no_init_bar,
                    init_bar_min_ms=args.init_bar_min_ms)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
