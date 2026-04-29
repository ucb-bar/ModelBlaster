"""Reconstruct a Gantt timeline from an xpurt execution trace.

The schedule-driven harness (`harness_xpurt`) emits a CSV block between
``=== AGENTS_XPURT_TRACE_BEGIN ===`` and ``=== AGENTS_XPURT_TRACE_END ===``
when built with ``-DAGENTS_XPURT_TRACE=ON``. Each row carries:

  entry_id, network, instance, dispatch_id, op, name, core_kind, hart,
  predicted_start_ms, predicted_duration_ms, worker_kind_idx,
  actual_start_cycles, actual_end_cycles

We render two stacked Gantt charts:
  * top: XPU-RT's predicted schedule (predicted_start_ms / duration_ms)
  * bottom: actual measured execution (actual_start/end_cycles divided
    by the assumed clock_mhz)

with bars colored by network and grouped on lanes by worker_kind_idx
(predicted side: by core_kind+hart). Red lines flag entries whose
actual end exceeded the predicted finish, the most actionable signal
for tuning the schedule against measured cost.

Usage:
    spike -p4 build/.../zephyr.elf > trace.log
    python -m agents.scripts.plot_xpurt_trace trace.log \\
        --clock-mhz 1000 --out plots/xpurt_trace.png

Or pipe directly:
    spike -p4 .../zephyr.elf | python -m agents.scripts.plot_xpurt_trace - \\
        --out plots/xpurt_trace.png
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass


_BEGIN = "=== AGENTS_XPURT_TRACE_BEGIN ==="
_END = "=== AGENTS_XPURT_TRACE_END ==="


@dataclass
class TraceEntry:
    entry_id: int
    network: str
    instance: int
    dispatch_id: int
    op: str
    name: str
    core_kind: str
    hart: int
    predicted_start_ms: float
    predicted_duration_ms: float
    worker_kind_idx: int
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
            "trace markers not found — was the binary built with "
            "-DAGENTS_XPURT_TRACE=ON, and was the full spike output "
            "captured into the input?"
        )
    start = text.index(_BEGIN) + len(_BEGIN)
    end = text.index(_END, start)
    return text[start:end].strip()


def parse_trace(text: str, clock_mhz: float) -> list[TraceEntry]:
    block = _extract_block(text)
    rows: list[TraceEntry] = []
    reader = csv.DictReader(io.StringIO(block))
    cycles_per_ms = clock_mhz * 1000.0  # cycles in 1 ms
    for r in reader:
        actual_start_cyc = int(r["actual_start_cycles"])
        actual_end_cyc = int(r["actual_end_cycles"])
        rows.append(TraceEntry(
            entry_id=int(r["entry_id"]),
            network=r["network"],
            instance=int(r["instance"]),
            dispatch_id=int(r["dispatch_id"]),
            op=r["op"],
            name=r["name"],
            core_kind=r["core_kind"],
            hart=int(r["hart"]),
            predicted_start_ms=float(r["predicted_start_ms"]),
            predicted_duration_ms=float(r["predicted_duration_ms"]),
            worker_kind_idx=int(r["worker_kind_idx"]),
            actual_start_ms=actual_start_cyc / cycles_per_ms,
            actual_end_ms=actual_end_cyc / cycles_per_ms,
        ))
    return rows


def write_csv(rows: list[TraceEntry], path: str) -> None:
    """Write a flat CSV that other tools (XPU-RT comparison, sheet
    pivots) can ingest. Same columns as the on-device printout, plus
    converted-to-ms variants."""
    cols = [
        "entry_id", "network", "instance", "dispatch_id", "op", "name",
        "core_kind", "hart", "worker_kind_idx",
        "predicted_start_ms", "predicted_duration_ms",
        "actual_start_ms", "actual_end_ms", "actual_duration_ms",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r.entry_id, r.network, r.instance, r.dispatch_id, r.op, r.name,
                r.core_kind, r.hart, r.worker_kind_idx,
                f"{r.predicted_start_ms:.6f}", f"{r.predicted_duration_ms:.6f}",
                f"{r.actual_start_ms:.6f}", f"{r.actual_end_ms:.6f}",
                f"{r.actual_duration_ms:.6f}",
            ])


def _summary(rows: list[TraceEntry]) -> str:
    if not rows:
        return "(no trace entries)"
    pred_makespan = max(r.predicted_start_ms + r.predicted_duration_ms for r in rows)
    actual_makespan = max(r.actual_end_ms for r in rows)
    mean_pred = sum(r.predicted_duration_ms for r in rows) / len(rows)
    mean_actual = sum(r.actual_duration_ms for r in rows) / len(rows)
    overruns = sum(
        1 for r in rows
        if r.actual_end_ms > r.predicted_start_ms + r.predicted_duration_ms + 0.001
    )
    return (
        f"entries: {len(rows)}\n"
        f"  predicted makespan: {pred_makespan:.3f} ms\n"
        f"  actual    makespan: {actual_makespan:.3f} ms "
        f"({actual_makespan / pred_makespan:.2f}x predicted)\n"
        f"  mean per-entry: predicted={mean_pred:.3f} ms, "
        f"actual={mean_actual:.3f} ms\n"
        f"  entries finishing later than predicted: "
        f"{overruns}/{len(rows)}"
    )


def render_plot(rows: list[TraceEntry], out_path: str,
                source: str = "spike") -> None:
    """Two stacked Gantt charts (predicted on top, actual below).

    `source` is the simulator that produced the trace — used in the
    bottom subplot title (e.g. "Actual execution on FireSim"). Free-form
    so future runners can pass their own label.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib not installed — `pip install matplotlib` "
            "or omit --out and use the printed CSV / summary instead.")

    # Color per network — same hue across both panels for easy
    # cross-reference.
    nets = sorted({r.network for r in rows})
    palette = plt.get_cmap("tab10")
    color_for = {n: palette(i % 10) for i, n in enumerate(nets)}

    # Predicted-side lanes: one per (core_kind, hart) it was scheduled to.
    pred_lane_keys = sorted(
        {(r.core_kind, r.hart) for r in rows},
        key=lambda x: (x[0], x[1])
    )
    pred_lane_idx = {k: i for i, k in enumerate(pred_lane_keys)}

    # Actual-side lanes: one per worker_kind_idx that ran something.
    actual_lane_keys = sorted({r.worker_kind_idx for r in rows})
    actual_lane_idx = {k: i for i, k in enumerate(actual_lane_keys)}

    pred_makespan = max(r.predicted_start_ms + r.predicted_duration_ms for r in rows)
    actual_makespan = max(r.actual_end_ms for r in rows)
    xmax = max(pred_makespan, actual_makespan) * 1.02

    fig, (ax_pred, ax_actual) = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True,
        gridspec_kw={"hspace": 0.35})

    bar_h = 0.6
    for r in rows:
        c = color_for[r.network]
        ax_pred.barh(pred_lane_idx[(r.core_kind, r.hart)],
                     r.predicted_duration_ms,
                     left=r.predicted_start_ms, height=bar_h,
                     color=c, edgecolor="black", linewidth=0.3)
        # Bar color the same; use a red border on actual when it
        # finishes later than predicted.
        pred_end = r.predicted_start_ms + r.predicted_duration_ms
        overrun = r.actual_end_ms > pred_end + 0.001
        edge = "red" if overrun else "black"
        ew = 1.0 if overrun else 0.3
        ax_actual.barh(actual_lane_idx[r.worker_kind_idx],
                       r.actual_duration_ms,
                       left=r.actual_start_ms, height=bar_h,
                       color=c, edgecolor=edge, linewidth=ew)

    ax_pred.set_yticks(list(pred_lane_idx.values()))
    ax_pred.set_yticklabels([f"{kind}#hart{hart}"
                             for kind, hart in pred_lane_keys])
    ax_pred.set_title("XPU-RT predicted schedule")
    ax_pred.set_xlim(0, xmax)
    ax_pred.invert_yaxis()
    ax_pred.set_axisbelow(True)
    ax_pred.grid(axis="x", alpha=0.3)

    ax_actual.set_yticks(list(actual_lane_idx.values()))
    ax_actual.set_yticklabels([f"worker[{i}]" for i in actual_lane_keys])
    ax_actual.set_title(
        f"Actual execution on {source} "
        f"(red border = ran past predicted finish)")
    ax_actual.set_xlabel("time (ms)")
    ax_actual.invert_yaxis()
    ax_actual.set_axisbelow(True)
    ax_actual.grid(axis="x", alpha=0.3)

    # Vertical line at predicted vs actual makespan.
    for ax in (ax_pred, ax_actual):
        ax.axvline(pred_makespan, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.axvline(actual_makespan, color="black", linestyle="-",
                   linewidth=1.0, alpha=0.6)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_for[n], label=n)
                      for n in nets]
    ax_pred.legend(handles=legend_handles, loc="upper right",
                   bbox_to_anchor=(1.0, 1.0), ncol=1, framealpha=0.9)

    fig.suptitle(
        f"xpurt timeline — predicted {pred_makespan:.2f} ms vs actual "
        f"{actual_makespan:.2f} ms ({actual_makespan/pred_makespan:.2f}x)",
        fontsize=11)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="spike output capture (- for stdin)")
    ap.add_argument("--clock-mhz", type=float, default=10.0,
                    help="MHz rate of the on-device timer used for the "
                         "trace cycles. Default 10 matches Zephyr's "
                         "CONFIG_SYS_CLOCK_HW_CYCLES_PER_SEC=10000000 "
                         "for spike — k_cycle_get_64 returns mtime ticks "
                         "at 10 MHz, NOT processor cycles. Set to your "
                         "target's actual mtime rate for correctness.")
    ap.add_argument("--out", default=None,
                    help="path for the rendered PNG (omit to skip the plot)")
    ap.add_argument("--csv", default=None,
                    help="path for a flat trace CSV (predicted+actual per entry)")
    ap.add_argument("--source", default="spike",
                    help="simulator label for the bottom-subplot title "
                         "(e.g. 'spike', 'FireSim', 'rtl-sim'). Default: spike.")
    args = ap.parse_args()

    text = _read_text(args.input)
    rows = parse_trace(text, args.clock_mhz)

    print(_summary(rows))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"wrote {args.csv}")
    if args.out:
        render_plot(rows, args.out, source=args.source)
        print(f"wrote {args.out}")
    elif not args.csv:
        # No outputs requested — print the trace to stdout for piping.
        for r in rows:
            print(f"  {r.entry_id:>3}  {r.network:<13} d{r.dispatch_id:>2} "
                  f"{r.op:<14} pred=[{r.predicted_start_ms:7.3f}+"
                  f"{r.predicted_duration_ms:6.3f}ms]  "
                  f"actual=[{r.actual_start_ms:7.3f}+"
                  f"{r.actual_duration_ms:6.3f}ms]  "
                  f"worker[{r.worker_kind_idx}]")


if __name__ == "__main__":
    main()
