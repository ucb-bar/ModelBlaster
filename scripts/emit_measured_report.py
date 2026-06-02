"""Convert a FireSim `xpurt_trace.csv` into an XPU-RT measured SchedulerReport.

Closes the agentic-loop's measured side: XPU-RT's `iterate_firesim.py`
produces a predicted-only `_report.json` (schema v2, per-dispatch
list). The harness then emits a per-dispatch actual-cycles trace
between `=== MODELBLASTER_XPURT_TRACE_BEGIN ===` /
`=== MODELBLASTER_XPURT_TRACE_END ===` markers. This script overlays
the actuals onto the predicted report so
`/scratch2/agustin/XPU-RT/xpu-rt/advisor.py --report <measured.json>`
can re-diagnose with real numbers.

The output preserves every field XPU-RT's advisor reads (makespan,
per-dispatch list, hardware_target, granularity, deadline, etc.) and
adds:
  - top-level `measured_makespan_us` and `clock_mhz`
  - per-dispatch `actual_start_us` / `actual_end_us` / `delta_us`

Usage:

    python3 scripts/emit_measured_report.py \\
        --predicted-report /scratch2/agustin/XPU-RT/schedules/scheduled__iter_heft_..._profiled_report.json \\
        --trace            artifacts/bundle/A2/xpurt_trace.csv \\
        --out              artifacts/bundle/A2/measured_report.json \\
        --clock-mhz 1000
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path


def _load_trace(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _index_predicted_dispatches(report: dict) -> list[dict]:
    """SchedulerReport v2 stores dispatches under different keys
    depending on emit path. Try the common ones."""
    for k in ("dispatches", "per_dispatch", "schedule"):
        v = report.get(k)
        if isinstance(v, list):
            return v
    raise SystemExit(
        f"predicted report has no per-dispatch list (looked for "
        f"'dispatches' / 'per_dispatch' / 'schedule'); keys: {list(report)}")


def _key_for(entry: dict) -> tuple[str, int, int]:
    """Match key for a dispatch: (network, instance, dispatch_id).

    Both the predicted report and the trace expose all three so the
    match is exact even with multi-instance periodic workloads."""
    return (
        str(entry.get("network", entry.get("job_name", ""))),
        int(entry.get("instance", 0)),
        int(entry.get("dispatch_id", entry.get("id", -1))),
    )


def _trace_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        row.get("network", "").strip(),
        int(row.get("instance", 0)),
        int(row.get("dispatch_id", -1)),
    )


def overlay(predicted: dict, trace: list[dict[str, str]],
            clock_mhz: float = 1000.0) -> dict:
    out = copy.deepcopy(predicted)
    trace_by_key = {_trace_key(r): r for r in trace}

    cycles_per_us = clock_mhz  # 1 cycle / (1/clock_mhz µs) = clock_mhz cycles/µs
    measured_last_end_us = 0.0

    dispatches = _index_predicted_dispatches(out)
    matched = 0
    for entry in dispatches:
        k = _key_for(entry)
        r = trace_by_key.get(k)
        if r is None:
            continue
        try:
            a_s = int(r["actual_start_cycles"])
            a_e = int(r["actual_end_cycles"])
        except (KeyError, ValueError):
            continue
        a_s_us = a_s / cycles_per_us
        a_e_us = a_e / cycles_per_us
        entry["actual_start_us"] = a_s_us
        entry["actual_end_us"] = a_e_us
        # Some XPU-RT reports store predicted as `start_time` / `duration` (ms);
        # surface a `delta_us` for both representations.
        pred_us = None
        if "start_time" in entry and "duration" in entry:
            pred_us = (float(entry["start_time"]) + float(entry["duration"])) * 1000.0
        elif "predicted_end_us" in entry:
            pred_us = float(entry["predicted_end_us"])
        if pred_us is not None:
            entry["delta_us"] = a_e_us - pred_us
        if a_e_us > measured_last_end_us:
            measured_last_end_us = a_e_us
        matched += 1

    out["measured_makespan_us"] = measured_last_end_us
    out["clock_mhz"] = clock_mhz
    out["measured_trace_matched"] = matched
    out["measured_trace_total"] = len(trace)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predicted-report", required=True, type=Path,
                   help="XPU-RT scheduled_*_report.json (the predicted report).")
    p.add_argument("--trace", required=True, type=Path,
                   help="xpurt_trace.csv extracted from the FireSim uartlog.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output measured report JSON.")
    p.add_argument("--clock-mhz", type=float, default=1000.0,
                   help="Clock for cycles->microseconds conversion. "
                        "Default 1000 MHz (FireSim chipyard default).")
    args = p.parse_args(argv)

    predicted = json.loads(args.predicted_report.read_text())
    trace = _load_trace(args.trace)
    merged = overlay(predicted, trace, clock_mhz=args.clock_mhz)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2))
    matched = merged["measured_trace_matched"]
    total = merged["measured_trace_total"]
    print(f"emit_measured_report: wrote {args.out} "
          f"(matched {matched}/{total} trace rows, "
          f"measured_makespan_us={merged['measured_makespan_us']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
