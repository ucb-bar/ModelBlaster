"""Convert a FireSim `xpurt_trace.csv` into an XPU-RT measured SchedulerReport.

Closes the agentic-loop's measured side: XPU-RT's `iterate_firesim.py`
produces a predicted-only `_report.json` (schema v2, per-dispatch
list). The harness then emits a per-dispatch actual-cycles trace
between `=== MODELBLASTER_XPURT_TRACE_BEGIN ===` /
`=== MODELBLASTER_XPURT_TRACE_END ===` markers. This script overlays
the actuals onto the predicted report so
`$XPURT_ROOT/xpu-rt/advisor.py --report <measured.json>`
can re-diagnose with real numbers.

The output preserves every field XPU-RT's advisor reads (makespan,
per-dispatch list, hardware_target, granularity, deadline, etc.) and
adds:
  - top-level `measured_makespan_us` and `clock_mhz`
  - per-dispatch `actual_start_us` / `actual_end_us` / `delta_us`

Usage:

    python3 scripts/emit_measured_report.py \\
        --predicted-report $XPURT_ROOT/schedules/scheduled__iter_heft_..._profiled_report.json \\
        --trace            artifacts/bundle/A2/xpurt_trace.csv \\
        --out              artifacts/bundle/A2/measured_report.json \\
        --clock-mhz 1000
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
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


_INSTANCE_RE = re.compile(r"^(?P<base>.*?)(?P<idx>\d+)$")

# XPU-RT writes `<job>_dispatch_<n>`; ModelBlaster writes `<job>$dispatch_<n>`.
_DISPATCH_SEPS = ("_dispatch_", "$dispatch_")


def _split_job_name(job: str, known: set[str] | None = None) -> tuple[str, int]:
    """Split "mlp_control3" into ("mlp_control", 3); no trailing index -> (job, 0).

    Mirrors `pipeline/ingest_xpurt_schedule._split_job_name`, including the
    longest-prefix match against `known`: a model whose own name ends in a
    digit ("yolov8_nano_640") would otherwise donate part of its name to the
    instance index. Keep the two in sync.
    """
    if known:
        for base in sorted(known, key=len, reverse=True):
            if job == base:
                return base, 0
            if job.startswith(base):
                rest = job[len(base):]
                if rest.isdigit():
                    return base, int(rest)
    m = _INSTANCE_RE.match(job)
    if not m:
        return job, 0
    return m.group("base"), int(m.group("idx"))


def _parse_entry_name(name: str,
                      known: set[str] | None = None) -> tuple[str, int, int] | None:
    """'mlp_control0_dispatch_7' -> ('mlp_control', 0, 7); None if unparseable.

    WHY THIS EXISTS. XPU-RT's SchedulerReport entries carry ONLY `name` --
    `profiling.py` writes `operation_name` and no `network` / `instance`
    fields at all -- while the trace CSV has all three as explicit columns.
    Keying the predicted side on the absent fields degraded every entry to
    ("", 0, id), which matches no trace row: the `matched 0/N` join that
    made the whole measured side of the loop report nothing.

    Accepts a trailing shard suffix (`..._dispatch_22_4`) by taking the
    FIRST integer after the separator as the dispatch id.
    """
    for sep in _DISPATCH_SEPS:
        if sep in name:
            job, _, post = name.partition(sep)
            head = post.split("_", 1)[0]
            if not head.isdigit():
                return None
            base, inst = _split_job_name(job, known)
            return base, inst, int(head)
    return None


def _key_for(entry: dict,
             known: set[str] | None = None) -> tuple[str, int, int]:
    """Match key for a dispatch: (network, instance, dispatch_id).

    Explicit fields win when present; otherwise each component falls back to
    what `name` encodes (see `_parse_entry_name`).
    """
    net = entry.get("network")
    inst = entry.get("instance")
    did = entry.get("dispatch_id", entry.get("id"))
    if net is not None and inst is not None and did is not None:
        return (str(net), int(inst), int(did))

    parsed = _parse_entry_name(str(entry.get("name", "")), known)
    if parsed is not None:
        p_net, p_inst, p_did = parsed
        return (
            str(net) if net is not None else p_net,
            int(inst) if inst is not None else p_inst,
            int(did) if did is not None else p_did,
        )
    return (
        str(net if net is not None else entry.get("job_name", "")),
        int(inst if inst is not None else 0),
        int(did if did is not None else -1),
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

    # The trace's `network` column is the authoritative list of network names
    # in this run, so it is what disambiguates a name whose own tail is a
    # digit when splitting `<job><instance>` off a predicted entry's `name`.
    known = {r.get("network", "").strip() for r in trace}
    known.discard("")

    cycles_per_us = clock_mhz  # 1 cycle / (1/clock_mhz µs) = clock_mhz cycles/µs
    measured_last_end_us = 0.0

    dispatches = _index_predicted_dispatches(out)
    matched = 0
    for entry in dispatches:
        k = _key_for(entry, known)
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
                        "Default 1000 MHz (FireSim chipyard default). "
                        "On the SpaceMiT K1 the trace's actual_*_cycles are "
                        "rdtime ticks and this MUST be 24, or every measured "
                        "time is under-reported by 41.7x.")
    p.add_argument("--allow-unmatched", action="store_true",
                   help="Exit 0 even when no trace row joins a predicted "
                        "dispatch. Off by default: a zero-row join means the "
                        "measured numbers describe nothing, and it used to "
                        "be reported as a clean success.")
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
    if total and not matched and not args.allow_unmatched:
        print(
            "emit_measured_report: ERROR -- 0 of "
            f"{total} trace rows joined a predicted dispatch, so every "
            "measured field in the output describes nothing. Check that the "
            "predicted report and the trace came from the SAME schedule: the "
            "join key is (network, instance, dispatch_id), taken from the "
            "trace's columns and from each report entry's `name`. Pass "
            "--allow-unmatched to accept this deliberately.",
            file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
