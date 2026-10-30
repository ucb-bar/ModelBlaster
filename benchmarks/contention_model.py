"""Derive per-op contention multipliers from captured xpurt_trace.csv files.

The multi-network HEFT/MOSEK scheduler uses solo single-network per-op cycles
from profile_db. Real FireSim runs in multi-network mode show 30-50× wall
inflation vs predicted, dominated by gemmini conv2d_s8 ops blowing up 80-100×
their solo cycles. The runtime overhead is real; this module quantifies it
empirically so the next solve can apply per-op multipliers.

Method:
  For each captured xpurt_trace.csv across multi-network runs:
    - For each row: actual_duration_ms = (actual_end - actual_start) / 1000
                    predicted_duration_ms = predicted_duration_ms (as stored)
                    ratio = actual / predicted
    - Bucket by (network, op_type, hardware_target) and take median ratio.
  Emit benchmarks/profile_db/contention_multipliers.json:
    { "version": 1, "source_runs": [...], "multipliers": {
        "yolov8_nano|conv2d_s8|gemmini": 9.5,
        "dronet|conv2d_s8|gemmini": 16.2,
        ... } }

Usage:
    python -m benchmarks.contention_model derive
    python -m benchmarks.contention_model show
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results" / "A"
DB_OUT = REPO_ROOT / "benchmarks" / "profile_db" / "contention_multipliers.json"


def _walk_traces(results_root: pathlib.Path):
    """Yield (cell, run_id, trace_path) for every multi-network capture."""
    if not results_root.exists():
        return
    for cell_dir in sorted(results_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        # Only multi-network cells have xpurt_trace.csv covering >=2 instances.
        name = cell_dir.name
        if not (name.startswith("3way_") or name.startswith("dronet_hetero")):
            continue
        for run_dir in sorted(cell_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "latest":
                continue
            trace = run_dir / "xpurt_trace.csv"
            if trace.exists():
                yield name, run_dir.name, trace


def _read_trace(path: pathlib.Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if not any(v.strip() for v in r.values() if isinstance(v, str)):
                continue
            try:
                ad = int(r["actual_end_cycles"]) - int(r["actual_start_cycles"])
                pd_ = float(r["predicted_duration_ms"])
                if pd_ <= 0:
                    continue
                rows.append({
                    "network": r["network"],
                    "op": r["op"],
                    "core_kind": r["core_kind"],
                    "actual_dur_ms": ad / 1000.0,  # mtime @ 1 MHz → ms
                    "predicted_dur_ms": pd_,
                    "ratio": (ad / 1000.0) / pd_,
                })
            except (KeyError, ValueError):
                continue
    return rows


def derive(results_root: pathlib.Path = RESULTS,
           out_path: pathlib.Path = DB_OUT,
           multi_only: bool = True) -> dict:
    """Walk all captured traces and emit a per (network, op_type, core_kind)
    multiplier file. `multi_only` excludes single-instance hetero runs
    (their predicted-vs-actual match too well and would skew the median
    DOWN; they're not representative of multi-network contention)."""
    by_key: dict[tuple, list[float]] = defaultdict(list)
    source_runs: list[str] = []
    for cell, run_id, trace_path in _walk_traces(results_root):
        # Skip the dronet_hetero_* runs which have only 1 instance.
        # They match predicted within 0.1% but don't tell us about
        # multi-network contention; we want 3way + sweep traces.
        if multi_only and cell.startswith("dronet_hetero"):
            continue
        source_runs.append(f"{cell}/{run_id}")
        for r in _read_trace(trace_path):
            key = (r["network"], r["op"], r["core_kind"])
            by_key[key].append(r["ratio"])

    multipliers: dict[str, dict] = {}
    for (net, op, core), ratios in sorted(by_key.items()):
        multipliers[f"{net}|{op}|{core}"] = {
            "n_samples": len(ratios),
            "median": round(statistics.median(ratios), 3),
            "mean":   round(statistics.mean(ratios), 3),
            "p90":    round(sorted(ratios)[max(0, int(len(ratios)*0.9)-1)], 3),
            "max":    round(max(ratios), 3),
        }

    out = {
        "version": 1,
        "source_runs": source_runs,
        "multipliers": multipliers,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def _print_summary(data: dict) -> None:
    print(f"=== Contention multipliers ({len(data['multipliers'])} (net, op, core) entries) ===")
    print(f"sources: {len(data['source_runs'])} runs")
    print()
    print(f"{'network':<14}{'op_type':<24}{'core':<10}{'n':>5}{'median':>10}{'p90':>10}{'max':>10}")
    print("-" * 90)
    for key, stats in sorted(data['multipliers'].items()):
        net, op, core = key.split("|")
        print(f"{net:<14}{op:<24}{core:<10}{stats['n_samples']:>5}"
              f"{stats['median']:>10.2f}{stats['p90']:>10.2f}{stats['max']:>10.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_d = sub.add_parser("derive", help="Walk captured traces, emit JSON.")
    p_d.add_argument("--results-root", type=pathlib.Path, default=RESULTS)
    p_d.add_argument("--out", type=pathlib.Path, default=DB_OUT)

    p_s = sub.add_parser("show", help="Print the saved JSON.")
    p_s.add_argument("--path", type=pathlib.Path, default=DB_OUT)

    args = ap.parse_args()
    if args.cmd == "derive":
        out = derive(args.results_root, args.out)
        _print_summary(out)
        print(f"\nwrote {args.out}")
    elif args.cmd == "show":
        with args.path.open() as f:
            data = json.load(f)
        _print_summary(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
