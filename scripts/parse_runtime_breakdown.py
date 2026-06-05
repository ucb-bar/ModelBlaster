#!/usr/bin/env python3
"""Phase G1 — parse MODELBLASTER_HART_ACC block from a v_N run.log and
emit breakdown.json with per-hart attribution + sum-of-categories
sanity check.

Usage:
    python scripts/parse_runtime_breakdown.py \
        artifacts/runtime_optimization/v9_baseline_instrumented/run.log
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

_BEGIN = "=== MODELBLASTER_HART_ACC_BEGIN ==="
_END = "=== MODELBLASTER_HART_ACC_END ==="
_WALL_RE = re.compile(r"\[(\w+)\] wall_clock_cycles=(\d+) \(mtime\)")
_NET_ERR_RE = re.compile(r"\[(\w+)\] max_abs_err=(\d+(?:\.\d+)?)")


def parse_hart_acc(text: str) -> list[dict]:
    if _BEGIN not in text:
        return []
    block = text.split(_BEGIN, 1)[1].split(_END, 1)[0]
    rows = list(csv.DictReader(io.StringIO(block.strip())))
    for r in rows:
        for k in list(r.keys()):
            if k.endswith("_us") or k in ("kind_idx", "entries_done"):
                try:
                    r[k] = int(r[k])
                except ValueError:
                    pass
    return rows


def parse_per_net(text: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in _WALL_RE.finditer(text):
        out.setdefault(m.group(1), {})["wall_clock_cycles_us"] = int(m.group(2))
    for m in _NET_ERR_RE.finditer(text):
        out.setdefault(m.group(1), {})["max_abs_err"] = float(m.group(2))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_log", type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run_log_dir>/breakdown.json")
    args = ap.parse_args(argv)

    text = args.run_log.read_text(errors="replace")
    harts = parse_hart_acc(text)
    nets = parse_per_net(text)

    if not harts:
        print("ERROR: no MODELBLASTER_HART_ACC block in", args.run_log,
              file=sys.stderr)
        return 1

    for h in harts:
        attributed = (h["kernel_us"] + h["dep_wait_us"]
                      + h["sync_overhead_us"] + h["target_gate_spin_us"]
                      + h["hart_idle_us"] + h["gemmini_cfg_emit_us"])
        h["attributed_us"] = attributed
        h["unattributed_us"] = h["wall_total_us"] - attributed
        h["attribution_pct"] = (
            100.0 * attributed / h["wall_total_us"] if h["wall_total_us"] else 0.0
        )

    out_path = args.out or args.run_log.parent / "breakdown.json"
    out_path.write_text(json.dumps(
        {"harts": harts, "per_network": nets}, indent=2))

    print(f"== Runtime breakdown ({args.run_log}) ==")
    print(f"{'kind':<12} {'wall':>8} {'kernel':>8} {'dep':>8} "
          f"{'sync':>8} {'gate':>8} {'idle':>8} {'cfg':>8} "
          f"{'unattr':>8} {'attrib%':>8}")
    for h in harts:
        print(f"{h['kind']:<12} {h['wall_total_us']:>8} {h['kernel_us']:>8} "
              f"{h['dep_wait_us']:>8} {h['sync_overhead_us']:>8} "
              f"{h['target_gate_spin_us']:>8} {h['hart_idle_us']:>8} "
              f"{h['gemmini_cfg_emit_us']:>8} {h['unattributed_us']:>8} "
              f"{h['attribution_pct']:>7.1f}%")
    print(f"\nPer-network: {json.dumps(nets, indent=2)}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
