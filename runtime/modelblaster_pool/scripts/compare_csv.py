#!/usr/bin/env python3
"""Compare two parallel trees of IREE-shape results.csv files.

Usage:
    compare_csv.py <baseline_root> <after_root> [--out <markdown_path>]

Walks <baseline_root> for every results.csv, finds the matching path
under <after_root>, and emits a summary table in markdown:

    | profile_dir | baseline_total_ns | after_total_ns | ratio |

Plus an aggregate row per (backend, model, topo).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def _read_total_ns(path: str) -> int:
    total = 0
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                total += int(float(row["mean_time_ns"]))
            except (KeyError, ValueError):
                pass
    return total


def _walk(root: str) -> dict[str, str]:
    """Return relative-path -> absolute-path for every results.csv."""
    out: dict[str, str] = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "results.csv":
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                out[rel] = os.path.join(dirpath, fn)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_root")
    ap.add_argument("after_root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    baseline = _walk(args.baseline_root)
    after = _walk(args.after_root)

    rows: list[tuple[str, int, int, float]] = []
    common = sorted(set(baseline) & set(after))
    for rel in common:
        b = _read_total_ns(baseline[rel])
        a = _read_total_ns(after[rel])
        ratio = (a / b) if b > 0 else float("nan")
        rows.append((rel, b, a, ratio))

    only_baseline = sorted(set(baseline) - set(after))
    only_after = sorted(set(after) - set(baseline))

    lines: list[str] = []
    lines.append("# pthreadpool vs modelblaster_pool — FireSim per-(model, pool) totals\n")
    lines.append("Per-row: sum of `mean_time_ns` across every dispatch in the "
                 "results.csv. Lower is better; ratio = after / baseline.")
    lines.append("")
    lines.append("Both runs use the same generated kernels.c — the only "
                 "variable is the parallel-for pool implementation: "
                 "pthreadpool (xnnpack-vendored) vs modelblaster_pool (raw "
                 "pthreads + k_sem).")
    lines.append("")
    lines.append("| profile_dir | baseline_ns (pthreadpool) | after_ns (modelblaster_pool) | ratio |")
    lines.append("|---|---:|---:|---:|")
    for rel, b, a, ratio in rows:
        lines.append(f"| `{rel}` | {b:,} | {a:,} | {ratio:.3f}x |")
    lines.append("")

    # Aggregate by topo (the cores/pool size encoding).
    by_topo: dict[str, list[float]] = defaultdict(list)
    for rel, b, a, ratio in rows:
        topo = "topo_0" if "topo_0/" in rel.replace(os.sep, "/") else (
            "topo_0_1" if "topo_0_1/" in rel.replace(os.sep, "/") else (
                "topo_0_1_2_3" if "topo_0_1_2_3" in rel else "?"))
        by_topo[topo].append(ratio)

    lines.append("## Aggregate per topo (mean ratio across both backends and models)")
    lines.append("")
    lines.append("| topo | n | mean ratio |")
    lines.append("|---|---:|---:|")
    for k in sorted(by_topo):
        rs = by_topo[k]
        lines.append(f"| {k} | {len(rs)} | {sum(rs)/len(rs):.3f}x |")
    lines.append("")

    if only_baseline:
        lines.append("## Only in baseline\n")
        for r in only_baseline:
            lines.append(f"- `{r}`")
        lines.append("")
    if only_after:
        lines.append("## Only in after\n")
        for r in only_after:
            lines.append(f"- `{r}`")
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
