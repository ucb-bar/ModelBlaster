"""Walk a gen/profile/<...>/results.csv tree and summarize.

Two views:

  * Per-(model, backend, topo) totals — useful to see end-to-end
    scaling across pool sizes and backends.
  * Per-dispatch comparison across topos — pivot a single (model,
    backend) on dispatch_id to show how each dispatch's cost scales
    when the pool size changes. This is what XPU-RT looks at when
    deciding how to place dispatches.

Usage:
  python -m modelblaster.scripts.summarize_profile gen/profile
  python -m modelblaster.scripts.summarize_profile gen/profile --model mlp_control --backend scalar
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _walk_profile_root(root: str):
    """Yield (model, backend, cpu, topo, csv_path) tuples."""
    for backend in sorted(os.listdir(root)):
        backend_dir = os.path.join(root, backend)
        if not os.path.isdir(backend_dir):
            continue
        for cpu in sorted(os.listdir(backend_dir)):
            cpu_dir = os.path.join(backend_dir, cpu)
            if not os.path.isdir(cpu_dir):
                continue
            for model in sorted(os.listdir(cpu_dir)):
                model_dir = os.path.join(cpu_dir, model)
                if not os.path.isdir(model_dir):
                    continue
                for spec in sorted(os.listdir(model_dir)):
                    spec_dir = os.path.join(model_dir, spec)
                    if not os.path.isdir(spec_dir):
                        continue
                    for inner in sorted(os.listdir(spec_dir)):
                        inner_dir = os.path.join(spec_dir, inner)
                        if not os.path.isdir(inner_dir):
                            continue
                        for topo in sorted(os.listdir(inner_dir)):
                            csv_path = os.path.join(inner_dir, topo, "results.csv")
                            if os.path.exists(csv_path):
                                yield (model, backend, cpu, topo, csv_path)


def _read(csv_path: str) -> list[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _summary_line(model: str, backend: str, cpu: str, topo: str,
                  rows: list[dict]) -> str:
    total_cycles = sum(int(r.get("cycles", 0)) for r in rows)
    total_ns = sum(float(r.get("mean_time_ns", 0)) for r in rows)
    n = len(rows)
    return (f"{model:<14} {backend:<6} {cpu:<8} {topo:<14} "
            f"n={n:<3} total_cycles={total_cycles:>11,d}  total_us={total_ns/1e3:>10.1f}")


def print_totals(rows_by_key: dict) -> None:
    """One line per (model, backend, cpu, topo) with end-to-end cycles."""
    if not rows_by_key:
        print("(no profile data found)")
        return
    print(f"{'model':<14} {'backend':<6} {'cpu':<8} {'topo':<14} {'n':<5} "
          f"{'total_cycles':>15}  {'total_us':>11}")
    print("-" * 80)
    for key in sorted(rows_by_key.keys()):
        model, backend, cpu, topo = key
        rows = rows_by_key[key]
        print(_summary_line(model, backend, cpu, topo, rows))


def print_dispatch_pivot(rows_by_key: dict, model: str, backend: str) -> None:
    """For a fixed (model, backend), pivot every dispatch on topo to
    show per-dispatch scaling across pool sizes."""
    keys = [k for k in rows_by_key if k[0] == model and k[1] == backend]
    if not keys:
        print(f"no data for model={model!r} backend={backend!r}")
        return
    topos = sorted({k[3] for k in keys})

    # Build dispatch_id -> { topo -> row }
    by_did: dict[int, dict[str, dict]] = defaultdict(dict)
    for k in keys:
        topo = k[3]
        for r in rows_by_key[k]:
            did = int(r["dispatch_id"])
            by_did[did][topo] = r

    # Pick a reference row per dispatch for shape/op text.
    print(f"\n--- {model} / {backend} per-dispatch cycles by topo ---")
    header = f"{'did':>3}  {'op':<14}  {'shape':<28}  "
    header += "  ".join(f"{t:>14}" for t in topos)
    if len(topos) > 1:
        header += f"  {'speedup '+topos[0]+' vs '+topos[-1]:>14}"
    print(header)
    print("-" * len(header))
    for did in sorted(by_did.keys()):
        any_row = next(iter(by_did[did].values()))
        op, shape = any_row["op"], any_row["shape"]
        cells = []
        cyc_first = cyc_last = None
        for t in topos:
            r = by_did[did].get(t)
            if r is None:
                cells.append(f"{'(missing)':>14}")
                continue
            c = int(r["cycles"])
            cells.append(f"{c:>14,d}")
            if cyc_first is None:
                cyc_first = c
            cyc_last = c
        row = f"{did:>3}  {op:<14}  {shape:<28}  " + "  ".join(cells)
        if len(topos) > 1 and cyc_first and cyc_last:
            row += f"  {cyc_first/cyc_last:>13.2f}x"
        print(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", default="gen/profile", nargs="?",
                    help="profile root (default: gen/profile)")
    ap.add_argument("--model", default=None,
                    help="if set, also pivot per-dispatch across topos for "
                         "this (model, backend)")
    ap.add_argument("--backend", default=None,
                    help="paired with --model for the per-dispatch pivot")
    args = ap.parse_args()

    rows_by_key: dict = {}
    for model, backend, cpu, topo, path in _walk_profile_root(args.root):
        rows_by_key[(model, backend, cpu, topo)] = _read(path)

    print(f"# profile root: {args.root}\n")
    print_totals(rows_by_key)

    pivot_targets: list[tuple[str, str]] = []
    if args.model and args.backend:
        pivot_targets.append((args.model, args.backend))
    elif args.model:
        # auto-pivot all backends for this model
        backends = sorted({k[1] for k in rows_by_key if k[0] == args.model})
        pivot_targets.extend((args.model, b) for b in backends)
    else:
        # auto-pivot every (model, backend) pair found
        seen = sorted({(k[0], k[1]) for k in rows_by_key})
        pivot_targets.extend(seen)

    for m, b in pivot_targets:
        print_dispatch_pivot(rows_by_key, m, b)


if __name__ == "__main__":
    main()
