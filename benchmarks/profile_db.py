"""Per-op cycle profile database.

Aggregates per-dispatch FireSim cycles across all single-network runs into a
queryable JSONL store. The scheduler reads from here instead of one CSV per
run, so it can median across reps and reason about all (network, op, target)
combinations.

Storage layout (one file per (network, target, quant)):
    benchmarks/profile_db/<network>__<target>__<quant>.jsonl

Each line is one (run, dispatch) record:
    {"network": ..., "target": ..., "quant": ..., "workload_id": ...,
     "run_id": ..., "dispatch_id": ..., "op_type": ..., "op_name": ...,
     "signature": ..., "cycles": ..., "git_sha": ..., "captured_at": ...}

Public API:
    ingest(results_root) -> int                  # records added/updated
    query(network, target, quant, ...) -> dict   # median cycles per dispatch
    coverage_report() -> dict                    # tuples present, gaps surfaced

CLI:
    python -m benchmarks.profile_db ingest
    python -m benchmarks.profile_db coverage
    python -m benchmarks.profile_db query --network dronet --target gemmini --quant int8
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Iterable, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_DB_ROOT = REPO_ROOT / "benchmarks" / "profile_db"

# (network, target, quant) tuples we expect to see for a "complete" matrix.
# Anything in this set without records becomes a MISSING row in coverage.
EXPECTED_MATRIX: list[tuple[str, str, str]] = [
    ("dronet", "gemmini", "int8"),
    ("dronet", "gemmini_q31", "int8"),
    ("dronet", "rvv_opu", "int8"),
    ("yolov8_nano", "gemmini", "int8"),
    ("yolov8_nano", "gemmini_q31", "int8"),
    ("yolov8_nano", "rvv_opu", "int8"),
    # mlp_control: all targets in int8. Currently blocked by `elu_s8`
    # (extract_graph.extract_int8 raises NotImplementedError on nn.ELU).
    # Once elu_s8 lands these become regular gap-fill cells.
    ("mlp_control", "gemmini", "int8"),
    ("mlp_control", "gemmini_q31", "int8"),
    ("mlp_control", "rvv_opu", "int8"),
]


def _db_path(db_root: pathlib.Path, network: str, target: str, quant: str) -> pathlib.Path:
    return db_root / f"{network}__{target}__{quant}.jsonl"


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _existing_keys(records: list[dict]) -> set[tuple[str, int]]:
    return {(r["run_id"], r["dispatch_id"]) for r in records}


def _read_run_meta(run_dir: pathlib.Path) -> Optional[dict]:
    rj = run_dir / "run.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text())
    except json.JSONDecodeError:
        return None


def _read_profile_csv(csv_path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "dispatch_id": int(r["dispatch_id"]),
                "op_name": r["name"],
                "op_type": r["op"],
                "signature": r["shape"],
                "cycles": int(r["cycles"]),
            })
    return rows


def ingest(
    results_root: pathlib.Path = DEFAULT_RESULTS_ROOT,
    db_root: pathlib.Path = DEFAULT_DB_ROOT,
    verbose: bool = False,
) -> int:
    """Walk all <cell>/<run-id>/profile_firesim.csv and append new records.

    Idempotent: existing (run_id, dispatch_id) records are skipped. Returns
    the number of NEW records added.
    """
    db_root.mkdir(parents=True, exist_ok=True)
    arm_a = results_root / "A"
    if not arm_a.exists():
        return 0

    # Group new records by (network, target, quant) so we write each JSONL once.
    new_by_file: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for cell_dir in sorted(arm_a.iterdir()):
        if not cell_dir.is_dir():
            continue
        for run_dir in sorted(cell_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "latest":
                continue
            csv_path = run_dir / "profile_firesim.csv"
            if not csv_path.exists():
                continue
            meta = _read_run_meta(run_dir)
            if meta is None:
                if verbose:
                    print(f"  skip (no run.json): {run_dir}", file=sys.stderr)
                continue
            network = meta.get("model")
            target = meta.get("target")
            quant = meta.get("quant")
            if not (network and target and quant):
                continue
            # Hetero cells have target="hetero_*"; skip those — their per-op
            # cycles are split across multiple targets and live in xpurt_trace.csv.
            if target.startswith("hetero"):
                continue

            key = (network, target, quant)
            file_path = _db_path(db_root, *key)
            existing = _existing_keys(_load_jsonl(file_path))

            run_id = meta.get("run_id") or run_dir.name
            git_sha = meta.get("git_sha", "")
            captured_at = meta.get("started_at", "")
            workload_id = meta.get("workload_id", cell_dir.name)

            rows = _read_profile_csv(csv_path)
            for row in rows:
                if (run_id, row["dispatch_id"]) in existing:
                    continue
                new_by_file[key].append({
                    "network": network,
                    "target": target,
                    "quant": quant,
                    "workload_id": workload_id,
                    "run_id": run_id,
                    "dispatch_id": row["dispatch_id"],
                    "op_type": row["op_type"],
                    "op_name": row["op_name"],
                    "signature": row["signature"],
                    "cycles": row["cycles"],
                    "git_sha": git_sha,
                    "captured_at": captured_at,
                })

    added = 0
    for key, recs in new_by_file.items():
        file_path = _db_path(db_root, *key)
        with file_path.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        added += len(recs)
        if verbose:
            print(f"  +{len(recs):>5} -> {file_path.name}")
    return added


def query(
    network: str,
    target: str,
    quant: str,
    op_type: Optional[str] = None,
    agg: str = "median",
    db_root: pathlib.Path = DEFAULT_DB_ROOT,
) -> dict[int, int]:
    """Return {dispatch_id -> aggregated cycles} for this (network, target, quant).

    `agg` is one of: median, mean, min, max. Aggregated across all reps
    (= unique run_ids) present in the DB for matching records.
    If `op_type` is given, only those dispatches are returned.
    """
    path = _db_path(db_root, network, target, quant)
    if not path.exists():
        return {}
    by_dispatch: dict[int, list[int]] = defaultdict(list)
    for r in _load_jsonl(path):
        if op_type is not None and r["op_type"] != op_type:
            continue
        by_dispatch[r["dispatch_id"]].append(r["cycles"])

    fn = {
        "median": statistics.median,
        "mean": statistics.mean,
        "min": min,
        "max": max,
    }.get(agg)
    if fn is None:
        raise ValueError(f"unknown agg: {agg!r}")
    return {did: int(fn(vs)) for did, vs in by_dispatch.items()}


def _all_records(db_root: pathlib.Path = DEFAULT_DB_ROOT) -> Iterable[dict]:
    if not db_root.exists():
        return
    for path in sorted(db_root.glob("*.jsonl")):
        yield from _load_jsonl(path)


def coverage_report(db_root: pathlib.Path = DEFAULT_DB_ROOT) -> dict:
    """Summary of what (network, target, quant, op_type) tuples are present.

    Returns:
        {
          "present": [{network, target, quant, op_type, n_runs, n_dispatches, median, min, max}, ...],
          "missing": [{network, target, quant}, ...]
        }
    """
    by_key: dict[tuple, dict] = defaultdict(lambda: {"runs": set(), "cycles": []})
    seen_combos: set[tuple[str, str, str]] = set()
    for r in _all_records(db_root):
        combo = (r["network"], r["target"], r["quant"])
        seen_combos.add(combo)
        k = (*combo, r["op_type"])
        by_key[k]["runs"].add(r["run_id"])
        by_key[k]["cycles"].append(r["cycles"])

    present = []
    for (network, target, quant, op_type), agg in sorted(by_key.items()):
        cycles = agg["cycles"]
        present.append({
            "network": network,
            "target": target,
            "quant": quant,
            "op_type": op_type,
            "n_runs": len(agg["runs"]),
            "n_dispatches": len(cycles),
            "median": int(statistics.median(cycles)),
            "min": min(cycles),
            "max": max(cycles),
        })

    missing = []
    for combo in EXPECTED_MATRIX:
        if combo not in seen_combos:
            missing.append({"network": combo[0], "target": combo[1], "quant": combo[2]})

    return {"present": present, "missing": missing}


def _print_coverage(report: dict) -> None:
    print(f"\n=== Profile DB Coverage ({len(report['present'])} present rows, "
          f"{len(report['missing'])} MISSING combos) ===\n")
    if report["present"]:
        print(f"{'network':<14}{'target':<14}{'quant':<6}{'op_type':<24}"
              f"{'runs':>6}{'disp':>6}{'median':>14}{'min':>14}{'max':>14}")
        print("-" * 112)
        for r in report["present"]:
            print(f"{r['network']:<14}{r['target']:<14}{r['quant']:<6}"
                  f"{r['op_type']:<24}{r['n_runs']:>6}{r['n_dispatches']:>6}"
                  f"{r['median']:>14,}{r['min']:>14,}{r['max']:>14,}")
    if report["missing"]:
        print("\nMISSING combos (in EXPECTED_MATRIX but no records):")
        for m in report["missing"]:
            print(f"  - {m['network']:<14}{m['target']:<14}{m['quant']}")


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Scan results/ and append new records.")
    p_ing.add_argument("--results-root", type=pathlib.Path, default=DEFAULT_RESULTS_ROOT)
    p_ing.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)
    p_ing.add_argument("-v", "--verbose", action="store_true")

    p_cov = sub.add_parser("coverage", help="Print coverage matrix.")
    p_cov.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)
    p_cov.add_argument("--json", action="store_true", help="emit JSON instead of table")

    p_q = sub.add_parser("query", help="Return per-dispatch cycles.")
    p_q.add_argument("--network", required=True)
    p_q.add_argument("--target", required=True)
    p_q.add_argument("--quant", required=True)
    p_q.add_argument("--op-type", default=None)
    p_q.add_argument("--agg", default="median",
                     choices=["median", "mean", "min", "max"])
    p_q.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)

    args = ap.parse_args()

    if args.cmd == "ingest":
        added = ingest(args.results_root, args.db_root, verbose=args.verbose)
        print(f"added {added} record(s)")
        return 0
    if args.cmd == "coverage":
        report = coverage_report(args.db_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_coverage(report)
        return 0
    if args.cmd == "query":
        out = query(args.network, args.target, args.quant,
                    op_type=args.op_type, agg=args.agg, db_root=args.db_root)
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
