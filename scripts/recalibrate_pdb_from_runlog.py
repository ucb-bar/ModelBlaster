"""Phase G3 — re-ingest measured per-dispatch cycles into the profile DB.

Reads a ModelBlaster run.log (e.g. the v20b FireSim run), extracts each
network's `MODELBLASTER_PROFILE_BEGIN [<net>] .. PROFILE_END [<net>]`
block, and updates the matching PDB CSV(s) for the canonical workload
networks (mlp_control, dronet, yolov8_nano).

Backend name mapping (uart → PDB):
  - gemmini_q31 → gemmini_q31
  - rvv_opu     → V256D128_rvv

Usage:
  python scripts/recalibrate_pdb_from_runlog.py \\
      --runlog artifacts/runtime_optimization/v20b_transpose_elim_retry/run.log \\
      --tag v20b
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

PDB_ROOT = Path("/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/profile/sweep_v8")
BACKEND_TO_PDB = {"gemmini_q31": "gemmini_q31", "rvv_opu": "V256D128_rvv"}

# PDB CSV layout: <backend>/firesim_rocket_saturn/<net>/<net>.int8/
#                 <net>_firesim_rocket_saturn_<backend>_<net>.int8/topo_0/results.csv
NETS = ["mlp_control", "dronet", "yolov8_nano"]


def parse_runlog(text: str) -> dict[str, dict[tuple[str, int], tuple[str, str, int, str]]]:
    """Return {net: {(backend, did): (op, shape, cycles, name)}}."""
    per_net: dict[str, dict] = {}
    cur_net: str | None = None
    begin_re = re.compile(r"=== MODELBLASTER_PROFILE_BEGIN \[(\w+)\] ===")
    end_re = re.compile(r"=== MODELBLASTER_PROFILE_END \[(\w+)\] ===")
    for line in text.splitlines():
        mb = begin_re.search(line)
        if mb:
            cur_net = mb.group(1)
            per_net.setdefault(cur_net, {})
            continue
        me = end_re.search(line)
        if me:
            cur_net = None
            continue
        if cur_net is None:
            continue
        if line.startswith(("gemmini_q31,", "rvv_opu,")):
            parts = line.split(",")
            if len(parts) < 6:
                continue
            backend = parts[0]
            try:
                did = int(parts[1])
            except ValueError:
                continue
            name, op, shape = parts[2], parts[3], parts[4]
            try:
                cyc = int(parts[-1])
            except ValueError:
                continue
            per_net[cur_net][(backend, did)] = (op, shape, cyc, name)
    return per_net


def pdb_path(net: str, backend: str) -> Path:
    pdb_backend = BACKEND_TO_PDB[backend]
    quant = "fp32" if net == "mlp_control" else "int8"
    return (PDB_ROOT / pdb_backend / "firesim_rocket_saturn" / net / f"{net}.{quant}"
            / f"{net}_firesim_rocket_saturn_{pdb_backend}_{net}.{quant}" / "topo_0"
            / "results.csv")


def update_pdb(pdb_csv: Path, measured: dict[tuple[str, int], tuple[str, str, int, str]],
               backend: str, backup_suffix: str) -> tuple[int, list[tuple]]:
    if not pdb_csv.is_file():
        return 0, []
    rows = list(csv.DictReader(open(pdb_csv)))
    if not rows:
        return 0, []
    fieldnames = list(rows[0].keys())
    backup = pdb_csv.with_suffix(f".csv.pre{backup_suffix}_backup")
    if not backup.exists():
        shutil.copy(pdb_csv, backup)
    n = 0
    deltas: list[tuple] = []
    for r in rows:
        try:
            did = int(r["dispatch_id"])
        except (KeyError, ValueError):
            continue
        key = (backend, did)
        if key not in measured:
            continue
        op, shape, cyc, name = measured[key]
        old_cyc = int(r.get("cycles") or 0)
        r["mean_time"] = f"{cyc/1e6:.6f}"
        r["mean_unit"] = "ms"
        r["mean_time_ns"] = f"{cyc:.6f}"
        r["cycles"] = str(cyc)
        r["source"] = "firesim"
        n += 1
        deltas.append((did, name, op, old_cyc, cyc))
    with open(pdb_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return n, deltas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runlog", required=True, type=Path)
    ap.add_argument("--tag", default="v20b",
                    help="suffix used for the PDB backup files (.csv.pre<tag>_backup)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.runlog.is_file():
        print(f"runlog not found: {args.runlog}", file=sys.stderr)
        return 1

    print(f"[G3] parsing {args.runlog}")
    per_net = parse_runlog(args.runlog.read_text())
    for net, entries in per_net.items():
        print(f"  {net}: {len(entries)} measured (backend, did) entries")

    grand_total_updated = 0
    grand_total_delta_cyc = 0
    for net in NETS:
        if net not in per_net:
            print(f"[G3] WARNING: {net} not in runlog, skipping")
            continue
        measured = per_net[net]
        for backend in ("gemmini_q31", "rvv_opu"):
            backend_measured = {k: v for k, v in measured.items() if k[0] == backend}
            if not backend_measured:
                continue
            pdb = pdb_path(net, backend)
            if not pdb.is_file():
                print(f"[G3] WARNING: PDB missing for {net}/{backend}: {pdb}")
                continue
            if args.dry_run:
                rows = list(csv.DictReader(open(pdb)))
                count = sum(1 for r in rows if (backend, int(r["dispatch_id"])) in backend_measured)
                print(f"[G3] DRY {net}/{backend}: would update {count} of {len(backend_measured)} rows")
                continue
            n, deltas = update_pdb(pdb, backend_measured, backend, args.tag)
            grand_total_updated += n
            sum_old = sum(d[3] for d in deltas)
            sum_new = sum(d[4] for d in deltas)
            grand_total_delta_cyc += sum_new - sum_old
            ratio = sum_new / sum_old if sum_old else float("nan")
            print(f"[G3] {net}/{backend}: updated {n} rows  "
                  f"sum_cyc {sum_old/1e6:.2f}M → {sum_new/1e6:.2f}M  ratio {ratio:.3f}")

    print(f"[G3] grand total: updated {grand_total_updated} rows, "
          f"net cycle delta {grand_total_delta_cyc/1e6:+.2f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
