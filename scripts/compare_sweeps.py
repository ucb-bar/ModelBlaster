#!/usr/bin/env python3
"""Q-rerun gate: compare two sweep runs row-by-row, flag any cell whose
metrics drift > 0.5%."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--tol", type=float, default=0.005,
                    help="Relative drift tolerance (default 0.5%)")
    args = ap.parse_args()

    def _load(p):
        with open(Path(p) / "grid_headline.csv") as f:
            return list(csv.DictReader(f))

    a = _load(args.run_a)
    b = _load(args.run_b)
    a_by_id = {r["cell_id"]: r for r in a}
    b_by_id = {r["cell_id"]: r for r in b}

    if set(a_by_id) != set(b_by_id):
        print("MISMATCH: cell sets differ")
        print("  only in A:", set(a_by_id) - set(b_by_id))
        print("  only in B:", set(b_by_id) - set(a_by_id))
        return 1

    n_pass = 0
    n_fail = 0
    drift_rows = []
    for cell_id, ra in sorted(a_by_id.items()):
        rb = b_by_id[cell_id]
        worst_drift = 0.0
        worst_field = ""
        for field in ("makespan_ms", "n_deadline_miss"):
            va = float(ra[field]) if ra[field] not in ("", "None") else None
            vb = float(rb[field]) if rb[field] not in ("", "None") else None
            if va is None or vb is None:
                continue
            if va == 0 and vb == 0:
                continue
            denom = max(abs(va), abs(vb), 1e-9)
            drift = abs(va - vb) / denom
            if drift > worst_drift:
                worst_drift = drift; worst_field = field
        ok = worst_drift <= args.tol
        status = "OK " if ok else "FAIL"
        print(f"{status}  {cell_id:<40s}  worst_drift={worst_drift*100:.3f}%  ({worst_field})")
        if ok:
            n_pass += 1
        else:
            n_fail += 1
            drift_rows.append((cell_id, worst_field, worst_drift))

    print()
    print(f"PASS: {n_pass}/{n_pass+n_fail}  FAIL: {n_fail}")
    if n_fail:
        print()
        print("Drift detail (rerun for these cells before declaring done):")
        for cell, fld, dr in drift_rows:
            print(f"  {cell}  {fld}  drift={dr*100:.2f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
