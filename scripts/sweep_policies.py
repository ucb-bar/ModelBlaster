#!/usr/bin/env python3
"""Phase D — headline-focused policy sweep.

Fixed 4 MLP + 2 Dronet + 1 Yolo mix on hetero, across:
  - 3 frequency configurations (canon 10/20, tight MLP 5/20, slack
    Dronet 10/33)
  - 4 policies (C1-C4)

= 12 cells. Each cell:
  1. frequency_feasibility precheck (Phase B1).
  2. Run policy.
  3. Render band-aware Gantt with red overruns.
  4. Append summary row to grid.csv.

Auxiliary mix ablation (no Gantt, summary only): MLP ∈ {2,4,8} ×
Dronet ∈ {1,2,4} × Yolo=1 at canonical frequency × 4 policies = 36
cells. Used to build a makespan-vs-mix line plot.

Usage:
    python scripts/sweep_policies.py \
        --out artifacts/sweeps/<date>/ \
        [--mix-ablation]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_XPURT = Path("/scratch2/agustin/XPU-RT")
sys.path.insert(0, str(_XPURT))
sys.path.insert(0, str(_XPURT / "xpu-rt"))


HEADLINE_CONFIGS = [
    # (label, mlp_period_ms, dronet_period_ms)
    ("canon",        10, 20),
    ("tight_mlp",     5, 20),
    ("slack_dronet", 10, 33),
]

HEADLINE_POLICIES = [
    "yolo_anchor",
    "periodic_anchor",
    "critical_path_first",
    "cpsat_unconstrained",
]


def headline_cells():
    return [(cfg_label, mlp, dr, policy)
            for cfg_label, mlp, dr in HEADLINE_CONFIGS
            for policy in HEADLINE_POLICIES]


def build_workload_variant(base_path: str, mlp_period: int,
                            dronet_period: int, label: str,
                            mlp_count: int = 4, dronet_count: int = 2,
                            ) -> str:
    """Materialize a workload JSON variant for the sweep cell."""
    import copy
    base = json.loads(Path(base_path).read_text())
    nets = base["networks"]
    if "mlp_control" in nets:
        nets["mlp_control"]["period"] = mlp_period
        nets["mlp_control"]["window_duration"] = mlp_period
        nets["mlp_control"]["num_instances"] = mlp_count
    if "dronet" in nets:
        nets["dronet"]["period"] = dronet_period
        nets["dronet"]["window_duration"] = dronet_period
        nets["dronet"]["num_instances"] = dronet_count
    out_path = _XPURT / "data" / "toplevel" / f"sweep_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, indent=2))
    return str(out_path)


def feasibility_check(workload_data, mlp_period, dronet_period) -> dict:
    """Lightweight B1 application. We don't have per-op profile data
    handy in this driver (it lives in the profile DB which the workload
    factory loads). As a proxy: compare the previously-observed
    decomposed makespan per network in the canonical run vs the new
    period. Returns a status string. The honest path long-term is to
    plumb profile_loader output here; for now we record the bookkeeping
    decision (feasible_estimated_from_canonical / unknown) so the cell
    isn't silently skipped.
    """
    return {
        "checked": False,
        "reason": "profile-DB driven feasibility deferred; "
                  "policy results provide the actual feasibility signal",
    }


def run_cell(policy_name, workload_path, out_dir):
    from policies import POLICIES
    fn = POLICIES[policy_name]
    workload_data = json.loads(Path(workload_path).read_text())
    return fn(workload_path, workload_data=workload_data, time_limit=60.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parents[1] /
                                "artifacts" / "sweeps" /
                                f"{time.strftime('%Y%m%d_%H%M%S')}"))
    ap.add_argument("--base-workload",
                    default=str(_XPURT / "data" / "toplevel" /
                                "networks_1yolo_4mlp_2dronet_firesim.json"))
    ap.add_argument("--mix-ablation", action="store_true",
                    help="Also run the 36-cell mix-ablation table")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    cells_dir = out_root / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    gantts_dir = out_root / "gantts"
    gantts_dir.mkdir(parents=True, exist_ok=True)

    from diagnostics import render_band_gantt

    # ---- HEADLINE: 12 cells ----
    rows = []
    for cfg_label, mlp_p, dr_p, policy_name in headline_cells():
        cell_id = f"{cfg_label}__{policy_name}"
        print(f"\n[cell] {cell_id} (MLP@{mlp_p}ms Dronet@{dr_p}ms)")
        wl_path = build_workload_variant(args.base_workload, mlp_p, dr_p,
                                          f"4m2d1y_{cfg_label}")
        workload_data = json.loads(Path(wl_path).read_text())

        feas = feasibility_check(workload_data, mlp_p, dr_p)

        t0 = time.perf_counter()
        result = run_cell(policy_name, wl_path, cells_dir / cell_id)
        wall = time.perf_counter() - t0

        if result.get("fixture_path"):
            # Copy / link fixture into cell dir for reproducibility.
            cell_dir = cells_dir / cell_id
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "fixture.json").write_text(
                Path(result["fixture_path"]).read_text()
            )
            (cell_dir / "policy_log.json").write_text(
                json.dumps(result, indent=2, default=str)
            )
            # Band Gantt.
            fixture = json.loads(Path(result["fixture_path"]).read_text())
            gantt_path = gantts_dir / f"{cell_id}.png"
            summary = render_band_gantt(
                fixture, workload_data, str(gantt_path),
                solver=policy_name,
                title=f"{policy_name}  MLP@{mlp_p}  Dronet@{dr_p}",
            )
        else:
            summary = {"n_dispatches": None, "makespan": None,
                       "n_release_violations": None,
                       "n_deadline_violations": None}

        rows.append({
            "cell_id": cell_id,
            "policy": policy_name,
            "mlp_period_ms": mlp_p,
            "dronet_period_ms": dr_p,
            "mlp_count": 4,
            "dronet_count": 2,
            "yolo_count": 1,
            "status": result.get("status", "?"),
            "makespan_ms": result.get("makespan"),
            "n_deadline_miss": result.get("n_deadline_miss"),
            "n_release_viol": result.get("n_release_viol"),
            "n_dispatches": result.get("n_dispatches"),
            "solve_wall_s": result.get("solve_wall_s"),
            "feasibility_checked": feas.get("checked"),
            "feasibility_note": feas.get("reason"),
        })

    # Write headline CSV.
    csv_path = out_root / "grid_headline.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nHeadline CSV -> {csv_path}")

    # ---- Aux mix-ablation: 36 cells (skip Gantts) ----
    if args.mix_ablation:
        aux_rows = []
        for mlp_c in (2, 4, 8):
            for dr_c in (1, 2, 4):
                for policy_name in HEADLINE_POLICIES:
                    cell_id = f"aux_m{mlp_c}_d{dr_c}__{policy_name}"
                    print(f"\n[aux cell] {cell_id}")
                    wl_path = build_workload_variant(
                        args.base_workload, 10, 20,
                        f"aux_m{mlp_c}_d{dr_c}",
                        mlp_count=mlp_c, dronet_count=dr_c,
                    )
                    workload_data = json.loads(Path(wl_path).read_text())
                    result = run_cell(policy_name, wl_path, cells_dir / cell_id)
                    aux_rows.append({
                        "cell_id": cell_id,
                        "policy": policy_name,
                        "mlp_count": mlp_c,
                        "dronet_count": dr_c,
                        "yolo_count": 1,
                        "mlp_period_ms": 10,
                        "dronet_period_ms": 20,
                        "status": result.get("status"),
                        "makespan_ms": result.get("makespan"),
                        "n_deadline_miss": result.get("n_deadline_miss"),
                        "n_dispatches": result.get("n_dispatches"),
                        "solve_wall_s": result.get("solve_wall_s"),
                    })
        aux_csv = out_root / "grid_aux_mix.csv"
        with aux_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(aux_rows[0].keys()))
            w.writeheader()
            w.writerows(aux_rows)
        print(f"\nAux mix CSV -> {aux_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
