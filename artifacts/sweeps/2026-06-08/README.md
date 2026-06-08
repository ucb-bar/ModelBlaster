# Phase D — Parametric sweep (2026-06-08)

12-cell grid: 3 frequency configs × 4 iterative policies on the
canonical workload **4 MLP + 2 Dronet + 1 Yolo** on the hetero
bitstream (Gemmini + Saturn-OPU). All solver predictions consume the
**G3-calibrated PDB** (v20b measured cycles, landed into `sweep_v8`
on 2026-06-08).

## Grid axes

| Axis           | Values                                                |
|:---------------|:------------------------------------------------------|
| Mix            | 4 MLP + 2 Dronet + 1 Yolo (fixed)                     |
| Frequency      | p10_20_100, p5_20_100, p10_33_200 (ms)                |
| Policy         | yolo_anchor, periodic_anchor, critical_path_first,    |
|                | cpsat_unconstrained                                   |

## Outputs

- `grid.csv` — one row per cell with `makespan_us`, `n_deadline_miss`,
  `n_release_viol`, `n_shards_applied`, `n_fuses_applied`,
  `n_dispatches`, `solve_wall_s`.
- `cells/<cell>/workload.json` — the materialised workload JSON for
  that frequency config.
- `cells/<cell>/fixture.json` — the solver output fixture.
- `cells/<cell>/gantt.png` — predicted Gantt rendered with
  `xpu-rt/plot_gantt.py::render_fixture_gantt`.
- `cells/<cell>/policy_result.json` — full policy return value.
- `composite.png` — 3×4 composite figure assembled by
  `scripts/render_sweep_composite.py`.

## Driver

`/scratch2/agustin/XPU-RT/scripts/sweep_workloads.py` (added by this
phase). Invocation:

```bash
cd /scratch2/agustin/XPU-RT
.venv/bin/python scripts/sweep_workloads.py \
    --base-workload data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json \
    --out /scratch2/agustin/ModelBlaster/artifacts/sweeps/2026-06-08 \
    --policies yolo_anchor,periodic_anchor,critical_path_first,cpsat_unconstrained \
    --time-limit 90 --freq-configs 3
```

## Honest-cycles invariant

Per the plan's Phase D3 gate, every Gantt bar duration in this sweep
traces to a measured-cycle entry in the calibrated PDB. The G3
recalibration (2026-06-08) ingested 235 rows from v20b's FireSim
run.log into sweep_v8 — see
`artifacts/runtime_optimization/v23_pdb_recalibrated/README.md`.

## What this sweep does NOT do

- It does NOT requeue FireSim per cell. The FireSim queue was busy
  with another agustin process during this window; for each cell we
  used the PDB-driven *predicted* makespan plus a predicted Gantt.
  Re-running each cell on real hardware is a follow-up
  (`g4_followup_firesim_per_cell`).
- It does NOT include MOSEK as a separate policy column; MOSEK is
  reachable via `hybrid_periodic_mosek_yolo` and `mosek_decomposed`
  in `policies/`, but on the 388-op composite workload at 2026-06-08
  MOSEK still hits the divergence diagnosed under Phase F (tracked
  separately).
- It does NOT cover the auxiliary 36-cell mix ablation grid
  (MLP∈{2,4,8} × Dronet∈{1,2,4}) — that grid is a single line in
  the next sweep run (4-policy × 9 mix points = 36 cells); the
  headline 3×4 was prioritised first.

## Cross-reference

- Phase A1 band invariant audit (14 solvers) at
  `artifacts/runtime_optimization/v23_pdb_recalibrated/band_compliance.csv`.
  Only `decomposed` achieves 0 deadline misses on the canonical cell;
  every heuristic in this sweep overshoots some deadline.
- Phase G3 PDB recalibration at
  `artifacts/runtime_optimization/v23_pdb_recalibrated/README.md`.
- v20b kernel-optimisation closeout at
  `artifacts/runtime_optimization/FINAL_COMPARISON.md`.
