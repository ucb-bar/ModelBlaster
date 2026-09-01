# Phase D — Sweep results (run01)

## Headline figure

`headline_3x4.png` — 3 frequency configs × 4 policies on the
4 MLP + 2 Dronet + 1 Yolo workload (hetero bitstream).

## Per-cell metrics (`grid_headline.csv`)

| Config | Policy | Makespan (ms) | DL misses | Dispatches | Solve wall (s) |
|:---|:---|---:|---:|---:|---:|
| canon (MLP@10, Dr@20) | yolo_anchor | 61.20 | 67 | 236 | 5.45 |
| canon | periodic_anchor | 75.57 | **0** | 248 | 2.97 |
| canon | critical_path_first | 54.43 | 88 | 245 | 1.77 |
| canon | cpsat_unconstrained | 111.17 | 34 | 230 | 62.05 |
| tight_mlp (MLP@5, Dr@20) | yolo_anchor | 65.23 | 82 | 250 | 9.62 |
| tight_mlp | periodic_anchor | 108.50 | **0** | 279 | 6.67 |
| tight_mlp | critical_path_first | 54.43 | 88 | 245 | 1.77 |
| tight_mlp | cpsat_unconstrained | 122.45 | 51 | 222 | 62.04 |
| slack_dronet (MLP@10, Dr@33) | yolo_anchor | 56.54 | 28 | 213 | 3.57 |
| slack_dronet | periodic_anchor | 92.59 | **0** | 213 | 3.03 |
| slack_dronet | critical_path_first | 54.43 | 58 | 245 | 1.76 |
| slack_dronet | cpsat_unconstrained | 117.63 | 32 | 224 | 62.05 |
| canon | **hybrid_periodic_mosek_yolo** | **70.00** | **0** | 240 | 228.92 |

## Correction (2026-06-05) — the 70 ms number is stale

The fixture below was generated against an older snapshot of the
FireSim PDB (the `.prebitexact_backup` set, predating Phase E's
bit-exact kernel re-ingestion). End-to-end FireSim validation
(`artifacts/firesim_runs/hybrid_v8/` and
`artifacts/runtime_optimization/v9_baseline_instrumented/`) measured
786 ms worst-hart wall on this same fixture. Re-running the policy's
Phase-1 decomposed solver against the CURRENT PDB predicts 782 ms —
agreement with measurement within 0.5%. The "9× gap" between
predicted 70 ms and measured 786 ms was a stale-PDB artifact, not a
solver-vs-runtime model gap. See
`artifacts/runtime_optimization/v10_calibrated_schedule/` for the
re-cost and re-solve evidence. The 70 ms row below is preserved for
provenance; the honest predicted-vs-measured headline is in the v10
artifacts.

## Headline answer (provenance — see correction above)

The `hybrid_periodic_mosek_yolo` policy is Pareto-best on the canonical
workload: **70 ms makespan with 0 deadline misses across all 240
dispatches**. Phase 1 reserves periodic instances via the `decomposed`
solver (3.0 s), phase 2 schedules yolov8 against the remainder via
MOSEK (225.9 s), phase 3 shifts yolov8 to avoid periodic-busy intervals
(no shift needed on this cell). Source: `artifacts/policies/
headline_hybrid_result.json`; Gantt:
`gantts/canon__hybrid_periodic_mosek_yolo.png`. End-to-end FireSim
validation pending — see `task #245`.

## Findings (answer to the user's research questions)

### 1. Which solvers respect frequency bands?

**Only the `decomposed` solver, via the `periodic_anchor` policy.**
Across all three frequency configurations, it returns 0/213-279
deadline misses. No other policy (yolo_anchor → greedy_periodic,
critical_path_first → heft, cpsat_unconstrained → cpsat) holds the
band invariant on any cell.

### 2. When does the policy choice change makespan?

- `critical_path_first` always produces makespan ≈ 54.4 ms, regardless
  of frequency config — the HEFT heuristic is dominated by the
  aperiodic critical path of yolov8 and is essentially insensitive to
  MLP/Dronet periods. But it pays for that with 58–88 deadline misses
  per cell.
- `periodic_anchor` makespan SCALES with periodicity: 75.57 ms at canon
  → 108.50 ms when MLP period tightens to 5 ms. That's the cost of
  honoring tighter bands: 4 MLP instances fit into a tighter window,
  forcing the aperiodic Yolo tail to extend.
- `slack_dronet` (dronet period 33 ms instead of 20 ms) helps
  `yolo_anchor` significantly: makespan drops 61.2 → 56.5 ms and
  deadline misses drop 67 → 28. Slackening one network's period
  reduces its scheduling pressure on the other lane.
- `cpsat_unconstrained` is the global-makespan optimizer in principle
  but in practice gets time-limited at this problem size (300 ops) and
  produces sub-optimal results (111–122 ms across cells). This is the
  Phase F1 motivation for the MOSEK reformulation.

### 3. What's the deadline-miss vs makespan trade-off?

Plotting the cells in (makespan, miss) space:

- Pareto frontier: `periodic_anchor` (miss=0, mksp=75-108) vs
  `critical_path_first` (miss=58-88, mksp=54.4).
- `yolo_anchor` and `cpsat_unconstrained` sit interior to this frontier
  — neither is Pareto-optimal on this workload.

### 4. Does compaction help, and when?

Compaction post-pass is applied by `_wrap_with_compaction` (Phase A3
band-safe). Effect on this run: the periodic_anchor (decomposed)
fixtures already produce tight schedules — compaction is essentially
a no-op there. On heft / cpsat fixtures it shifts ops earlier but
cannot move periodic instances past their release times. The
band-safety patch from Phase A3 ensures compaction never inadvertently
pushes a downstream op past a deadline.

## Honesty notes

- Every bar in every Gantt traces to the profile DB loaded by
  `run_xpurt_schedule.py --use-profiled`. No bookkeeping fictions.
- All 12 cells reused the same `firesim_rocket_saturn` profile (the
  hetero bitstream the user-confirmed scope is locked to).
- `cpsat_unconstrained` used a 60-s wall-clock budget per cell. It did
  not converge to optimum; the reported makespan is the best-known
  feasible at that budget. The Phase F1 binary-search diagnostic will
  characterize whether CPSAT can converge given more time, or whether
  the workload at this op count is genuinely intractable for the
  current CPSAT formulation.
- Cold-rerun gate (Phase Q-rerun) NOT yet executed — these numbers
  come from the first sweep. Phase Q's cold re-run will produce
  numbers within 0.5% or surface a reproducibility bug.

## Open follow-ups

- Aux mix-ablation (36 cells: MLP ∈ {2,4,8} × Dronet ∈ {1,2,4} ×
  4 policies at canonical frequency) — `sweep_policies.py
  --mix-ablation` ready to run; deferred for time.
- Phase F1 — diagnose why CPSAT can't honor `max_end_t` on this
  workload (it should be a hard constraint).
- Phase E2 — Bedrock kernels for the 117 conv→BN→silu fusion gap
  exposed by E1.
- Phase C5 — wire policies into the decision-loop driver so per-policy
  decision loops can produce comparable rounds.
