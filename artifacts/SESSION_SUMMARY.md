# Session summary: frequency-as-band + decision formulas + iterative policies + sweep + diagnostics

**Date:** 2026-06-03.
**Plan:** `/home/agustin/.claude/plans/buzzing-wiggling-pretzel.md`
(updated this session to reflect F2 expansion into six convergence-aid
tracks per user request).

## Headline answer to the user's research questions

### Q1. Which solvers / policies respect frequency bands?

**Only `decomposed` (via `periodic_anchor` policy).** Across 14
solvers and 12 sweep cells (3 frequency configs × 4 policies on
4 MLP + 2 Dronet + 1 Yolo), this is the only one that returns 0
deadline misses. Evidence:

- `artifacts/audit/band_compliance.csv` — solver-level audit, 13 of 14
  miss bands.
- `artifacts/sweeps/run01/grid_headline.csv` — policy-level sweep,
  periodic_anchor is the only band-clean column.
- Band-aware Gantts in `artifacts/audit/gantts/` and
  `artifacts/sweeps/run01/gantts/` make the overruns visible as red
  rectangles.

### Q2. When does sharding help / hurt — with a formula?

`xpu-rt/decision_formulas.py:shard_benefit` gives the closed form:

```
optimal_fraction f* = c_home / (c_home + c_alt)
optimal_finish     = c_home * c_alt / (c_home + c_alt)     # harmonic mean
contention adj.    = max(0, alt_soonest_free - op_ready)
```

When contention exceeds breakeven (`f* ≤ 0`), the shard hurts and
the formula returns `expected_delta = 0` with reason `"alt contended
past breakeven"`. The agent calls this BEFORE invoking a solver — no
solver round wasted on a candidate that the formula says is dead on
arrival. 22 unit tests in `tests/test_decision_formulas.py` exercise
adversarial cases.

### Q3. Which iterative policy wins on which workload class?

The 12-cell sweep (`artifacts/sweeps/run01/`) gives the Pareto frontier:

| Frontier point | Policy | Makespan | DL misses |
|:---|:---|---:|---:|
| Best deadline compliance | periodic_anchor | 75.6 ms | 0 |
| Best makespan | critical_path_first | 54.4 ms | 88 |
| Interior (no Pareto win) | yolo_anchor | 61.2 ms | 67 |
| Interior + slow | cpsat_unconstrained | 111.2 ms | 34 |

`yolo_anchor` and `cpsat_unconstrained` are NOT Pareto-optimal on
this workload — they trade poorly on both axes. The structural
choice is between "honor deadlines" (decomposed/periodic_anchor) and
"squeeze makespan" (HEFT/critical_path_first). Frequency variations
show that:

- Tightening MLP period (5 ms) pushes periodic_anchor from 75.6 →
  108.5 ms makespan but preserves 0 deadline misses.
- Slackening dronet period (33 ms) helps yolo_anchor (61.2 → 56.5 ms,
  misses 67 → 28).

### Q4. Does compaction help, and when?

Compaction is now band-safe (Phase A3 patch): a compaction shift
cannot push an op past its `max_end_t`. On this workload it acts
purely as a clean-up — exact solvers (cpsat) already produce tight
schedules, list schedulers (heft/peft/edf) leave small slack that
compaction removes. The B5 formula `compaction_eligible` exposes
this per-op: when `gap > 0 AND band_safe AND downstream_safe`, the
shift is applicable.

## Deliverables shipped

### Phase A — Frequency-as-band
- A1: `xpu-rt/diagnostics/band_invariant.py` + `scripts/audit_band_compliance.py`
- A2: HEFT-family `deadline_miss` honest-marking + fixture propagation
- A3: band-safe `compaction.py` + instance-aware `automerge.py` +
  tests/test_band_safe_postpass.py (6/6 PASS)
- A4: band-aware Gantt with period overlays + red overruns
  (`xpu-rt/diagnostics/plot_band_gantt.py`)

### Phase B — Analytical decision formulas
- `xpu-rt/decision_formulas.py`: 6 closed-form formulas (B1–B6)
- `tests/test_decision_formulas.py`: 22 unit tests, all PASS

### Phase C — Iterative scheduling policies
- 4 policies under `xpu-rt/policies/`: yolo_anchor, periodic_anchor,
  critical_path_first, cpsat_unconstrained
- `scripts/decision_loop.py --policy` integrates policies into the
  measurement-grounded loop

### Phase D — Sweep + headline figure
- `scripts/sweep_policies.py`: 12-cell driver
- `artifacts/sweeps/run01/`: grid CSV + per-cell band Gantts +
  3×4 comparison figure (`headline_3x4.png`)
- Phase D3 measured-cycles guard built into the band Gantt renderer
- Phase Q-rerun: cold sweep `artifacts/sweeps/run02_cold/` in progress

### Phase E — Bedrock fused-kernel expansion
- E1: `scripts/kernel_gap_survey.py` + `artifacts/kernel_gap_survey.json`
  — 188 candidate fuse pairs, top gaps `conv2d→batchnorm` (60) and
  `batchnorm→silu` (57). 117 candidates concentrated in the top 2
  gaps. Existing fused kernels: `conv2d_silu_s8`, `linear_s8_elu_s8`.
- E2/E3/E4: framework documented in `artifacts/kernels/README.md`;
  actual Bedrock generation is budget-gated and deferred to a
  next-budget-approved run.

### Phase F — MOSEK convergence rework
- F1: MOSEK confirmed structurally divergent on this workload at any
  practical scale. CPSAT 62 s vs MOSEK > 180 s on cap=1 (~100 ops).
  Diagnosis: cvxpy canonicalization + MOSEK presolve dominate
  wall-clock, neither bounded by `MSK_DPAR_OPTIMIZER_MAX_TIME`.
- F2: 6 convergence-aid tracks documented in
  `artifacts/mosek_rework/README.md`:
  - F2a warm-start (framework module written)
  - F2b variable elimination (pre-fix singletons)
  - F2c symmetry-breaking constraints
  - F2d coarse-fine time discretization
  - F2e solver-parameter sweep (script ready)
  - F2f time-indexed reformulation
  - F2g Lagrangian decomposition by network
- F3/F4 (re-enable in audit): audit now records MOSEK as
  `mosek_diverged` cleanly, capping wall at 120 s instead of hanging.

### Quality
- 28 unit tests pass (`pytest`).
- All Gantts use measured profile data (no bookkeeping fictions).
- Per-phase evidence trail in `artifacts/quality_log.md`.

## Honest gaps

1. **Phase E2 Bedrock kernels** — gap identified (117 candidate
   conv→BN→silu pairs); generation deferred to a budget-approved run.
   Triple-fused `conv2d_s8_bn_silu_s8` is the highest-value target.
2. **Phase F2 actual implementation** — F1 diagnosed structural
   divergence and prioritized F2b/F2f/F2g. Each requires a
   scheduler.py refactor of varying depth; F2a's framework module is
   in place but activation needs the refactor too. Documented and
   tracked.
3. **Decision-loop policy seeding** — Phase B formulas are unit-tested
   but not yet wired as candidate seeds inside `scripts/decision_loop.py`.
   The hook (`--policy`) is in; the seed pass is the next iteration.

## Files added / modified this session

XPU-RT:
- xpu-rt/diagnostics/{__init__.py,band_invariant.py,plot_band_gantt.py}
- xpu-rt/decision_formulas.py
- xpu-rt/policies/{__init__.py,_common.py,yolo_anchor.py,periodic_anchor.py,critical_path_first.py,cpsat_unconstrained.py}
- xpu-rt/scheduler_mosek_warmstart.py
- xpu-rt/tests/test_decision_formulas.py
- xpu-rt/tests/test_band_safe_postpass.py
- scripts/audit_band_compliance.py
- scripts/mosek_divergence_diagnose.py
- scripts/mosek_param_sweep.py
- (modified) xpu-rt/{scheduler_heft.py, compaction.py, automerge.py, postprocessing.py}

ModelBlaster:
- scripts/sweep_policies.py
- scripts/render_headline_grid.py
- scripts/kernel_gap_survey.py
- (modified) scripts/decision_loop.py
- artifacts/{audit, policies, kernels, mosek_rework, sweeps/run01}/ with
  per-phase READMEs, CSVs, Gantts, the 3×4 grid figure

## Next session entry point

If a follow-up session continues this work, the natural starting
points are:

- **E2** — generate the `conv2d_s8_bn_silu_s8` kernel via Bedrock.
  117 candidates close on success. Budget approval needed.
- **F2b** — pre-fix singleton-feasible binary vars. Reduces MOSEK
  canonicalization size. Operational, no algorithm work.
- **F2f / F2g** — Lagrangian decomposition is the most-promising
  structural answer to F1's diagnosis.
- **Cold-rerun verification** — once `artifacts/sweeps/run02_cold/`
  completes, diff against `run01/` to verify reproducibility.
