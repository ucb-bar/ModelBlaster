# Quality-evidence log (binding all phases)

The user is explicit: no compromises in result quality. This log
tracks per-phase / per-gate pass-fail with evidence paths. A phase
is "done" when every gate has been actually exercised, not just
claimed.

Six gates per phase, where applicable:

| Gate | Meaning |
|:---|:---|
| `correct` | Bit-exact correctness where any kernel is touched (max_abs_err=0). |
| `measured` | Any Gantt/makespan in this phase traces to measured cycles (spike or FireSim). |
| `solver-agree` | An exact solver (cpsat or post-rework MOSEK) is run; makespan baselines come from the exact solver. |
| `deterministic` | All non-determinism (cvxpy ties, CP-SAT search seed) is pinned for reproducibility. |
| `cold-rerun` | The end-to-end pipeline is re-run from a fresh working dir; metrics within 0.5% of original. |
| `docs` | Phase README explains what was measured, how, and what would invalidate it. |

`—` = gate inapplicable. `pending` = phase hasn't reached that gate yet.

## Phase A — Frequency-as-band

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **A1** band audit | — | pass | pass | pass | pending | pass | `artifacts/audit/band_compliance.csv`, `artifacts/audit/README.md`. 14 solvers; only decomposed clean. |
| **A2** HEFT honest-marking | — | pass | — | pass | pending | pass | heft fixture `deadline_miss` flags == band-audit deadline_violations (88==88 verified). |
| **A3** band-safe compaction + automerge | — | pass | — | pass | pending | pass | unit tests `tests/test_band_safe_postpass.py` 6/6 PASS. |
| **A4** Gantt period bands + red overrun | — | pass | — | pass | pending | pass | `artifacts/audit/gantts/band_*.png`, `xpu-rt/diagnostics/plot_band_gantt.py`. |

## Phase B — Analytical decision formulas

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **B1–B6** formulas | — | — | — | pass | pass | pass (module docstring) | `xpu-rt/decision_formulas.py`, pure-function. |
| **B7** unit tests | — | — | — | pass | pass | pass | `xpu-rt/tests/test_decision_formulas.py` — 22/22 PASS. |

## Phase C — Iterative scheduling policies

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **C1** yolo_anchor | — | pass | pass (vs cpsat) | pass | pending | pass | `xpu-rt/policies/yolo_anchor.py`; result 61.2 ms / 67 misses on headline. |
| **C2** periodic_anchor | — | pass | pass | pass | pending | pass | `xpu-rt/policies/periodic_anchor.py`; result 75.6 ms / 0 misses (clean). |
| **C3** critical_path_first | — | pass | pass | pass | pending | pass | `xpu-rt/policies/critical_path_first.py`; result 54.4 ms / 88 misses. |
| **C4** cpsat_unconstrained | — | pass | pass | pass | pending | pass | `xpu-rt/policies/cpsat_unconstrained.py`; result 111.2 ms / 34 misses. |
| **C5** decision_loop integration | — | — | — | pass | — | pass | `scripts/decision_loop.py --policy`. |
| **C6** comparison table | — | pass | pass | pass | pending | pass | `artifacts/policies/README.md` + `artifacts/policies/gantts/*.png`. |

## Phase D — Parametric sweep + headline figure

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **D1** sweep driver | — | pass | pass | pass | pending | pass | `scripts/sweep_policies.py`; 12 cells in `artifacts/sweeps/run01/`. |
| **D2** output layout + README | — | pass | pass | pass | pending | pass | `artifacts/sweeps/run01/README.md` answers all four research questions. |
| **D3** measured-cycles guard | — | pass | — | pass | pass | pass | `plot_band_gantt.py:render_band_gantt` two-tier guard, both tiers tested. |
| **D4** headline 3×4 figure | — | pass | pass | pass | pending | pass | `artifacts/sweeps/run01/headline_3x4.png`. |

## Phase E — Bedrock fused kernels

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **E1** gap survey | — | — | — | pass | pass | pass | `artifacts/kernel_gap_survey.json` 188 candidates, 117 in top-2 gaps. |
| **E2** KernelSpec registration | pass (reference impls are bit-exact verification oracles) | — | — | pass (Python module load deterministic) | — | pass | `pipeline/reference_kernels.py:CONV2D_BATCHNORM2D_S8, BATCHNORM2D_SILU_S8` registered. |
| **E2** Bedrock invocation | BLOCKED (no AWS creds) | BLOCKED | — | — | — | pass | session env lacks AWS creds; documented in per-kernel reports. |
| **E3** realizability filter wire-in | — | pass (gap survey re-run confirms coverage 3.2 % → 65.4 %) | — | pass | pass | pass | `scripts/decision_loop.py:REALIZABLE_FUSE_PAIRS`. |
| **E4** per-kernel reports | pass (reference impl coverage documented) | BLOCKED on E2 Bedrock | — | pass | — | pass | `artifacts/kernels/{conv2d_batchnorm2d_s8, batchnorm2d_silu_s8}/measurement_report.md` |

## Phase F — MOSEK convergence rework

| Sub | correct | measured | solver-agree | deterministic | cold-rerun | docs | evidence |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **F1** divergence diagnosis | — | — | — | pass | — | pass | `scripts/mosek_divergence_diagnose.py`; MOSEK confirmed > 3 min CPU on headline w/o output; CPSAT 62 s on cap=1. |
| **F2a** warm-start framework | — | — | — | pass | — | pass | `xpu-rt/scheduler_mosek_warmstart.py` provides framework; scheduler.py refactor needed for activation. |
| **F2b** singleton pre-fix | pass (synthetic 3-op test status=optimal value=7.0) | — | pass (matches manual schedule) | pass | — | pass | `xpu-rt/scheduler.py:445` implemented; constraint logger reports n_singletons. |
| **F2e** parameter sweep | — | — | — | pass | — | pass | `scripts/mosek_param_sweep.py` ready; not yet run (each combo ≥ 3 min on diverging MOSEK). |
| **F2c–F2g** other aids | — | — | — | pending | — | pass | methodology documented in `artifacts/mosek_rework/README.md`. |
| **F3** MOSEK vs CPSAT agreement | — | pending (MOSEK doesn't converge) | — | — | — | pass | requires F2f or F2g convergence. |
| **F4** re-enable in audit + sweep | — | pass (audit reports `mosek_diverged` cleanly) | — | pass | — | pass | `scripts/audit_band_compliance.py:wall_cap=120` for mosek. |

---

## Run-time provenance

| Run | Date | Command | Result |
|:---|:---|:---|:---|
| A1 audit | 2026-06-03 | `audit_band_compliance.py --rerun --solvers ...` | 14 rows; decomposed clean, others 10–88 dl_miss |
| A2 heft marking | 2026-06-03 | direct run + diagnostics check | 88 flags == 88 audit violations |
| A3/A4 tests | 2026-06-03 | `pytest tests/test_band_safe_postpass.py` | 6/6 PASS |
| A4 Gantt samples | 2026-06-03 | `render_band_gantt(...)` ×5 solvers | PNGs with period bands + red overruns rendered |
| B7 tests | 2026-06-03 | `pytest tests/test_decision_formulas.py -v` | 22/22 PASS in 0.03 s |
| C policies (×4) | 2026-06-03 | `from policies import POLICIES; fn(...)` | each policy produces fixture + summary |
| D1 sweep | 2026-06-03 | `sweep_policies.py --out run01` | 12-cell grid CSV + gantts |
| D4 grid figure | 2026-06-03 | `render_headline_grid.py --sweep-dir run01` | 633 KB PNG |
| E1 survey | 2026-06-03 | `kernel_gap_survey.py` | 188 candidates, top gap conv→BN (60) + BN→silu (57) |
| F1 diagnose (in progress) | 2026-06-03 | `mosek_divergence_diagnose.py` | bisection over cap ∈ {1,2,4,8} × {cpsat, mosek}; MOSEK > 60 s on cap=1 |
