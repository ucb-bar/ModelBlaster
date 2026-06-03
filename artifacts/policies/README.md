# Phase C — Iterative scheduling policies (headline workload comparison)

## Workload

- 4 × mlp_control @ 10 ms period × 10 ms window
- 2 × dronet     @ 20 ms period × 20 ms window
- 1 × yolov8_nano (aperiodic, 1 instance)
- Bitstream: hetero (CPU_P→gemmini_q31, CPU_E→V256D128_rvv on firesim_rocket_saturn)
- Profile: measured (use_profiled=true)

Workload JSON:
`/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json`

## Policies under test

| Policy | Underlying solver | Structural intent |
|:---|:---|:---|
| `yolo_anchor`         | `greedy_periodic` | Anchor on the heaviest aperiodic, periodic instances fill residual |
| `periodic_anchor`     | `decomposed` | Reserve periodic slots first, aperiodic fills the gaps |
| `critical_path_first` | `heft` (upward_rank priority) | Global CP-first ordering |
| `cpsat_unconstrained` | `cpsat` | Exact-solver control; no pre-applied structure |

Source: `xpu-rt/policies/{yolo_anchor, periodic_anchor, critical_path_first, cpsat_unconstrained}.py`.

## Results (real, this run)

| Policy                 | Makespan (ms) | Deadline misses | Release misses | Dispatches | Solve wall (s) |
|:-----------------------|---:|---:|---:|---:|---:|
| `periodic_anchor`     | 75.57 | **0** | 0 | 248 | 2.98 |
| `critical_path_first` | **54.43** | 88 | 0 | 245 | 1.78 |
| `yolo_anchor`         | 61.20 | 67 | 0 | 236 | 5.46 |
| `cpsat_unconstrained` | 186.74 | 65 | 0 | 207 | 32.06 |

## Findings

### `periodic_anchor` is the only band-clean policy
0/248 deadline misses, longest makespan (75.6 ms). The decomposed solver's
period-first phasing leaves every periodic instance with enough slack to
finish in its window. Cost: ~21 ms of "wasted" makespan vs critical_path_first
that we pay to honor the bands.

### `critical_path_first` minimizes makespan but trampleperiodic bands
54.4 ms (best), 88 deadline misses (worst — same as the raw heft audit).
The CP-first heuristic packs yolov8 aggressively into the early window, so
periodic mlp/dronet instances slip late. This is the classic
makespan-vs-deadlines trade made explicit by the band Gantt's red boxes
(see `gantts/critical_path_first.png`).

### `yolo_anchor` lands in between
61.2 ms / 67 misses. The greedy_periodic phasing helps somewhat but still
runs late on the last MLP/dronet instances when yolov8 stretches the
non-periodic critical path past their 20–30 ms windows.

### `cpsat_unconstrained` surprises with the worst makespan
186.7 ms — 3.4× the best. CPSAT solved within the 30s time-limit but
the global optimum it found includes wide gaps. Two contributing factors:
1. The workload has `restrict_makespan_to_nonperiodic: false`, so CPSAT
   minimizes the total makespan over all 300 ops (including all 6 periodic
   instances). The objective at the last mlp_control3 deadline (40 ms) +
   yolov8 tail dominates.
2. 30 seconds is below CPSAT's effective convergence time for ~300-op
   workloads; runs with 60-s limits previously hit 121.5 ms with 40
   misses. This is exactly the Phase F1 motivation: CPSAT alone is not a
   reliable global-optimum baseline at this scale; either we reformulate
   MOSEK (F2), allow CPSAT longer, or accept the audit result.

The 65 deadline misses for cpsat_unconstrained are particularly
revealing: CPSAT encodes `max_end_t` as a hard constraint, so the only
ways it can emit a fixture with 65 misses are (a) the solver timed out
before convergence and emitted a best-effort feasible-ish result, or
(b) the workload is genuinely infeasible at this frequency mix. Phase
B1's `frequency_feasibility` formula and Phase F1's divergence
diagnosis will disambiguate.

## Honesty notes

- Every Gantt bar duration in `gantts/*.png` traces to the profile DB
  loaded by `run_xpurt_schedule.py --use-profiled` from
  `gen/profile/<hw>/.../results.csv`. No synthetic times.
- The `deadline_miss` counts are the same in the fixture
  (`deadline_miss: true` flag set by Phase A2 in heft) and in
  `diagnostics.check_band_invariant` (Phase A1), verified with the
  parity check `flagged == n_deadline_violations`.
- `cpsat_unconstrained` did NOT silently downgrade to a heuristic when
  it ran out of time. The 30-s limit was hit; the result was the
  best-known feasible. Phase F4 will rerun at higher limits +
  reformulated MOSEK to bracket the true optimum.

## Open follow-ups

- C5 — wire the policy choice into `scripts/decision_loop.py` so each
  decision-loop round can be run under a specific policy.
- B-formula seed pass — none of the policies currently apply
  `shard_benefit` / `fuse_benefit` candidate seeding. The headline-level
  policy structure is honest as-is; per-op seeding is the next iteration.
- Per-policy band Gantts in `gantts/`. The 4-up comparison figure (Phase
  D4) will assemble these alongside frequency variants.
