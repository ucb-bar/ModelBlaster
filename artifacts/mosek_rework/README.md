# Phase F — MOSEK convergence rework

The user's directive (2026-06-03): *"if mosek diverges we should find
methods to help it converge to good results based on logic, tools or
patterns"*.

The plan now pursues SIX parallel convergence-aid tracks (F2a–F2g)
rather than the original two-formulation choice. Pick the
combination that converges first; the rest serve as documented
fallback.

## F1 — Divergence diagnosis (run01)

`scripts/mosek_divergence_diagnose.py` bisects op count from a tiny
(mlp_instances=1) variant up to the headline (mlp_instances=4) and
runs both CPSAT and MOSEK at each scale. Records (status, wall, obj)
per cell.

Output: `artifacts/audit/mosek_diagnose.csv`.

Initial observation (headline scale): MOSEK runs > 3 minutes of CPU
time without producing a fixture, despite `time_limit=60` being passed
as `MSK_DPAR_OPTIMIZER_MAX_TIME`. This means the wall-clock is spent
either in:
1. CVXPY formulation/canonicalization (which the time_limit doesn't
   bound), OR
2. MOSEK presolve before optimization (which separate params control).

Per-cap bisection results will land in the CSV when F1 completes;
the README will be filled in with the actual scale-vs-status data.

## F2 — Convergence aids (six tracks, ranked by implementation cost)

### F2a — Warm-start from CPSAT/HEFT (operational)

Solve CPSAT first (~30 s), feed its (t, alpha) as initial values to
MOSEK. CVXPY exposes `warm_start=True` plus the MOSEK-specific
`MSK_IPAR_MIO_CONSTRUCT_SOL=ON` parameter. Even when MOSEK can't
improve, it has a feasible primal.

Implementation status: framework module
`xpu-rt/scheduler_mosek_warmstart.py` provides the wrapper
(`warmstart_from_fixture`) that reconstructs (t, alpha) from a CPSAT
fixture. The actual injection into cvxpy's variable values requires a
scheduler.py refactor (it currently builds the cvxpy `Problem` and
calls `.solve()` in one path; warm-start needs the build/solve split
so we can set `t.value=t_init` between them). The refactor is bounded
and documented inline in the module.

### F2b — Pre-fix obvious placements (variable elimination)

For ops whose `infeasible_combinations` set leaves only one feasible
combo, hard-fix `alpha[i,k]=1` before MOSEK touches them. Cuts
hundreds of binary variables. Implementation: add `--prefix-singletons`
flag to scheduler.py that walks the workload pre-solve, identifies
singleton-feasibility ops, and adds equality constraints
`alpha[i,k]=1` for those.

### F2c — Symmetry-breaking constraints

Periodic instances of the same network are interchangeable. Add per-
network-pair ordering: for instance k and k+1 of network N, constrain
`t[N_inst_k, first_op] ≤ t[N_inst_(k+1), first_op]`. Cuts the
branch-and-bound search by `num_instances!`. Standard technique from
scheduling-MILP literature.

### F2d — Coarse→fine time discretization

First solve at 1 ms time buckets (low big-M), use that schedule to
constrain a 0.1 ms refinement step. Each step is small. The two-stage
approach is standard for time-indexed formulations.

### F2e — Solver parameter sweep

Sweep MOSEK params at headline scale, keep the row that converges
fastest:
- `MSK_DPAR_MIO_TOL_REL_GAP`: 0.0001 → 0.01 (accept a 1% gap)
- `MSK_IPAR_PRESOLVE_USE`: ON → OFF (presolve sometimes hurts on
  big-M dense formulations)
- `MSK_IPAR_MIO_HEURISTIC_LEVEL`: 1 → 5 (aggressive heuristic
  emphasis finds primal feasible faster)
- `MSK_IPAR_NUM_THREADS`: cap at physical cores
- `MSK_DPAR_OPTIMIZER_MAX_TIME`: ensure it bounds optimizer time,
  not formulation time.

### F2f — Time-indexed reformulation

Replace big-M assignment with `x[op, machine, t] ∈ {0,1}` at coarse
time granularity. Eliminates the worst big-M coefficients. Classical
alternative; published convergence guarantees on small-disjunctive
problems.

### F2g — Lagrangian decomposition by network

Decompose the global problem into per-network subproblems, coordinate
via shared CPU_P/CPU_E capacity duals. Each subproblem is small enough
for MOSEK in seconds; the outer ADMM loop converges in O(20)
iterations on similar shapes. Heaviest implementation lift; deepest
guaranteed convergence.

## Quality gates (binding for F3)

For any aid (or combination) to be declared the winner:

- Reported objective must be within **1 %** of CPSAT on the same
  workload (Phase F3 codifies this as `test_mosek_cpsat_agreement.py`).
- Solver returns OPTIMAL or OPTIMAL_INACCURATE — not TIME_LIMIT,
  not USER_OBJ_CUT, not SOLVER_ERROR.
- 60-s wall-clock budget for the converged path on the headline
  workload.
- Bit-identical band-compliance audit row to CPSAT.

If no single track meets these, the README documents which
combination DID, with the runtime evidence pinned in
`artifacts/audit/`. The cold-rerun gate (Phase Q-rerun) applies.

## Honest fallback

If none of F2a-F2g get MOSEK to converge at headline scale within
budget, the deliverable is:
- A documented failure mode (which gates each track hit).
- A practical recommendation: **CPSAT is the global-optimum
  reference at this scale**, MOSEK is reserved for smaller subsets
  (e.g. periodic-only workloads where instance counts are low).

The framework supports both outcomes; the worst case is not a
silent fallback but an explicit "MOSEK formulation does not scale
to headline op-counts even with these aids" entry in the report.
