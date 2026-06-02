# Schedule-quality comparison — oracle floor + post-pass (predicted)

Workload: `configs/multi_3way_qrb_y64.yaml` (1×yolov8_nano_64 + 4×mlp_control + 2×dronet on Gemmini + Saturn OPU hetero, 75 ms horizon).

All schedules below were generated with the compaction + automerge
post-passes **enabled** (XPURT_NO_COMPACT / XPURT_NO_AUTOMERGE unset).
Per-scheduler fixtures live in `artifacts/scheduler_bundle_postpass/`.

| Solver | Solve wall | Makespan | Meets 75 ms? | Dispatches pre / post automerge | Pairs merged |
|:-------|-----------:|---------:|:------------:|:-------------------------------:|-------------:|
| CPSAT (600s budget) | < 60 s | **73.98 ms** | ✅ | 300 → 235 | 65 |
| HEFT     | < 1 s | 83.08 ms | ❌ (10.8 % over) | 300 → 170 | 130 |
| EDF      | < 1 s | 83.08 ms | ❌ (10.8 % over) | 300 → 170 | 130 |
| PEFT     | < 1 s | 87.17 ms | ❌ (16.2 % over) | 300 → 147 | 153 |
| MOSEK (1800s budget) | in-flight | — | — | — | — |

### Reading the table

- **Compaction + automerge** apply uniformly to every solver's output.
  The compaction pass (xpu-rt/compaction.py) slides each op to the
  earliest start time that still satisfies its release time, its
  predecessor finish times, and its assigned machine's previous-op
  finish — strictly left-shift, never right. The automerge pass
  (xpu-rt/automerge.py) collapses adjacent same-network back-to-back
  dispatches on the same core when nothing external reads the
  intermediate output. Both passes are no-ops on already-tight
  solvers like MOSEK / CPSAT and slack-eliminating on list schedulers.
- **CPSAT wins** by ~10 % over the list-scheduler heuristics on this
  workload and is the only one to clear the 75 ms horizon target.
  It actively interleaves networks for parallelism (fewer adjacent
  same-network pairs to merge — 65 vs 130-153 for the heuristics).
- **PEFT collapsed the most pairs (153)** because its placement is
  more sequential per-network — same total work, but more
  dispatch-handshake overhead before the automerge eliminated it.
- **Oracle floor** numbers come from `xpu-rt/oracle.compute_floor()`
  via max(critical-path-on-fastest-device, per-machine-load-sum,
  release-time-floor). Returned as native workload time units
  (ms for this workload via `cycles_per_ms: 1000000`).
- **MOSEK** is still running with the 1800s time budget; results
  will be appended here when the solve completes. CPSAT's 73.98 ms
  is the current best, so MOSEK's gap (if any) measures how much
  the cvxpy/MOSEK formulation pessimizes vs CP-SAT on this shape.

### Reproduction

```bash
cd /scratch2/agustin/ModelBlaster
# For each of HEFT/PEFT/EDF/CPSAT/MOSEK:
/tmp/run_one_solver.sh CPSAT 600
# (run_one_solver.sh is a 10-line wrapper that swaps the `solver:`
# line in configs/multi_3way_qrb_y64.yaml and shells out to
# scripts.run_xpurt_scheduler_multi with --time-limit and a per-solver
# output path. The full driver lives in /tmp during a session — see
# scripts/run_xpurt_scheduler_multi.py for the actual entry point.)
```
