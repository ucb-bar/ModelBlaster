# FireSim-measured scheduler captures — status & closure (#163)

## Task

#163 originally asked for measured FireSim captures of MOSEK +
greedy_periodic on the 1+4+2 demo workload
(`networks_1yolo_4mlp_2dronet_firesim.json`).

## What we actually have, end of session

| Scheduler | Predicted (postpass) | Measured (FireSim) | Notes |
|:----------|---------------------:|-------------------:|:------|
| baseline (decomposed) | 75.57 ms | **13.5 ms (388 disp)** | longrun/baseline, 1.0 rep, complete |
| HEFT     | 83.08 ms | error after 6984 s | b00lwnji9, infrasetup OK but uart hang |
| EDF      | 83.08 ms | not attempted | low signal vs HEFT (same makespan) |
| PEFT     | 87.17 ms | not attempted | worst predicted; not informative |
| CPSAT    | **73.98 ms** ✓ | **not converged on FireSim** | predicted-only |
| MOSEK    | cvxpy SolverError | N/A | formulation diverges at 300 ops |
| greedy_periodic | 61.20 ms | partition_gantt OK | uart hang during runworkload (>60 min) |

## Root cause

Two independent blockers, both surfaced this session:

1. **MOSEK formulation diverges.** With 300 ops × 2 machine-combo
   choices the big-M no-overlap formulation produces a cvxpy
   numerical-error after ~50 min of branch-and-bound (see
   `artifacts/scheduler_bundle_postpass/MOSEK.log`). CPSAT is the
   feasible optimum surrogate at this scale; documented in
   `oracle_table.md`. **Not fixable from the loop side** — would
   need a tighter MILP encoding (e.g. CHC formulation, time-window
   decomposition).

2. **FireSim bitstream stability on the 1+4+2 workload.** The
   Gemmini+Saturn-OPU hetero bitstream + the full
   `1×yolov8_nano_64 + 4×mlp_control + 2×dronet` workload hangs
   inside FireSim's simulator for runs >60 min. We've reproduced this
   on HEFT (b00lwnji9, 6984 s and exited error), MOSEK (separate
   attempts in prior session), and greedy_periodic. The simulator's
   UART output freezes even though the FPGA continues; cancelling
   restores the queue.

## What we did instead

- Curated `configs/agentic_fuse_split_demo.yaml` — smaller workload
  (no yolo) where both the MILP formulation converges AND the
  FireSim runtime stays stable. See `artifacts/agentic_fuse_split/`
  for the 4-scheduler comparison on that workload.
- Documented the formulation-divergence and bitstream-hang issues
  in:
  - `artifacts/bundle/walkthrough_v2/oracle_table.md` (MOSEK
    convergence note).
  - `artifacts/bundle/walkthrough_v2/README.md` Section 9
    (compaction + automerge + CPSAT result).

## Closure

**#163 closed-with-caveat.** The measured-FireSim captures of MOSEK
+ greedy_periodic on the 1+4+2 baseline are not currently producible
due to two independent infra blockers above. CPSAT (the optimum
solver that DOES converge) is captured predicted. For measured-
schedule story we already have the baseline 388-dispatch FireSim
trace at `artifacts/bundle/longrun/baseline/`, which validates the
loop end-to-end on the same workload via the decomposed solver.

If/when the bitstream stability issue is resolved (likely needs a
firesim simulator upgrade or a different hetero bitstream), re-run:

```bash
# Once bitstream is stable on 1+4+2:
PYTHONPATH=/scratch2/agustin/XPU-RT/xpu-rt:/scratch2/agustin/ModelBlaster/src \
    /scratch2/agustin/miniforge3/envs/merlin-dev/bin/python \
    -m scripts.run_xpurt_scheduler_multi \
    --config configs/multi_3way_qrb_y64.yaml \
    --output artifacts/bundle/cpsat/firesim_batch.json \
    --time-limit 600   # CPSAT
# Then:
bash scripts/run_bundle_firesim.sh artifacts/bundle/cpsat/manifest.json
```
