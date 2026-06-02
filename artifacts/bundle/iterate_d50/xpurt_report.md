# Iterative scheduling improvement

Deadline budget: **50.0 us** (explicit).

| id | config | axis | profile_hw | makespan_us | meets | granularity | bottleneck |
|----|--------|------|-----------|------------:|:-----:|-------------|-----------|
| A3 🏆 | milp/heft | scheduler | V256D128_rvv+gemmini_q31 | 54.43 | ⚠️ | too_fine | CPU_P#0 |
| A4 | milp/peft | scheduler | V256D128_rvv+gemmini_q31 | 56.24 | ⚠️ | too_fine | CPU_P#0 |
| A6 | greedy | scheduler | V256D128_rvv+gemmini_q31 | 60.48 | ⚠️ | too_fine | CPU_P#0 |
| baseline | decomposed | baseline | V256D128_rvv+gemmini_q31 | 75.57 | ⚠️ | too_fine | CPU_E#0 |
| B3 | decomposed | backend | V256D128_rvv+gemmini_q31 | 75.57 | ⚠️ | too_fine | CPU_P#0 |
| A5 | milp/edf | scheduler | V256D128_rvv+gemmini_q31 | 116.78 | ⚠️ | too_fine | CPU_E#0 |

Skipped (did not finish within the per-candidate budget):
- `A1` milp/mosek on V256D128_rvv+gemmini_q31 — error: n(              ^^^^^^^^^^^^^   File "/scratch2/agustin/XPU-RT/xpu-rt/schedulers.py", line 29, in _mosek     from scheduler import schedule as milp_schedule   File "/scratch2/agustin/XPU-RT/xpu-rt/scheduler.py", line 5, in <module>     import cvxpy as cp ModuleNotFoundError: No module named 'cvxpy' 
- `A2` milp/cpsat on V256D128_rvv+gemmini_q31 — error: _schedule     cp_model = _lazy_cp_model()                ^^^^^^^^^^^^^^^^   File "/scratch2/agustin/XPU-RT/xpu-rt/scheduler_cpsat.py", line 50, in _lazy_cp_model     raise RuntimeError( RuntimeError: ortools is required for the CP-SAT scheduler. Install with `pip install ortools` or add to env.yml. 
- `B1` decomposed on V256D128_rvv — timeout (rvv emits many more dispatches — a fusion/coarsen candidate)
- `B2` decomposed on gemmini_q31 — timeout

**Winner: `A3` (milp/heft)** — makespan 54.43 us vs baseline 75.57 us (**28.0% lower**); is within the deadline.

## Advisor on baseline

```
Scheduler advisor — solver=decomposed
Deadline: MISSED by 25.6 us (51.1%)   (makespan 75.6 us, deadline 50.0 us)

Diagnosis:
  • Bottleneck backend: CPU_E#0; idle: none.
  • Granularity: too_fine.
  • deadline_miss_count=0.

Recommendations (ordered by expected impact):
  1. [coarsen] fusion_threshold = 1000 cycles
      why: 388/388 dispatches (100.0%) fall below 1000 cycles. Fusing them collapses the dispatch-overhead tail.
```

## Profiler/backend comparison (axis B)

- **cpu_e=V256D128_rvv, cpu_p=gemmini_q31**: best makespan 54.43 us (milp/heft, misses).
- **cpu_e=gemmini_q31, cpu_p=V256D128_rvv**: best makespan 75.57 us (decomposed, misses).
- _Homogeneous backends (V256D128_rvv, gemmini_q31) exceeded the predicted scheduling budget (dispatch-count explosion) — compare them on the real FireSim batch, and/or apply axis-C fusion first._

## Granularity/fusion (axis C — ModelBlaster)

Advisor flagged `too_fine`. Emitted fusion hints for 3 network(s): mlp_control (1 groups), dronet (2 groups), yolov8_nano (6 groups).
See `firesim_batch.json` candidate `C1` for the hint payload; ModelBlaster realizes it (re-extract/re-gen kernels on spike, re-profile on FireSim).

## Next: FireSim batch

`firesim_batch.json` lists 10 candidates for the ModelBlaster session to build + run in one batched FireSim session.
