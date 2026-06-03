# Provenance ledger — every metric in this thread

The user (correctly) pushed back on imprecise claims. This file
tracks every number quoted in our conversation by its true source.

## Direct FireSim measurements this session (real wall-clock)

| Measurement | Value | Path | Notes |
|:---|:---|:---|:---|
| yolov8 unfused total cycles | 6,985,278,800 (raw) | `baseline_yolov8_rvv_opu/profile.csv` | 204 per-op rows |
| yolov8 unfused wall (mtime) | 6.985 ms | `baseline_yolov8_rvv_opu/run.log:1318` | mtime cycles ÷ 1 GHz |
| yolov8 fused total cycles | 7,044,461,818 (raw) | `fused_yolov8_rvv_opu/profile.csv` | 147 per-op rows (57 fused) |
| yolov8 fused wall (mtime) | 7.044 ms | `fused_yolov8_rvv_opu/run.log` | |
| Fused bit-exactness on FPGA | max_abs_err=0 | `fused_yolov8_rvv_opu/run.log:1322` | bit-exact vs PyTorch golden |

Both runs used `RUNNER=firesim TARGET=rvv_opu BACKEND=reference
FIRESIM_QUEUE=1`. yolov8 was run isolated — not as part of the multi-
network schedule.

**Conclusion:** fused kernel is bit-exact on FPGA but ~0.85 % slower
because the per-call LUT precomputation eats the dispatch-overhead
savings. Vectorized RVV variant is the next step.

## Spike-hetero functional verifications this session (correctness only)

| Kernel | Result | Path |
|:---|:---|:---|
| `batchnorm2d_silu_s8` (algo 0) | max_abs_err=0 after 6 Bedrock attempts | `artifacts/kernels/batchnorm2d_silu_s8/.../spike_loop_summary.json` |
| `conv2d_batchnorm2d_s8` (algo 0) | max_abs_err=0 on attempt 1 | `artifacts/kernels/conv2d_batchnorm2d_s8/.../spike_loop_summary.json` |

Spike-hetero per memory caveat: cycle counts NOT meaningful on
OPU/Gemmini ops. Only used for correctness here.

## Profile-DB-derived scheduler-model results (NOT measured this session)

Per-op cycles for the scheduler model came from
`gen/profile/<hw>/firesim_rocket_saturn/yolov8_nano/.../results.csv`,
which was populated by FireSim runs in **prior sessions** (the
`mean_time_ns` column has nanosecond-equivalent measurements).

The scheduler then computed a per-policy placement, summed per-op
times along the makespan-defining chain, and reported a makespan. The
cycles are real measurements; the makespan is a model output.

| Policy | "Makespan" reported | Reality |
|:---|---:|:---|
| periodic_anchor | 75.6 ms | scheduler model output, NOT a measured wall |
| critical_path_first (heft) | 54.4 ms | same |
| yolo_anchor | 61.2 ms | same |
| MOSEK F2g | 51.1 ms | same |
| cpsat_unconstrained | 111-187 ms | same (also non-deterministic) |
| **hybrid_periodic_mosek_yolo (NEW)** | 70.0 ms | same |

All "deadline miss" counts are also model outputs — they come from
the `band_invariant` check on the scheduler's placement.

## Sharding / asymmetric-split work (NOT measured this session)

Last session demonstrated that the scheduler PLACES tiles on
different cores (cross-accelerator parallelism). The per-tile cycles
that the scheduler used were `parent_cycles × tile_oc_fraction` — a
derived prediction, NOT a measurement. The actual tile kernels
(OC=13 conv2d on gemmini, OC=3 conv2d on rvv_opu, etc.) were never
generated, never compiled, never run.

So "tile_0 = 1.71 ms / tile_1 = 1.61 ms" is a SCHEDULING
FEASIBILITY result, not a kernel-measurement result.

## End-to-end hetero measurement IN PROGRESS this session

`scripts/run_xpurt_bundle.py` (via `examples/xpurt_demo/run.sh` with
`RUNNER=firesim`) takes a scheduled fixture and runs the FULL
hetero workload (4 MLP + 2 Dronet + 1 Yolo, with ops dispatched to
gemmini_q31 / rvv_opu per the schedule's placement). Currently:

| Policy | Status | Output dir |
|:---|:---|:---|
| periodic_anchor | building | `artifacts/firesim_runs/policy_periodic_anchor/` |
| hybrid_periodic_mosek_yolo | not yet submitted | — |

When these complete, the "makespan" in the policy table will be
replaced by direct mtime-wall measurements from the FPGA — true
end-to-end measurement, not scheduler-model output.
