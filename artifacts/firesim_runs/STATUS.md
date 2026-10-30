# FireSim closed-loop status (this session)

## What's measured

### Baseline yolov8 (unfused, 212 ops) — COMPLETE

- Submitted: 2026-06-03 ~13:50 PDT via FIRESIM_QUEUE=1
- Completed: 2026-06-03 14:00 PDT (~10 min wall, no queue wait)
- Status: FireSim PASSED at 14,312,017,337 sim cycles; in-binary verify
  reports `max_abs_err=5` (expected — int8 quantized rvv_opu vs PyTorch
  golden is normally non-zero on this network).
- Output: `artifacts/firesim_runs/baseline_yolov8_rvv_opu/`
  - `run.log` — full pipeline log
  - `profile.csv` — 204 per-op cycle measurements
- Wall clock (mtime): 6,985,323 cycles ≈ 7.0 ms on a 1 GHz Rocket
  (note: this is the rocket harness mtime; the per-op cycle counts in
  `profile.csv` are from a different counter and total ~7 B raw cycles;
  unit reconciliation deferred — both measurements are from the SAME
  run so internal ratios are valid).

### Fused yolov8 (57 BN+SiLU fused, 155 ops) — IN PROGRESS

- Submitted: 2026-06-03 14:18 PDT
- IR: `examples/yolov8_nano/int8/generated/graph.fused.json` (212 → 155
  ops via 57 batchnorm2d_silu_s8 fusions).
- Cached kernel pulled in: `examples/yolov8_nano/int8/cache/rvv_opu/
  rvv_opu_batchnorm2d_silu_s8_bn_silu_per_channel_register_fused.c`
  (the spike-hetero-verified Bedrock kernel from this session)
- generate_skeleton emitter for `batchnorm2d_silu_s8` was missing —
  added this session (`pipeline/generate_skeleton.py:1309`)
- Currently in stage 5 (FireSim running).

### Hybrid policy (periodic-reservation + MOSEK yolov8) — COMPLETE (against older profile DB)

- Implementation: `xpu-rt/policies/hybrid_periodic_mosek_yolo.py`
- Wall: ~230 s (88 s yolov8 MOSEK + 3 s periodic reservation + stitch)
- Result on the existing profile DB (NOT yet re-run with fresh
  FireSim cycles):
  - **70.0 ms makespan**
  - **0 deadline misses**
  - **0 release violations**
- This is the new Pareto-best deadline-safe option, beating
  periodic_anchor (75.6 ms / 0 miss).
- Caveat: 240 dispatches in the fixture — the workload's instance
  expansion produces 8 mlp + 4 dronet instances (because horizon is
  ~75 ms / period_mlp=10 ms).

## Honest status

What I claimed before this exchange:
- "75.6 ms / 0 miss" for periodic_anchor — measurement-grounded but
  via OLDER FireSim data (mean_time_ns column from
  `gen/profile/.../results.csv`), NOT from this session's run.

What's measured this session:
- The fresh `profile.csv` from the baseline run (raw cycle counter,
  204 ops).
- The fused profile.csv (in progress).
- Spike-hetero bit-exact verification of both new fused kernels
  (max_abs_err=0).

What's needed to close the loop:
- Convert the fresh `profile.csv` cycles into the profile DB's
  expected `mean_time_ns` format. The raw-cycle ratio between fresh
  and old (~8x for heavy ops) is the unit-conversion factor.
- Re-run periodic_anchor + hybrid policies with the FUSED measured
  cycles. Compare against baseline measured cycles to get the actual
  fusion speedup.

## What the fused run will tell us

When the fused FireSim run completes (expected: ~10-30 min wall after
queue), the deliverable is:

1. Per-op cycles for the 155-op fused IR (57 fused conv→silu chains).
2. Compared against the baseline 204-op profile, per-fused-op speedup.
3. After re-scheduling with the fused cycles: actual makespan numbers
   on both `periodic_anchor` and `hybrid_periodic_mosek_yolo` policies.

This will give the first end-to-end measured answer to:
- Does the spike-hetero-verified fused kernel actually run faster on
  FireSim than the unfused chain?
- How much makespan does fusion shave under the deadline-safe hybrid
  policy?
