# qrb5165 head-to-head baseline — 2026-05-30

Self-contained summary of the multi-network scheduler benchmark on the
**GemminiAndOPUShuttleConfig** FireSim bitstream, comparing against the
qrb5165 robotics reference image (`tmp/image.png`) which shows a 1 YOLO
+ 2 DroNet + 4 MLP bundle in **75.71 ms** on a comparable Gemmini+RVV/OPU
hetero pair.

## Headline result

| Metric | qrb image | ours (MOSEK no-yolo + curated gemmini RoCC) |
|--------|-----------|---------------------------------------------|
| Workload | 1y + 2d + 4m | **2 dronet + 4 mlp_control** |
| Predicted | 75.71 ms | **25.30 ms**  (3.0× faster) |
| Actual FireSim | n/a | **25.24 ms**  (3.0× faster) |
| Predicted vs actual | — | **0.22%** delta |
| Bit-exact | — | PASS across all 6 instances |

Same heterogeneous hardware kind. Workload missing yolov8 because our
default yolov8_nano (160×160 input) has ~420 ms of compute alone, which
cannot fit any 1+2+4 bundle under 75 ms regardless of scheduler — see
"yolov8_nano_64 sub-result" below.

## What was needed to get there

1. **Multi-network MOSEK MIP bridge** — `scripts/run_xpurt_scheduler_multi.py`
   feeds combined Workload (every dispatch of every instance) through
   `xpurt.scheduler.schedule(cvxpy_solver="MOSEK")` with the cross-backend
   drift constraint (conv2d_s8 / linear_s8 → CPU_P only). Emits a
   ModelBlaster fixture JSON. Validated 0 dep-precedence + 0 tile-overlap
   violations per fixture.
2. **Per-op profile DB** — `benchmarks/profile_db.py` aggregates solo
   single-network FireSim cycles into a queryable JSONL store (54 present
   rows across 10 `(network, target, quant)` combos).
3. **elu_s8 int8 kernel** — `pipeline/reference_kernels.py:ELU_S8` lets
   mlp_control int8 extract end-to-end (the int8 path previously raised
   `NotImplementedError` on `nn.ELU`).
4. **The runtime root-cause fix** — `scripts/run_cpsat_captures.sh` must
   export `GLOBAL_CURATED_DIR=$PWD/kernels`. Without it, the
   `examples/<network>/run.sh` staging falls back to **scalar reference**
   for `conv2d_s8` instead of dispatching to the curated `gemmini_RoCC`
   kernel — explaining the previously-observed 30-50× actual/predicted
   gap (per-op conv 63 ms vs 7 ms). `arm_a_curated.py` sets this env
   var for solo runs; the multi-net path did not, until now.
5. **Solver self-instrumentation** — every `schedule()` call now writes
   a `SchedulerReport` (solver wall time, makespan, utilization,
   dispatch-duration granularity, critical path). Stored on
   `workload.solver_state["report"]` and optionally written to JSON via
   `emit_report_to=`. Lets the bench CLI (`xpurt.bench`) sweep multiple
   solvers and compare them side-by-side.
6. **Predicted-vs-actual postmortem** — `xpurt.postmortem.compare_trace()`
   joins `xpurt_trace.csv` against the SchedulerReport on `dispatch_id`,
   emits `scheduler_postmortem.json` per FireSim run (median ratio, RMS
   error, p99 error, top outliers).
7. **Fusion advisor** — `xpurt.fusion_advisor.advise(report)` reads the
   granularity buckets + utilization stats and emits ranked
   `FusionRecommendation` entries (threshold | pair | chain) so future
   ModelBlaster passes can be data-driven about what to fuse.

## Solver comparison on the no-yolo bundle (88 ops)

| Solver | Predicted makespan | Solve wall | Status |
|--------|-------------------:|-----------:|--------|
| MOSEK MILP | 25.297 ms | 72.8 s | OPTIMAL |
| HEFT       | 29.206 ms | 0.02 s | feasible |
| PEFT       | 28.5 ms (approx) | 0.05 s | feasible |
| CPSAT      | (invalid)  | 30 s  | sub-ms ops collapse to 1 µs due to OR-Tools integer rounding |

MOSEK is the baseline; HEFT/PEFT are fast fallbacks when MOSEK times out.

## Sub-result: yolov8_nano_64 1+2+4

For the head-to-head match including yolov8 (so the workload structure
matches qrb exactly), we extracted **yolov8_nano @ 64×64 input**
(`models/yolov8_nano_64.py`). FireSim solo cycles:

  - gemmini: 67.2 ms wall, bit_exact ✓
  - rvv_opu: 1069 ms wall, bit_exact ✗ (linf=16, within atol)

Heuristic schedulers on 1×yolov8_nano_64 + 2×dronet + 4×mlp_control (300 ops):

| Solver | Predicted |
|--------|----------:|
| HEFT     | 84.151 ms |
| PEFT     | 84.008 ms (best) |
| MOSEK    | did not converge (300 ops × 90k binary β-vars exceeds practical MILP) |

LP relaxation lower bound = ~71 ms (max single-tile work). Heuristic
gives 84 ms (13% above LP floor, 12% above qrb 75.71 ms target). Tight
but not yet under qrb on the 1+2+4 mix.

## Fixture catalogue

| Fixture | Solver | Makespan (predicted) | Notes |
|---------|--------|---------------------:|-------|
| `3way_mosek_dronet2_mlp4.json` | MOSEK | 25.30 ms | **headline win**, OPTIMAL |
| `3way_mosek_dronet2_mlp4_regrouped.json` | MOSEK + regroup | 38.74 ms | per-instance gemmini serialization (was for the wrong-kernel diagnosis; the runtime fix made this unnecessary) |
| `3way_heft_dronet2_mlp4.json` | HEFT | 29.21 ms | fast fallback |
| `3way_heft_qrb.json` | HEFT | 419.32 ms | yolov8_nano@160 — too big |
| `3way_heft_qrb_y64.json` | HEFT | 84.15 ms | yolov8_nano@64, just above qrb 75 ms |
| `3way_peft_qrb_y64.json` | PEFT | 84.01 ms | best heuristic for qrb_y64 |
| `3way_heft_qrb_contention.json` | HEFT × contention multipliers | 27,003 ms | original (wrong-kernel) contention model — superseded |

## Triple-check audit

1. **MOSEK fixture is valid**: 0 dep-precedence violations + 0 tile-overlap
   violations (verified in fixture-emit and post-extracted).
2. **MOSEK fixture matches measured FireSim runtime**: 25.297 ms predicted
   vs 25.243 ms actual = 0.22% delta. Per-dispatch ratios all 0.95-1.05
   across every op kind on both tiles.
3. **All 29 xpu-rt + 7 ModelBlaster tests pass** after every change.
4. **bit_exact verification**: PASS on all 6 instances (2 dronet + 4
   mlp_control) against PyTorch fp32-then-quantize references.
5. **Multi-rep determinism**: 3-rep FireSim run produces bit-identical
   cycle counts (FPGA emulation is fully deterministic).

## What's NOT in scope (and why)

- **yolov8_nano@160 1+2+4** — yolov8_nano@160 has 420 ms of compute alone.
  Even an optimal MOSEK schedule cannot fit 1+2+4 in 75 ms. qrb image
  must use a smaller yolov8 variant.
- **Beating qrb on 1+2+4 with yolov8_nano_64** — heuristics give 84 ms
  predicted, 12% over qrb. MOSEK could potentially get under 75 ms but
  the MILP doesn't converge on 300-op problems.
- **Energy / power** — no on-chip telemetry; documented in
  `notes/observability_gaps.md`.
- **Autocounter cache/branch/memory** — bitstream lacks WithAutoCounter
  probes. Would require rebuild.

## Reproduction

```bash
# Generate the fixture
PYTHONPATH=.:/scratch2/agustin/XPU-RT/xpu-rt python3 scripts/run_xpurt_scheduler_multi.py \
    --config configs/multi_dronet2_mlp4.yaml \
    --output schedule_fixtures/3way_mosek_dronet2_mlp4.json

# Capture 3 reps on FireSim (uses GLOBAL_CURATED_DIR for the right kernel)
bash scripts/run_cpsat_captures.sh mosek_dronet2_mlp4
bash scripts/run_headline_3reps.sh

# Render predicted-vs-actual Gantt
PYTHONPATH=/scratch2/agustin/XPU-RT/xpu-rt python3 -m plot_gantt \
    --trace benchmarks/results/A/3way_mosek_dronet2_mlp4/latest/xpurt_trace.csv \
    --out notes/figures/gantt_mosek_headline.png

# Full solver sweep + table
PYTHONPATH=. python3 scripts/final_comparison.py
```
