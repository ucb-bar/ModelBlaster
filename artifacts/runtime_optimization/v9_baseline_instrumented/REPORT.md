# v9 — instrumented baseline (Phase G1)

## Result

End-to-end FireSim of the hybrid_periodic_mosek_yolo schedule with
per-hart attribution counters added to the walker codegen. Build
clean in 9 s, simulation PASSED in 11.62 B cycles. Per-network
correctness identical to v8 (within rounding):

| Network | max_abs_err | per-net worst-instance wall (mtime, µs) |
|---|---|---|
| mlp_control | 0 | 77,783 |
| dronet | 72 | 178,280 |
| yolov8_nano | 22 | 638,187 |

## The breakdown (the headline)

```
kind        wall(µs)  kernel  dep_wait    sync  gate  idle   cfg
gemmini      708,799 109,080   599,085     116   448    47     0
rvv_opu      786,534 732,620    53,586     234     0    54     0
```

The data flips the prior runtime-overhead hypothesis on its head:

1. **rvv_opu is doing 732 ms of real kernel work** — 93% of its wall.
   It is the saturated lane.
2. **gemmini is dep-blocked for 599 ms** waiting for rvv_opu
   producers — 85% of its wall is idle wait, only 15% is real
   compute (109 ms).
3. **All four "runtime overhead" categories — sync, gate, idle, cfg
   — sum to under 1 ms across both harts.** Pthread sync, fence
   instructions, the existing target_start gate spin, and inter-iter
   gaps are NOT the bottleneck. They are not even on the leaderboard.

## What G2 *was* going to fix

The original Phase G plan ranked four fixes against this 9× wall
gap:

| Phase | Target | v9-measured potential save |
|---|---|---|
| G2a — R_k gating per instance | dep_wait | already in place; saved 0 |
| G2b — Gemmini config caching | gemmini kernel time | ≤ 20 ms (gemmini total is 109) |
| G2c — atomic-flag signaling | sync_overhead | ~0.2 ms (sync is 350 µs) |
| G2d — fanout signaling | dep_wait | ~0 ms (dep_wait is rvv_opu blocking) |

None of these move the needle on 786 ms wall. Updated task tracker:
G2a marked complete (already in place), G2b/c/d deleted.

## What the gap actually is

The 732 ms of kernel work on rvv_opu is dominated by **scalar
fallback kernels** that the Saturn-OPU rvv_opu lane should be
vectorizing:

```
op_kind         shape                                  per-call   ×n   total
conv2d_s8       IC=128;IH=4;IW=4;OC=128;3×3 stride-1    47.9 ms   ×2   95.9 ms
silu_s8         n=25600                                 7.1 ms   ×9   64.7 ms
silu_s8         n=51200                                14.2 ms   ×3   42.6 ms
silu_s8         n=102400                               29.3 ms   ×1   29.3 ms
batchnorm2d_s8  N=1;C=16;H=80;W=80                      8.2 ms   ×1    8.2 ms
conv2d_s8       IC=64;IH=7;IW=7;OC=128;3×3 stride-2    24.0 ms   ×1   24.0 ms
conv2d_s8       IC=80;IH=10;IW=10;OC=80;1×1            20.3 ms   ×1   20.3 ms
```

Inspecting `examples/yolov8_nano/int8/generated/rvv_opu/kernels.c`
confirms: the conv2d_s8 implementation is a direct sliding-window
nested loop over `int8_t`. No `__riscv_vsetvl_*`, no
`__riscv_vfmacc_*`, no `__riscv_vle8_*`. The Saturn-OPU vector unit
is sitting idle while the scalar core grinds through `for (oh) for
(ow) for (ic) ...` loops.

Same for silu_s8 — the source has no AlgorithmCandidate registered
in `pipeline/reference_kernels.py`, so the skeleton emits the
reference scalar impl every time.

The PDB has these costs accurately recorded (verified: silu_s8
n=102400 = 29.10 ms in PDB / 29.34 ms in v9, within 1%). So **the
solver chose this placement knowing the costs**. That's a separate
investigation, tracked under #253 (G3 — verify what cycle source
the solver actually consumed; the policy may have used a different
PDB than the FireSim one).

## Where the leverage actually lives

Ranked by approximate wall-clock reduction:

| Fix | Expected save | Path |
|---|---|---|
| RVV-vectorize silu_s8 (3 shapes) | ~100-130 ms | new AlgorithmCandidate + Bedrock or hand-roll |
| RVV-vectorize / OPU conv2d_s8 (3 shapes) | ~140 ms | bigger lift — proper OPU outer-product kernel |
| Solver replays w/ measured PDB (G3) | unknown | could rebalance off rvv_opu OR confirm 70 ms was based on different cycles |
| Gemmini config caching (G2b) | ≤ 20 ms | small relative to above |

The Phase G plan in the repo plan file has been updated to reflect
this re-ranking. The instrumentation itself (G1) was correct and
load-bearing — it told us where to actually point the optimization.

## Artifacts

- `run.log` — full FireSim run with HART_ACC block and per-net
  records dump
- `breakdown.json` — parsed per-hart attribution and per-network
  walls + correctness
- This `REPORT.md`

## Reproducing

```bash
cd /scratch2/agustin/ModelBlaster
source scripts/setup_benchmark_env.sh
export PYTHONPATH=/scratch2/agustin/ModelBlaster/src
export SCHEDULE_JSON=/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json
export MODELS=mlp_control,dronet,yolov8_nano
export REGISTRY=/scratch2/agustin/ModelBlaster/cores/chipyard_gemmini_opu_hetero.json
export BACKENDS=gemmini_q31,rvv_opu CPU_P_KIND=gemmini CPU_E_KIND=rvv_opu
export RUNNER=firesim FIRESIM_QUEUE=1 MAX_ACCURACY_CLASS=bit_exact
export QUANT=int8 FORCE_REGEN=0
bash examples/xpurt_demo/run.sh

python3 scripts/parse_runtime_breakdown.py \
    artifacts/runtime_optimization/v9_baseline_instrumented/run.log
```
