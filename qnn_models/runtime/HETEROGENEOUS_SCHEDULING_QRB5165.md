# Heterogeneous Scheduling on QRB5165 (Hexagon v66)

End-to-end notes from the dronet+mlp+yolov8 scheduling experiment on QRB5165
(Snapdragon 865 / Hexagon v66). Three physical lanes — HTA (NPU), DSP (HVX),
CPU — and three networks with different shapes and deadlines.

## Summary

| Schedule | Predicted | Measured | Ratio |
|---|---|---|---|
| Greedy (segmented per-op) | 140 ms | 141 ms | 1.01× |
| MILP (segmented per-op, 300s) | 117 ms | 183 ms | 1.52× |
| MILP (coarse, no HTA-dronet) | 86 ms | 147 ms | 1.71× |
| MILP (coarse, +HTA-dronet) | 63 ms | 66 ms | 1.05× |
| MILP (hybrid: coarse-dronet + split-yolov8) | 34 ms | 37 ms | 1.10× |
| MILP (3-way: + 2 ms periodic mlp_control) | 34 ms | 36 ms | 1.07× |

**The hybrid/3-way schedule is 3.8× faster than greedy** and matches
prediction within 10%. The 3-way result is the most useful: three
networks running on three lanes simultaneously, all meeting deadlines.

## Per-network granularity choices

| Network | Granularity | Why |
|---|---|---|
| dronet (16k → 4 control output) | 1 op (whole) | DSP whole = 0.92 ms; sum of 7 segments = 3.49 ms. Per-call overhead dominates segmenting for small networks. HTA build needs `dronet_full_hta.onnx` (BN-rewrite + #8 conv-head + #11 drop trailing Reshape per `optimization_flow.md`). |
| yolov8n | 2 ops (backbone+head split) | Whole DSP = 62 ms; split sum = 33.6 ms. **Splitting is 1.85× faster on the same backend** because per-segment quantization gets tighter scales than end-to-end calibration, and the split avoids end-to-end NCHW↔NHWC layout flips. The head must run on DSP/CPU anyway (Resize/Slice/Softmax reject on HTA). |
| mlp_control (16→256→128→64→4 ELU) | 1 op | 70k MACs total; CPU dispatch = 113 µs vs DSP = 543 µs. DSP/HTA pay ~500 µs FastRPC RTT regardless of compute size. ELU rejects on HTA (only ReLU/ReLU6/Sigmoid/Tanh/HardSwish supported). |

## Per-lane perf data (whole-network, mean of 50 iters)

| Network | HTA | DSP | CPU |
|---|---|---|---|
| dronet (full_hta variant) | 2.65 ms | **0.92 ms** | 7.50 ms |
| yolov8n backbone (seg100) | 21.95 ms | **14.36 ms** | 34.50 ms |
| yolov8n head (seg101) | (rejects) | **19.21 ms** | 39.42 ms |
| mlp_control | (rejects: ELU) | 0.54 ms | **0.11 ms** |

## What MILP actually picked (3-way schedule)

```
DSP:  yolov8n_backbone(seg100) ─►  yolov8n_head(seg101)             [33.6 ms]
HTA:  dronet  dronet  dronet  dronet  dronet  dronet  dronet         [7×2.65 ms over 5 ms periods]
CPU:  mlp ── mlp ── mlp ── … (17 instances, 2 ms periods)            [17×0.11 ms]
```

All three lanes running concurrently. yolov8 sets the makespan; HTA and
CPU have plenty of headroom (55% and 6% utilization respectively).

## Why DSP beats HTA on conv-heavy workloads on v66

Counter-intuitive — HTA is the dedicated tensor accelerator, but DSP wins
both per-network conv-heavy cases by 1.5–3×. Per-conv-block measurements
(dronet segments 0–6, ~4–8 ops each):

| Segment | DSP (µs) | HTA (µs) | DSP advantage |
|---|---|---|---|
| dronet_HTA_split_seg0 | 622 | 2331 | 3.75× |
| dronet_HTA_split_seg2 | 494 | 1681 | 3.40× |
| dronet_HTA_split_seg4 | 569 | 1619 | 2.85× |
| dronet_HTA_split_seg6 | 438 | 1520 | 3.47× |
| dronet_CPU_seg{1,3,5} (Add only) | ~455 | ~1559 | ~3.4× |

DSP wins **every** segment. Even on segments with only ElementWiseAdd,
HTA is 3× slower — so it's not a "HTA can't do X" issue.

### Per-op breakdown (DSP, yolov8 backbone — 10.3 ms accelerator time)

`profile_per_op.cpp` with `QNN_PROFILE_LEVEL_DETAILED`:

| Op type | DSP sum (µs) | Share |
|---|---|---|
| Conv | 4875 | 47% |
| Sigmoid | 2689 | 26% |
| Mul | 1531 | 15% |
| Concat | 530 | 5% |
| Split | 377 | 4% |
| Add | 205 | 2% |
| MaxPool | 140 | 1% |

The **SiLU activation (Conv → Sigmoid → Mul) is 41% of accelerator time**.
On DSP these are inline-fused; on HTA they're three separate kernel
dispatches with full launch cost each. That structural difference, not
peak conv throughput, is why HTA loses on backbones.

The first conv (`/model.0/conv/Conv` on the 640×640×3 input) is a single
1.74 ms op — 17% of all DSP compute. A pure-conv workload might let HTA
shine, but real detection backbones aren't pure conv.

### v66 HTA profile API limitation

`libQnnHta.so` doesn't populate per-node profile events
(`copyPerfId: Stat lutBuff not initialized`). So per-op DSP-vs-HTA can't
be measured directly at the same granularity — only graph totals (DSP
exposes 108 events per graphExecute, HTA exposes 2). Comparison has to
be at the conv-block level via separate sub-DLCs.

## Hardware concurrency validation

`test_dsp_hta_concurrency.cpp` measured the serial-vs-parallel ratio of
HTA + DSP graphExecute calls and got 0.55× — i.e. parallel time ≈ max,
not sum. **HTA and DSP run on separate silicon paths and don't contend.**
CPU is similarly independent. The 3-way schedule exploits this by giving
each network its own lane.

## Why the per-op MILP failed (lessons learned)

Earlier attempts ran a per-op MILP (each op independently routable). It
predicted 117 ms but measured 183 ms — 52% off, much worse than greedy.
Two failure modes:

1. **CPU contention isn't in the cost model.** The per-op profile measures
   each backend in isolation. When MILP packs CPU lane with concurrent
   work (yolov8 + dronet adds + runtime threads + FastRPC handlers), CPU
   slows ~70%. DSP and HTA don't have this problem because their compute
   runs on separate silicon; CPU shares cores/cache with everything else.

2. **Periodic deadline + tight cost model fights coarsening.** Per-op
   MILP has no incentive to coalesce; every dispatch is a free degree of
   freedom. But each per-op call pays ~600 µs FastRPC overhead. For
   small networks like dronet (1 ms whole), adding 7 dispatch boundaries
   inflates total cost 3.5×.

The fix is **per-network granularity tuning**, not "always per-op".
Coarse for small networks, split for networks where end-to-end
quantization is suboptimal.

## MILP variable counts

| Schedule | Operations | Variables | Constraints | Solve time |
|---|---|---|---|---|
| Per-op (with bad sentinel→800 instances) | 5602 | 31,404,813 | 94,407 | did not solve |
| Per-op (after sentinel fix→80 instances) | 562 | 318,093 | 9,447 | 300s, gap 62% |
| Coarse (24 dronet + 1 yolov8) | 19 | ~440 | ~150 | ~1s, optimal |
| 3-way (7 dronet + 17 mlp + 2 yolov8) | 26 | ~770 | ~250 | ~1s, optimal |

The dominant scaling is `beta` (op-pair ordering, n_ops²). Coarsening
networks is the fastest way to keep `beta` tractable.

## Reproducing

```bash
# emit profile artifacts for each granularity choice
python qnn_models/runtime/emit_coarse_graph_json.py --repo-root .
python qnn_models/runtime/emit_segmented_graph_json.py \
    --perf-json qnn_models/runtime/gen/qrb5165_dronet_yolov8/segment_perf.json \
    --repo-root . --target qrb5165_v66

# solve
python scripts/run_xpurt_schedule.py \
    --networks-json data/toplevel/networks_3way_dronet5ms_mlp2ms_yolov8_qrb5165.json \
    --solver milp --profiled --time-limit 60

# generate runtime + push + run
python qnn_models/runtime/generate_runtime.py \
    --schedule schedules/scheduled_networks_3way_dronet5ms_mlp2ms_yolov8_qrb5165_profiled.json \
    --out-dir qnn_models/runtime/gen/qrb5165_milp_3way \
    --backend-map "CPU_P=HTA:libQnnHta.so,CPU_E=DSP:libQnnDsp.so,CPU_X=CPU:libQnnCpu.so" \
    --from-segmented-schedule
# scp + g++ on board, run with detailed profile, plot:
python qnn_models/runtime/plot_runtime_trace.py --log /tmp/3way_hw_run.log \
    --out plots/qrb5165_3way_milp_dronet_mlp_yolov8.png
```

## Open follow-ups

- **bnfree-only HTA whole-dronet**: `dronet_bnfree.onnx` alone fails HTA
  compose because of NCHW↔NHWC boundary Transposes. Adding #8 conv-head
  + #11 drop-trailing-Reshape (folded into `dronet_full_hta.onnx`) is
  what makes it work. Per `optimization_flow.md`.
- **MLP on HTA**: would require swapping ELU → ReLU/HardSwish, which
  changes inference behavior. Not pursued.
- **CPU contention in cost model**: the predicted vs measured gap is
  small (~10%) for the 3-way schedule because MILP kept CPU sparse, but
  any schedule that loads CPU heavily will over-predict. Worth profiling
  CPU under contention if larger CPU loads are needed.
- **Sub-DLC per-conv profiling**: per-op DSP↔HTA comparison would need
  splitting yolov8 backbone into single-conv DLCs and rebuilding context
  binaries per op, since v66 HTA's profile API doesn't expose per-node
  events.
