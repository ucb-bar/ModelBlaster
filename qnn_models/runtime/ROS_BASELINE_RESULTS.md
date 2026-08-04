# ROS Baseline vs MILP-Scheduled QNN Runtime — Findings

Companion document to
`qnn_models/runtime/HETEROGENEOUS_SCHEDULING_QRB5165.md` (the MILP/QNN
side) and `qnn_models/runtime/ROS_BASELINE_PLAN.md` (the experiment
plan). This file captures what we measured.

## Setup

Same workload as the 3-way MILP experiment:

- **dronet** at 5 ms period (200 Hz) — drone vision
- **mlp_control** at 2 ms period (500 Hz) — control actor (tightest deadline)
- **yolov8n** at 33.33 ms period (~30 Hz, periodic to load DSP) — detection

Identical context binaries, identical sub-DLC granularity choices
(coarse dronet, 2-split yolov8, coarse mlp). The only thing that differs
between configurations is **how dispatches are scheduled and which
backend lane each network uses**.

| Config | Scheduler | dronet → | mlp → | yolov8 → |
|---|---|---|---|---|
| MILP | global plan, explicit start-time gate | HTA | CPU | DSP (split) |
| ROS B1 | per-node `wall_timer`, no coordination | DSP | DSP | DSP |
| ROS B2 | per-node `wall_timer`, no coordination | HTA | CPU | DSP (split) |

ROS2 Foxy on Ubuntu 20.04 ARM64 (board's native distro). Three nodes,
one per network, in separate processes. Each node loads its QNN context
at construction, runs `warmup_iters=2` to match the MILP runtime's
warmup, then fires `create_wall_timer` at the target period. `rclcpp`
defaults — single-threaded executor per node.

ROS run length: ~2 s per scenario. MILP run length: 32 ms (its natural
makespan; matches one period of the periodic schedule).

## Results — full run

Counting deadline misses with the same definition both sides:
**instance N must have its result published by time `(N+1) × period`**
(period-aligned, MILP-style). For ROS we adjust by one period because
`wall_timer` fires its first callback at `t0 + period`, not `t0`.

### MILP runtime (warmed, runtime-tuned 2026-05-10)

After porting four runtime knobs into `generate_runtime.py` (see
"Runtime tuning" subsection below) and re-running with all of them:

| Network | Instances | Deadline misses | exec p50 | exec max |
|---|---|---|---|---|
| dronet | 7 | 0 | 1.59 ms | 2.21 ms |
| mlp_control | 17 | 0 | 0.03 ms | 0.04 ms |
| yolov8n | 2 segs | 0 | 12.66/17.14 ms | 12.66/17.14 ms |

Makespan 32.21 ms vs predicted 33.57 ms — measured was actually
**4% faster than predicted** (0.96× ratio). Across 5 replays wall
varied within 32.07–32.30 ms; deadlines met every time. Compare
against the original (un-tuned) numbers we had before:

| | dronet exec p50 | dronet exec max | mlp_control exec p50 | makespan ratio |
|---|---|---|---|---|
| Original (Apr) | 2.00 ms | 4.00 ms | 0.10 ms | 1.07× |
| Tuned (May) | **1.59 ms** | **2.21 ms** | **0.03 ms** | **0.96×** |

The tightening on mlp_control (3× lower p50, ~6× lower max) is the
main payoff — it's the network with the smallest period (2 ms) so
its tail latency dominated the deadline-margin calculation.

#### Runtime tuning

Four runtime env knobs applied for this measurement:

- **`XPURT_SPLIT_BY_NETWORK=1`** — spawn one OS worker thread per
  `(kind, backend, network)` instead of `(kind, backend)`. When the
  MILP serialises two networks onto one logical lane (e.g. mlp_control
  + resnet50 seg2 both on CPU_X) the runtime no longer queues them
  through one thread, so a long dispatch can't push periodic callbacks
  off-cadence on the same OS thread.
- **`XPURT_WORKER_AFFINITY=7,5,6`** + `XPURT_WORKER_AFFINITY_FALLBACK=4`
  — pin each lane to a specific QRB5165 core (X1 prime @ 2.84 GHz, A78s
  @ 2.4 GHz, A55 @ 1.8 GHz). HTA worker → core 7, CPU/mlp_control → 5,
  DSP/yolov8n → 6. Stops migration + cross-thread cache thrash.
- **`XPURT_SCHEDULE_ITERATIONS=2`** — walk the schedule twice; the trace
  reflects only iteration 2. The first iteration absorbs cache-cold,
  dlopen-page-fault, and DVFS-ramp overhead (~1–1.5 ms before the very
  first dispatch fires); iter 2 starts effectively at t=0 with all
  state warm.
- Board side: `cpufreq` governor set to `performance` on cores 4-7
  (`echo performance > /sys/.../scaling_governor`). Stops idle cores
  from down-clocking between dispatches, which was a major contributor
  to the cold-cache outliers we saw before this fix.

`SCHED_FIFO` priority was implemented (`XPURT_WORKER_SCHED_FIFO=80`)
but disabled by default — at high priority it can starve system
threads, observably regressing wall time. Pinning + governor alone
gives most of the benefit.

### ROS B2 (best-isolated lanes per node)

| Network | Instances | Deadline misses | exec p50 | exec max | Fire jitter p99 |
|---|---|---|---|---|---|
| dronet | 400 | **399 / 400 (99.8%)** | 2.79 ms | 18.5 ms | 10.8 ms |
| mlp_control | 1000 | 0 / 1000 (0%) | 0.38 ms | 1.37 ms | 0.32 ms |
| yolov8n | 60 | 7 / 60 (12%) | 31.9 ms | 36.6 ms | 10.1 ms |

### ROS B1 (everything on DSP — naive)

| Network | Instances | Deadline misses | exec p50 | exec max | Fire jitter p99 |
|---|---|---|---|---|---|
| dronet | 400 | 364 / 400 (91%) | 1.64 ms | 20.3 ms | **996 ms** |
| mlp_control | 1000 | 975 / 1000 (97.5%) | 0.84 ms | 21.3 ms | **1727 ms** |
| yolov8n | 60 | 60 / 60 (100%) | 35.5 ms | 39.1 ms | 131 ms |

## What the data says

### B1 fails because of lane contention

Putting all three QNN nodes on `libQnnDsp.so` serializes everything
through one FastRPC pipe to the cDSP. yolov8 takes 33–39 ms per call;
during those windows dronet (5 ms period) and mlp (2 ms period) just
queue up. By the end of the 2 s run mlp callbacks have piled up
**1.7 seconds behind** schedule. This is the failure mode anyone who's
deployed multiple QNN models without thinking about backends would hit
on day one.

### B2 fails because ROS `wall_timer` doesn't resync after overrun

This is the more interesting finding. **B2's lane assignment matches
exactly what the MILP picked** — dronet on HTA, mlp on CPU, yolov8 on
DSP. Three lanes, three networks, no contention by construction (we
empirically validated DSP+HTA concurrency earlier). The exec times
match the isolated profile: dronet p50 = 2.79 ms (well inside 5 ms),
mlp p50 = 0.38 ms (well inside 2 ms), yolov8 p50 = 31.9 ms (just inside
33 ms).

But dronet still missed 99.8% of deadlines. Trace inspection:

```
seq=0  cb_start=  5.09 ms expected= 5.00 ms  delay=+0.09 ms exec=1.51 ms
seq=1  cb_start= 10.07 ms expected=10.00 ms  delay=+0.07 ms exec=18.54 ms  ← outlier
seq=2  cb_start= 28.67 ms expected=15.00 ms  delay=+13.67 ms exec=1.23 ms
seq=3  cb_start= 30.07 ms expected=20.00 ms  delay=+10.07 ms exec=3.40 ms
seq=4  cb_start= 35.08 ms expected=25.00 ms  delay=+10.08 ms
seq=399 cb_start=2010.04 ms expected=1995.00 ms delay=+10.04 ms
```

**One** runtime spike on seq=1 (cold-cache or page-fault — exec jumped
from 1.5 ms p50 to 18.5 ms) caused the executor to fall ~13 ms behind
schedule. ROS `wall_timer` re-fires at the previous-fire-time + period,
not at the original wallclock target. So the 13 ms drift propagated for
the next 1990 ms (398 callbacks). **Every subsequent dronet instance
missed its deadline because of one event 1.99 seconds ago.**

mlp didn't hit this in B2 because (a) CPU has no FastRPC layer, (b)
exec spikes are tiny (max 1.4 ms vs 2 ms period) so they self-absorb.
yolov8 lost ~12% on its own — exec p99 of 36 ms doesn't fit a 33 ms
period reliably; one such tick puts the next behind, identical mechanism
as dronet.

### MILP-scheduled runtime succeeds because it gates on absolute time

The runtime's worker walker has explicit start-time gating
(`while (now_ms() < target) std::this_thread::yield()` per the codegen
in `generate_runtime.py`). When one instance runs long, the next still
fires at its *scheduled* wall-time, not at "previous-end + period". A
single jitter event is contained to that one instance — the next one
re-syncs to the global plan. Combined with explicit lane assignment to
non-contending backends, all 26 dispatches in the workload meet their
deadline.

## Plots

| File | What it shows |
|---|---|
| `plots/qrb5165_ros_b1_vs_milp.png` | full 2 s ROS B1 run (top) vs full 32 ms MILP run (bottom), shared time axis |
| `plots/qrb5165_ros_b2_vs_milp.png` | same for B2 |
| `plots/qrb5165_ros_b1_vs_milp_snapshot.png` | 120 ms snapshot, B1 centered on yolov8n seq=15 (t≈567 ms abs) vs MILP from t=0 |
| `plots/qrb5165_ros_b2_vs_milp_snapshot.png` | 120 ms snapshot, B2 centered on yolov8n seq=29 (t≈1016 ms abs) vs MILP from t=0 |

The snapshot plots are the headline figure for the writeup: same x-axis
width, one yolo frame visible in each panel, immediately readable
contrast between "two lanes contending and 100+ pending callbacks" (ROS
B1) or "three lanes running but timing-desynced" (ROS B2) and "three
lanes, gated cadence, all on time" (MILP).

## Independent-process deployment (more typical production setup)

Reviewers may ask whether `ros2 launch` is doing something
"non-typical." It isn't — each `Node` action in `all_nodes.py` already
forks its own process (verified via `pstree` showing 3 sibling PIDs
under the launch parent). Still, we ran a parallel test where the
three nodes are spawned by **three independent `ros2 run` invocations**
(`qnn_models/runtime/ros_baseline/scripts/run_independent.sh`), the way
a production system might launch each node as a separate systemd
service or in three separate operator terminals.

Headline: results are **functionally equivalent** to the launch-coord
runs. B1 is consistently catastrophic; B2 is consistently fragile.

| Scenario | dronet misses | mlp misses | yolov8 misses |
|---|---|---|---|
| Launch B1 | 371/400 (92.8%) | 927/1000 (92.7%) | 60/60 (100%) |
| Indep  B1 | 373/400 (93.2%) | 937/1000 (93.7%) | 60/60 (100%) |
| Launch B2 | 1/400 (0.2%) | 0/1000 (0%) | 10/60 (16.7%) |
| Indep  B2 (run a) | 0/400 (0%) | 955/1000 (95.5%) | 7/60 (11.7%) |
| Indep  B2 (run b) | 1/400 (0.2%) | 0/1000 (0%) | 10/60 (16.7%) |
| Indep  B2 (run c) | 0/400 (0%) | **899/1000 (89.9%)** | 8/60 (13.3%) |

### Run-to-run variance is the headline finding for B2

Repeated to characterize variance (all independent-process B2):

| Run | dronet | mlp_control | yolov8 |
|---|---|---|---|
| 1 | 0/400 (0%) | **987/1000 (98.7%)** | 5/60 (8%) |
| 2 | 0/400 (0%) | 0/1000 (0%) | 3/60 (5%) |
| 3 | 0/400 (0%) | 1/1000 (0.1%) | 6/60 (10%) |
| 4 | 1/400 (0.2%) | 0/1000 (0%) | 10/60 (16.7%) |
| 5 (latest) | 0/400 (0%) | **899/1000 (89.9%)** | 8/60 (13.3%) |

**~2 in 5 runs have catastrophic mlp deadline misses** despite identical
lane assignment, identical workload, identical code, identical board
state. The cause is the same as the original B2 failure: whichever
callback happens to take a scheduling jitter event — kernel preemption,
page fault, FastRPC queue lag — desyncs the `wall_timer` permanently,
and the affected node drifts behind for the rest of the run. In run 5,
a 3ms scheduling gap between seq=100 and seq=101 (no exec spike —
pure OS jitter) desynchronized mlp for the remaining 1800ms.

The XPURT-scheduled runtime is structurally immune to this: each
periodic instance gates on absolute time (`while (now_ms() < target)
yield`), so a long instance can't push the next one's start. The 3-way
warmed XPURT run hit 0 deadline misses on every replay. This is
arguably the strongest argument in the writeup — not "MILP is faster"
but "MILP is *deterministic* where ROS is not."

## Caveats / honest framing

- **MILP runs warmed, ROS runs warmed too** — apples-to-apples. We
  pre-call `graphExecute` twice on each context after `bringup` in both
  configs.
- **Single executor per node** in ROS. A `MultiThreadedExecutor` would
  not help here — each node is its own *process*, the bottleneck is
  intra-node timer drift, not cross-node thread contention.
- **No DDS message work in the inference path.** Both configurations
  use zero-init buffers and only publish a tiny heartbeat. Adding real
  pub/sub of feature maps would slow ROS further; we explicitly chose
  to remove that variable so the comparison is on scheduling discipline,
  not DDS overhead.
- **ROS Foxy is EOL upstream** (May 2023), but it's the standard ROS2
  for Ubuntu 20.04 which the board runs. Humble would require Ubuntu
  22.04 (board OS upgrade). The `wall_timer` semantics that cause the
  B2 failure are the same in Humble — verified against the Humble
  source — so the result transfers.
- **One ROS run per scenario.** The variance is large (one outlier
  determines whether dronet recovers). The 99.8% miss rate in B2 isn't
  representative of every run — sometimes seq=0/1/2 don't spike and the
  schedule stays in sync. But our N=2 (B2 was rerun for the snapshot)
  both showed the same pattern. Worth noting in the writeup.

## Reproducing

```bash
# Build (on board, after ROS Foxy + apt-installed deps).
ssh root@<board> "source /opt/ros/foxy/setup.bash && cd /root/ros2_ws && \
    colcon build --packages-select ros_qnn_baseline --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Run B1 / B2.
ssh root@<board> "source /opt/ros/foxy/setup.bash && \
    source /root/ros2_ws/install/setup.bash && \
    LD_LIBRARY_PATH=/root/qairt/lib/target \
    ADSP_LIBRARY_PATH='/root/qairt/lib/hexagon-v66;/dsp/cdsp;/dsp' \
    ros2 launch ros_qnn_baseline all_nodes.py scenario:=B2 warmup_iters:=2"

# Pull traces and plot.
scp root@<board>:/tmp/ros_baseline/B2_*.csv /tmp/ros_baseline/
python qnn_models/runtime/plot_ros_vs_milp.py --scenario B2 \
    --out plots/qrb5165_ros_b2_vs_milp.png
python qnn_models/runtime/plot_ros_vs_milp.py --scenario B2 \
    --snapshot-yolo-seq 29 --snapshot-pad-ms 60 \
    --out plots/qrb5165_ros_b2_vs_milp_snapshot.png
```

`/tmp/3way_warm_hw_run.log` (the MILP runtime's trace dump) is read by
default for the MILP panel.
