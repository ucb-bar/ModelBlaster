# ROS Baseline for the QRB5165 Heterogeneous-Scheduling Comparison

## Why this baseline

Reviewers want ROS as the apples-to-apples comparison: a baseline that
matches what robotics teams actually write today (one ROS node per
inference network, periodic timer per node, network-API call for QNN).
Our MILP-scheduled runtime should beat it convincingly — that's the
whole pitch — but we need the comparison numbers, not just argue the
theory.

The expected story: a naive ROS deployment with three nodes, no
coordination, each picking a backend independently, will hit at least
one of the contention failure modes we already saw on hardware
(CPU-saturation, DSP-pile-up, missed deadlines for the tight 2 ms MLP
period). Our MILP-emitted runtime puts each network on its best lane
with explicit pre-warmup, and all three lanes run in parallel.

## Plan

### 1. Environment

QRB5165 has only stub vendor pieces of ROS2 Foxy under `/opt/ros/foxy/`
— no `rclcpp`, no full distro. Three options, pick one before building:

| Option | Pros | Cons |
|---|---|---|
| **A.** Install ROS2 Humble (or Jazzy) from `apt` on the board | Standard, what reviewers expect | Older Ubuntu on the board may not have packages; may need source build |
| **B.** Build ROS2 from source (`colcon`) on the board | Works regardless of distro | 2-3 hour build, ~10 GB |
| **C.** Run ROS2 on the host, push timer events via TCP/UDP to a board-side stub that calls QNN | Avoids on-board ROS install | Adds network-RTT noise; reviewers may push back ("not a real ROS deployment") |

Recommendation: **A** if the board's distro has packages, else **B**.
Worth a 10-minute check first (`apt-cache search ros-humble-rclcpp`).

### 2. Node design

One C++ node per inference network, all in one ROS package
(`ros_qnn_baseline`). Each node:

- Loads its QNN context binary at construction (`contextCreateFromBinary`
  on the chosen backend lib). One context per node, lifetime = node
  lifetime — same as the runtime we already have.
- Runs a `wall_timer` at the target frequency (`200 Hz` for dronet,
  `500 Hz` for mlp_control, periodic at e.g. `30 Hz` for yolov8n).
- Timer callback does `graphExecute` on zero-init buffers (or real input
  if pulling from a publisher), records `now()` before/after, publishes
  the output tensor on a `std_msgs/Float32MultiArray` topic.
- Pre-warmup: optional. Default to ON (matches our MILP runtime) so the
  first instance doesn't blow the deadline; can A/B with OFF to show how
  quickly things fail without it.

Node skeleton (C++):

```cpp
class DronetNode : public rclcpp::Node {
public:
    DronetNode() : Node("dronet_node") {
        bringup_qnn_("/root/qnn_runtime_ctx/ctx_dronet_full_seg0__Hta.bin",
                     "libQnnHta.so");
        warmup_(2);
        pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
            "/dronet/out", 10);
        timer_ = create_wall_timer(5ms, [this]{ tick(); });
    }
    void tick() {
        auto t0 = now();
        QnnGraphExecute(graph_, ...);
        auto t1 = now();
        // record (t0, t1) into a per-node CSV; publish result.
    }
};
```

Same pattern for `MlpControlNode` (CPU lib) and `Yolov8Node` (DSP lib,
runs the segmented two-dispatch chain — see §4).

### 3. Backend assignment

Two scenarios — both interesting, run both:

- **B1 — naive (default-DSP):** every node uses `libQnnDsp.so`. Mirrors
  "the dev didn't think about backends, used what worked first."
  Expected outcome: yolov8 hogs DSP for ~33 ms each call, 2 ms MLP
  deadline gets crushed.
- **B2 — best-isolated-backend per node:** each node uses the backend
  that wins in isolation: dronet→HTA, yolov8→DSP, mlp→CPU. Mirrors
  "the dev picked the obvious lane per network, but no global plan."
  Expected outcome: backends don't directly contend, but ROS thread
  scheduling jitter causes occasional misses; cold-start spikes if
  warmup is off.

(B2 is essentially what our MILP picked, minus the explicit period
sequencing. The remaining gap measures what global scheduling
contributes vs lane assignment.)

### 4. Sub-DLC inputs

Reuse what's already on the board:

- `ctx_dronet_full_seg0__{Hta,Dsp,Cpu}.bin` (whole-network HTA-friendly
  variant, dronet_full_hta build)
- `ctx_yolov8n_HTA_split_seg{100,101}__{Dsp,Cpu}.bin` (2-split, the
  sweet-spot granularity per `HETEROGENEOUS_SCHEDULING_QRB5165.md`)
- `ctx_mlp_control_full_seg0__{Dsp,Cpu}.bin`

For yolov8 specifically: the node has to chain two `graphExecute` calls
(backbone → head, with the boundary tensors as a producer-consumer
hand-off). Easiest path: keep both contexts loaded in one node, run
them back-to-back in the same callback. Don't try to publish the
intermediate over a topic — that adds unrelated DDS noise that dwarfs
the actual work.

### 5. Measurement

Per-callback CSV from each node, columns:
`{seq, callback_start_us, exec_start_us, exec_end_us, callback_end_us}`.
Post-process to compute:

- per-instance latency = `exec_end - exec_start`
- per-instance deadline slack = `(seq+1)*period - callback_end`
- jitter = stddev of `callback_start - seq*period`
- deadline miss rate over a fixed window (e.g. 2 s of run time)

Plus one global trace from `ros2 bag` of all topics so we have a
ground-truth timeline for the gantt-style comparison plot. The plot
uses the same darkened xpurt palette as the MILP gantts so the
comparison reads cleanly side-by-side.

For the side-by-side: same 2 s wall-time window as the MILP run, same
networks, same backends (in B2). Plot two gantts vertically — top is
MILP, bottom is ROS — same x-axis.

### 6. Predicted outcomes (a sanity bound)

If the model holds and there's no Pareto improvement we missed:

| Scenario | dronet@5ms misses | mlp@2ms misses | yolov8 latency | What kills it |
|---|---|---|---|---|
| MILP (current) | 0 / ~7 | 0 / ~17 | ~33 ms | n/a |
| ROS B1 (all on DSP) | most | almost all | ~62 ms (whole) | DSP serial |
| ROS B2 (split lanes) | maybe a few | 1-3 cold-start | ~33 ms | thread jitter, no warmup |

If ROS B2 turns out to *match* MILP, that's also a useful negative
result — it'd say "the explicit MILP wasn't needed; lane assignment is
what actually matters." Worth being honest about it either way.

### 7. Non-goals / scoping

- No real ROS DDS workload (camera input, image preprocessing). Use
  zero-init or pre-loaded sample inputs same as the runtime — we want
  apples-to-apples on the inference-call cost, not the perception
  pipeline.
- No QoS tuning beyond defaults; reviewers want to see what people get
  out of the box, not what an expert can squeeze out.
- No multi-machine ROS, no DDS-over-network. Localhost only.
- No power measurement (separate ask if it comes up).

### 8. Risks worth surfacing now

1. **QNN backend handle sharing across processes.** Each ROS node is
   its own process, each calling `backendCreate` on the same backend
   lib. We hit "Context handle already exists!" on HTA when two threads
   in *one* process tried that; across processes via FastRPC it should
   work, but worth verifying with a 30-line POC before committing to
   the full plan. If it doesn't, fall back to component composition
   (multiple nodes in one process, shared backend handle — basically a
   thin wrapper over the runtime we already have).
2. **ROS install on board.** If neither A nor B works in a reasonable
   timeframe, fallback C (host-side ROS, board-side stub) is the
   compromise. Note in the writeup that the host hop is a real source
   of noise, not just lab convenience.
3. **Cold-start fairness.** If we run MILP with warmup ON and ROS with
   warmup OFF, that's not apples-to-apples. Run both with warmup ON for
   the headline comparison, then a "cold-start" panel as a separate
   point ("MILP can pre-warm because it knows the schedule; ROS doesn't,
   so the first frame after launch is gated by the slowest cold path").
4. **Timer drift.** `rclcpp::create_wall_timer` runs in the executor's
   callback thread. With a `SingleThreadedExecutor` (default), three
   timers in one process serialize. Use `MultiThreadedExecutor` and
   put each timer in its own callback group, or run nodes in separate
   processes. Document which we picked.

### 9. Concrete deliverable list

| File | Purpose |
|---|---|
| `qnn_models/runtime/ros_baseline/CMakeLists.txt` | colcon package |
| `qnn_models/runtime/ros_baseline/src/dronet_node.cpp` | dronet ROS node |
| `qnn_models/runtime/ros_baseline/src/mlp_control_node.cpp` | mlp ROS node |
| `qnn_models/runtime/ros_baseline/src/yolov8n_node.cpp` | yolov8 ROS node (chains backbone+head) |
| `qnn_models/runtime/ros_baseline/launch/all_nodes.py` | launch all three with B1 vs B2 backend choice via param |
| `qnn_models/runtime/ros_baseline/scripts/run_and_collect.sh` | run for 2s, gather CSVs + bag, scp back |
| `qnn_models/runtime/plot_ros_vs_milp.py` | side-by-side gantt + deadline-miss table, dark-palette consistent |
| `plots/qrb5165_ros_b1_vs_milp.png` | output |
| `plots/qrb5165_ros_b2_vs_milp.png` | output |

### 10. Sequencing — minimum viable

Suggested order so we can bail out early if something looks broken:

1. **30-min POC**: one C++ binary that creates two QNN contexts in
   separate processes (no ROS yet) and sees if they coexist on the
   same backend lib. Validates risk #1.
2. **ROS install** (option A → B → C). Verify `ros2 run demo_nodes_cpp
   talker` works.
3. **Single-node build**: dronet node only, B2 backend (HTA), no MLP/
   yolov8 yet. Run for 2 s, dump CSV. Eyeball deadlines.
4. **Add the other two nodes**, B2 first. If that already shows a
   deadline blowup (mlp at 2 ms is the most fragile), we have the
   "ROS scheduling jitter is the problem" headline immediately.
5. **B1 (all-DSP)** as the punchline — expected to be worse than B2,
   shows the "no lane assignment" cost.
6. **Plot + writeup**: side-by-side gantt, deadline-miss table, narrative.

If step 1 fails, pivot to "in-process multi-node baseline" (component
composition) and note the limitation honestly in the writeup.

## Open questions for the user

- ROS install path: try **A (apt humble)** first, fall back to **B
  (source build)** if packages aren't there?
- Is yolov8n in this baseline periodic (e.g. 30 Hz) or one-shot?
  The MILP comparison treated it as non-periodic / runs-once-per-window
  — if ROS runs it periodically that's a fair load increase but changes
  the workload definition.
- Do we want a CPU contention scenario (B1') where everything runs on
  CPU? It's a bit synthetic but reviewers love seeing the worst case.
