# 3-Network Micro-ROS Baseline — Build & Run Notes

## Goal
Run all three networks (yolov8_nano + dronet + mlp_control) on the micro-ROS
harness as a fixed-pinning baseline comparison to the xpurt scheduler.

**Pinning Config A**: yolov8 → gemmini/hart0, dronet → rvv/hart1,
mlp_control → gemmini/hart0.

## Changes Made

### 1. `agents/harness_microros/prj.conf`
```diff
-CONFIG_MICROROS_NODES="2"
-CONFIG_MICROROS_PUBLISHERS="2"
+CONFIG_MICROROS_NODES="4"
+CONFIG_MICROROS_PUBLISHERS="4"
```
**Why**: The `libmicroros.mk` Makefile's `configure_colcon_meta` target reads
Kconfig values via `update_meta_from_zephyr_config` and overwrites whatever is
in `colcon.meta`. With MAX_NODES=2, the 3rd `rclc_node_init_default` call
failed because the global node pool was full. Setting to 4 allows up to 4
simultaneous ROS nodes.

### 2. `agents/harness_microros/src/main.c` — NET_C support (MICROROS_3NET)
- Added `#include "net_c_includes.h"`, dispatch table extern, output buffer,
  state struct, thread stack, thread struct for net_c.
- Added `run_graph_c()` dispatch loop (mirrors `run_graph_a` pattern).
- Added `timer_cb_c()` for rclc executor.
- Added `net_a_ready` atomic for serialized init (broker can't handle
  simultaneous CREATE handshakes).
- Init order: net_b first → sets `net_b_ready` → net_a proceeds → sets
  `net_a_ready` → net_c proceeds.
- Thread creation/start/join for net_c gated by `#ifdef MICROROS_3NET`.
- `irq_lock()` around `run_graph_a` and `run_graph_c` to prevent V-state
  context-switch corruption (same as existing `run_graph_b` protection).

### 3. `agents/harness_microros/CMakeLists.txt`
- Extended to accept 2 or 3 models.
- When 3 models: defines `MICROROS_3NET=1`, `NET_C_NAME`, `NET_C_BACKEND`,
  `NET_C_HART`, `NET_C_PERIOD_MS`.
- Generates `net_c_includes.h`.

### 4. `agents/examples/microros_demo/run.sh`
- Changed model count check from `-ne 2` to `-lt 2 || -gt 3`.
- Added `QUANTS` env variable: comma list of per-model quant modes
  (parallel to MODELS). Falls back to uniform `$QUANT` if unset.
- Updated staging loop and `MODEL_DIRS_BASE` construction to use
  `QUANT_LIST[$idx]` per model, allowing mixed fp32/int8 builds.
- Updated firesim_runner args to use `QUANT_LIST[0]`.

## How to Build & Run

### Prerequisites
```bash
cd /scratch2/dima/misc_sw/FreshScheduler/zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
```

### Full Clean Rebuild (if libmicroros needs regeneration)
Delete these under `third-party/micro_ros_zephyr_module/modules/libmicroros/`:
- `micro_ros_src/build/`, `micro_ros_src/install/`, `micro_ros_src/log/`
- `libmicroros.a`, `include/`, `configured_colcon.meta`, `zephyr_toolchain.cmake`

Then run the command below — colcon will rebuild libmicroros.a from source.

### Run Command
```bash
MODELS=yolov8_nano,dronet,mlp_control \
BACKENDS=gemmini_q31,rvv \
PIN_BACKENDS=gemmini_q31,rvv,gemmini_q31 \
PIN_HARTS=0,1,0 \
PERIODS_MS=0,20,10 \
QUANTS=int8,int8,fp32 \
RUNNER=firesim \
FORCE_REGEN=0 \
MICROROS_NO_PUBLISH=1 \
FIRESIM_TIMEOUT=900 \
bash agents/examples/microros_demo/run.sh
```

### Env Variables Explained
| Variable | Value | Meaning |
|---|---|---|
| MODELS | yolov8_nano,dronet,mlp_control | Networks in pin order |
| PIN_BACKENDS | gemmini_q31,rvv,gemmini_q31 | Backend per network |
| PIN_HARTS | 0,1,0 | Hart assignment per network |
| PERIODS_MS | 0,20,10 | yolov8=one-shot, dronet=20ms, mlp=10ms |
| QUANTS | int8,int8,fp32 | Per-model quantization mode |
| MICROROS_NO_PUBLISH | 1 | Mode E (no rcl_publish → broker) |
| FORCE_REGEN | 0 | Skip model code regeneration (use existing) |

## Updates (2026-05-11)

### Hart 1 serialization with rclc multi-timer executors — confirmed software issue
Comprehensive testing established the following root cause:

1. **2-net (yolov8 + dronet, separate executors)** works correctly — 91% of
   hart 1's dronet dispatches run concurrently with yolov8 on hart 0.

2. **3-net Config B** (3 executors, mlp+dronet share hart 1) shows hart 1
   essentially idle while yolov8 runs (0.4% concurrent). Adding a third
   executor or third timer triggers the serialization.

3. **Ruled out** as causes:
   - HW bus contention: Spike test (no bus contention) shows the same gap pattern.
   - yolov8's `irq_lock`: removing it changes nothing (`MICROROS_NO_LOCK_A=1`).
   - `K_FP_REGS` context-switch cost on hart 1: removing it changes nothing
     (`MICROROS_NO_FPREGS_C=1`).
   - Multi-executor competing on one hart: collapsing to one executor with
     two timers (`MICROROS_2EXEC_BC=1`) shows the same gap.

4. **Confirmed** as cause: `rclc_executor_spin_some`'s internal `rcl_wait`
   blocks on something that only unblocks when hart 0's executor activity
   stops (yolov8 finishes).
   - `MICROROS_2EXEC_NORCLC=1` (bypass rclc, raw `run_graph` loop on hart 1)
     runs hart 1 dispatches continuously concurrent with yolov8.
   - `MICROROS_2EXEC_FUSE_BC=1` (1 executor, 1 timer that runs dronet+2×mlp
     internally) restores partial concurrency (total runtime ~10.88ms vs
     15.6ms for dual-timer).

5. **Discovered side issue**: mlp_control's generated kernel writes past
   `buf_mlp_control_mlp_0` (1024 bytes) into adjacent memory. This is what
   corrupted the first ~13 KB of `ros_trace[]` (slots 0-230) and produced
   the `n=64`/`mlp.X` patterns. Verified with `ROS_TRACE_MAGIC` sentinel —
   231/1524 slots lose the magic value. Heap-allocating ros_trace exposed
   the overflow (hit a function pointer → mcause=1). The real fix is in
   mlp's generated dispatch code; for now the harness skips slots whose
   magic is missing or pointer fields don't look like .rodata.

### Knobs added during this investigation
| Knob | Effect |
|---|---|
| `MICROROS_NO_LOCK_A=1` | Remove `irq_lock` around yolov8's `run_graph_a` |
| `MICROROS_NO_FPREGS_C=1` | Create mlp's thread without `K_FP_REGS` |
| `MICROROS_2EXEC_BC=1` | Collapse dronet+mlp onto one executor/thread on hart 1 |
| `MICROROS_2EXEC_FIRE_FAST=1` | Both timer periods → 1ns (always due) |
| `MICROROS_2EXEC_NORCLC=1` | Bypass rclc entirely on hart 1 (raw `run_graph` loop) |
| `MICROROS_2EXEC_FUSE_BC=1` | One timer, callback runs `run_graph_b` once + `run_graph_c` twice |
| `MICROROS_SKIP_TRACE=1` | Skip `emit_trace_block()` entirely |
| `ROS_TRACE_MAGIC` | 64-bit sentinel in each slot to detect external clobber |

### Recommended baseline for xpurt comparison
Use **2-net (yolov8+dronet) microros** as the reference for "real microros
concurrent execution" — it's the largest configuration that demonstrably
runs hart 0 and hart 1 in parallel. For 3-net comparisons, FUSE_BC mode
(`MICROROS_2EXEC_FUSE_BC=1`) gives partial concurrency, total runtime
~10.88ms on FireSim Q31.

## Current State (2026-05-10)

### What Works
- Build completes successfully (libmicroros with MAX_NODES=4, 3-network harness).
- All 3 sessions register with the broker successfully.
- All 3 networks run to completion on FireSim Q31:
  - yolov8_nano: 406,827 cycles (1 iter)
  - dronet: 17,474 cycles per iter (30 iters, hit periodic cap)
  - mlp_control: 414 cycles per iter (20 iters)

### Current Issue: HTIF Stalls on Trace Dump
After all threads complete, the `emit_trace_block()` starts printing the CSV
trace via HTIF. After outputting the header + first line, HTIF appears to stall
(uartlog stops growing). The sim is still alive (process running) but no output
is produced.

The trace data itself is also garbled (first line shows `n=64,6446` instead of
a valid network name), suggesting the shared `ros_trace[]` buffer was corrupted
by concurrent writes from 3 threads without proper synchronization on the
trace_record path.

### Possible Fixes (TODO)
1. **Trace corruption**: Add a spinlock to `trace_record()`, or use per-thread
   trace buffers that are merged at emit time.
2. **HTIF stall**: May be related to hart0 having leftover interrupt state from
   the irq_lock pattern. Or the trace dump is simply too large — try reducing
   `ROS_TRACE_MAX` or emitting fewer entries.
3. **Alternative**: Skip the trace entirely for the baseline measurement —
   the `wall_cycles` values (printed AFTER the trace) are the primary metric.
   Could add a `MICROROS_SKIP_TRACE=1` knob that skips `emit_trace_block()`.

## Key Architecture Notes

### libmicroros Build System Flow
```
prj.conf  →  Zephyr .config  →  libmicroros.mk `configure_colcon_meta`
                                  reads CONFIG_MICROROS_NODES from .config
                                  calls update_meta_from_zephyr_config()
                                  writes configured_colcon.meta
                                  ↓
                                colcon build --metas configured_colcon.meta
                                  ↓
                                rmw_microxrcedds rebuilt with MAX_NODES=N
                                  ↓
                                libmicroros.a (all .o's merged)
                                  ↓
                                include/ (installed headers with config.h)
```

Editing `colcon.meta` alone does NOT work — the Kconfig values in `prj.conf`
override it via `update_meta_from_zephyr_config`.

### V-State Corruption
The Zephyr V-register save/restore code (`arch/riscv/core/v.c`) has a bug
where `saved_v_context.vreg` pointer can be NULL if the thread hasn't saved V
state yet. When 2+ threads share a hart and both use K_FP_REGS, a context switch
triggers `vle8.v` from a NULL pointer → mcause=5 at offset sizeof(struct fields).

Workaround: `irq_lock()` around the dispatch loops prevents preemption during
kernel execution. This is acceptable for the baseline measurement since it
measures sequential per-network execution time (not preemptive multitasking).
