# Evidence: agentic loop end-to-end

Concrete evidence each piece of the XPU-RT ⇄ ModelBlaster iterative
scheduling loop works. Each row points at a reproducible artifact or
command.

## 1. XPU-RT predicted side (axes A, B, C decisions)

| claim | evidence |
|-------|----------|
| Iterative driver produces a candidate bundle from a baseline | `/scratch2/agustin/XPU-RT/artifacts/iterate/firesim_batch.json` (Contract 1 — 8 candidates: A1-A4 scheduler, B1-B3 backend, C1 fusion). |
| Advisor diagnoses granularity / bottleneck / deadline | `xpu-rt/advisor.py --report scheduled_..._report.json` on the baseline returns `granularity_verdict=too_fine`, `bottleneck_backend=CPU_E#0`, `recommendations=[coarsen at fusion_threshold=1000 cycles]`. |
| Axis-C decision (fuse) is emitted as Contract-2 hint | `granularity_loop.py --emit-hint` writes `/scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json` with `fuse_groups` for mlp_control / dronet / yolov8_nano. |
| Predicted Gantts per candidate | `artifacts/bundle/baseline/predicted_gantt.png` (75.57 ms), `A2/predicted_gantt.png` (54.43 ms, the winner), `predicted_greedy_gantt.png` (60.48), `predicted_peft_gantt.png` (56.24), `predicted_edf_gantt.png` (116.78), `partition_gantt.png` (61.20 — the greedy_periodic *partition* schedule shape). |
| Predicted before/after composite | `/scratch2/agustin/XPU-RT/artifacts/iterate/before_after_gantt.png`. |

Reproduce: `cd /scratch2/agustin/XPU-RT && bash scripts/demo_iterate_firesim.sh`.

## 2. ModelBlaster IR rewrite (Phase 1a)

| claim | evidence |
|-------|----------|
| Pure IR transform — collapses fuse_groups into `__fused__` ops | `pipeline/apply_fusion_hint.py`; 11/11 unit tests in `pipeline/tests/test_apply_fusion_hint.py`. |
| Works on the actual XPU-RT hint | `python3 -m pipeline.apply_fusion_hint --hint .../granularity_hint.json --model mlp_control ...` collapses `mlp_control` from 7 dispatches to 2 (fused chain [0..5] + trailing op 6). Output stamp in `artifacts/bundle/axis_c_spike/graph.fused.json`. |
| Disjoint groups + topological order + branch-out detection | Test cases `test_overlapping_groups`, `test_out_of_order`, `test_intermediate_consumed_outside` cover the rejection logic. |

Reproduce:
```bash
python3 -m unittest pipeline.tests.test_apply_fusion_hint
python3 -m pipeline.apply_fusion_hint \
    --hint /scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json \
    --model mlp_control \
    --ir   examples/mlp_control/int8/generated/graph.json \
    --out  /tmp/graph.fused.json
```

## 3. ModelBlaster codegen for fused ops (Phase 1b)

| claim | evidence |
|-------|----------|
| Kernel picker emits all sub-op kernels | `pipeline/generate_kernels.py` `_expand_op_kinds` walks `__fused__` op.sub_ops; pre-existing tests still green. |
| Skeleton emits ONE fused dispatcher per fused op | `pipeline/generate_skeleton.py` adds the `__fused__` branch + `_emit_sub_op_call` helper; non-fused path untouched. The generated C body chains sub-kernel calls back-to-back inside a single `dispatch_<mid>_<id>` function. |
| Generated C compiles | `examples/mlp_control/int8/build/rvv_opu/zephyr/zephyr.elf` links cleanly under `west build`. |

## 4. ModelBlaster axis-C end-to-end on spike (Phase 1c)

| claim | evidence |
|-------|----------|
| Fused mlp_control passes bit-exact verify | `artifacts/bundle/axis_c_spike/spike_output.log` — `=== MODELBLASTER_VERIFY === max_abs_err=0 max_rel_err=0 n=4`. |
| Per-dispatch profile reflects the fusion | Same log: `dispatch_id=0 op=__fused__linear_s8__elu_s8__linear_s8__elu_s8__linear_s8__elu_s8 shape=fused(6) cycles=726023 (99.4%)` + `dispatch_id=1 op=linear_s8 cycles=4036 (0.6%)`. |

Reproduce:
```bash
# Apply hint (backup first)
cp examples/mlp_control/int8/generated/graph.json /tmp/orig.bak
python3 -m pipeline.apply_fusion_hint \
    --hint /scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json \
    --model mlp_control \
    --ir   examples/mlp_control/int8/generated/graph.json \
    --out  examples/mlp_control/int8/generated/graph.json

# Build + run on spike
RUNNER=spike TARGET=rvv_opu QUANT=int8 bash -c '
  source scripts/setup_benchmark_env.sh
  uv run bash examples/mlp_control/run.sh
'

# Restore
cp /tmp/orig.bak examples/mlp_control/int8/generated/graph.json
```

## 5. Bundle driver + measured-report adapter (Phase 2)

| claim | evidence |
|-------|----------|
| Resolves all 8 candidates from Contract-1 | `bash scripts/run_bundle_firesim.sh --batch ... --dry-run --include baseline,A1,A2,A3,A4,B1,B2,B3,C1` exits 0; manifest shows fixtures for A1-A4 + baseline + B3 resolved, B1/B2 `missing-fixture`, C1 `skipped` (now: skipped pending the hint-application step in the driver — Phase 1b kernel/codegen is in). |
| Trace extraction from uartlog works | `scripts/run_xpurt_bundle.py:_extract_trace` parsed the partial baseline uartlog from the earlier 1hr-timeout run and produced `artifacts/bundle/baseline/xpurt_trace.csv` (200 dispatches; first ~8.7% of the 75.6ms schedule). |
| Trace → measured `SchedulerReport` adapter works | `artifacts/bundle/baseline/measured_report.json` produced by `scripts/emit_measured_report.py` from that partial trace. |
| Advisor accepts measured report | `artifacts/bundle/baseline/measured_advice.json` shows the advisor ran on the measured report — `granularity_verdict=too_fine`, `recommendations=[coarsen]`, matching the predicted-side verdict. |

## 6. Skills (Phase 3)

| claim | evidence |
|-------|----------|
| ModelBlaster `/realize-hint` and `/realize-and-run` | `.claude/skills/realize-hint/SKILL.md`, `.claude/skills/realize-and-run/SKILL.md`. |
| XPU-RT `/close-loop` | `/scratch2/agustin/XPU-RT/.claude/skills/close-loop/SKILL.md`. |

The skills wrap the underlying scripts (`run_xpurt_bundle.py`,
`close_xpurt_loop.py`, `demo_iterate_firesim.sh`); see the SKILL.md
files for the runbook each one drives.

## 7. End-to-end measured loop (Phase 4)

**Round 1 — partial.** `artifacts/bundle/round1_report.md` covers
the predicted side completely (6 Gantts, advisor verdict) plus the
baseline measured prefix (200 of 388 dispatches before the
firesim-queue default 1hr timeout). A re-capture with
`FIRESIM_QUEUE_TIMEOUT=14400` is currently running (see
`artifacts/bundle/longrun/`) — once it finishes,
`close_xpurt_loop.py` will produce full predicted-vs-actual
Gantts for both baseline and A2, plus the advisor re-run on the
complete measured numbers. The mechanism is the same that already
produced the partial-trace artifacts; the only thing changing is
the trace coverage.

**Round 2** isn't run yet — the loop is one-round-at-a-time. The
spec for round 2: feed the measured `measured_advice.json` back to
XPU-RT's `bundle.propose_bundle`, generate a Round-2
`firesim_batch.json`, repeat. With the longrun trace in hand and
axis-C now realizable (Phase 1b/1c done), Round 2 should include a
fused-mlp_control candidate the harness can actually run.
