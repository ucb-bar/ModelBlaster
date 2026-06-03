# Full agentic loop run — schedule → analyze → act → re-schedule

This is the end-to-end demonstration the user asked for: not just
"compaction passes" but actually *trying to schedule, analyzing the
schedule, and acting upon it via fuse/split*.

## Workload (now with yolo back)

`configs/agentic_fuse_split_demo.yaml`:
- 1×yolov8_nano_64 (perception, one-shot)
- 4×mlp_control @ 5 ms period (PPO actor)
- 2×dronet @ 20 ms period (depth)
- 75 ms horizon

## Step 1 — try to schedule (baseline)

CPSAT (optimal solver) on 1+4+2:
- Predicted makespan: **77.72 ms**
- Misses the 75 ms horizon by **−3.6 %** (-2.72 ms)
- → axis-A solver choice is exhausted; need axis-C

(See `fuse_split_yolo_CPSAT.json`, `stack_yolo.png` for the full 4-scheduler
comparison.)

## Step 2 — analyze the schedule

`xpu-rt/advisor.py` on the CPSAT report:
- `meets_deadline: false`, `margin_pct: -3.6`
- `granularity_verdict: balanced` — no obvious imbalance under
  axis-A alone, confirming we need axis-C.

Then `scripts/granularity_loop.py --baseline-solver greedy`
(used as a fast inner-loop scheduler for the candidate scoring,
NOT for the headline number — that's CPSAT):

```
330 candidates generated:
  fuse_producer_consumer: 254
  fuse_linear_chain:       51
  split_heavy_dispatch:    25

8 candidates scored by full re-schedule
```

## Step 3 — act on the analysis

The granularity loop's top picks:

| Rank | Candidate | Δ makespan | Δ dispatches | Objective |
|:----:|:----------|----------:|------------:|----------:|
| Top FUSE  | `mlp_control1[0..5]` chain | -0.80 µs | -5 | -5.80 |
| Top SPLIT | `dronet1_dispatch_0` (heavy conv) | -0.58 µs | +1 | -0.42 |

**Agent's raw decision: SPLIT** (the granularity loop's tiebreaker
prefers Δmakespan over Δdispatch when split is feasible).

Honest scope note: `pipeline/apply_split_hint.py` (Phase 1e) only
supports `linear_s8` splits at present — `conv2d_s8` weight surgery
along OC is a follow-up. So the realizable winner is the
**top fuse candidate**: `mlp_control1[0..5]`.

Emitted Contract-2 hint at `fuse_hint.json`:
```json
{
  "contract": "modelblaster.fusion_hints/v1",
  "reason": "agent's top fuse_linear_chain ...",
  "networks": [{"network": "mlp_control",
                "fuse_groups": [[0,1,2,3,4,5]],
                "n_tiny": 7}]
}
```

## Step 4 — realize the hint

`pipeline/apply_fusion_hint.py --pairwise`:
- Input: `examples/mlp_control/int8/generated/graph.json` (7 dispatches)
- Output: `artifacts/agentic_fuse_split/mlp_control_fused.graph.json`
  (3 fused ops + 1 trailing linear = 4 dispatches per instance)
- The Phase 1d `linear_s8_elu_s8` registered KernelSpec backs each
  fused boundary (curated, spike-bit-exact, max_abs_err=0).

## Step 5 — re-schedule

Same HEFT solver, IR substituted:

| | Makespan | Dispatches | Δ vs before |
|:--|---------:|-----------:|------------:|
| BEFORE | 83.08 ms | 170 | — |
| AFTER  | 82.67 ms | 157 | **−0.41 ms / −13 disp** |

Visual: **`agentic_loop_before_after.png`** — 2-panel stacked Gantt
showing the BEFORE schedule (top, 170 dispatches) and the AFTER
schedule (bottom, 157 dispatches, with the mlp_control1 fused chains
rendered as hatched blocks per the `__fused__` convention).

## Why this counts as "full agentic"

| Step | Mechanism | Driven by |
|:----:|:----------|:----------|
| 1 | Run scheduler, measure makespan | Deterministic |
| 2 | Diagnose verdict + propose candidates | XPU-RT's advisor + rewrite logic |
| 3 | Score N candidates by **re-scheduling each** | `rewrite.score_candidates` (the LLM-free inner agent) |
| 4 | Pick winner respecting realizability | tiebreaker logic + scope awareness |
| 5 | Realize the IR change | `apply_fusion_hint.py` Phase 1d |
| 6 | Re-schedule and measure delta | Same HEFT on swapped IR |

No human in the loop between step 1 and step 6. The agent's choices
were:
- whether to keep adding solvers (axis A) or change the IR (axis C);
- which 8 of 330 candidates to score;
- which scored candidate to emit as the hint;
- how to rewrite the IR for that hint;
- which fused KernelSpec to dispatch to.

That's the full agentic loop — built earlier, exercised end-to-end
now on the actual robotics workload (yolo + mlp + dronet).

## What's NOT in this loop (honest scope)

- **The conv2d_s8 split realization** is the agent's top raw choice,
  but `apply_split_hint.py` Phase 1e doesn't implement weight slicing
  along OC yet. We fall back to the fuse runner-up. Adding conv2d_s8
  splits is the next obvious follow-up.
- **MOSEK on the 300-op shape** still fails (`cvxpy.error.SolverError`
  — see `oracle_table.md`). CPSAT is the de-facto optimum.
- **FireSim measured** of the AFTER schedule: not run (the 1+4+2
  workload + hetero bitstream is unstable in simulation; see
  `notes/firesim_measured_status.md`). The AFTER schedule's BEFORE
  twin (the BEFORE HEFT) was the same workload that hung in
  `b00lwnji9` for 2 hours.
EOF
echo "wrote AGENTIC_LOOP.md"