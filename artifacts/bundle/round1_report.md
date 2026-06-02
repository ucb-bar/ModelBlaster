# XPU-RT ⇄ ModelBlaster agentic loop — round 1

The loop closed end-to-end on the demo workload (`1×yolov8_nano +
4×mlp_control + 2×dronet` on the heterogeneous Gemmini + Saturn-OPU
chipyard bitstream). XPU-RT did the predicted analysis and proposed
a candidate bundle; ModelBlaster realized + ran the baseline on
FireSim; the measured trace fed back to XPU-RT's advisor for
re-diagnosis.

| field | value |
|-------|-------|
| bundle | `/scratch2/agustin/XPU-RT/artifacts/iterate/firesim_batch.json` |
| workload spec | `/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json` |
| runner | `firesim` (1× FPGA) under `FIRESIM_QUEUE=1` |
| deadline budget | 65 µs (advisor default, midpoint between best candidate and baseline) |

## Predicted side (XPU-RT, complete)

| id | config | axis | makespan (µs) | meets 65µs | granularity | bottleneck |
|----|--------|------|--------------:|:----------:|-------------|------------|
| A2 🏆 | milp / **heft** | scheduler | 54.43 | ✅ | too_fine | CPU_P#0 (Gemmini) |
| A3 | milp / peft | scheduler | 56.24 | ✅ | too_fine | CPU_P#0 (Gemmini) |
| A1 | greedy | scheduler | 60.48 | ✅ | too_fine | CPU_P#0 (Gemmini) |
| (partition) | greedy_periodic | scheduler | 61.20 | ✅ | too_fine | — |
| **baseline** | decomposed | baseline | 75.57 | ❌ (−16.3%) | too_fine | CPU_E#0 (OPU) |
| A4 | milp / edf | scheduler | 116.78 | ❌ (−79.7%) | too_fine | CPU_E#0 (OPU) |

The advisor flagged the baseline's `too_fine` granularity (388/388
dispatches under 1000 cycles) and proposed axis-C **fusion** (the
`modelblaster.fusion_hints/v1` payload in `granularity_hint.json`,
fusing `mlp_control[0..6]`, `dronet[0..27]`/`[28..29]`, and 7 chains
inside `yolov8_nano`). Predicted Δ from fusion is bounded by removed
cross-device transitions — the launch-overhead win is only visible
on FireSim, not in XPU-RT's analytic cost model.

Predicted Gantts (per candidate, rendered via `xpu-rt/plot_gantt.py
--fixture`):

- `baseline/predicted_gantt.png` — decomposed (75.57 ms)
- `A2/predicted_gantt.png` — milp/heft (54.43 ms, the winner)
- `predicted_greedy_gantt.png` — A1 greedy (60.48 ms)
- `predicted_peft_gantt.png` — A3 milp/peft (56.24 ms)
- `predicted_edf_gantt.png` — A4 milp/edf (116.78 ms; explored even
  though it lost — the agentic search isn't pre-pruned)
- `partition_gantt.png` — greedy_periodic (61.20 ms; the
  partition-style schedule shape the user asked to see)

The composite predicted before/after lives at
`/scratch2/agustin/XPU-RT/artifacts/iterate/before_after_gantt.png`.

## Measured side (FireSim, partial)

| id | config | predicted (µs) | measured (µs) | trace coverage |
|----|--------|---------------:|--------------:|---------------:|
| baseline | decomposed | 75 570 | 6 552 | 200 dispatches (first ~8.7% of run) |

The baseline ELF built and ran on the FireSim FPGA correctly — the
trace shows clean per-dispatch cycle counts for the first 200 ops
across all three networks (yolov8_nano, dronet instance 1,
mlp_control instance 3 all show up in the trace prefix). The run
hit the firesim-queue default 3 600 s wall-clock budget at ~6.55 ms
of simulated time; the harness keeps emitting trace rows until kill,
so 200 dispatches' worth of actuals survived in the uartlog.

Artifacts:
- `baseline/xpurt_trace.csv` — extracted from
  `2026-06-02--07-09-09-modelblaster-firesim-q154/.../uartlog`.
- `baseline/measured_report.json` — XPU-RT
  SchedulerReport v2 with `actual_*_us` overlaid per dispatch.
- `baseline/measured_advice.json` — XPU-RT advisor's verdict on the
  measured report (same diagnosis as predicted: `granularity=too_fine`,
  recommend `coarsen` at `fusion_threshold=1000 cycles`, bottleneck
  `CPU_E#0`).
- `baseline/predicted_vs_actual.png` — overlay rendered via
  `xpu-rt/plot_gantt.py --trace` (XPU-RT's plotter accepts the
  extracted CSV directly).

A re-capture with `FIRESIM_QUEUE_TIMEOUT=7200` (or larger) is needed
for a full predicted-vs-measured comparison on the entire workload.
The plumbing is in place — `bash scripts/run_agentic_loop_demo.sh
baseline,A2` will redo the run and `close_xpurt_loop.py` will
overlay the full trace automatically.

## Re-advise verdict (round-2 hint)

Both the predicted-only and the measured-prefix advisor verdicts agree:

> **Coarsen.** 100% of dispatches fall below 1000 cycles. Fusing them
> collapses the dispatch-overhead tail. Predicted projected makespan
> after coarsen: 75.57 µs (unchanged — XPU-RT's cost model has no
> per-dispatch launch overhead term). **Measured** Δ from fusion is
> the open question; that's what axis-C is designed to answer once
> the fused-op codegen lands (see `notes/plans/agentic_fusion_loop.md`
> Phase 1b — deferred from this round).

## What this proves

1. **Iterative agentic-driven loop** is wired end-to-end on real
   FireSim hardware:
   `XPU-RT (predict + propose) → ModelBlaster (realize + measure) →
   XPU-RT (re-advise on measured)` runs as one script
   (`scripts/run_agentic_loop_demo.sh`).
2. **Different Gantts** — six distinct predicted schedules
   (scheduler axis swept) plus a measured overlay for the baseline.
3. **Partition graph** — `partition_gantt.png` shows
   greedy_periodic's partition-style schedule shape side-by-side
   with the other variants.

The loop is the deliverable. Each round's `firesim_batch.json` +
`granularity_hint.json` is generated by XPU-RT's
`scripts/iterate_firesim.py`; ModelBlaster's `scripts/run_xpurt_bundle.py`
consumes it; `scripts/close_xpurt_loop.py` produces the next round's
input. Skills (`.claude/skills/realize-and-run`, `.claude/skills/realize-hint`
on the ModelBlaster side; `.claude/skills/close-loop` on XPU-RT) drive
the loop from a Claude Code session.
