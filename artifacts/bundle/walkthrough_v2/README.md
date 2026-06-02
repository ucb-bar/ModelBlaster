# XPU-RT ⇄ ModelBlaster agentic loop — complete demo walkthrough

Branch: `feat/agentic-fusion-loop` (rebased on `main` via PR #1 / commit
`405d1e6`).
This document indexes every PNG / artifact produced by the loop and the
exact command that reproduces it.

---

## 1. The loop, end to end

```
XPU-RT (predicted)                       ModelBlaster (realized + measured)
─────────────────                        ─────────────────────────────────────
iterate_firesim ─► firesim_batch.json ──►  run_xpurt_bundle ─► harness_xpurt ELF
                                                ▼                       ▼
granularity_loop ─► granularity_hint   ──►  apply_fusion_hint  ──►  FireSim run
(or split_hint)     (Contract-2)         OR apply_split_hint           ▼
                                                                   xpurt_trace.csv
                                                                       ▼
advisor ◄────────────────  measured SchedulerReport  ◄── close_xpurt_loop
   ▼
propose_bundle (round 2) ─►  next firesim_batch.json
```

Single-command reproduction:
```bash
bash scripts/run_agentic_loop_demo.sh baseline,A2,A_periodic
```
(invokes XPU-RT iterate → ModelBlaster bundle build/run → close-loop +
per-step PNG rendering automatically.)

---

## 2. The 7-scheduler predicted comparison

**`predicted_stack_all.png`** — all 7 schedulers on the same x-axis with
the 65 ms deadline marker overlaid. Lets you directly read off:

- the shape of each scheduler's packing (gaps, bottleneck core,
  inter-dispatch contention),
- which meet vs miss the deadline,
- where MOSEK lands when its solver-time budget is too short.

| solver | scheduler | makespan (ms) | meets 65 ms? |
|---|---|---:|:--:|
| decomposed | (baseline) | 75.57 | ❌ |
| greedy | — | 60.48 | ✅ |
| milp | **heft** 🏆 | 54.43 | ✅ |
| milp | peft | 56.24 | ✅ |
| milp | edf | 116.78 | ❌ |
| milp | mosek (60 s budget) | 144.24 | ❌ (hit time limit) |
| greedy_periodic | (partition) | 61.20 | ✅ |

`scripts/render_compare_gantt.py` produced this PNG. It auto-detects
fixtures (JSON) vs traces (CSV) by suffix.

---

## 3. Tighter-deadline scenario (50 µs) — axis-A alone isn't enough

`artifacts/bundle/iterate_d50/` — same workload, deadline tightened to
50 µs. EVERY axis-A scheduler misses (HEFT 54.43 µs misses by 8.86 %);
only axis-C fusion (C1) closes the gap predicted.

- `xpurt_report.md` — XPU-RT's advisor + comparison table.
- `firesim_batch.json` — Contract-1 bundle.
- `granularity_hint.json` — Contract-2 hint, picks `mlp_control [0..5]`
  with Δmakespan = -3 µs (the **specific** chain the advisor scored, not
  blanket "fuse everything").
- `<id>/predicted_gantt.png` — one per candidate (baseline, A1, A3, A4,
  A5, A6, B3, C1). Auto-rendered by `scripts/render_per_step.py`.

The advisor's choice is data-driven: it scored every possible fuse and
split candidate via `rewrite.py` + re-schedule, then picked the one with
the best Δmakespan. Top split candidates (yolov8_nano dispatches) all
scored Δmakespan = 0 — splitting doesn't help on this workload (no idle
core), so the advisor doesn't propose them.

---

## 4. Axis-C fusion (Phase 1b/1d) — measured + bit-exact

- **Phase 1b — dispatch-overhead fusion.** Chains N back-to-back kernel
  calls inside one harness dispatch. Eliminates the N-1
  worker-thread handshakes; total compute is identical. Built from the
  IR `__fused__<sub0>__<sub1>__...` synthetic op produced by
  `pipeline/apply_fusion_hint.py`. Bit-exact on spike for `mlp_control`
  (artifacts/bundle/axis_c_spike/spike_output.log,
  `max_abs_err=0 max_rel_err=0`).

- **Phase 1d — true compute fusion.** `linear_s8_elu_s8` registered
  KernelSpec (pipeline/reference_kernels.py) does the linear MAC +
  requantize tail + ELU **in register**, no intermediate-tensor
  write/read. Bit-exact on spike for pairwise-fused mlp_control
  (artifacts/bundle/axis_c_phase1d/spike_pairwise_fused.log,
  `max_abs_err=0 max_rel_err=0`). LLM-codegen seeds describe rvv_opu
  (VOPMACC + VRGATHER LUT in vector registers) and gemmini
  (tiled_matmul_auto epilogue) variants — `BACKEND=llm TARGET=rvv_opu`
  invokes Bedrock to generate target-specific optimized variants
  gated against the reference impl.

Visual walkthrough at `artifacts/bundle/walkthrough/`:
- `step1_baseline_predicted.png` / `step2_heft_predicted.png` —
  axis-A loop, deadline-miss / bottleneck-flip explanation.
- `step3_baseline_measured.png` — full FireSim trace, 388 dispatches,
  13.5 ms measured (5.6× faster than predicted 75.57 ms because the
  harness's work-conserving walker skips periodic gaps; see FAQ in
  `walkthrough/README.md`).
- `step4_fused_mlp.png` — same trace with `mlp_control[0..5]` collapsed
  into hatched fused chains; 24 worker handshakes eliminated.
- `measured_compare.png` / `measured_compare_zoom.png` — baseline
  vs axis-C fused on the same x-axis; zoom 0-2 ms makes the fused
  hatched blocks visually obvious.

---

## 5. Axis-C split (Phase 1e/1f/1g) — synthetic too_coarse demo

`artifacts/bundle/walkthrough_v2/split_demo/`:
- `before.json` — synthetic too_coarse fixture, 3 dispatches, 5.0 ms
  makespan, CPU_E idle 4 ms of 5 (~50 % utilization).
- `after.json` — same workload after splitting `linear_big` into 2
  tiles: 4 dispatches, 3.0 ms makespan (**-40 %**), both cores busy.
- `split_demo.png` — stacked before/after Gantts on the same axis. The
  large CPU_P-only orange bar at 0-4 ms in the BEFORE panel gets
  replaced by two parallel tile bars on both cores 0-2 ms in the AFTER
  panel, exactly the "idle core gets filled" transform.

`pipeline/apply_split_hint.py` (Phase 1e) is the IR-level transform that
realizes this for `linear_s8` ops along the N (output-feature)
dimension. 6/6 unit tests pass. Conv2d_s8 splits (weight slicing along
OC) are the follow-up.

---

## 6. Measured FireSim runs

- `artifacts/bundle/longrun/baseline/` — **complete** 388-dispatch
  trace, 13.5 ms measured, advisor re-run on measured numbers
  (`measured_advice.json`).
  - `predicted_gantt.png`, `measured_gantt.png`,
    `predicted_vs_actual.png` auto-rendered by
    `scripts/render_per_step.py`.
- `artifacts/bundle/round1_report.md` — predicted-vs-measured table +
  re-advise verdict.
- `artifacts/bundle/round2/firesim_batch.json` — Round-2 bundle
  generated from the measured baseline (proves the loop closes:
  `bundle.propose_bundle` accepts the measured `SchedulerReport`).
- `artifacts/bundle/round2/firesim_batch_inf_budget.json` — same Round-2
  with MOSEK + CPSAT opted in (9 candidates).

Notes on stuck FireSim jobs: greedy_periodic on FireSim hung for >60 min
in RUNNING phase — UART output froze even though the simulator was
still computing. Cancelled. The `mlp_control + dronet + yolov8` schedule
on this hetero bitstream is at the edge of what completes; a tighter
workload or longer FIRESIM_QUEUE_TIMEOUT would be the fix. Predicted
Gantts for greedy_periodic are still in
`artifacts/bundle/partition_gantt.png` and `predicted_stack_all.png` (the
schedule shape is captured even though the FPGA run didn't finish).

---

## 7. Skills + commands

`.claude/skills/`:
- ModelBlaster — `/realize-hint`, `/realize-and-run`.
- XPU-RT — `/close-loop`.

`scripts/`:
- `run_agentic_loop_demo.sh baseline,A2,...` — one-shot orchestrator
  (XPU-RT iterate → MB bundle → close-loop with per-step PNGs).
- `run_xpurt_bundle.py --batch ...` — bundle driver (per-candidate ELF
  build + FireSim queue).
- `render_per_step.py --manifest ...` — auto per-candidate Gantts
  (predicted + measured + side-by-side).
- `render_annotated_gantt.py` — single annotated Gantt.
- `render_compare_gantt.py --panel ... --panel ...` — stacked
  multi-scheduler comparison (auto-detects fixture vs trace by suffix).
- `close_xpurt_loop.py --manifest ...` — post-process bundle (measured
  reports + Gantts + advisor re-run).
- `budget_check.py --estimate-usd N` — enforce $150 LLM cap with
  $17.38 actual-vs-recorded offset; refuses if projected total > cap.

---

## 8. Tasks complete vs roadmap

Done (commits on the branch):
- Phase 0, 1a, 1b, 1c, 1d, 1e — IR rewrite + fusion + split realized,
  bit-exact on spike where measurable.
- Phase 2 — bundle driver, measured-report adapter, demo orchestrator.
- Phase 3 — skills in both repos.
- Phase 4 — end-to-end demo on real FireSim hardware (baseline trace
  captured, advisor re-run, Round-2 bundle proposed).
- Per-step PNG rendering wired into the orchestrator.
- Budget enforcement gate before any LLM-codegen launch.

Roadmap (tasks still open):
- More FireSim captures (axis-A and axis-C measured) — blocked on the
  hetero bitstream's runtime stability on this workload, not the loop.
- Conv2d_s8 split (weight slicing along OC) — extension of Phase 1e.
- MOSEK/CPSAT under 600 s budget — running in `/tmp/mosek_600s.log` and
  `/tmp/cpsat_600s.log`; should match-or-beat HEFT once converged.
- LLM-generated kernel variants — `BACKEND=llm` path is ready; budget
  gate at `scripts/budget_check.py` enforces the $150 cap.
