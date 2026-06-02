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

---

## 9. Schedule-quality post-pass + oracle floor (Phase A+B)

Three additive XPU-RT-side changes (committed as `9c9eaf6`) make every
scheduler's output **honest**:

1. **Left-shift compaction** (`xpu-rt/compaction.py`). After any solver
   returns `(t, alpha)`, we walk in dispatch-order and slide each task's
   start to `max(release_time, max(dep.end), prev_on_core.end)`.
   Strictly safe — never violates deps, releases, or per-core
   no-overlap. **Idempotent** on MOSEK/CPSAT (already tight);
   slack-eliminating on HEFT/PEFT/EDF/greedy_periodic. Plugged into the
   registry via `_wrap_with_compaction` so every scheduler gains it
   uniformly. Disable with `XPURT_NO_COMPACT=1`.

2. **Same-network adjacent auto-merge** (`xpu-rt/automerge.py`). Operates
   on the emitted fixture dict. Collapses back-to-back same-network
   same-core dispatches when no external reader/writer is in the way,
   mirroring the `__fused__` convention so `render_compare_gantt`'s
   hatched rendering applies for free. On this workload: HEFT/EDF
   collapsed 130 pairs each (300 → 170 dispatches), PEFT collapsed 153,
   CPSAT only 65 (because CP-SAT already interleaves for parallelism).
   Disable with `XPURT_NO_AUTOMERGE=1`.

3. **Oracle lower-bound floor** (`xpu-rt/oracle.py`). Pure function of
   the workload — no solver in the loop. Returns
   `{critical_path_us, load_us, release_us, oracle_floor_us}` where
   the floor = `max` of the three. Plumbed into `SchedulerReport` as
   schema-v3 additive fields (`oracle_floor_us`, `oracle_gap_pct`).
   Also computed directly in
   `scripts/run_xpurt_scheduler_multi.py` so HEFT/PEFT/EDF/greedy
   (which don't go through `scheduler.schedule`) get the oracle gap
   surfaced too.

See `oracle_table.md` for the headline result. **CPSAT is the only
solver that hit the 75 ms horizon target** on this workload (73.98 ms
predicted) — beating HEFT/EDF by 11 % and PEFT by 15 %. MOSEK is still
running and will be added when the solve converges.

Visual: `artifacts/scheduler_bundle_postpass/stack_postpass.png`
(stacked HEFT / PEFT / EDF on the same x-axis with period boundaries
overlaid; the visible per-scheduler hatched fused blocks reflect the
automerge pass).

---

## 10. LLM-generated kernels (Phase C)

The Bedrock `BACKEND=llm` path was broken for `rvv_opu` until
commit `275f0b7`. Root cause: `cores/saturn_opu/include/saturn_opu.h`
unconditionally `#undef`'d the register-name macros `m0..m3` /
`v0..v31` at the end of the header — so `OPMVINBCAST(m1, v0)` called
in a kernel body expanded to `asm volatile(".insn r 0x57,0x6,0x59, "
m1 ", x0, " v0)` with `m1`/`v0` as undefined C identifiers, hitting
`expected ':' or ')' before 'm1'` at the C parse level. The curated
`kernels/rvv_opu/rvv_opu_*_outerprod.c` reference kernels were
equally broken — they just weren't reachable on the demo workload
because the eligibility gate falls back to scalar for `N > mlmax`.

**Fix:** wrap the `#undef`s in `#ifndef SATURN_OPU_KEEP_REGISTER_MACROS`,
and update the `linear_s8` / `matmul_s8` AlgorithmCandidate
`reference_impl` and the rvv backend guide to instruct LLMs to
`#define SATURN_OPU_KEEP_REGISTER_MACROS` before the include.

**Result:** Bedrock generated a bit-exact `linear_s8/outerprod`
kernel for `mlp_control` on the first try after the fix
(`max_abs_err=0 max_rel_err=0`, saved at
`artifacts/bundle/walkthrough_v2/llm_kernels/mlp_control_linear_s8_opu.c`).

| Operation | Shape | Scalar cycles | OPU cycles | Speedup |
|:----------|:------|-------------:|----------:|--------:|
| linear_s8 | K=16, N=256  |  68,944 | 12,881 |  5.4× |
| linear_s8 | K=256, N=128 | 495,313 | 41,042 | **12.1×** |
| linear_s8 | K=128, N=64  | 124,818 | 11,347 | 11.0× |
| linear_s8 | K=64,  N=4   |   4,036 |    497 |  8.1× |
| **Total mlp_control** | | **730,069** | **102,725** | **7.1×** |

Logs:
- `llm_kernels/run.log` — pre-fix attempts (4 failures, all `bad value
  for funct2 field` / `illegal operands` errors).
- `llm_kernels/run_v2.log` — post-fix success on first attempt.

### Budget tracking gap (fixed)

The `LLM_PROVIDER=bedrock BACKEND=llm` path through
`examples/<model>/run.sh` was constructing the Bedrock client with
`log_path=None` because `_run_lib.sh` wasn't exporting
`BEDROCK_CALLS_LOG`. Every call bypassed `benchmarks/tools/cost_monitor.py`
(mb-cost) entirely. Same commit (`275f0b7`) adds the export pointing
to `benchmarks/results/<MODEL>_<TARGET>_<QUANT>/<UTC>/llm_calls.jsonl`,
so future runs are tracked. The manual offset in
`benchmarks/.budget.json` was bumped by +$3.00 to cover today's
untracked attempts (offset now $22.88, remaining $63.50 of $150
cap).
