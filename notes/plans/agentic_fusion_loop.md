# Plan: Close the XPU-RT ⇄ ModelBlaster agentic granularity loop

## Status (current)

- **Phase 0** (branch hygiene) — done. `baseline` branch off
  `main@405d1e6`; `feat/agentic-fusion-loop` is the active branch.
- **Phase 1a** (IR-level fusion realization) — done.
  `pipeline/apply_fusion_hint.py` + tests (11/11 pass).
  Smoke-tested on the real XPU-RT hint: `mlp_control` 7 dispatches
  → 2 (one `__fused__linear_s8__elu_s8__...` + the trailing linear).
- **Phase 1b** (fused-op codegen integration) — done.
  `generate_kernels.py` expands `__fused__` ops into sub-op kinds so
  the kernel picker emits each constituent kernel.
  `generate_skeleton.py` adds a top-of-loop branch that emits a
  single `dispatch_<mid>_<id>` function chaining the sub-kernel
  calls back-to-back, routed through a small `_emit_sub_op_call`
  helper. Strictly additive — non-fused IRs (baseline / A2 / every
  existing example) go through the unchanged if/elif. Sub-op kinds
  supported today: `linear_s8`, `elu_s8`, `relu_s8`. Other kinds
  raise a loud `NotImplementedError` pointing at the right branch
  to copy from the main chain.
- **Phase 1c** (spike verify on `mlp_control_int8`) — done.
  Applied the actual XPU-RT hint to `examples/mlp_control/int8/generated/graph.json`,
  ran `RUNNER=spike TARGET=rvv_opu QUANT=int8 bash examples/mlp_control/run.sh`.
  Result: **PASS** with `max_abs_err=0 max_rel_err=0` (bit-exact vs reference).
  Profile shows the fused dispatch (`mlp_control.fused_0_5`,
  `__fused__linear_s8__elu_s8__linear_s8__elu_s8__linear_s8__elu_s8`,
  shape `fused(6)`) accounting for 99.4% of cycles + the trailing
  standalone `linear_s8` (`mlp.6`) for 0.6%.
- **Phase 2** (bundle driver + measured-report adapter + shell
  wrapper) — done.
  `scripts/run_xpurt_bundle.py`, `scripts/run_bundle_firesim.sh`,
  `scripts/emit_measured_report.py`, `scripts/close_xpurt_loop.py`.
- **Phase 3** (skills) — done.
  ModelBlaster `.claude/skills/{realize-hint,realize-and-run}/SKILL.md`;
  XPU-RT `.claude/skills/close-loop/SKILL.md`.
- **Phase 4** (end-to-end demo) — running.
  Baseline (`decomposed`, 388 dispatches, predicted 75.57 ms) and
  HEFT candidate A2 (300 dispatches, predicted 54.43 ms) queued on
  FireSim under `FIRESIM_QUEUE=1`. Predicted Gantts already rendered
  under `artifacts/bundle/{baseline,A2}/predicted_gantt.png`; the
  greedy-periodic partition variant is in
  `artifacts/bundle/partition_gantt.png` for reference. Measured
  Gantts + advisor verdicts will land via `close_xpurt_loop.py` once
  the FireSim traces come back.

## Context

`feat/benchmark-harness` is merged (PR #1, `405d1e6` on `main`). The
measurement infra side is done — kernels, schedule fixtures, FireSim
runner, contention model, eager-V on the hetero bitstream. The
**iteration side** is what's still open: XPU-RT can already diagnose
schedules, propose A/B/C candidate bundles, and emit fusion / split
hints (Contracts 1 & 2 in
`/scratch2/agustin/XPU-RT/docs/iterative_firesim_loop.md`), but
**ModelBlaster cannot yet realize axis-C** — rewrite a graph to fuse
sub-1k-µs dispatch chains into a single coarser op. Without that,
the predicted-vs-measured loop never closes for granularity.

This plan does three things:
1. Mark the current shipped state as a `baseline` branch and open a
   new feature branch for the loop work.
2. Add IR-level fusion realization in ModelBlaster so a Contract-2
   `modelblaster.fusion_hints/v1` file becomes a fused graph + fused
   kernels + new dispatch graph + spike-verified ELF.
3. Wire the closed loop end-to-end and surface it as agentic skills
   on **both** repos (XPU-RT skill orchestrates; ModelBlaster skill
   owns realization), reusing the existing trace ingest
   (`benchmarks/ingest/xpurt_trace.py`) to feed measured numbers back
   into XPU-RT's advisor.

Outcome: one Claude session can run
`bash /scratch2/agustin/XPU-RT/scripts/demo_iterate_firesim.sh`
→ apply hint → build candidate bundle ELFs → FireSim batch
→ measured Gantt + re-advise, with the full 1×yolov8_nano +
4×mlp_control + 2×dronet workload on the Gemmini+OPU hetero
bitstream.

---

## Phase 0 — Branch hygiene (one shot, no code yet)

```bash
git fetch origin
git checkout main && git pull --ff-only origin main      # 405d1e6
git branch baseline                                       # immutable snapshot of merged harness
git push -u origin baseline
git checkout -b feat/agentic-fusion-loop                  # new work lands here
git push -u origin feat/agentic-fusion-loop
```

`baseline` is the post-merge marker — no edits ever land there.
All work below lands on `feat/agentic-fusion-loop`.

---

## Phase 1 — Realize Contract-2 fusion hints (IR-level)

The hint contract from XPU-RT is local-op-id chains per network:
```json
{"contract": "modelblaster.fusion_hints/v1",
 "networks": [{"network": "mlp_control",
               "fuse_groups": [[0,1,2],[7,8]], "n_tiny": 9}]}
```

**Strategy:** post-process the IR after `extract_graph.py` writes
`graph.json` — *don't* fork the FX/torch.export tracers. Each
`fuse_group` (a topologically-ordered list of `dispatch_id`s in a
single network's graph.json) collapses into one new op whose name
encodes the chain, with reassigned dispatch_ids and rewired
`depends_on`. A matching fused KernelSpec must exist (or be
codegen-emittable) in `reference_kernels.py`.

### 1a. New module: `pipeline/apply_fusion_hint.py`

CLI:
```bash
python -m modelblaster.pipeline.apply_fusion_hint \
    --hint   /scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json \
    --model  mlp_control \
    --ir     examples/mlp_control/int8/generated/graph.json \
    --out    examples/mlp_control/int8/generated/graph.fused.json
```

Logic:
1. Load `graph.json` (list of ops with `dispatch_id`, `op`, `name`,
   `depends_on`, shape info).
2. For each `fuse_group` in the hint that targets this network:
   - Verify all ids are present and form a connected weakly-dependent
     chain: `depends_on` only references members of the group or
     producers outside. Reject groups with branches that escape and
     re-enter — fusion is for linear chains only in this pass.
   - Collect external inputs (producers outside the group) and
     external outputs (consumers outside the group). These define
     the fused op's signature.
   - Synthesize a single op:
     - `op` = `fused_<op1>_<op2>_..._<opN>` (e.g.
       `fused_linear_s8_bias_s8_relu_s8`).
     - `name` = `<chain.name>_fused_<id_low>_<id_hi>` (stable).
     - `depends_on` = union of group members' external `depends_on`.
     - Carries a `fused_from: [orig_ids...]` field so codegen and
       the schedule-time trace can attribute cycles back to
       constituent ops.
3. Reassign dispatch_ids contiguously (fused op replaces the first
   member's id; downstream ops shift down).
4. Emit the rewritten IR JSON.

### 1b. Fused kernel coverage in `pipeline/reference_kernels.py`

The XPU-RT demo's `granularity_loop.py` chose to fuse `mlp_control`
ops `[0..5]` (`n_tiny=9`). Read the actual fuse_groups it emits and
register the **minimum set of fused KernelSpecs needed**:

- Identify the constituent op names per chosen group from
  `examples/mlp_control/int8/generated/graph.json`.
- Register one fused spec per distinct pattern (e.g.
  `linear_s8_bias_s8_relu_s8`, `gelu_s8_chain`).
- Each KernelSpec: signature matches the fused-op buffer list;
  `reference_impl` chains constituent reference impls in-order;
  optional algorithmic variants only after the demo loop is working.

Reuse the existing compound-pattern recognizer in `extract_graph.py`
(the one that already folds Swish/GELU chains): the fused-KernelSpec
name should match the convention that recognizer uses, so a curated
kernel is preferred over an LLM re-codegen when one exists.

### 1c. Re-emit dispatch graph + spike verify (existing tooling)

After 1a/1b, the existing pipeline is unchanged:
```bash
python -m modelblaster.pipeline.emit_dispatch_graph \
    --ir <fused_graph.json> --target generic_riscv64 --hw <hw>
RUNNER=spike QUANT=int8 TARGET=rvv_opu bash examples/mlp_control/run.sh
```
The spike PASS is the inner-loop oracle (fast functional check
before committing a candidate to the FireSim batch).

### 1d. Tests

- `pipeline/tests/test_apply_fusion_hint.py` — golden IR with a
  hand-crafted hint, assert the fused IR is well-formed, dispatch_ids
  contiguous, depends_on rewired, fused_from preserved.
- One end-to-end smoke (spike) that runs apply_fusion_hint →
  emit_dispatch_graph → run.sh and asserts `PASSED` on
  `dronet_rvv_opu_int8` with a 2-op identity-style fuse_group
  (sanity).

---

## Phase 2 — Bundle driver + measured loop-back

XPU-RT writes `artifacts/iterate/firesim_batch.json` (Contract 1).
Today, nothing in ModelBlaster consumes it. Add a thin driver and
a re-advise hook — most of the heavy lifting already exists.

### 2a. `scripts/run_xpurt_bundle.py` (new)

For each candidate in `firesim_batch.json`:

- `realizable_by: "xpurt"` (axes A/B) — pull the matching
  `schedules/scheduled_<...>_profiled.json` (already produced
  XPU-RT-side; reference by absolute path), then call the existing
  `pipeline/ingest_xpurt_schedule.py` + `generate_xpurt_main.py` +
  `examples/xpurt_demo/run.sh` to build a tagged ELF per candidate.
- `realizable_by: "modelblaster"` (axis C) —
  1. Run `apply_fusion_hint` on the candidate's networks per
     `candidate.hints`.
  2. Re-emit the dispatch graph for each fused network.
  3. **Profile-aware update:** for the first cut, re-use the existing
     profile for non-fused ops and assign the fused op a synthetic
     cycle count equal to
     `sum(constituent_predicted_cycles) - transition_overhead_estimate`
     so the schedule fixture is buildable. The *measured* truth comes
     from FireSim after the batch — we deliberately don't iterate
     spike for granular cycle counts here.
  4. Re-run `scripts/run_xpurt_scheduler_multi.py` (existing) with
     the new dispatch graphs to produce a candidate
     `scheduled_*_profiled.json` for the fused variant.
  5. Build the harness_xpurt ELF.

Output: `artifacts/bundle/<candidate_id>/zephyr.elf` plus a small
`artifacts/bundle/manifest.json` listing all candidates.

### 2b. Batched FireSim run

Reuse `validation/firesim_runner.py` per candidate, but queue them
under **one** infrasetup using `FIRESIM_QUEUE=1` (the shared FPGA
queue serializes runworkload across users). A new
`scripts/run_bundle_firesim.sh`:
```bash
FIRESIM_QUEUE=1 \
for c in $(jq -r '.candidates[].id' artifacts/bundle/manifest.json); do
    python -m modelblaster.validation.firesim_runner \
        --elf artifacts/bundle/$c/zephyr.elf \
        --tag $c --capture-trace artifacts/bundle/$c/xpurt_trace.csv
done
```
The trace CSV format already matches both
`scripts/plot_xpurt_trace.py` (predicted-vs-actual Gantt) and
XPU-RT's `xpu-rt/plot_gantt.py --trace`. **No new parsing.**

### 2c. Loop-back to XPU-RT advisor

`benchmarks/ingest/xpurt_trace.py` already exposes makespan,
per-kind busy cycles, etc. Add one small adapter
`scripts/emit_measured_report.py` that:
- reads `xpurt_trace.csv` for a candidate,
- writes a `scheduled_<candidate>_measured_report.json` mimicking
  XPU-RT's `SchedulerReport` schema v2 with `actual_*_cycles` filled
  from the trace,
- so `python3 /scratch2/agustin/XPU-RT/xpu-rt/advisor.py --report
  <measured_report.json> --deadline-us <N>` re-diagnoses on
  **measured** timings and the loop produces a Round-2 bundle.

This is the only piece of new XPU-RT-facing glue and it lives on
the ModelBlaster side because it owns the trace format.

---

## Phase 3 — Agentic skills (both repos)

### 3a. XPU-RT side — extend existing `.claude/skills/`

New skill: `/close-loop`. SKILL.md mirrors the frontmatter shape
already used by `diagnose-schedule`, `sweep-schedulers`,
`compare-runs`:
1. Run `bash scripts/demo_iterate_firesim.sh` — produces
   `artifacts/iterate/{firesim_batch.json, granularity_hint.json,
   before_after_gantt.png, report.md}`.
2. Hand off to the ModelBlaster session: invoke `/realize-and-run`
   (Phase 3b) with the batch + hint paths.
3. When traces come back, run
   `xpu-rt/advisor.py --report <measured_report.json>` on each and
   `xpu-rt/plot_gantt.py --trace` to render measured Gantts.
4. Decide next-round bundle (call `bundle.propose_bundle` again or
   stop if the deadline is met on measured timings).

### 3b. ModelBlaster side — new `.claude/skills/`

Create `.claude/skills/` and add two skills:

- **`/realize-hint`** — apply a single Contract-2 hint:
  1. `python -m modelblaster.pipeline.apply_fusion_hint ...` per
     network in the hint.
  2. Re-emit dispatch graph; spike-verify the fused ELF.
  3. Report PASSED/FAILED + the fused KernelSpecs touched.

- **`/realize-and-run`** — full bundle driver:
  1. Build all candidates (`scripts/run_xpurt_bundle.py`).
  2. `FIRESIM_QUEUE=1 scripts/run_bundle_firesim.sh`.
  3. Parse traces; emit measured reports; render
     predicted-vs-actual Gantts via the existing
     `scripts/plot_xpurt_trace.py`.
  4. Print a one-line summary per candidate
     (`baseline: X µs → candidate <id>: Y µs (axis=<a>)`) and
     return paths to the measured reports for the XPU-RT skill to
     pick up.

### 3c. MCP surface (optional, time-boxed)

If skills aren't enough for an interactive demo, expose the three
ModelBlaster CLIs (`apply_fusion_hint`, `run_xpurt_bundle`,
`emit_measured_report`) as tools via a small MCP server modeled on
`/scratch2/agustin/XPU-RT/scripts/compgen-mcp.sh`. Defer unless the
skill-only demo can't carry the narrative.

---

## Phase 4 — End-to-end demo

Workload: `data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json`
(unchanged, on the Gemmini+OPU hetero bitstream).

Demo path, single recorded session:

1. `cd /scratch2/agustin/XPU-RT && bash scripts/demo_iterate_firesim.sh`
2. Invoke `/realize-and-run artifacts/iterate/firesim_batch.json
   artifacts/iterate/granularity_hint.json` (ModelBlaster skill).
3. Wait for one batched FireSim session (FIRESIM_QUEUE=1
   serializes against other users).
4. Inspect `before_after_gantt.png` (predicted, XPU-RT-side) and
   `artifacts/bundle/<winner>/predicted_vs_actual.png` (measured,
   ModelBlaster-side).
5. Re-advise via `/close-loop` Step 3; capture the Round-2 verdict
   in `artifacts/bundle/round2_report.md` and stop.

---

## Critical files

**New (Phase 1, ModelBlaster):**
- `pipeline/apply_fusion_hint.py` — IR rewrite + fused-op synthesis.
- `pipeline/tests/test_apply_fusion_hint.py` — unit tests.
- New KernelSpec entries in `pipeline/reference_kernels.py` (one per
  fused pattern the demo actually requests; expect 2–3).

**New (Phase 2, ModelBlaster):**
- `scripts/run_xpurt_bundle.py` — drives Contract-1 candidates.
- `scripts/run_bundle_firesim.sh` — batched FireSim runner.
- `scripts/emit_measured_report.py` — trace → XPU-RT
  `SchedulerReport` adapter.

**New (Phase 3, ModelBlaster):**
- `.claude/skills/realize-hint/SKILL.md`.
- `.claude/skills/realize-and-run/SKILL.md`.

**New (Phase 3, XPU-RT):**
- `.claude/skills/close-loop/SKILL.md`.

**Reuse, do not duplicate:**
- `pipeline/extract_graph.py` — compound-pattern recognizer (fused
  KernelSpec names should match its convention so curated kernels
  win over LLM re-codegen).
- `pipeline/emit_dispatch_graph.py`, `pipeline/ingest_xpurt_schedule.py`,
  `pipeline/generate_xpurt_main.py` — unchanged.
- `examples/xpurt_demo/run.sh` — already multi-network, multi-backend.
- `harness_xpurt/` — already V-eager + hart-pinned (`430f8be`).
- `benchmarks/ingest/xpurt_trace.py` — already parses
  `MODELBLASTER_XPURT_TRACE_BEGIN/END` blocks; reused by the
  loop-back adapter.
- `scripts/plot_xpurt_trace.py` and
  `/scratch2/agustin/XPU-RT/xpu-rt/plot_gantt.py --trace` — same CSV
  schema; both render predicted-vs-actual.
- `/scratch2/agustin/XPU-RT/scripts/granularity_loop.py`,
  `iterate_firesim.py`, `bundle.py`, `advisor.py` — unchanged.

---

## Verification

End-to-end gates, each must pass before the next:

1. **Branch hygiene.** `git ls-remote origin baseline` resolves to
   `405d1e6` (the merge commit). `feat/agentic-fusion-loop` is
   tracked at origin.

2. **Phase 1 unit + smoke.**
   - `pytest pipeline/tests/test_apply_fusion_hint.py` green.
   - `apply_fusion_hint` on a toy IR produces a well-formed
     `graph.fused.json` (dispatch_ids contiguous, `depends_on`
     consistent, `fused_from` preserved).
   - `RUNNER=spike` smoke on `mlp_control_int8` with a 2-op
     identity-style fuse_group: spike says `PASSED`.

3. **Phase 1 full hint.** Apply the actual hint from
   `/scratch2/agustin/XPU-RT/artifacts/iterate/granularity_hint.json`;
   `RUNNER=spike` on all touched models PASSES with
   `max_abs_err ≤ atol OR max_rel_err ≤ rtol`.

4. **Phase 2 bundle build.** `scripts/run_xpurt_bundle.py`
   produces N ELFs (one per candidate) under
   `artifacts/bundle/<id>/zephyr.elf`; CMake config logs show
   `CONFIG_RISCV_ISA_EXT_V=y` + `CONFIG_SCHED_CPU_MASK_PIN_ONLY=y`
   for OPU candidates.

5. **Phase 2 batched FireSim.**
   `FIRESIM_QUEUE=1 scripts/run_bundle_firesim.sh` completes within
   one infrasetup; per-candidate uartlog has both
   `MODELBLASTER_VERIFY` PASS and a full
   `MODELBLASTER_XPURT_TRACE_BEGIN/END` block.

6. **Phase 2 loop-back.** `emit_measured_report.py` produces JSON
   that `xpu-rt/advisor.py` accepts; advisor verdict differs from
   the predicted-only verdict where the measured numbers warrant
   it (i.e. the loop actually moved).

7. **Phase 3 skills.** Invoking `/realize-and-run` (in a fresh
   ModelBlaster session) and `/close-loop` (in a fresh XPU-RT
   session) chains through Phases 1–2 end-to-end without manual
   shelling beyond the skill steps themselves.

8. **Phase 4 demo.** Final artifacts present:
   `artifacts/iterate/before_after_gantt.png` (predicted),
   `artifacts/bundle/<winner>/predicted_vs_actual.png` (measured),
   `artifacts/bundle/round2_report.md` (re-advise verdict).

Success = all eight gates pass on the full 1+4+2 workload on the
hetero bitstream. Anything less is a regression of the closed loop
and gets explicitly called out.

---

## Constraints honored

- **Hetero bitstream only** for scheduling work: Gemmini+scalar /
  RVV+OPU pair (no q31 / scalar-only / rvv-only variants).
- **`FIRESIM_QUEUE=1`** on every FireSim invocation so the shared
  FPGA queue serializes runworkload across users.
- **`tmp/` docs are authoritative** for anything ambiguous about
  ModelBlaster internals (see `tmp/01_pipeline.md`,
  `tmp/04_xpurt_integration.md`, `tmp/07_gotchas.md`).
