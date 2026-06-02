---
name: realize-and-run
description: Drive an XPU-RT candidate bundle (Contract 1 firesim_batch.json) end-to-end on ModelBlaster — build harness_xpurt for each axis A/B/C candidate, queue them under one FireSim infrasetup, parse the xpurt_trace.csv per run, and emit measured SchedulerReports for XPU-RT to re-advise. Use to close one iteration of the granularity loop.
---

# realize-and-run

Take a `firesim_batch.json` (`xpurt.candidate_bundle/v1`) produced by
XPU-RT's `iterate_firesim.py` and run all candidates on FireSim in a
single batched session, returning per-candidate measured timings that
feed back into the advisor for the next round.

This is the ModelBlaster half of the predicted-vs-measured loop. The
XPU-RT half is `/close-loop` in `/scratch2/agustin/XPU-RT/.claude/skills/`.

## Steps

1. **Read the bundle and classify candidates by axis:**
   ```bash
   BATCH=/scratch2/agustin/XPU-RT/artifacts/iterate/firesim_batch.json
   jq -r '.candidates[] | "\(.id) axis=\(.axis) realizable_by=\(.realizable_by)"' "$BATCH"
   ```
   - `realizable_by: "xpurt"` (axes A/B) — schedule fixture is already
     produced by XPU-RT; just build harness_xpurt for it.
   - `realizable_by: "modelblaster"` (axis C) — apply the fusion hint
     via `/realize-hint`, re-emit the dispatch graph, re-run the
     XPU-RT scheduler, then build the harness.

2. **Build one ELF per candidate:**
   ```bash
   python3 scripts/run_xpurt_bundle.py \
       --batch "$BATCH" --out-dir artifacts/bundle
   ```
   Produces `artifacts/bundle/<id>/zephyr.elf` plus
   `artifacts/bundle/manifest.json`. The CMake configure log should
   show `CONFIG_RISCV_ISA_EXT_V=y` and
   `CONFIG_SCHED_CPU_MASK_PIN_ONLY=y` for OPU-using candidates.

3. **Run the batched FireSim session** (one infrasetup, N
   runworkloads). Always set `FIRESIM_QUEUE=1` so the shared FPGA
   queue serializes against other users:
   ```bash
   FIRESIM_QUEUE=1 bash scripts/run_bundle_firesim.sh \
       artifacts/bundle/manifest.json
   ```
   Per-candidate uartlogs must contain both `MODELBLASTER_VERIFY
   PASS` (numerical correctness) and a full
   `MODELBLASTER_XPURT_TRACE_BEGIN/END` block (per-dispatch actual
   timings). Without the trace block, the loop-back step has nothing
   to feed XPU-RT.

4. **Emit measured SchedulerReports** so XPU-RT's advisor can
   re-diagnose on real numbers:
   ```bash
   for c in $(jq -r '.candidates[].id' artifacts/bundle/manifest.json); do
       python3 scripts/emit_measured_report.py \
           --trace artifacts/bundle/$c/xpurt_trace.csv \
           --schedule artifacts/bundle/$c/scheduled_*.json \
           --out artifacts/bundle/$c/measured_report.json
   done
   ```

5. **Render predicted-vs-actual Gantts** per candidate (sanity check
   before handing back to XPU-RT):
   ```bash
   for c in $(jq -r '.candidates[].id' artifacts/bundle/manifest.json); do
       python3 -m scripts.plot_xpurt_trace artifacts/bundle/$c/xpurt_trace.csv \
           --out artifacts/bundle/$c/predicted_vs_actual.png
   done
   ```
   Use `scripts/plot_xpurt_trace.py` (ModelBlaster) or
   `/scratch2/agustin/XPU-RT/xpu-rt/plot_gantt.py --trace` (XPU-RT);
   both consume the same CSV schema.

6. **Summarize for the user**, one line per candidate:
   `<id> axis=<a>  predicted: X µs  measured: Y µs  Δ=Z%`
   Highlight the winning candidate (best measured makespan that
   passes verify). Hand back the paths to the measured reports so
   `/close-loop` (XPU-RT side) can propose the next round.

## Rules

- **Always set `FIRESIM_QUEUE=1`.** The shared FPGA host serializes
  runworkload across users; bypassing the queue corrupts other
  users' sessions.
- **Use the hetero bitstream only** (Gemmini+scalar / RVV+OPU pair).
  Multi-net scheduling is undefined on q31 / scalar-only / rvv-only
  variants.
- **Verify must pass before reporting a measurement.** A candidate
  that runs faster but fails `MODELBLASTER_VERIFY` is not a winner —
  log it as a regression and skip it in the summary.
- **Don't iterate spike for cycle estimates** of fused kernels —
  spike's per-dispatch cycle counts are not RTL-accurate. The
  measured truth comes from FireSim; the predicted estimate is
  XPU-RT's profile model.
- **Hand back trace.csv paths**, not just summary numbers — XPU-RT's
  `plot_gantt --trace` and the advisor want the raw trace.
