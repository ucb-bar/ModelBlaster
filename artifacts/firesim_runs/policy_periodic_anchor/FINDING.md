# periodic_anchor end-to-end FireSim measurement — ROOT-CAUSE FOUND

## What the run showed (Jun 3 2026, 14:55 PDT)

The hetero schedule (4 MLP + 2 Dronet + 1 Yolo on
CPU_P=gemmini, CPU_E=rvv_opu) was submitted to FIRESIM_QUEUE=1
and executed end-to-end on the FPGA. Verify reported:

| Network | output | max_abs_err | Result |
|:---|:---|---:|:---|
| mlp_control | all zeros | 125 | FAIL |
| dronet | all zeros | 127 | FAIL |
| yolov8_nano | first 8 = 0, last 8 = -42 | 45 | FAIL |

The "all zeros" pattern was the giveaway: the output buffer never
got written. That's not random FPGA corruption — that's the static
zero-init showing through because the kernel call never happened.

## Root cause — silent dispatch drop from build-config mismatch

`pipeline/ingest_xpurt_schedule.py:_resolve_target` stores the
registry's `kind` field as `.core_kind` in every dispatch table
entry. For the hetero bitstream registry
(`cores/chipyard_gemmini_opu_hetero.json`), that's `"gemmini"` and
`"rvv_opu"`.

`pipeline/generate_xpurt_main.py:_emit` (pre-fix) used the
`--backends` list for both (a) the dispatch fn symbol suffix and
(b) the strcmp against `e_->core_kind` in the dispatch branch.

The periodic_anchor binary was built with
`BACKENDS=gemmini_q31,rvv_opu`, so the generated branch was:

```c
if      (strcmp(e_->core_kind, "gemmini_q31") == 0) { /* call */ }
else if (strcmp(e_->core_kind, "rvv_opu")    == 0) { /* call */ }
else { printf("xpurt: WARN unknown core_kind '%s'..."); }
```

Every gemmini entry's `core_kind == "gemmini"` fell through to the
else clause — 100 dispatches dropped silently. The model output
buffer stayed at its static zero init; verify saw it as
quant-zero-distance from the golden (max_abs_err = the int8 quant
range for that network).

The 100 "WARN unknown core_kind 'gemmini'" lines are in
`run.log:10727+`.

yolov8's "first 8 zeros, last 8 -42" pattern is consistent: the few
rvv_opu-targeted yolov8 ops did run and partially wrote the output;
the gemmini-targeted ops dropped, leaving the rest of the buffer at
its zero init (= -42 after sigmoid-LUT post-process for that
quantization, head still zero).

## Fix (committed)

`pipeline/generate_xpurt_main.py` now:
1. Pairs `core_kinds[i]` with `backends[i]` by index.
2. Emits `strcmp(e_->core_kind, "<core_kinds[i]>")` (schedule-side
   tag).
3. Dispatches through `MODEL_..._DISPATCH_FNS_<backends[i].upper()>`
   (compiled-symbol tag).
4. The else clause is now FATAL (`sys_reboot(SYS_REBOOT_COLD)`) so
   a future mismatch fails loud instead of producing zero-output.

This restores the originally-intended kind→backend separation
documented in the top-of-file docstring at lines 75-79.

## Implication for previously-reported policy makespans

The pre-fix run executed almost no kernels — only the rvv_opu
half of the hetero schedule's dispatches actually ran. So the
9.077 GC sim wall and the 3.018 ms yolov8 in-hetero wall in this
run reflect ONLY the rvv_opu work, with all gemmini work skipped.
Those numbers are not comparable to single-network baselines and
are now superseded.

Every previously-quoted policy makespan in
`PROVENANCE.md`'s scheduler-model table (periodic_anchor 75.6 ms,
heft 54.4 ms, MOSEK 51.1 ms, etc.) is from the scheduler model
and was never measured end-to-end. With the codegen fix in place,
those measurements can now be made.

## Next

Re-run periodic_anchor (and the other policies) with the patched
codegen. Same `BACKENDS=gemmini_q31,rvv_opu` is fine — the new
dispatch branches strcmp against the registry kind ("gemmini") but
dispatch through the compiled-symbol tag ("gemmini_q31"), so no
rebuild of the model object libs is required.

## Update: v2 rerun (Jun 3 17:xx) — codegen fix verified, second bug exposed

The codegen fix landed and the rerun (`policy_periodic_anchor_v2/run.log`)
confirms:

- No `WARN unknown core_kind` messages — the gemmini-side branches
  now dispatch.
- Sim cycles **31.3 GC** (vs the pre-fix 9.07 GC) — 3.5× more
  execution work fired.
- yolov8 wall in hetero context: **10.055 ms** (vs the pre-fix
  3.018 ms); it now executes real gemmini work too.
- yolov8 verify shows max_abs_err=30 (vs the pre-fix 45 with the
  all-zero head). Output is structured data, not the static init.
- dronet output[0] is now -33 (vs 0) — partial chain ran.

But mlp_control + dronet outputs are still bad. The cause is now a
*separate* bug visible underneath: the schedule fixture
`scheduled_networks_1yolo_4mlp_2dronet_firesim_greedy_periodic_profiled.json`
is itself incomplete:

| Network | IR ops | Schedule ops | Missing |
|:---|---:|---:|:---|
| mlp_control | 7 | 1-2 per instance | indices 1,3,4,5,6 (incl. final writeback) |
| dronet | 32 | 23 per instance | indices 3,6,11,14,19,22,29,30,31 (incl. final 3) |
| yolov8_nano | 212 | 158 | 54 ops (incl. zero-cost view/chunk2 plus others) |

The XPU-RT scheduler's profile-loader is filtering ops that have no
measured cycles in the profile DB (silu/relu fused into the prior op
during profiling, or view/chunk2 zero-cost reshapes). The harness
walker requires the FULL op chain — including zero-cost passthroughs
— to reach the final dispatch that writes `state.output`. When the
schedule omits the tail, the output buffer stays at its static init
for the missing ops' downstream consumers.

Two ways to close this gap:

1. **Harness-side**: walk IR + schedule together; for IR ops not in
   the schedule, dispatch on whichever core's worker reaches them
   first (or on a designated default core for zero-cost ops). This
   keeps the schedule as a placement hint and the IR as the source
   of truth for chain completeness.

2. **Scheduler-side**: emit a synthetic zero-cycle schedule entry
   for every filtered IR op so the round-trip is complete. The
   harness already handles dispatch_id=-1 (view/chunk2 zero-cost
   sentinel) — extend that path to cover all ops the scheduler
   currently drops.

The user's measurement-first request hits either way: until one of
these is done, multi-network FireSim cannot produce honest
end-to-end walls for mlp_control / dronet, only for yolov8 (which
runs almost-complete already).
