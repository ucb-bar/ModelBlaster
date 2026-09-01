# ⚠️  REJECTED DEMONSTRATIONS — DO NOT CITE

The files in this directory marked with the `dima_*` prefix produced
a numerical claim that does NOT hold up under examination:

> "BEFORE 75.57 ms / 248 dispatches → AFTER 64.78 ms / 258 dispatches,
>  Δ −10.79 ms / −14.3% from agent-applied fuse hint."

The agentic-loop **decision** trace (granularity_loop scored 330
candidates, picked `mlp_control[0..5]` fuse, emitted Contract-2 hint)
is real and reproducible. The **scheduling result** is not, for two
compounding reasons documented here so the bug doesn't recur:

## Reason 1 — IR rewrite stripped work the profile DB didn't know about

To produce the AFTER schedule I overwrote
`/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/vmfb/mlp_control/.../mlp_control.fp32_dispatch_graph.json`
with a hand-edited version that deleted `dispatch_1..5` and left
`dispatch_0` with no internal dependencies.

Dima's `scripts/run_xpurt_schedule.py` then read this IR and saw
mlp_control as needing only 2 dispatches per instance (dispatch_0 and
dispatch_6) instead of 7. Each instance's "work" dropped from
527,645 cycles to 52,842 cycles in the scheduler's view.

But the *actual compute* — three linear layers + three ELUs, totaling
523,890 cycles per mlp_control instance — still has to happen. There
is no fused kernel in
`/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/vmfb/mlp_control/.../`
that takes the merged work and runs it as one dispatch. The schedule
counted the work as gone; the hardware would still need to do it.

## Reason 2 — the "honest patch" silently no-op'd

After identifying Reason 1 I tried to patch the profile so dispatch_0
carries the SUM of the original dispatch_0..5 cycles (498,890 after
subtracting a 5×5µs handshake-elimination credit). The result was
saved as `dima_after_fuse_HONEST.json`.

That ALSO doesn't prove anything, because the scheduler's
`cycles_source: db` mode reads cycles from a profile DB ingest path,
NOT from the
`zephyr-chipyard-sw/gen/profile/sweep_v8/.../results.csv` I edited.
The patched CSV was a dead file. The "HONEST" makespan of 64.73 ms is
essentially identical to the bookkeeping-fiction 64.78 ms because
both runs read the same unmodified DB.

Verification of the no-op: `dima_after_fuse_HONEST.json` shows
mlp_control0's total work as **0.0593 ms** in the fixture (the
unchanged dispatch_0 at ~0.050 ms + dispatch_6 at ~0.003 ms), not the
~0.50 ms it would be if the patch had applied.

## What's still valid in this directory

- `granularity/granularity_result.json` — the real candidate-scoring
  output from `granularity_loop.py`. The decision logic itself is
  reproducible. **The Δmakespan numbers in the granularity result are
  PREDICTED from re-scheduling the in-memory workload**, not measured;
  they suffer from the same accounting issue but the candidate ranking
  is still useful as a starting point.
- `dima_hint.json` — the Contract-2 fusion hint the agent emitted.
  Real Contract-2 output; another implementation could apply it
  correctly via measured cycles.
- `dima_reference.json` / `dima_reference.png` — the BEFORE schedule
  from Dima's command. Valid as the proven decomposed baseline.

## What a HONEST AFTER measurement requires

1. A fused kernel that actually does the work of the merged ops, built
   for the workload's target. For Dima's workload that's
   gemmini_q31 / fp32 RVV — we do NOT have this kernel; the
   Bedrock-generated `linear_s8_elu_s8` is for rvv_opu / int8 only.
2. Build the fused kernel into a harness ELF.
3. Run on FireSim, capture xpurt trace, extract per-dispatch cycles.
4. Substitute those measured cycles into the scheduler's profile
   ingest path (not a side-file).
5. Re-run the scheduler. Compare the resulting makespan to the
   baseline.

Steps 1-5 are what Phase 2 of the current plan
(`~/.claude/plans/buzzing-wiggling-pretzel.md`) builds out properly.

## What IS proven from this session

- Compaction + automerge + oracle floor post-passes (xpu-rt/, 88
  tests green, idempotent on MOSEK).
- Saturn-OPU header fix (`SATURN_OPU_KEEP_REGISTER_MACROS`).
- Bedrock-generated `linear_s8` kernel for mlp_control on rvv_opu int8:
  spike PASS with `max_abs_err=0 max_rel_err=0`, **measured 7.1x
  cycles speedup** vs scalar reference (730,069 → 102,725 cycles
  total network). See `artifacts/bundle/walkthrough_v2/llm_kernels/`.

None of those rely on the rejected demonstration in this directory.
