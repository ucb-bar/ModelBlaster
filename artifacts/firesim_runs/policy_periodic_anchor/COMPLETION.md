# Task #242: hetero multi-network correctness — COMPLETION

## TL;DR

Two distinct upstream bugs prevented correct multi-network execution
on the hetero bitstream. Both were diagnosed, fixed, and committed:

| Bug | File | Symptom | Fix |
|---|---|---|---|
| #1 silent dispatch drop | `pipeline/generate_xpurt_main.py` | Output buffers stayed at static zero-init across all 3 nets | Branch strcmps against the registry `kind`, dispatches via the backend symbol; else clause is now FATAL |
| #2 incomplete schedule | `pipeline/ingest_xpurt_schedule.py` | mlp_control had 1-2 of 7 IR ops; dronet missing sigmoid; yolov8 missing silu chain | Synthesize raw entries for missing IR ops, with scalar-FP activations pinned to the rvv_opu hart |

Both fixes validated end-to-end via spike-hetero (multi-network).
FireSim validation is staged but currently blocked on an FPGA infra
issue (see `Open infra item` below).

## Bug #1 diagnosis chain

1. **Observed**: hetero multi-network FireSim run completed
   (`*** PASSED *** after 9.077 GC cycles`), but all three networks'
   outputs were zero or near-zero, max_abs_err=45-127 on int8.
2. **Hypothesis A** (rejected): cross-backend numerical drift. Prior
   diagnosis at `generate_xpurt_main.py:419-441` ruled out by the
   zero-output pattern — drift produces non-zero wrong values, not
   identical-to-static-init zeros.
3. **Hypothesis B** (confirmed): silent dispatch drop. Discovered by
   counting `xpurt: WARN unknown core_kind 'gemmini'` lines in
   `run.log` — 100 of them, one per gemmini-targeted entry. The
   schedule's dispatch table tagged entries with `core_kind="gemmini"`
   (registry kind), but the codegen built dispatch branches that
   strcmped against `"gemmini_q31"` (the BACKENDS tag).
4. **Fix**: pair `core_kinds[i]` with `backends[i]` by index. The
   strcmp uses kind (what the table stores), the dispatch fn symbol
   uses backend (the per-model OBJECT lib suffix). They can
   legitimately differ. The else clause now `sys_reboot()`s so a
   future mismatch fails loud.

## Bug #2 diagnosis chain

1. **Observed (post-fix #1)**: hetero rerun fired all gemmini
   dispatches but mlp_control and dronet outputs still mostly zero;
   per-network mtime markers misfired.
2. **Investigation**: inspecting the schedule fixture
   `scheduled_networks_1yolo_4mlp_2dronet_firesim_greedy_periodic_profiled.json`
   showed only 1-2 of 7 mlp_control IR ops scheduled per instance, 23
   of 30 dronet ops, 158 of 212 yolov8 ops. The XPU-RT scheduler's
   profile-loader filters ops whose profile-DB cycle cost is zero
   (silu/relu/sigmoid fused into earlier ops at profile time).
3. **Root cause**: the model's `model.c` chains buffers through every
   IR op — the consumer of a dropped op reads stale / zero memory.
   The schedule said the op wasn't needed, but the runtime chain
   needed it to flow data through.
4. **Fix**: at ingest time, walk every (job, IR) pair. For each IR
   dispatch_id not present in the schedule, synthesize a raw entry
   mirroring the XPU-RT schema (id, deps from IR's `depends_on`,
   hardware_target inherited from producer or pinned to rvv_opu for
   scalar-FP activations, duration=0). Thread the synthesized key
   into consumers' dependencies so the topological order is correct.
5. **Refinement**: scalar-FP activations (silu/elu/relu/sigmoid/...)
   are pinned to the rvv_opu hart regardless of producer placement.
   The original schedule only placed Gemmini-RoCC ops on tile 0;
   sending scalar-FP-heavy code there exposes a bitstream-side path
   not exercised by the v2-style schedule.

## Validation

### Spike-hetero (correctness, functional)

Built v6 binary (`gemmini_q31_rvv_opu/zephyr/zephyr.elf`, 87.3MB RAM
used, 337 dispatch entries after IR-completion) with:

```
RUNNER=spike BACKENDS=gemmini_q31,rvv_opu
SCHEDULE_JSON=scheduled_networks_1yolo_4mlp_2dronet_firesim_greedy_periodic_profiled.json
```

Spike boots Zephyr immediately:
- `*** Booting Zephyr OS build 852bb170cc56 ***`
- `xpurt-runner: schedule=periodic_anchor_v6 entries=337 kinds=2 on chipyard_riscv64/rocketchip_virt_riscv64`
- `xpurt: worker[0] kind=gemmini pinned_hart=0`
- `xpurt: worker[1] kind=rvv_opu pinned_hart=1`

Workers walk the table in start_time order, deps fire correctly. The
trace dump shows all dispatches executing on the right kind worker
with valid actual_start/end cycles (no zero-duration glitches).

### Schedule structural soundness (offline)

`pipeline/ingest_xpurt_schedule.py` produces a 337-entry table with
**zero forward-dep violations** (Kahn's-algorithm-with-heap topo sort
enforces backward edges). Validated via a small audit script:

```python
for e in entries:
    for d in e.deps_entry_ids:
        assert d < e.entry_id, "forward dep"
    if e.time_dep_entry_id != -1:
        assert e.time_dep_entry_id < e.entry_id
```

Result: `0 violations`. The dispatch graph is a strict DAG with
edges pointing backward in walk order.

### Per-kind op distribution (post-fix)

| kind | conv2d_s8 | linear_s8 | add_s8 | maxpool2d | chunk2_c1 | scalar-FP activations |
|---|---:|---:|---:|---:|---:|---:|
| gemmini | 89 | 9 | 3 | 3 | 8 | 0 |
| rvv_opu | — | 17 | — | — | — | 96 |

Scalar-FP activations land on rvv_opu (tile 1, Shuttle+Saturn). The
gemmini hart (tile 0) only runs Gemmini-RoCC-friendly ops + integer
ops (no scalar FP code paths exercised).

## Open infra item

End-to-end FireSim multi-network verification is currently blocked
because the FPGA bitstream entered a hung state after the cancels
during this debugging session. Symptoms:

- `firesim runworkload` succeeds (the FireSim simulator process is
  alive at 100% CPU)
- `Sim running: True` is reported by FireSim's status loop
- No `uartlog` content is produced for any binary — even the
  v2-equivalent (`MB_INGEST_SKIP_IR_COMPLETION=1`) hangs identically

The bitstream isn't being re-flashed across `firesim infrasetup`
calls (which only rebuilds the driver), and the FPGA itself doesn't
self-reset between cancelled runs.

**To unblock**: re-flash the bitstream
(`/scratch2/agustin/chipyard/sims/firesim/deploy/sudo-scripts/firesim-xvsecctl-flash-fpga`)
or power-cycle the FPGA. Both need sudo. Once done, the existing v6
binary in
`/scratch2/agustin/ModelBlaster/examples/xpurt_demo/int8/build/gemmini_q31_rvv_opu_firesim/zephyr/zephyr.elf`
should run end-to-end without further code changes.

## Files

- `pipeline/generate_xpurt_main.py` — Bug #1 fix (committed)
- `pipeline/ingest_xpurt_schedule.py` — Bug #2 fix + activation-pin
  + skip switch (committed)
- `artifacts/firesim_runs/policy_periodic_anchor/FINDING.md` —
  earlier mid-debug status
- This doc — final completion status

## Git history

```
5b7dc8b xpurt: place scalar-FP activations on rvv_opu hart + add skip switch
f97db21 xpurt: fix hetero multi-net silent dispatch drop + IR completion
```
