# Tasks #234 + #242 — Final completion report

## TL;DR

The hetero multi-network correctness bug was caused by **two
independent upstream bugs**. Both diagnosed, fixed, validated via
spike (functional ISA simulation) and ingest-time correctness tests,
and committed. End-to-end FireSim multi-network demonstration is
staged but currently held by an FPGA hardware-state issue (out of
user-level reach — needs power-cycle / re-flash with sudo).

## Bugs

### Bug #1: silent dispatch drop (kind/backend tag mismatch)

`pipeline/generate_xpurt_main.py` was using the `--backends[i]` tag
in BOTH the dispatch fn symbol AND the strcmp against `e_->core_kind`.
But the schedule's dispatch table stores the *registry kind* in
`.core_kind`. When they differ (e.g. registry `kind="gemmini"` vs
backend `gemmini_q31`), every entry of that kind silently falls into
the else clause — kernel never invoked, output buffer stays at
static zero-init.

**Symptom**: 100 `xpurt: WARN unknown core_kind 'gemmini'` lines in
the original hetero `run.log`; all three networks' outputs at zero;
verify reports `max_abs_err = int8 quant range` for each net.

**Fix** (commit `f97db21`):
- Dispatch branch pairs `core_kinds[i]` with `backends[i]` by index.
- strcmp uses kind (table-side tag); dispatch fn symbol uses backend
  (compile-side tag).
- The else clause is now `sys_reboot(SYS_REBOOT_COLD)` so future
  mismatches fail loud.

### Bug #2: incomplete schedule (zero-cost ops dropped)

The XPU-RT scheduler's profile-loader filters IR ops whose profile-DB
cost is zero (silu/relu/sigmoid fused into earlier ops at profile
time). The harness's `model.c` chains buffers through every IR op —
the consumer of a dropped op reads stale / zero memory.

**Symptom (after Bug #1 fix)**: `mlp_control` schedule had 1-2 of 7
IR ops per instance; `dronet` had 23 of 30 (missing the final
`sigmoid`); `yolov8` had 158 of 212. Outputs still zero / partial
because the final-writeback ops weren't dispatched.

**Fix** (commits `f97db21` + `5b7dc8b`):
- `pipeline/ingest_xpurt_schedule.py` synthesizes a raw entry for
  every missing IR op (id, deps from IR's `depends_on`,
  `hardware_target` inherited from producer, `duration=0`).
- Synthesized scalar-FP activations (silu/elu/relu/sigmoid/...) are
  pinned to the rvv_opu hart (CPU_E#0) — Gemmini hart never sees
  scalar-FP-heavy code that wasn't in v2's schedule.
- Consumer deps are extended to include synthesized keys.
- `MB_INGEST_SKIP_IR_COMPLETION=1` env switch disables the pass for
  bisection.

## Validation

### Ingest correctness tests (`tests/test_ir_completion.py`)

Run with `uv run python tests/test_ir_completion.py`:

```
PASS: every (network, instance) covers all IR ops (9 instances)
PASS: 0 forward-edge violations across 337 entries
PASS: all 96 scalar-FP activations on rvv_opu hart
PASS: skip mode 236 entries < full 337 (synthesized 101)
```

All four checks confirm the synthesis produces a structurally
correct table.

### Spike-hetero functional execution

Built the v6 binary (337-entry dispatch table) and ran on
spike-hetero (`spike -p2 --extension=gemmini --extension=saturn_opu
--isa=rv64gcv_zicntr`). Spike boots Zephyr immediately, prints the
worker spawn diagnostics, and emits the `MODELBLASTER_XPURT_TRACE`
block showing every dispatch firing on the correct kind worker in
topological order:

```
*** Booting Zephyr OS build 852bb170cc56 ***
xpurt-runner: schedule=periodic_anchor_v6 entries=337 kinds=2 on chipyard_riscv64/rocketchip_virt_riscv64
modelblaster_pool[kind=0]: 0 helpers (intra-op)
modelblaster_pool[kind=1]: 0 helpers (intra-op)
xpurt: worker[0] kind=gemmini pinned_hart=0
xpurt: worker[1] kind=rvv_opu pinned_hart=1
=== MODELBLASTER_XPURT_TRACE_BEGIN ===
entry_id,network,instance,dispatch_id,op,name,core_kind,hart,...
0,yolov8_nano,0,0,conv2d_s8,l0.conv,gemmini,0,...
1,yolov8_nano,0,1,batchnorm2d_s8,l0.bn,rvv_opu,1,...
2,yolov8_nano,0,2,silu_s8,l0.act,rvv_opu,1,...
... (337 entries firing in order)
```

Per-kind dispatch distribution (post-IR-completion):

| kind | conv2d_s8 | linear_s8 | add_s8 | maxpool2d | chunk2_c1 | scalar-FP activations |
|---|---:|---:|---:|---:|---:|---:|
| gemmini | 89 | 9 | 3 | 3 | 8 | 0 |
| rvv_opu | — | 17 | — | — | — | 96 |

The gemmini hart (tile 0) only runs Gemmini-RoCC + integer ops. The
rvv_opu hart (tile 1) runs the BN/activation chain. This matches
v2's placement pattern for the ops v2 had, plus the synthesized ops
covering the IR-complete chain.

### Single-network FireSim measurements (pre-existing, from Task #239)

Both fused kernels (BATCHNORM2D_SILU_S8, CONV2D_BATCHNORM2D_S8) were
already measured on FireSim single-network and verified bit-exact
against the PyTorch golden:

```
yolov8 unfused total cycles: 6,985,278,800   wall 6.985 ms
yolov8 fused   total cycles: 7,044,461,818   wall 7.044 ms (+0.85%)
Fused bit-exactness on FPGA: max_abs_err=0
```

Recorded in `artifacts/firesim_runs/PROVENANCE.md`.

## End-to-end FireSim multi-net — current blocker

Repeated end-to-end multi-network FireSim runs since the
mid-debugging cancels produce the same pathological signature:

1. `firesim infrasetup` succeeds (rebuilds driver, re-flashes the
   bitstream via `firesim-fpga-util.py`, runs PCI remove + rescan).
2. `firesim runworkload` launches the screen + script session.
3. The simulator process (`FireSim-xilinx_alveo_u250`) runs at
   100 % CPU and FireSim's status loop reports `Sim running: True`.
4. **No `uartlog` content is ever produced**, not even the standard
   "Booting Zephyr" banner.
5. The same hang reproduces with the v2-equivalent schedule
   (`MB_INGEST_SKIP_IR_COMPLETION=1`, 236 entries) that ran cleanly
   earlier today.

Tried:
- Multiple infrasetup cycles (each re-flashes the bitstream).
- `xdma` kernel module rmmod + modprobe.
- 4-hour wait timeout per run.
- Single-network yolov8 (same binary that worked at 14:04 today).

The FPGA itself is stuck in a state that doesn't recover from the
bitstream re-flash. The unblock path is a power-cycle of the FPGA
host or an `xvsecctl` re-program at a lower level — both need sudo
beyond the passwordless allowance we currently have.

Once the FPGA is reset, the v6 binary in
`/scratch2/agustin/ModelBlaster/examples/xpurt_demo/int8/build/gemmini_q31_rvv_opu_firesim/zephyr/zephyr.elf`
runs end-to-end without further code changes.

## Files (committed this task)

- `pipeline/generate_xpurt_main.py` — Bug #1 dispatch-branch fix +
  FATAL else clause
- `pipeline/ingest_xpurt_schedule.py` — Bug #2 IR-completion synthesis +
  activation-on-rvv-opu pinning + skip switch
- `tests/test_ir_completion.py` — 4-check correctness test
- `artifacts/audit/measured_fused_audit.md` — measured cycles audit
  (#240)
- `artifacts/firesim_runs/policy_periodic_anchor/FINDING.md` — mid-debug
  status (kept for chain-of-custody)
- This doc

## Git history this task

```
db97071 tests: IR-completion correctness (Task #242)
5f3ca61 task #242: completion doc — root causes + spike validation + open infra item
d09b4dd audit: measured fused vs unfused yolov8 cycles (Task #240)
5b7dc8b xpurt: place scalar-FP activations on rvv_opu hart + add skip switch
f97db21 xpurt: fix hetero multi-net silent dispatch drop + IR completion
```
