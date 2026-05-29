# Known issues during benchmark capture

## dronet × rvv_opu × firesim — FIXED (2026-05-27 evening)

**Resolution**: removed `CONFIG_RISCV_ISA_EXT_V=y` (and `_V_LAZY=n`) from
`harness/backends/rvv_opu.conf`. With Zephyr's V context-switch disabled
(matching merlin's working `build_dronet_gem_opu` setup), Zephyr no
longer issues `vsetvli` during boot init on hart 0 — the trap-into-loop
that caused the silent hang is gone. V codegen still works for the
generated kernels via their own `-march=rv64gcv` cflags; Zephyr core +
libc compile without V (which is what we want anyway under
`-fno-tree-vectorize`).

First clean run: 295,424 cycles, bit_exact=true (3 reps still TODO).
Compares cleanly with dronet × rvv × spike (275,800 cycles) — OPU is
slightly slower than plain RVV on this workload, much faster than
scalar (4,541,250 cycles), still slower than Gemmini (18,620 cycles).

Historical write-up preserved below for the diagnostic trail.

---

### Historical: dronet × rvv_opu × firesim — hangs in runworkload (pre-fix)

**Symptom**: `dronet_rvv_opu_int8` on FireSim never completes. Every
attempt (6 across two timeout configs) has been killed mid-run:

- With `FIRESIM_QUEUE_TIMEOUT=900` (15 min): SIGTERM'd at 900s with
  rc=-15. Three reps timed out identically.
- Without a timeout cap: simulator stays in the `RUNNING` phase
  indefinitely. Last observed at ~115 min wall with no progress in
  the live UART (`/scratch2/agustin/FIRESIM_RUNS_DIR/sim_slot_0/uartlog`
  last touched ~16 min into the run, then silent).

**What we know**:
- gemmini, gemmini_q31, scalar, and rvv all complete cleanly on the
  same dronet binary path. So this is OPU-specific, not a general
  pipeline issue.
- The infrasetup phase completes (the per-job `results-workload/<...>/`
  dir gets created).
- No uartlog ever materializes — the sim either hangs immediately on
  workload start, or before its first printk. SIGTERM-cancelled runs
  rsync no output, so we have nothing to grep.

**Hypotheses to investigate** (top hypothesis identified post-diag):

**TOP**: Zephyr's eager V-extension init runs on hart 0 (Gemmini tile),
which doesn't have V. The trap on V-state init never completes, so
Zephyr never reaches its first `printk`. Evidence:

- Kconfig diff vs gemmini (which boots fine on the same bitstream)
  shows the only meaningful difference is `CONFIG_RISCV_ISA_EXT_V=y`
  + `CONFIG_RISCV_ISA_EXT_V_LAZY=n` (rvv_opu) — both unset for gemmini.
- The bitstream is `GemminiAndOPUShuttleConfig`: 2 harts, tile 0 =
  stock Shuttle (no V), tile 1 = Saturn OPU (has V + OPU).
- The dts overlay `harness/boards/chipyard_riscv64.overlay` disables
  cpus 2-7 but leaves cpu@0 (no V) AND cpu@1 (V) enabled. Zephyr boots
  on hart 0 by default → V-init traps → hang.
- The 30-min diagnostic with no timeout confirmed: live uartlog stops
  growing at "Commencing simulation. tsibridge_t::tick skipping tick"
  (host-side driver chatter only) — zero output from the simulated CPU.

**Fix paths** (none small):
1. Per-target dts overlay that disables cpu@0 for rvv_opu / hetero
   builds, forcing Zephyr to boot directly on hart 1.
2. Zephyr-side CPU-feature detection that skips V init on harts that
   don't advertise it.

Either fix needs Zephyr / device-tree knowledge; not a 1-hour change.

**Diagnostic path** (when we revisit):
- Submit one `runworkload-full` with no timeout, but `tail -f` the
  shared live uartlog at `FIRESIM_RUNS_DIR/sim_slot_0/uartlog` in
  another terminal during the run. Look for the LAST line written —
  that tells us where it hung.
- If the sim never prints anything, instrument the Zephyr main with
  an early `printk("alive")` to see if it boots at all.
- Compare the rvv_opu kernels.c against the rvv kernels.c — the OPU
  kernels are RVV + OPU-specific intrinsics; isolating the first OPU
  call would localize the bug.

**Workaround for the current benchmark cycle**: rvv_opu cells are
skipped. The matrix collects scalar, rvv, gemmini, gemmini_q31,
hetero — enough to demonstrate the harness across accelerators.
Hetero also exercises the OPU tile (the schedule places some ops on
Saturn), but with Gemmini sharing the FPGA — if hetero completes
cleanly while rvv_opu doesn't, that suggests the standalone-OPU
runtime path is what's broken, not the OPU compute units themselves.

## FPGA out of service — kernel upgrade broke xdma.ko (2026-05-28 15:13)

**Status**: blocking. All FireSim runs fail at `firesim infrasetup` →
`firesim-load-xdma-module` → `insmod` with no .ko file argument.

**Cause**: the running kernel got upgraded to `6.8.0-117-generic` during
the session. xdma.ko was only built for older kernels: newest
available is `/lib/modules/6.8.0-101-generic/updates/xdma.ko`. The
loader script `/usr/local/bin/firesim-load-xdma-module` does:

```bash
insmod $(find /lib/modules/$(uname -r) -name "xdma.ko") poll_mode=1
```

`find` returns empty → command becomes `insmod poll_mode=1` →
"could not load module poll_mode=1: No such file or directory".

**Fix paths** (need host sudo, out of scratch):
1. Rebuild xdma.ko for `6.8.0-117-generic` from the xdma source tree
   that built the earlier .ko's.
2. Reboot the host into kernel `6.8.0-101-generic` (the newest one
   with a matching xdma.ko).
3. Pin the kernel via `apt-mark hold linux-image-generic` once the
   above lands, so the next apt update doesn't break it again.

**Impact on the baseline**: data captured BEFORE 15:13 today is still
valid. Captures launched after (MOSEK reps 2+3, 3-way reps 2+3) all
failed during infrasetup. The baseline at `notes/baseline_2026-05-28.md`
documents what's in hand.

---

## dronet × hetero_gemmini_opu × firesim — FIXED (2026-05-28)

**Fix landed:** `schedule_fixtures/dronet_hetero_gemmini_opu.json` updated
so all conv2d_s8 and linear_s8 dispatches route to CPU_P#0 (gemmini).
Structural ops (maxpool, bn, relu, add, sigmoid) stay on CPU_E#0
(rvv_opu). Result: `OVERALL: PASS`, `max_abs_err=0 max_rel_err=0`,
bit-exact against PyTorch golden. End-to-end cycle count 18.7M (wall
18206 mtime ticks) — essentially the same as standalone gemmini.

The schedule still exercises both tiles meaningfully: 12 gemmini
dispatches, 18 rvv_opu dispatches, exercising RoCC + Saturn OPU in
real cross-tile data flow.

Below is the full diagnostic trail kept for the lessons.

---

### Real root cause: cross-backend per-op numerical drift

Each backend (gemmini, rvv_opu) is bit-exact END-TO-END against
PyTorch golden, but their intermediate per-op outputs differ slightly
because the int8 multiply-and-requantize policies differ:
gemmini's systolic uses round-half-to-even, RVV's `vsmul` rounds
half-up, and the shift sequence isn't identical across both. Each
backend's full chain compounds its own micro-drift in a way that
cancels into a golden-matching final answer; mixing chains breaks
that property — the consumer kernel was tuned against its OWN
backend's per-op output, not the OTHER backend's, so the drift
accumulates.

The structural ops (maxpool, bn, relu, add, sigmoid) don't
requantize — they preserve the int8 input layout and pass values
through (or do non-quantization-affecting transforms). They can
safely run on either backend without breaking the chain.

### What the wrong hypotheses taught us

The bug LOOKED like cache coherence: deterministic per-rep error,
correct routing per xpurt_trace, fence-and-flush instincts. We
spent time on:

1. **`fence rw,rw` before + after each dispatch** — no effect on
   correctness (kept in `generate_xpurt_main.py` as good hygiene
   for Gemmini DMA per merlin's loader recipe).
2. **L1 dcache eviction via 128KB scratch sweep** — no effect.
3. **Zicbom `cbo.inval`** — illegal-instruction trap (mcause=2,
   mtval=0x7a00f); the 2026-05-06 bitstream wasn't built with
   Zicbom support.

Identical failure mode across three increasingly aggressive cache
remedies disproved the cache hypothesis. The next experiment —
forcing ALL dispatches to one tile — produced bit-exact output,
confirming the per-op-drift hypothesis (one consistent numerical
regime = correct).

### Implications for future hetero workloads

When authoring a new hetero schedule fixture, **place all
requantization-producing ops (conv2d_*, linear_*, depthwise_*,
matmul_*) on a single backend.** Structural / passthrough ops
(maxpool, bn, relu, add, sigmoid, view, chunk, cat) can mix
freely across backends. Document this in the fixture's
`_provenance.notes`.

If we ever want convs+linears split across backends (to actually
leverage two MACs concurrently), we'd need to align the per-op
requantize policies between the kernels — sketch in
generate_kernels.py's `output_multiplier`/`output_shift` math
plus possibly the inline rounding intrinsics. Multi-week work;
not on the critical path.

---

### Historical: dronet × hetero_gemmini_opu × firesim — verify FAIL (older, superseded by above)

**FINAL DIAGNOSIS, 2026-05-28 after extended debug session**

Original guess (cache coherence) was WRONG. The real cause is
cross-backend per-op numerical drift.

### Evidence that ruled out cache coherence

Tried three increasingly aggressive cache remedies, all with the
EXACT SAME failure mode (max_abs_err=52, actual=[-108,127] vs
golden=[-56,127]):

1. **`fence rw,rw` before + after each dispatch**. Standard memory
   ordering fence. No effect on output.

2. **L1 dcache eviction via 128KB scratch sweep** before each
   dispatch. Forces every L1 line to evict (assuming reasonable
   L1 size). Adds ~6k cycles overhead per call. No effect on output.

3. **Zicbom `cbo.inval`**. Standard RISC-V cache-block-management
   instruction. Triggered illegal-instruction trap (mcause=2,
   mtval=0x7a00f) — the May 6 bitstream was synthesized WITHOUT
   Zicbom support.

Identical failures across three very different cache strategies
disprove the cache-staleness hypothesis. If the consumer hart were
reading stale L1 lines, at least ONE of those remedies would have
moved the needle.

### Actual cause: cross-backend numerical drift

Each backend (gemmini, rvv_opu) is bit-exact END-TO-END against
PyTorch golden:
  - dronet_gemmini_int8 (all gemmini): 18620 cycles, bit_exact=True
  - dronet_rvv_opu_int8 (all rvv_opu): 295424 cycles, bit_exact=True

But each backend's INTERMEDIATE per-op outputs differ slightly due
to backend-specific rounding and quantization paths. Gemmini's
systolic-array rounding rounds-half-to-even, RVV's vector multiply
uses VSMUL which rounds-half-up; gemmini's requantize takes a
single-rounding path, RVV does two-step shift+round. Each backend's
full chain compounds its own micro-drift in a way that happens to
yield the same final golden answer. Mixing chains — one op's output
in one rounding regime, fed to the next op which expects the other
regime — doesn't preserve that property and accumulates a real,
deterministic, non-zero error.

The all-gemmini-routed test (force every dispatch to tile 0) PASSED
bit-exactly because it kept the entire chain in one numerical regime.

### Implication for heterogeneous execution

`OVERALL: PASS` against a strict bit-exact `atol=0 rtol=0` is the
wrong success criterion for hetero. Mixed-backend chains will not
agree bit-exactly with single-backend golden by definition.

Two viable paths forward:
1. **Loosen verify tolerance for hetero cells.** Define an
   acceptable `linf` threshold (e.g., 1-2 quant units), tune
   per-cell. Hetero cells then report verify=PASS within that
   bound. Honest about the model the silicon implements.
2. **Cross-backend numerical alignment.** Pick one canonical
   rounding/requantize policy (e.g., match PyTorch's quantization)
   and patch both backends' kernels to obey it. Likely a lot of
   per-kernel work — multiple weeks if you want to do it right.

For the current benchmark cycle, hetero is excluded from the
baseline. The diagnostic value of (1) is high since it lets the
hetero cycle profile be captured + dashboard'd, with the verify
caveat surfaced in the metric column.

### What WAS load-bearing during this debug

`fence rw,rw` before+after dispatch — kept in
`pipeline/generate_xpurt_main.py`. It's the right thing for Gemmini
DMA hygiene per merlin's `embedded_elf_loader.c`, even though it
didn't fix hetero. Cost is ~zero; semantic value is real.

---

### Historical: dronet × hetero_gemmini_opu × firesim — verify FAIL (cross-tile L1 dcache staleness, diagnosis complete) [SUPERSEDED]

**FULL DIAGNOSIS, 2026-05-27 evening**

### Confirmed cross-tile bug

Ran an isolation experiment: temporarily swapped the schedule fixture
to route ALL 30 dispatches to a single tile (`CPU_P#0` =
Gemmini/Shuttle tile 0). Result: `OVERALL: PASS`, output bit-exact,
total cycles 18,031,198 — identical to standalone gemmini's 18,620
mtime cycles. So the kernels, the schedule machinery, the buffer
chain, and the xpurt walker are ALL CORRECT.

The bug appears ONLY when dispatches split across harts. Hetero with
the original 9-gemmini-21-rvv_opu schedule produces consistent
`max_abs_err=52, max_rel_err=0.929` — deterministic across reps,
meaning it's a structural data-flow corruption, not a race.

### Mechanism

After Zephyr's BSS-zero init at boot, both harts' L1 dcaches have
the `buf_dronet_*` lines cached in Shared state (zeros). When
dispatch 0 (Gemmini conv2d on tile 0) runs, Gemmini's RoCC DMA
writes to `buf_dronet_conv_modules_0` via the SoC coherence point —
bypassing tile 0's L1. The write lands in L2/DRAM, but tile 1's L1
copies of those lines are **not invalidated** by the Gemmini DMA
write. When dispatch 1 (maxpool on tile 1) reads
`buf_dronet_conv_modules_0`, tile 1's L1 HITS on the stale zeros.
The rest of the network cascades from zero input + biases, producing
a deterministically-wrong output that's close to golden in magnitude
(`max=127` matches) but offset on the negative side (`-108` vs
`-56`).

### What I tried that did NOT fix it

- **`fence rw,rw` before+after each dispatch** (committed to
  `generate_xpurt_main.py`): correctly flushes the calling hart's
  store buffer, doesn't invalidate the other hart's L1.
- The merlin recipe of fence rw,rw works for merlin because their
  `merlin_hetero_runner` does **two independent whole-model
  inferences** (one per hart), not per-op dispatch across harts.
  Merlin avoids cross-tile data flow by design.

### Fix paths (none small)

1. **Explicit L1 invalidate before cross-tile reads.** Issue a
   `cbo.inval` (Zicbom) or vendor-specific cache-management
   instruction on the consumer hart for the buffer range it's
   about to read. Requires confirming the bitstream supports
   the relevant Zicbom/Zicboz/T-Head xtheadcmo extension — the
   May 6 bitstream was built without those Kconfigs explicitly
   set so support is unverified. Worst case: rebuild bitstream.

2. **Move cross-tile buffers to non-cacheable memory.** Add a
   linker section like `.noncacheable` mapped to a PMP region
   marked NC. Place all `buf_dronet_*` symbols there. Trivial
   software fix but requires PMP setup in the chipyard board's
   early init.

3. **Force per-tile sub-inference (merlin pattern).** Split each
   inference into a tile-local sub-inference and merge results
   at a synchronization point. Defeats the per-op heterogeneous
   scheduling goal — only useful as a stopgap.

4. **Buffer pre-touch from consumer hart.** Before the dispatch
   loop, have each consumer hart read every buffer it will later
   read. This puts the lines in L1 in Shared state ONCE — but
   the issue isn't the first read, it's that subsequent producer
   writes don't invalidate. Doesn't help.

### Recommended next step

When this is picked up: check chipyard's `BaseXilinxAlveoU250Config`
for whether L1 dcaches participate in TileLink coherence on the
GemminiAndOPUShuttle. If they do, the bug is somewhere unexpected
(maybe Gemmini's DMA port doesn't get acknowledged correctly). If
they don't, the fix is option 2 (non-cacheable region) — implement
it in `harness/boards/chipyard_riscv64.overlay` + matching linker
fragment.

---

### Historical: dronet × hetero_gemmini_opu × firesim — verify FAIL (older)

**Refined diagnosis (after the V-off fix + xpurt trace inspection)**:
Routing is CORRECT — `xpurt_trace.csv` confirms gemmini got its 9
dispatches (5,980 cycles total), rvv_opu got its 21 (22,820 cycles).
Per-op profile from `MODELBLASTER_PROFILE_BEGIN` shows reasonable
backend-specific cycle counts (gemmini at hundreds of K-cycles per
conv, opu at millions for the same shape — expected delta). The
verify still fails (`max_abs_err=52, max_rel_err=0.929`).

This is **not** a routing bug; it's **data corruption at the tile
boundary** between Gemmini (tile 0) and Saturn OPU (tile 1) on the
`alveo_u250_firesim_shuttle_gemmini_opu` bitstream. Gemmini writes
`buf_dronet_conv_modules_0` then OPU's maxpool reads the same buffer
— if the SoC's interconnect isn't cache-coherent between L1s, OPU
sees stale L2/DRAM contents.

**Diagnostic path (when we revisit)**:
1. Add a `fence rw, rw` + cache-flush sequence at the harness's tile-
   boundary points in `examples/xpurt_demo/.../xpurt_demo_schedule_main.c`.
2. Compare merlin's `build_dronet_gem_opu` driver — they have a
   working hetero on the same bitstream; whatever cache mgmt they
   do is the recipe.
3. Alternative: place all cross-tile buffers in a non-cacheable
   memory region by overriding the linker script for buffers tagged
   `produced_by != consumed_by`. Heavier change.

**Workaround for the current benchmark cycle**: hetero cells stay
excluded from the baseline. We have clean baselines for scalar, rvv,
gemmini_int8, gemmini_q31, and rvv_opu individually — that's enough
to drive the modification demo and characterize each accelerator
independently. Hetero (combining accelerators) is a follow-up.

---

### Historical: dronet × hetero_gemmini_opu × firesim — verify FAIL

**Symptom**: All 3 reps run to completion on FireSim (5-6 min each,
339s wall), produce a full per-op cycle profile (`xpurt_trace.csv`,
~32M total cycles), but the verify step fails with
`max_abs_err=52, max_rel_err=0.929` — output is nowhere near the
golden reference.

**What we know**:
- gemmini_int8 alone is bit-exact (3 reps × 18620 cycles).
- The hetero per-op cycle profile shows conv ops at ~3-7M cycles each,
  which is far higher than gemmini's expected per-conv cost and closer
  to RVV. Suggests hetero is routing ALL compute to RVV (not actually
  exploiting Gemmini tiles), AND the routing has a data-flow bug.
- Per `workloads.yaml`, the hetero cell uses a schedule fixture at
  `schedule_fixtures/yolov8n_hetero_gemmini_opu.json` (despite the
  cell being dronet_hetero — that path looks wrong, may be a copy-paste).
- The cell is tagged `blocked_by: P2.1-schedule` in workloads.yaml —
  i.e. it's a known WIP. Re-running it now confirms it's still WIP.

**Diagnostic path**: examine `xpurt_trace.csv` to see which tile each
dispatch went to. If everything went to tile 0 (RVV) when it should
have been split between tiles 0 and 1, the schedule emitter is the
bug. Otherwise check `pipeline/ingest_xpurt_schedule.py` for dispatch
ID handling — the schedule fixture references dispatch IDs that may
have shifted between IR versions.

**Workaround for the current benchmark cycle**: hetero cells are
excluded from the baseline. Cells we do have clean data for: scalar,
rvv, gemmini_int8 (bit-exact), gemmini_q31 (deterministic but bit_exact=False).

---

## dronet × gemmini_q31 — bit_exact=False (documented Q31 drift)

Three reps each produce 9039 cycles, all deterministic. Output
differs from the spike-emulated reference (linf=72.0 typical).
This is **expected** — the Q31 quantization path's integer
arithmetic differs from spike's emulated Q31, by a documented amount.
The number is stable, the run is correct on its own terms, just not
bit-identical to the reference. Per `tmp/07_gotchas.md` #6.
