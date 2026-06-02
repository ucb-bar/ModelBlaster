# FireSim integration for the benchmark harness

After the spike-side matrix completed (commit `9c7e3b2`), the remaining
holes in the baseline coverage are all FireSim-only by design:

1. **Authoritative accelerator cycle counts** for
   `dronet × {rvv_opu, gemmini, gemmini_q31, hetero}` +
   `yolov8n × {rvv_opu, gemmini, gemmini_q31, hetero}`. Spike treats
   Gemmini / Saturn-OPU custom instructions as atomic (zero-cycle)
   per `tmp/07_gotchas.md` #6; the cycle numbers we have on those
   cells are functional-only. FireSim runs the actual
   `GemminiAndOPUShuttleConfig` (`alveo_u250_firesim_shuttle_gemmini_opu`)
   bitstream so cycles are authoritative.

2. **`gemmini_q31` numerics**. Spike's `libgemmini.so` is built with
   `acc_scale_t = float`; our Q31 kernels emit Q0.31 int multipliers
   that get reinterpreted as float bits → `linf=9..72` drift. FireSim
   on a Q31GemminiRocketConfig bitstream is the authoritative path.

3. **`hetero_gemmini_opu` verify**. spike-hetero emulates Gemmini + OPU
   via `libgemmini.so` and `libsaturn_opu.so`; the emulated
   `acc_scale_t` doesn't match the bitstream's, so verify drifts
   (`linf=52`). Authoritative on FireSim.

## Status of FireSim from this shell

**Verified reachable** (commit `7464c0a`). Activation flow:

```bash
source scripts/setup_benchmark_env.sh   # now also activates firesim env
firesim kill   # smoke-test; ~5s, exits clean on empty FPGA host state
```

The setup script:
- Sources `chipyard/sims/firesim/sourceme-manager.sh` (puts `firesim` on PATH)
- Loads `~/.ssh/firesim` into `ssh-agent` (FireSim's "Non-AWS Host" path SSHes to localhost)

`config_runtime.yaml::default_hw_config` is pinned to
`alveo_u250_firesim_shuttle_gemmini_opu` — the
`GemminiAndOPUShuttleConfig` bitstream (2-tile Shuttle SoC: tile 0
Gemmini RoCC, tile 1 Saturn OPU vLen=128). This is the "all-in-one"
hetero config the user referenced.

## What's still needed

### 1. Zephyr sample for `chipyard_riscv64` that hosts the harness

Today's `harness/` builds for `spike_riscv64`. For FireSim we need an
equivalent that targets `chipyard_riscv64/rocketchip_virt_riscv64`
(the board the merlin sample uses). The harness has to:

- Emit the same `MODELBLASTER_PROFILE_*` / `MODELBLASTER_VERIFY_*` /
  `MODELBLASTER_WALL_CYCLES_*` markers the spike runner already
  consumes (`validation/spike_runner.py:_parse_*`).
- Pin per-tile execution: hart 0 (Gemmini tile) runs Gemmini-affined
  ops, hart 1 (Saturn-OPU tile) runs rvv_opu-affined ops. Mirror the
  `MERLIN_CPU_FEATURES` mask trick in
  `merlin/benchmarks/firesim_shuttle/run_hetero.sh`.
- Use the `chipyard_riscv64.overlay` from
  `/scratch2/agustin/zephyr-chipyard-sw/samples/merlin_hetero_runner/boards/`
  (disables harts 2-7 so Zephyr's SMP wakeup doesn't try to ping
  non-existent CPUs on the 2-tile SoC).

Estimated effort: 1-3 days. Most of the per-hart pinning + the marker
emission can be cloned from `merlin_hetero_runner`.

### 2. Multi-binary bundling in `validation/firesim_runner.py`

Today the runner stages exactly one ELF per `firesim infrasetup +
runworkload + kill` cycle. The Explore agent's read of the code says
the lifecycle is correctly wired (lines 204-341); only the per-call
signature needs to change.

```python
# current
run_firesim(elf="...", ...)

# proposed
run_firesim(elfs=["a.elf", "b.elf", ...], ...)
#  - stages each ELF into workloads/<id>/
#  - emits one workload JSON with all of them listed
#  - calls infrasetup ONCE, runworkload ONCE, kill ONCE
#  - parses N sets of MODELBLASTER_*_BEGIN/END markers from uartlog
```

`firemarshal`'s workload JSON spec supports `jobs: [...]` for
multi-binary sweeps (per the chipyard docs). The `merlin` script uses
the single-binary path but builds one ELF that bundles N test cases
internally (the `--MERLIN_JOBS=32` flag). Either pattern works; the
single-binary-with-internal-bundle is simpler from FireSim's POV
because `runworkload` just runs the ELF.

Estimated effort: 0.5-1 day once the Zephyr sample lands.

### 3. Cell capture flow

```bash
# Build N binaries (one per cell in the same workload bundle):
for cell in dronet×gemmini  dronet×gemmini_q31  yolov8n×gemmini  ...; do
    examples/<m>/run.sh ... RUNNER=firesim TARGET=<target>
done

# Bundle + run:
uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
    --workload dronet_gemmini_int8     \
    --workload yolov8n_gemmini_int8    \
    --workload dronet_gemmini_q31_int8 \
    --workload yolov8n_gemmini_q31_int8 \
    --bundle --runner firesim

# That ONE FireSim cycle would capture 4 cells × however-many-reps.
```

Estimated firesim wall time per bundle: 15-30 minutes (the
infrasetup is the slow part; once the bitstream is loaded, swapping
binaries via the embedded harness's job loop is fast).

## Cells that close once FireSim is wired

| Cell | Currently | After FireSim |
|---|---|---|
| dronet × rvv_opu | spike functional-only (4.55M) | real cycles |
| dronet × gemmini | spike functional-only (188k) | real cycles |
| dronet × gemmini_q31 | linf=72 drift on spike | authoritative + verify_pass |
| dronet × hetero | spike-hetero linf=52 | authoritative + verify_pass |
| yolov8n × rvv_opu | spike functional (104M) | real cycles |
| yolov8n × gemmini | spike functional (2.83M) | real cycles |
| yolov8n × gemmini_q31 | linf=9 drift | authoritative |
| yolov8n × hetero | blocked_by P2.1-schedule | needs schedule fixture too |

8 of 10 declared cells go from yellow to green.

## Why this didn't land in the 2026-05-27 session

- Setup verification (firesim CLI on PATH, SSH key loaded,
  `firesim kill` round-trip) took ~30 min of debugging the
  source-this-not-that env-vars rabbit hole. That's now a one-line
  `source scripts/setup_benchmark_env.sh` away — saves the next
  session's first hour.
- The actual Zephyr sample + multi-binary refactor is a multi-day
  build (see effort estimates above). Better as a dedicated session
  than a tail-end-of-day push.
- FireSim wall time per attempt (~15-30 min for infrasetup +
  runworkload) makes iteration on the harness change slow; you
  really want the Zephyr sample working in isolation FIRST, with
  the multi-binary refactor landing after one bundle is proven to
  run end-to-end.

## What's next

When ready to resume:

1. Read `/scratch2/agustin/zephyr-chipyard-sw/samples/merlin_hetero_runner/`
   in detail. That sample is the closest existing analogue and the
   pinning / marker emission patterns transfer directly.
2. Clone it into `zephyr-chipyard-sw/samples/modelblaster_firesim/`
   adapted to consume our IR's generated kernels + emit our markers.
3. Wire `examples/_run_lib.sh` to dispatch to the new sample when
   `RUNNER=firesim` and target is in
   `{rvv_opu, gemmini, gemmini_q31, hetero_gemmini_opu}`.
4. Extend `validation/firesim_runner.py` to take `elfs: list[str]`
   and bundle into one workload JSON.
5. Capture: 1 cell first (e.g. `dronet × gemmini` FireSim) to prove
   the path end-to-end; then bundle all 8 into one infrasetup.
