# micro-ROS baseline flow — end-to-end

This documents the build → run → analyze pipeline for the `microros_demo`
harness on FireSim Q31, and lists steps that are still manual today.

## 1. Pipeline

```
                                         ┌────────────────────────────┐
prj.conf (Kconfig)                      │ libmicroros.mk              │
    CONFIG_MICROROS_NODES, _PUBLISHERS ──→│  update_meta_from_zephyr   │
                                         │  → configured_colcon.meta  │
                                         │  → colcon build            │
                                         │  → libmicroros.a + headers │
                                         └────────────┬───────────────┘
                                                      │
modelblaster/examples/microros_demo/run.sh ─── west build ──┴── zephyr.elf
    MODELS, BACKENDS, PIN_BACKENDS,
    PIN_HARTS, PERIODS_MS, QUANTS,
    MICROROS_* knobs                                  ┌──────────────┐
                                                      │ firesim_runner│
                                                      │  kill prior  │
                                                      │  infrasetup  │
                                                      │  stage elf   │
                                                      │  runworkload │
                                                      └──────┬───────┘
                                                             │
                                       uartlog (HTIF, ~6 KB/s)
                                                             │
              ┌─── grep MODELBLASTER_ROS_TRACE_{BEGIN,END}  ───────┘
              │      WALL_CYCLES per net
              ▼
scripts/plot_microros_trace.py ──→ Gantt PNG (per-hart lanes)
scripts/plot_microros_vs_xpurt.py ──→ side-by-side panes (microros + xpurt CSV)
```

## 2. Configuration surface (env vars to `run.sh`)

| var | values | purpose |
|---|---|---|
| `MODELS` | csv of 2 or 3 model names | which networks to bundle |
| `BACKENDS` | csv of unique kinds | enables kernel build per backend |
| `PIN_BACKENDS` | csv same length as `MODELS` | each net's chosen backend |
| `PIN_HARTS` | csv same length as `MODELS` | each net's hart pin |
| `PERIODS_MS` | csv, `0` = one-shot | rclc timer period per net |
| `QUANTS` | csv same length as `MODELS` | per-net quant (`int8` / `fp32`) |
| `RUNNER` | `firesim` / `spike` | execution target |
| `FORCE_REGEN` | `0` / `1` | regenerate per-net model.c |
| `FIRESIM_TIMEOUT` | seconds | uartlog polling cap |

Behavioural knobs (build-time `-D` flags forwarded from env):

| knob | effect |
|---|---|
| `MICROROS_NO_PUBLISH` | skip `rcl_publish` calls (Mode E) |
| `MICROROS_NO_BROKER` | skip broker thread entirely (Mode D) |
| `MICROROS_2EXEC_BC` | collapse dronet + mlp onto a single rclc executor on hart 1 |
| `MICROROS_2EXEC_FUSE_BC` | single timer; callback runs `run_graph_b` + 2× `run_graph_c` |
| `MICROROS_2EXEC_FIRE_FAST` | both timer periods = 1 ns (always due) |
| `MICROROS_2EXEC_NORCLC` | bypass rclc on hart 1, raw `while { run_graph_b(); run_graph_c(); }` |
| `MICROROS_FUSE_BC_NO_C` | fused callback skips mlp (debug-only) |
| `MICROROS_MASK_{ALL,TIMER,IPI,EXT}` | which `mie` bits `run_graph_b` masks (default MSIE only) |
| `MICROROS_NO_LOCK_A` | remove `irq_lock` around `run_graph_a` (yolov8) |
| `MICROROS_NO_FPREGS_C` | drop `K_FP_REGS` from mlp's thread |
| `MICROROS_SKIP_TRACE` | skip `emit_trace_block()` entirely |

## 3. Canonical recipe (3-net Config B with the irq-mask fix)

Prereqs:
```bash
cd /scratch2/dima/misc_sw/FreshScheduler/zephyr-chipyard-sw
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
```

Run:
```bash
MODELS=yolov8_nano,dronet,mlp_control \
BACKENDS=gemmini_q31,rvv \
PIN_BACKENDS=gemmini_q31,rvv,rvv \
PIN_HARTS=0,1,1 \
PERIODS_MS=0,40,20 \
QUANTS=int8,int8,fp32 \
RUNNER=firesim \
FORCE_REGEN=0 \
MICROROS_NO_PUBLISH=1 \
MICROROS_2EXEC_BC=1 \
MICROROS_2EXEC_FUSE_BC=1 \
FIRESIM_TIMEOUT=1500 \
bash modelblaster/examples/microros_demo/run.sh
```

Analyze:
```bash
cp /scratch2/dima/chipyard-fsim/sims/firesim/firesim_rundir/sim_slot_0/uartlog \
   data/microros_3net_q31_firesim_configB.txt

python3 scripts/plot_microros_trace.py \
  --uartlog data/microros_3net_q31_firesim_configB.txt \
  --clock-mhz 1 \
  --out plots/microros_3net_configB.png \
  --title "3-net Config B (clean)"
```

## 4. Clock conversion

Trace cycles in the harness are `mtime` ticks → **1 tick = 1 µs** at the
1 GHz simulated SoC frequency. FireSim's FPGA host runs at ~60 MHz but
the modeled timing is exact at 1 ns/cycle. So `--clock-mhz 1` for any
plot script reading the `start_cycles`/`end_cycles` columns.

## 5. Key invariants

* `RESET_PROFILE(net, backend)` is called at the top of each
  `run_graph_X` so per-model `records_[]` doesn't overflow into
  `ros_trace[]`. Without it, the static `n_` counter is never zeroed.
* `run_graph_b` and `run_graph_c` must use the **MSIE-only** `csrrc mie`
  mask, never full `irq_lock()`. Two harts simultaneously full-locking
  loses the SMP system tick on both and stalls rclc spin loops until
  one releases. (Fixed 2026-05-12; see
  `data/microros_vs_xpurt_3net_summary.md`.)
* `run_graph_a` (yolov8) still uses full `irq_lock()` for V-state
  protection. Hart 0 alone can do this safely as long as hart 1 doesn't
  collide on a full lock.

## 6. Manual / agent-driven steps still NOT scripted

These are things that get done by hand each time and would benefit from
either being baked into `run.sh`, a new helper script, or `firesim_runner.py`.

1. **uartlog snapshot.** After every successful run we manually `cp
   /scratch2/dima/chipyard-fsim/.../uartlog
   data/microros_<config>.txt`. The runner could optionally save its
   uartlog under `data/` with a `--snapshot-label` flag.

2. **Trace cleanup when FireSim times out.** With high periodic
   iter counts (or a `MICROROS_FUSE_BC` run that never reaches yolov8's
   cap quickly), FireSim hits its timeout before the harness emits the
   final `=== MODELBLASTER_ROS_TRACE_END (skipped=N corrupted) ===` line. We
   then have to hand-append a synthetic END marker for plot scripts.
   The plot scripts should treat EOF or non-CSV lines as an implicit
   end-of-trace.

3. **Min-bar-width for plot_microros_trace.** Sub-ms dispatches (mlp's
   ~18 µs ops on a ~1500 ms axis) were invisible until we added a min
   render width. The hack is in place but the threshold (`0.15%` of
   span) was eyeballed. Likely needs a `--min-bar-ms` flag for users
   who want strict timing.

4. **Knob plumbing.** Adding a new `MICROROS_*` knob requires touching:
   - `modelblaster/harness_microros/CMakeLists.txt` `foreach(knob ...)` list
   - `modelblaster/examples/microros_demo/run.sh` `for _knob in ...` list
   - source code in `main.c`
   A single Kconfig file driving both would remove two of the three
   forward-edits.

5. **libmicroros invalidation.** When changing `CONFIG_MICROROS_NODES`
   (or any value in `configure_colcon_meta`'s output), we must `rm -rf`
   six directories under
   `third-party/micro_ros_zephyr_module/modules/libmicroros/` (built
   tree, install tree, log, configured_colcon.meta, libmicroros.a,
   zephyr_toolchain.cmake). The Makefile target should be sensitive to
   prj.conf changes; today it isn't, so we manually wipe.

6. **Plot regeneration after model recompile.** Today `run.sh` always
   re-runs the full firesim flow but doesn't auto-replot. After every
   trace we run `plot_microros_trace.py` and (sometimes)
   `plot_microros_vs_xpurt.py` by hand. A `--plot` flag on `run.sh`
   would close the loop.

7. **Summary doc updates.** `data/microros_vs_xpurt_3net_summary.md` is
   maintained by hand; each new clean run requires re-typing range
   values. A small `report_microros_run.py` that ingests a uartlog and
   emits markdown for that run would let the summary be an append-only
   log.

8. **Cross-run comparison.** Currently `plot_microros_vs_xpurt.py`
   accepts multiple `--microros label=path` and `--xpurt label=path`
   args. We invoke it manually per comparison. There is no
   "comparison fixture file" (e.g. yaml/json) listing the runs that
   belong in the canonical comparison; that would let CI emit the same
   PR-quality plots without copy-paste.

9. **Firesim post-run cleanup.** Occasionally a sim is left running
   after a SIGKILL from our background task wrapper. We sometimes
   `firesim kill` by hand. `firesim_runner.py` already has a
   "kill prior sim" step at start; an `atexit` finalize would prevent
   stragglers entirely.

10. **CONFIG_MICROROS_NODES sizing.** The Kconfig integer is set from
    `prj.conf` but the harness can deduce the required count from
    `len(MODELS)`. We could autogenerate prj.conf from the env vars at
    `run.sh` start (write a `prj.conf.gen`, include it).

## 7. Files of record

* Source: `modelblaster/harness_microros/src/main.c`,
  `modelblaster/harness_microros/prj.conf`,
  `modelblaster/harness_microros/CMakeLists.txt`
* Build wrapper: `modelblaster/examples/microros_demo/run.sh`
* Plot scripts: `scripts/plot_microros_trace.py`,
  `scripts/plot_microros_vs_xpurt.py`, `scripts/plot_microros_compare.py`
* Notes: `modelblaster/examples/microros_demo/NOTES_3NET.md` (historical
  debug log), `data/microros_vs_xpurt_3net_summary.md` (canonical
  result + fix writeup)
