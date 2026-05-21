# Heterogeneous backend — `hetero_gemmini_opu`

Status: **dispatch wired, awaiting schedule fixtures**.

Adds a 2-tile chipyard configuration (Gemmini RoCC + Saturn-OPU
vector) as a first-class workload target. The harness routes
``target: hetero_gemmini_opu`` workloads through
``examples/xpurt_demo/run.sh`` instead of the per-model ``run.sh``,
links both backends into a single Zephyr ELF, and dispatches per-op
according to an XPU-RT schedule.

## What's wired

- ``cores/chipyard_gemmini_opu_hetero.json`` — 2-tile registry
  matching ``chipyard.HeteroConfigs.GemminiAndOPUShuttleConfig``
  (tile 0 Shuttle+Gemmini, tile 1 Shuttle+Saturn-OPU vLen=128).
- ``benchmarks/arms/_common.py::execute_run_sh`` detects
  ``target in HETERO_TARGETS`` and shells out to ``xpurt_demo/run.sh``
  with MODELS / BACKENDS / REGISTRY / SCHEDULE_JSON / RUNNER set from
  the workload row. ``MODELBLASTER_HETERO_SPIKE`` is auto-pointed at
  ``/scratch2/agustin/merlin/tools/spike-hetero/spike-hetero`` when
  the runner is spike and the wrapper exists.
- ``benchmarks/config/workloads.yaml`` rows for ``dronet_hetero_int8``,
  ``yolov8n_hetero_int8``, and ``vint_hetero_fp16`` carry the
  ``xpurt_cores_registry`` + ``xpurt_backends`` + ``xpurt_schedule_path``
  fields the dispatch reads.

## What's gated

- **Schedule JSON fixtures**: each hetero workload's
  ``xpurt_schedule_path`` points at a file under
  ``schedule_fixtures/`` that does not exist yet. The dispatch
  surface refuses to run without a valid schedule and prints the
  path the workload row references. Generate via FreshScheduler
  (``/scratch2/dima/misc_sw/FreshScheduler/scripts/run_xpurt_schedule.py``)
  against a profiled run of the model on the gemmini+rvv_opu
  topology; check the resulting JSON into
  ``schedule_fixtures/<workload>.json`` (or stage it locally and
  point the workload row there).

  Workload-side scheduler input shape (from
  ``notes/04_xpurt_integration.md``, but inlined here for
  reference):

  ```jsonc
  {
    "machines": {
      "GEMMINI": "gemmini",
      "OPU":     "rvv_opu"
    },
    "networks": [
      {"name": "dronet", "period_ms": 50}
    ],
    "profile_target": "firesim_chipyard_dual_gemmini_opu"
  }
  ```

- **FireSim bitstream**: the
  ``firesim_chipyard_dual_gemmini.conf`` overlay targets the
  Alveo-U250 dual-rocket-saturn-**Gemmini** bitstream where BOTH
  harts have Gemmini attached. The actual
  ``GemminiAndOPUShuttleConfig`` (tile 0 Gemmini, tile 1 OPU) does
  not have a built FireSim bitstream today. Spike via spike-hetero
  is the only functional simulator until the bitstream lands.

- **Spike-on-hetero cycle counts**: the merlin
  ``spike-hetero`` wrapper loads both Gemmini and Saturn-OPU
  extensions, but both extensions execute atomically — cycle counts
  on accelerator ops are not authoritative. The aggregator's
  cycle-source-honesty policy enforces this: ``cycles_spike`` on
  ``hetero_gemmini_opu`` is correctness-only; only ``cycles_firesim``
  (once the bitstream exists) counts toward arm-vs-arm cycle deltas.

## End-to-end smoke (once a schedule is staged)

```bash
# 1. Build a schedule for dronet on the 2-tile config (one-shot).
python /scratch2/dima/misc_sw/FreshScheduler/scripts/run_xpurt_schedule.py \
    --workload <workload-config-pointing-at-GEMMINI+OPU> \
    --profile  <profiled-dronet-on-gemmini-rvv_opu-results-csv> \
    --output   schedule_fixtures/dronet_hetero_gemmini_opu.json

# 2. Drop schedule_fixtures/dronet_hetero_gemmini_opu.json's path into
#    the workload row (already done; just create the file).

# 3. Drop the blocked_by entry on the workload row.

# 4. Run any arm; the driver auto-detects the hetero target.
uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
    --workload dronet_hetero_int8
```

## Architectural notes

- The arm drivers stay provider-agnostic and workload-agnostic; the
  hetero routing is a property of the workload, not the arm.
- ``xpurt_demo/run.sh`` already handles per-backend codegen
  (per-model x per-backend) and links both into one ELF, so each
  hetero workload only adds two artifacts: the schedule.json and the
  per-workload row in workloads.yaml.
- Pricing / token tracking is unchanged: ``BEDROCK_CALLS_LOG`` /
  ``GEMINI_CALLS_LOG`` / ``CLAUDE_CODE_CALLS_LOG`` env vars are
  propagated into the xpurt_demo subprocess, so per-op LLM kernel
  synthesis on hetero workloads (Arms B-*) attributes cost the same
  way it does on single-target workloads.
