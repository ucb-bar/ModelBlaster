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

## Schedule fixtures

Two routes to a schedule, in order of preference:

1. **FreshScheduler (canonical, awaits profile data)**

   ``/scratch2/dima/misc_sw/FreshScheduler/scripts/run_xpurt_schedule.py``
   reads a workload.json + profile.csv (per-op cycles measured on the
   target backends) and emits an XPU-RT schedule. Workload-side
   scheduler input shape:

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

   Requires per-op profiling against single-target Gemmini and
   single-target rvv_opu runs first (chicken-and-egg with the harness
   we're standing up).

2. **`scripts/gen_hetero_schedule.py` (hand-authored stop-gap)**

   Generates a VALID schedule (DAG matches the IR, hardware_target
   labels resolve, start_time monotonic) from a `graph.json` IR
   alone, with placeholder per-op durations. Useful for smoking the
   dispatch path before FreshScheduler is wired up. Replace with a
   FreshScheduler-emitted fixture when real cycle data exists.

   ```bash
   # Extract dronet's IR (pure Python, no zephyr required).
   uv run python -m modelblaster.pipeline.extract_graph \
       --model dronet --quant int8 \
       --out-dir /tmp/dronet-graph

   # Author the schedule under the gemmini_main_opu_skip policy:
   #   - main-path conv2d_s8 / linear_s8 -> CPU_P (gemmini)
   #   - residual-skip conv2d_s8         -> CPU_E (rvv_opu)
   #   - all elementwise / norm / pool / activation -> CPU_E
   uv run python scripts/gen_hetero_schedule.py \
       --ir /tmp/dronet-graph/graph.json \
       --out schedule_fixtures/dronet_hetero_gemmini_opu.json \
       --job-name dronet
   # -> wrote schedule_fixtures/dronet_hetero_gemmini_opu.json
   #    (30 dispatches: CPU_E#0=21, CPU_P#0=9)
   ```

   For yolov8n / vint, use the equivalent extract_graph[_export]
   invocation and rerun the generator. yolov8n's pretrained-weight
   loader requires `ultralytics` installed (or
   `MODELBLASTER_YOLOV8N_PRETRAINED=0` to use random init for trace
   purposes); vint requires
   `MODELBLASTER_VINT_CFG=<path-to-visualnav-transformer-vint.yaml>`
   plus the corresponding checkpoint and sys.path for `vint_train`.

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

## End-to-end smoke

The dronet fixture is committed; the workload row is unblocked. From
a zephyr-activated shell:

```bash
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh

uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
    --workload dronet_hetero_int8
```

The arm driver detects the hetero target, shells into
`examples/xpurt_demo/run.sh` with the right env (BACKENDS, REGISTRY,
SCHEDULE_JSON, MODELBLASTER_HETERO_SPIKE), and the rest of the path
is the existing xpurt_demo flow.

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
