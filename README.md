# `agents/` — PyTorch → optimized Zephyr/RISC-V binaries

End-to-end flow for taking a PyTorch model through quantization,
per-target kernel generation (reference, hand-curated, or LLM-written),
Zephyr build, and validation on spike, RTL sim, or FireSim. Plus the
multi-model + XPURT-schedule layer that runs N networks on M cores in
one binary with explicit core pinning and inter-network synchronization.

> **Full canonical pipeline diagram (workload JSON → scheduler → codegen
> → FireSim → trace plot):** `agents/notes/pipeline_overview.md`.
> **Parent-repo cross-link** to the XPURT side of the same flow:
> `../../docs/end_to_end_xpurt_firesim.md`.

## Quick orientation

```
agents/
  models/            PyTorch model classes (one .py per model)
  pipeline/          codegen — extract IR, emit C, pick kernels, build, profile
  reference_kernels  KernelSpec per op: signature, semantics, scalar C oracle,
                     AlgorithmCandidate list (the "alternatives" the picker
                     and the LLM are allowed to consider)
  kernels/           hand-curated kernels, organized by HW target
  cores/             vendored target SDKs (gemmini.h, saturn_opu.h, ...)
  harness/           single-model Zephyr app template
  harness_multi/     N-models-in-one-ELF harness (for pool sweeps)
  harness_xpurt/     schedule-driven multi-model harness (XPU-RT execution)
  harness_microros/  micro-ROS variant (DDS broker + N model nodes)
  validation/        spike + firesim runners; profile CSV writer
  examples/          per-model run.sh + cached artifacts
  notes/             working design docs (deep-dives by topic)
```

## Quick start

One-time per shell:

```bash
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
source ../set_api_keys.sh   # only for BACKEND=llm or --optimize
```

The simplest single-model run:

```bash
# scalar fp32 reference kernels on spike — fastest sanity check
bash agents/examples/mlp_generic/run.sh

# int8 PTQ, rvv backend, with curated kernels probed before LLM fallback
QUANT=int8 TARGET=rvv BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/agents/kernels \
  bash agents/examples/dronet/run.sh

# fp16 + RVV+Zvfh widening on spike (the rvv_f16 backend)
QUANT=fp16 TARGET=rvv BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/agents/kernels \
  bash agents/examples/vint/run.sh

# Saturn OPU integer matmul, spike via the custom OPU extension
QUANT=int8 TARGET=rvv_opu BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/agents/kernels \
  bash agents/examples/dronet/run.sh

# FireSim runtime (any backend; runner copies the elf, runs infrasetup +
# runworkload, tails uartlog until OUTPUT_END markers)
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash agents/examples/dronet/run.sh
```

## Pipeline at a glance

Each stage's outputs are deterministic, on disk, and re-enterable.

```
[1] extract_graph[_export]   PyTorch → graph.json + weights.npz + io.npz
                             (per-quant; int8 PTQ + fp16 cast + mixed-prec
                              auto-cast all live in extract)

[2] generate_skeleton        IR → model.{c,h} + weights.{c,h} + test_io.h
                             + buffers.c (per-net scratch; extern-shared
                             across backends for the het multi-net path)

[3] generate_kernels         IR → kernels.{c,h}
                             three sources, in priority order:
                                global curated dir (agents/kernels/)
                                per-model LLM cache
                                LLM generation (--backend llm)
                             fastest-wins among curated + cached
                             optional --optimize beam-search per op

[4] west build               agents/harness + generated/<target>/* → .elf
                             (or harness_multi / harness_xpurt for the
                              multi-net + schedule paths)

[5] spike or firesim         run the elf, parse OUTPUT/PROFILE/WALL
                             markers, compare to PyTorch / int8-sim
                             golden, write profile.csv
```

Single-model orchestration: `agents/examples/<model>/run.sh` sources
`agents/examples/_run_lib.sh`. Multi-net + schedule: `multi_demo/run.sh`
and `xpurt_demo/run.sh` chain the per-model flow then run a single
harness build linking N models.

## Targets supported

Registered in `agents/pipeline/backends.py::BACKENDS`. Each is a
`Backend(...)` declaration plus a `harness/backends/<name>.conf`
overlay; nothing else hard-codes per-target logic.

| target | base ISA | extras | verify path |
|---|---|---|---|
| `scalar` | rv64imafdc | — | host ctypes |
| `scalar_f16` | rv64imafdc | Zfh | host ctypes |
| `rvv` | rv64gcv | — | spike harness |
| `rvv_f16` | rv64gcv | Zfh + Zvfh | spike harness |
| `rvv_opu` | rv64gcv | Saturn OPU custom .insn (i8 outer-product) | spike harness (needs OPU spike fork — see below) |
| `gemmini` | rv64imafdc | Gemmini int8 RoCC (DIM=16, f32 acc_scale) | chipyard spike harness |
| `gemmini_q31` | rv64imafdc | Gemmini int8 RoCC + Q0.31 mvout requantize | chipyard spike harness |

## Quant axes supported

| `QUANT` | what it does |
|---|---|
| `fp32` | no quantization; reference and curated kernels operate on `float`. |
| `fp16` | extract path casts to fp16; needs an `_f16` backend variant. |
| `int8` | per-tensor symmetric PTQ; one calibration sample drives scale choice. |
| `int8` + `--per-channel` | per-output-channel weight scales for conv/linear (CMSIS-NN / TFLite convention). |
| **Mixed precision** | per-op overrides via `get_precision_spec()` in the model file. Walker inserts `cast_i8_to_f16` / `cast_f16_to_i8` at dtype boundaries. Most useful when one op family (e.g. ViNT's goal-encoder linear) has too-wide range for int8 but the rest of the net is fine. See `agents/notes/mixed_precision_plan.md`. |

`_run_lib.sh` auto-promotes `TARGET=rvv` → `rvv_f16` (or `scalar` →
`scalar_f16`) when the IR contains any `_f16` op, so mixed-precision
runs don't need extra environment variables.

## Models in scope

Each example dir contains a tiny `run.sh` plus a `<quant>/cache/<target>/`
of post-verify curated/LLM kernels that persist in git.

| example | what it is | notable |
|---|---|---|
| `mlp_generic` | random-init 16→32→32→10 MLP | smallest demo |
| `mlp_control` | trained rsl_rl PPO actor (steering) | trained weights |
| `lenet` | random-init LeNet-5 | first int8 PTQ smoke target |
| `mobilenet_v2` | MobileNetV2 stem (no classifier) | exercises depthwise + SE |
| `dronet` | trained DroNet (3×112×112, steer+collision) | the canonical "real" model |
| `yolov8_nano` | YOLOv8 nano stem | int8 + RVV cache-blocked conv2d_s8 |
| `vint` | ViNT visual-navigation transformer | torch.export path, mixed-precision, mt-attention |
| `microros_demo` | micro-ROS broker + N model nodes | runtime + DDS integration |
| `multi_demo` | run N model in one ELF | pool-size sweeps, profile amortization |
| `xpurt_demo` | run an XPURT schedule.json on the harness | het core pinning, k_sem chains |
| `fp16_smoke`, `gemmini_smoke`, `gemmini_unittests`, `kernelbench`, `v_save_smoke` | targeted unit tests | each isolates one ISA / quant feature |

## Environment variables (single-model `run.sh`)

| var | values | default | notes |
|---|---|---|---|
| `BACKEND` | `reference`, `llm` | `reference` | source of impls. `reference` also probes `GLOBAL_CURATED_DIR`. |
| `TARGET` | `scalar`, `rvv`, `rvv_f16`, `rvv_opu`, `gemmini`, `gemmini_q31`, ... | `scalar` | HW backend. |
| `QUANT` | `fp32`, `fp16`, `int8` | `fp32` | quant pass at extract. |
| `OPTIMIZE` | `0`, `1` | `0` | beam-search after correctness. requires `BACKEND=llm`. |
| `ALGORITHMS` | `all`, csv | `all` | per-op algorithm filter (e.g. `direct,im2col_gemm`). |
| `BEAM`, `EXPANSIONS`, `ITERATIONS` | int | `2`, `3`, `2` | beam-search knobs. |
| `GLOBAL_CURATED_DIR` | path | unset | enables the `agents/kernels/` probe; safe to leave on. |
| `RUNNER` | `spike`, `firesim` | `spike` | downstream simulator. |
| `FIRESIM_TIMEOUT` | seconds | `600` | wallclock cap for `firesim runworkload`. |
| `FIRESIM_SKIP_INFRASETUP` | `0`, `1` | `0` | skip `firesim infrasetup` (advanced — only when the bitstream + driver are known fresh). |
| `MAX_ACCURACY_CLASS` | `bit_exact`, `numeric_drift`, `approximate` | unset | tighten verify (drop curated algos with looser declared class). |
| `FIRESIM_EVAL`, `CACHE_AWARE_PROMPT` | `0`, `1` | `0` | optimize-phase FireSim re-rank + cache-aware prompt. See `notes/firesim_eval_design.md`. |

Bedrock model id is `meta.llama4-maverick-17b-instruct-v1:0` by default,
overridable via `MODEL`.

## Where artifacts land

```
agents/examples/<model>/<quant>/
  generated/
    graph.json
    weights.npz
    io.npz                       PyTorch reference input/output + golden
    profile.csv                  last spike/firesim run's per-kernel cycles
    <target>/                    per-backend codegen output
      model.{c,h}                run_model() driver + rdcycle profile array
      weights.{c,h}              packed const arrays (per-backend layout)
      kernels.{c,h}              per-op implementations
      buffers.c                  scratch storage (extern-shared in het multi-net)
      test_io.h                  model_test_input + model_test_golden
      optimize_summary.json      beam-search history (--optimize only)
  build/<target>/                west build tree → zephyr.elf
  cache/<target>/                PASSing kernels keyed <target>_<op>_<algo>.c
                                 (committed to git so re-runs skip LLM)
```

`generated/` and `build/` are regenerated by `run.sh` and gitignored.
`cache/` is **not** gitignored — successful kernels persist across
machines. `agents/kernels/` is the *global* curated dir (hand-authored
kernels reusable across models); the per-model `cache/` is its
LLM-iterated cousin.

## Workflow: profiling kernels

Three depths of profile, from cheapest to most accurate:

### 1. Spike per-kernel cycle CSV (every run)

Default — no extra flags. `run_model()` brackets each kernel call with
`rdcycle()` (read of the `mcycle` CSR — 1 insn). `spike_runner` parses
the `AGENTS_PROFILE_BEGIN/END` block from stdout and writes
`generated/profile.csv` with `(dispatch_id, name, op, shape, cycles)`.

```bash
QUANT=int8 TARGET=rvv BACKEND=reference \
  bash agents/examples/dronet/run.sh
cat agents/examples/dronet/int8/generated/profile.csv
```

### 2. FireSim per-kernel cycles (real RTL)

Same flow with `RUNNER=firesim`. Runner takes care of XDMA chmod,
runs `firesim infrasetup`, then `runworkload`, tails the uartlog
until expected `AGENTS_WALL_CYCLES` count is hit.

```bash
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash agents/examples/dronet/run.sh
```

The same profile.csv format is produced; the spike vs firesim
difference is the cycle counts (FireSim reflects pipeline, cache
locality, etc. that spike can't model).

### 3. IREE-shape per-dispatch profile (for scheduler ingest)

When `PROFILE_OUT_ROOT` is set, the runner additionally writes IREE-
schema `results.csv` files at
`gen/profile/<backend>/<cpu>/<model>/.../topo_<cores>/results.csv`.
XPU-RT consumes these directly. See `agents/notes/profile_emission.md`.

```bash
PROFILE_OUT_ROOT=gen/profile \
PROFILE_CPU=firesim_rocket_saturn PROFILE_CORES=0,1,2,3 \
PROFILE_CLOCK_MHZ=1000.0 \
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash agents/examples/dronet/run.sh
```

### Profile sweeps across pool sizes / cores (multi-model)

`agents/examples/multi_demo/run.sh` builds one ELF that runs every
constituent model under each pool size in succession — useful for
amortizing FireSim infrasetup across a sweep:

```bash
MODELS=dronet,yolov8_nano TARGET=rvv QUANT=int8 \
  POOL_SIZES=1,2,4 RUNNER=firesim \
  bash agents/examples/multi_demo/run.sh
```

Produces one `topo_<cores>/results.csv` per pool size, side by side.

### Optimize phase (beam-search per op)

With `BACKEND=llm OPTIMIZE=1`, each op's algorithms run through a
beam-search:

```bash
BACKEND=llm OPTIMIZE=1 TARGET=rvv \
  bash agents/examples/lenet/run.sh
```

Each candidate must verify AND have lower cycles than its parent to
survive. Winners are written into the per-model `cache/<target>/`.
With `FIRESIM_EVAL=1`, the top-K spike survivors get re-ranked on
FireSim for cache-locality wins spike misses. See
`notes/firesim_eval_design.md`.

## Workflow: integrating with XPURT (schedule generation + execution)

The single-model flow above produces the inputs the XPURT scheduler
needs (per-dispatch IREE-shape profile CSVs). The schedule comes back
as a JSON; `xpurt_demo` consumes it.

### Step 1 — profile every (model, backend) pair under realistic pool sizes

```bash
# For each model and each candidate HW backend, emit one IREE-schema
# profile CSV at the chosen pool size:
for m in dronet yolov8_nano; do
  for t in scalar rvv gemmini_q31; do
    PROFILE_OUT_ROOT=gen/profile \
    PROFILE_CPU=firesim_rocket_saturn PROFILE_CORES=0,1,2,3 \
    PROFILE_CLOCK_MHZ=1000.0 \
    RUNNER=firesim QUANT=int8 TARGET=$t \
      bash agents/examples/$m/run.sh
  done
done
```

### Step 2 — write the workload spec

Edit a top-level `data/toplevel/<workload>.json` in the parent
FreshScheduler repo:

```jsonc
{
  "machines": { "CPU_P": "rvv", "CPU_E": "scalar", "GEMMINI": "gemmini_q31" },
  "networks": [
    {"name": "dronet",      "period_ms": 50},
    {"name": "yolov8_nano", "period_ms": 100}
  ],
  "profile_target": "firesim_rocket_saturn"
}
```

### Step 3 — run the scheduler

```bash
# From the parent FreshScheduler repo root:
python scripts/run_xpurt_schedule.py \
  data/toplevel/<workload>.json \
  --scheduler greedy_periodic \
  --out schedules/scheduled_<workload>.json
```

The scheduler reads `gen/profile/.../results.csv` per (network,
backend), runs the MILP / greedy assignment, and emits
`schedules/scheduled_<workload>.json` plus a predicted-timeline plot.

### Step 4 — build and run the scheduled binary

```bash
SCHEDULE_JSON=$PWD/schedules/scheduled_<workload>.json \
MODELS=dronet,yolov8_nano \
BACKENDS=scalar,rvv,gemmini_q31 \
QUANT=int8 \
RUNNER=firesim \
XPURT_TRACE=1 \
bash agents/examples/xpurt_demo/run.sh
```

The harness links every (model × backend) object library; the dispatch
table generated from the schedule selects the right one per entry. With
`XPURT_TRACE=1`, the uartlog includes per-entry begin/end timestamps
that `agents/scripts/plot_xpurt_trace.py` renders as a Gantt vs the
predicted timeline.

See `agents/notes/scheduler_investigation.md` for the schedule.json
format and `agents/notes/dispatch_and_cores.md` for the core-registry
and pinning model.

## Adding a new HW backend

`pipeline/backends.py` is usually the only Python file that changes.
Register a `Backend(...)` entry:

```python
NEW_TGT = Backend(
    name="new_tgt",
    description="…",
    kernel_cflags=("-march=…", "-mabi=lp64d", "-DAGENTS_NEW_TGT=1"),
    kernel_includes=("<some_header.h>",),
    prj_conf_overlay="new_tgt.conf",
    spike_args=("--isa=…",),                    # if any
    optimization_guide="optimization_guide_new_tgt.md",
    verify_method=VERIFY_SPIKE_HARNESS,         # or VERIFY_HOST_CTYPES
    # atol_override / rtol_override if needed
)
BACKENDS[NEW_TGT.name] = NEW_TGT
```

Then drop the supporting files:

```
agents/harness/backends/new_tgt.conf            # Kconfig overlay
agents/pipeline/prompts/optimization_guide_new_tgt.md   # LLM guide (optional —
                                                          can reuse scalar/rvv)
agents/cores/new_tgt/include/...                # vendored SDK headers (optional)
agents/kernels/new_tgt/                         # curated kernels go here
```

If the backend has a vendored SDK (gemmini, OPU) include paths use the
`<repo_root>` placeholder — `Backend.resolved_kernel_cflags()`
substitutes at build time. If the backend needs a custom spike fork
(gemmini, rvv_opu), wire the `--spike` lookup in
`agents/examples/_run_lib.sh` (mirror the existing `AGENTS_GEMMINI_SPIKE`
/ `AGENTS_OPU_SPIKE` env knobs).

Worked example: `agents/notes/saturn_opu_backend.md` documents the
full set of changes to add the OPU backend, end to end.

## Adding a new model

1. Drop `agents/models/<name>.py` with `get_model()` (returns a torch
   `nn.Module` with weights loaded) and `get_sample_input()` (returns
   the calibration / golden input tensor). For trained models, load
   weights inside `get_model()` from a checkpoint path. Optionally
   define `get_precision_spec()` for per-op mixed-precision overrides
   (`{"default": "int8", "fp16_upstream_of": ["op_name"], "fp16_ops": [...]}`).

2. Register the name in `agents/pipeline/extract_graph.py`'s `--model`
   choices, OR — for models that don't FX-trace (anything with
   `nn.TransformerEncoder` internals, `len(...)`, etc.) — add a
   `torch.export` branch in `agents/pipeline/extract_graph_export.py`.
   See ViNT's example for the export-path pattern.

3. Copy `agents/examples/mlp_generic/run.sh` to
   `agents/examples/<name>/run.sh` and change `MODEL_NAME=<name>`.

4. If the model uses ops not yet registered, add them — see below.

5. (Optional) calibration data: drop `agents/datasets/<spec>.json`
   pointing at a list of input tensors; the extractor's per-channel
   activation calibration consumes it.

## Adding a new op kind

1. New `KernelSpec` in `agents/pipeline/reference_kernels.py::KERNEL_SPECS`:
   - `signature` (exact string used in `kernels.h`)
   - `semantics` (English description for the LLM prompt)
   - `reference_impl` (correct naive scalar C — the verify oracle and
     the `--backend reference` output)
   - `extra_shapes` (verify shapes beyond what the IR happens to have)
   - `argtypes_factory` (ctypes signature for host verify)
   - `algorithms` list (optional — `AlgorithmCandidate`s with
     `target_affinity`, `weight_layout`, `accuracy_class`)
2. Wire the op in `extract_graph[_export].py` (FX/export node → IR op)
   and `generate_skeleton.py` (IR op → kernel call site).
3. For int8 op kinds, add the matching path in `extract_graph.py`'s
   integer pipeline simulator so the bit-exact golden stays in sync.
4. Verify with `--backend reference` first; then write a curated
   kernel for the relevant target.

## Adding a curated kernel

A curated kernel is a hand-written `.c` file at
`agents/kernels/<target>/<target>_<op>_<algo>.c`. The pipeline picks
it up automatically when `GLOBAL_CURATED_DIR` is set, as long as the
algorithm name is registered in `reference_kernels.py` with
`target_affinity=("<target>",)`.

Minimum recipe:

1. Write the `.c` file. First two lines must be:
   ```c
   /* source: curated */
   /* algorithm: <algo_name> */
   ```
   Body implements the canonical signature from the `KernelSpec`.
2. Add an `AlgorithmCandidate` in the matching spec:
   ```python
   AlgorithmCandidate(
       name="<algo_name>",
       target_affinity=("<target>",),
       description="…",
       reference_impl="",  # the curated file supplies it
   ),
   ```
3. Run any example with the matching target — the log shows
   `curated swap from .../<file>.c` when the kernel gets picked up.

Worked examples in this repo:
- **`rvv_f16` widening MAC** (linear / conv2d / depthwise) —
  `agents/kernels/rvv_f16/`. Ported from the canonical scalar fp16
  reference, vectorized via `vfwmacc`.
- **`gemmini_q31` tiled conv + linear** — `agents/kernels/gemmini_q31/`.
  Routes through gemmini RoCC with bit-exact Q0.31 requantize.
- **`rvv_opu` outer-product matmul + linear** —
  `agents/kernels/rvv_opu/`. Ported from upstream saturn
  `benchmarks/opu-gemm/kernel.h::i8_mm_bme_sq`; cited in the file
  headers. Exercises the Saturn OPU custom .insn programming model.

See `agents/kernels/README.md` for the curated-vs-cache distinction
and the picker priority order.

## Curated kernel + spike correctness loop

For backends with custom instructions, you need a spike build that
decodes them. Two existing paths:

- **gemmini** — chipyard ships a `--extension=gemmini` spike fork.
  `agents/examples/_run_lib.sh` finds it via `AGENTS_GEMMINI_SPIKE`
  env, defaults to `/scratch2/dima/chipyard-fsim/.conda-env/...`.
- **rvv_opu** — custom spike extension at
  `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/saturn_opu.cc`
  (in-repo functional model of `VOPACC` / `OPMVINBCAST` / `VMV_VR` /
  `VMV_RV`). `_run_lib.sh` finds the built spike via `AGENTS_OPU_SPIKE`.
  See `notes/saturn_opu_spike_support.md` for build instructions.

For a brand-new accelerator, the path is the same: extend
`riscv-isa-sim/customext/` with a functional model, register via
`REGISTER_EXTENSION`, and point `_run_lib.sh` at the built binary.

## Notes / deep-dives

The `agents/notes/` directory holds focused design notes per topic.
Highlights for this README's surface area:

| topic | note |
|---|---|
| Canonical pipeline diagram | `pipeline_overview.md` |
| int8 PTQ flow | `int8_quantization_flow.md` |
| Mixed-precision plan + experiments | `mixed_precision_plan.md`, `vint_mixed_precision_experiments.md` |
| Per-dispatch profile schema (IREE-shape) | `profile_emission.md` |
| FireSim re-rank in the optimize loop | `firesim_eval_design.md` |
| XPURT schedule format + the dispatch table | `scheduler_investigation.md`, `dispatch_and_cores.md` |
| Multi-model threading + agents_pool | `multi_model_threading.md` |
| POSIX affinity on Zephyr | `posix_affinity_investigation.md` |
| Saturn OPU backend status | `saturn_opu_backend.md` |
| Saturn OPU spike extension design | `saturn_opu_spike_support.md` |
| Gemmini extension status | `gemmini_extension_plan.md`, `gemmini_firesim_status.md` |
| Gemmini LUT optimization (FPGA-side) | `gemmini_lut_optimization.md` |
| Saturn FP-precision stripping (FPGA area) | `saturn_fp_precision_stripping.md` |
| Conv weight layout (OIHW / HWIO / IHWOC) | `conv_weight_layout_decisions.md` |
| Caveats from real bugs (Saturn strided memop, V context, FireSim quirks) | `saturn_strided_memop_bug.md`, `firesim_*` |

## Known limitations / open issues

- **Spike is an ISA simulator with flat memory.** Cycle counts reward
  pipeline-pattern wins (multiple accumulators, breaking fp dependency
  chains, unrolling); they're blind to cache locality. Use
  `RUNNER=firesim` for memory-realistic profiling.
- **Reference impls are the trusted oracle**, not the `signature`
  strings in `kernels.h`. If you change a `KernelSpec.signature`, the
  reference impl's first line must match or host-ctypes verify will
  silently misalign.
- **conv2d_s8 RVV via OPU im2col** — not yet curated; conv2d on the
  OPU backend currently falls back to scalar reference.
- **Saturn OPU bitstream availability** — the V256D128 OPU+Q31Gemmini
  config exists in scala but the FireSim bitstream side is in flux
  (see `saturn_opu_backend.md`). Spike + verilator paths work today.
- **Stale Vitis cmake on PATH** — `run.sh` prepends `/usr/bin` to dodge
  it; do the same if you invoke `west` outside `run.sh`.
