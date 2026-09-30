# `modelblaster/` — PyTorch → optimized Zephyr/RISC-V binaries

End-to-end flow for taking a PyTorch model through quantization,
per-target kernel generation (reference, hand-curated, or LLM-written),
Zephyr build, and validation on spike, RTL sim, or FireSim. Plus the
multi-model + XPURT-schedule layer that runs N networks on M cores in
one binary with explicit core pinning and inter-network synchronization.

> **Full canonical pipeline diagram (workload JSON → scheduler → codegen
> → FireSim → trace plot):** `modelblaster/notes/pipeline_overview.md`.
> **Parent-repo cross-link** to the XPURT side of the same flow:
> `../../docs/end_to_end_xpurt_firesim.md`.

## Origin: built on KernelBlaster

ModelBlaster is an applied derivative of **KernelBlaster**, the original
research project that defines the underlying methodology. KernelBlaster
introduces continual, cross-task CUDA kernel optimization via
memory-augmented in-context reinforcement learning — an LLM that learns
from prior optimization attempts to improve generated kernels across
hardware targets. ModelBlaster carries that core idea — LLM-driven,
verify-then-optimize kernel generation with a persistent cache of winning
implementations — over to the PyTorch → quantized → Zephyr/RISC-V
embedded flow, and adds the multi-model XPU-RT scheduling layer.

To understand *why* the pipeline is shaped the way it is (the
generate → verify → beam-search-optimize loop, the per-op algorithm
candidates, the cached-kernel reuse model), read the KernelBlaster paper:

- **Paper:** *KernelBlaster: Continual Cross-Task CUDA Optimization via
  Memory-Augmented In-Context Reinforcement Learning* —
  https://arxiv.org/abs/2602.14293
- **Source:** vendored here as a git submodule under `third_party/KernelBlaster/`
  (NVlabs/KernelBlaster).

## Working with XPU-RT (the scheduler)

ModelBlaster compiles and runs; XPU-RT schedules. They form a closed loop, and
recent work landed in **both repos together** — read
XPU-RT's `docs/Feature/the_loop.md` for which script owns which
arrow, and XPU-RT's `docs/K1/k1_board.md` to run any of it on
hardware. The complete composite-target contract is in
[`docs/xpurt_schedule_sharding.md`](docs/xpurt_schedule_sharding.md).

What XPU-RT expects from this side, and where it lives here:

| XPU-RT asks for | ModelBlaster provides |
|---|---|
| a measured per-dispatch profile | `pipeline/profile_writer.py` → IREE-shape `results.csv` |
| a dispatch graph | `pipeline/emit_dispatch_graph.py` |
| a graph rewritten to a hint | `pipeline/apply_{fusion,split,unfuse,shard}_hint.py` |
| a binary honouring a schedule | `pipeline/ingest_xpurt_schedule.py` + `generate_xpurt_main.py` |
| live telemetry | `MB_XPURT_STREAM=1` → one JSON line per dispatch end |

Three things worth knowing before changing any of them:

* **`apply_shard_hint.py` does not rewrite the graph.** It annotates one
  dispatch with a core width; the dispatch count, ids and edges are unchanged.
  Its contract spells the count `n_shards`, not `n_splits`, so a hint fed to
  the wrong applier fails rather than quietly doing the other verb. For a
  schedule-selected width, `schedule_shards.py` derives the same per-dispatch
  annotation from the composite `hardware_target` before code generation.
* **The walker selects its kernel by the schedule's `impl`**, not by
  `core_kind`. That is what makes a heterogeneous schedule mean something at
  run time; before it, such a schedule produced a binary that ran one backend
  everywhere and reported the runtime it got.
* **`_weight_name` suffixes every symbol with its backend, unconditionally.**
  Without it two backends' `weights.c` define the same non-static symbol with
  differently-packed data, only one gets linked, and every other backend reads
  through the wrong layout — dronet at `max_abs_err=51`, no link error. Any new
  emitter must pass `backend`.

Environment: XPU-RT's `docs/Artifact/environment.md`. Run this
repo's tests **from this directory** (the curated-kernel verify cross-compiles
with repo-relative includes), with `CROSS` exported:

```bash
eval "$(../scripts/setup_spacemit_toolchain.sh)"
../.venv/bin/python -m pytest tests pipeline/tests -q
```

Scripts that reach outside this repo take the location from the environment rather
than a built-in path:

| variable | default | read by |
|---|---|---|
| `XPURT_ROOT` | the superproject (`..`) | the XPU-RT-facing scripts in `scripts/` (PDB, schedule, Gantt, sweep helpers) |
| `MODELBLASTER_VITFLY_MODELS` | none (required) | `models/_fused_loader.py`: the `vitfly/models` dir holding `fused_model.py` |
| `MODELBLASTER_FUSED_CKPT` | `../sims/models/warehouse/nav_fused_v12_cnn.pt` | `models/_fused_loader.py` |
| `CHIPYARD_ROOT` | none | `scripts/setup_benchmark_env.sh` (conda env, FireSim install, `env.sh`), `scripts/bedrock_spike_loop.py` (required), `scripts/spike_verify_kernel.py` |
| `ZEPHYR_BASE` | none | `scripts/setup_benchmark_env.sh` |
| `FIRESIM_ROOT` | `$CHIPYARD_ROOT/sims/firesim` | the FireSim capture scripts (`scripts/run_*_capture.sh`, `run_*_reps.sh`, `run_3way_sweep.sh`), `scripts/run_xpurt_bundle.py` |
| `FIRESIM_QUEUE_BIN` | `firesim-queue` on `PATH` | `validation/firesim_runner.py` |
| `RISCV` | `$CHIPYARD_ROOT/.conda-env/riscv-tools`, else tools on `PATH` | `scripts/spike_verify_kernel.py`, `scripts/bedrock_spike_loop.py` (bare-metal gcc, spike, pk) |
| `MODELBLASTER_HETERO_SPIKE` | `spike-hetero` on `PATH` | `scripts/bedrock_spike_loop.py` |
| `PY` / `XPURT_PY` | the running interpreter | `scripts/wallclock_split_eval.py` |

## The SpaceMiT K1 target

The SpaceMiT K1 (BananaPi) is an 8-hart riscv64 Linux board: cluster 0 (harts 0–3) and cluster 1
(harts 4–7) both run RVV at VLEN=256, and cluster 0 alone carries the IME int8 matrix engine
(`smt.vmadot`). `cores/spacemit_k1.json` is the measured core registry. Two backends target it,
both cross-compiled on the host and verified on the board (no simulator models `smt.vmadot`):

| backend | kernels | what it is |
|---|---|---|
| `rvv_x60` | `kernels/rvv/` | rv64gcv, VLEN=256, Zfh/Zvfh; runs on either cluster |
| `ime_x60` | `kernels/ime/`, then `kernels/rvv/` | the IME kernels where they exist, RVV for every other op; **cluster 0 only** |

### Toolchain: `CROSS`

Every board build uses the GCC cross toolchain named by the `CROSS` prefix. It defaults to
`riscv64-unknown-linux-gnu-`, i.e. whatever `riscv64-unknown-linux-gnu-gcc` is found on `PATH`.
Set it to use a specific toolchain; the SpaceMiT GCC 14.3 build is the one to use, because GCC 13.2
reorders `vsetvl` calls in RVV intrinsic code and the result traps on the board. From an XPU-RT
checkout, `eval "$(../scripts/setup_spacemit_toolchain.sh)"` fetches it and exports `CROSS`.

| variable | default | read by |
|---|---|---|
| `CROSS` | `riscv64-unknown-linux-gnu-` (on `PATH`) | `scripts/run_xpurt_k1.sh`, `scripts/run_model_k1.sh`, `scripts/ime_conv_verify_bench.py`, `scripts/ime_fused_conv_bench.py`, `scripts/k1_verify_curated_rvv.py`, `scripts/k1_verify_fused_conv_rvv.py` |
| `MODELBLASTER_K1_HOST` | `k1` (an ssh config entry) | the same scripts |
| `MODELBLASTER_K1_REMOTE_ROOT` | `/root/mb_k1` | the staging directory on the board |
| `MB_IME_FORCE` | `0` | `pipeline/generate_kernels.py` via `pipeline/ime_cost.py` (below) |
| `MB_XPURT_STREAM` | `0` | `scripts/run_xpurt_k1.sh`: one JSON line per dispatch end, for XPU-RT's streaming feedback |

`scripts/run_xpurt_k1.sh --schedule <scheduled_*.json>` runs an XPU-RT schedule on the board end to
end: extract the int8 IR, generate kernels per backend, ingest the schedule into a dispatch table,
cross-build `harness_xpurt_linux/`, run it pinned over ssh and pull back stdout and the per-dispatch
trace. `scripts/run_model_k1.sh` is the single-model counterpart.

### The IME kernels and when they are used

`kernels/ime/` holds int8 `vmadot` kernels on a 4×4×8 micro-tile: `linear`, `matmul`, `conv2d`
(im2col → `vmadot`), and the fused `conv2d_batchnorm2d_silu_s8`. The fused kernel exists because
the deployed YOLO graph fuses Conv→BN→SiLU into one op and curated kernels are looked up by exact op
name, so without it none of those convolutions could reach the engine. Its MAC is integer and its
BN and SiLU stages are the RVV kernel's expressions in the same order, so its output is
byte-identical to the RVV fused kernel's; that is the verification contract (`max_abs_err=0`).

`pipeline/ime_cost.py` decides, per op and shape, whether the IME kernel is used. The rule is
**only if measured faster** than the RVV kernel that would otherwise run:

* convolutions use a measured table per op kind, each against its own RVV baseline —
  `artifacts/ime_conv/ime_vs_rvv_conv.csv` for `conv2d_s8`,
  `artifacts/ime_fused_conv/ime_vs_rvv_fused_conv.csv` for the fused op. A shape absent from the
  table stays on RVV;
* `linear` / `matmul` use a speedup curve over M measured on the board (0.25× at M=7, 2.30× at
  M=128);
* the kernel picker (`generate_kernels.py`) and `pipeline/apply_ime_hint.py` both consult it.

`MB_IME_FORCE=1` is the explicit override for measuring the blanket deployment: the picker then takes
the IME kernel for every op that has one, whatever the tables say, and records that in
`kernel_picks.json` (`ime_force`, `table_guided: false`, `ime_forced_ops`). It does not relax
verification, it does not change placement (a forced build still runs on harts 0–3 only), and
nothing but the kernel build reads it. `pipeline/tests/test_ime_force.py` covers both modes.

The per-shape tables are produced on the board by `scripts/ime_conv_verify_bench.py` (IME conv vs
the standalone RVV conv, both checked against a scalar oracle) and `scripts/ime_fused_conv_bench.py`
(the fused IME kernel vs the deployed RVV fused kernel, byte-identical outputs required). Each
compiles one binary with `CROSS`, runs it pinned to cluster 0 and writes its CSV under
`artifacts/`.

### Which IME instructions the K1 implements

`artifacts/ime_isa_probe/FINDINGS.md` records a probe of the three instruction families in the IME
specification on all eight harts. On cluster 0, `vmadot` (all four signedness variants) and the
sliding-window `vmadot1` execute; on cluster 1 every one raises `SIGILL`; the fp16/bf16 `vfmadot`
family raises `SIGILL` everywhere, under SEW=8 and SEW=16. So the engine is int8 only, cluster 0
only, and the sliding-window family is present but not used by any kernel here. The page also
analyses which of the measured losing shapes a sliding window could affect. The probe sources and
the raw output are beside it.

### The worker pool and per-shard tracing

`runtime/modelblaster_pool/` is the POSIX thread pool the generated code uses to split one op across
harts (`parallelize_1d`). It can optionally record where every slice ran: the caller supplies a
buffer of `struct modelblaster_pool_shard` (rdtime start/end, the hart from `sched_getcpu()`, the
worker index, the call counter and the slice range) and arms it with
`modelblaster_pool_trace_arm(pool, buf, n)`; `modelblaster_pool_trace_count()` and
`modelblaster_pool_trace_reset()` read and clear it, and `n == 0` disarms. Recording is lock-free and
costs two rdtime reads per slice. XPU-RT's traced ROS 2 node (`board/k1_ros_mb/ros_mb_chain_traced.cpp`)
uses it to place each YOLO and nav shard on the hart that executed it.

How these pieces are measured on the deployed chain, and what the IME is worth there, is in XPU-RT's
`docs/K1/ime_kernel_reproduction.md`; the ROS 2 baseline with the
IME kernels is XPU-RT's `docs/Baselines/ros_with_ime.md`.

## Quick orientation

```
modelblaster/
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
  harness_linux/     single-model Linux harness (the K1)
  harness_xpurt_linux/  schedule-driven multi-model Linux harness (the K1)
  runtime/           modelblaster_pool (the worker pool, optional per-shard tracing)
  validation/        spike + firesim runners; profile CSV writer
  scripts/           runners and benches (run_xpurt_k1.sh, ime_*_bench.py, ...)
  artifacts/         measured tables and probe results the pipeline reads (ime_conv/, ime_fused_conv/, ime_isa_probe/)
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
bash modelblaster/examples/mlp_generic/run.sh

# int8 PTQ, rvv backend, with curated kernels probed before LLM fallback
QUANT=int8 TARGET=rvv BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
  bash modelblaster/examples/dronet/run.sh

# fp16 + RVV+Zvfh widening on spike (the rvv_f16 backend)
QUANT=fp16 TARGET=rvv BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
  bash modelblaster/examples/vint/run.sh

# Saturn OPU integer matmul, spike via the custom OPU extension
QUANT=int8 TARGET=rvv_opu BACKEND=reference \
  GLOBAL_CURATED_DIR=$PWD/modelblaster/kernels \
  bash modelblaster/examples/dronet/run.sh

# FireSim runtime (any backend; runner copies the elf, runs infrasetup +
# runworkload, tails uartlog until OUTPUT_END markers)
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash modelblaster/examples/dronet/run.sh
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
                                global curated dir (modelblaster/kernels/)
                                per-model LLM cache
                                LLM generation (--backend llm)
                             fastest-wins among curated + cached
                             optional --optimize beam-search per op

[4] west build               modelblaster/harness + generated/<target>/* → .elf
                             (or harness_multi / harness_xpurt for the
                              multi-net + schedule paths)

[5] spike or firesim         run the elf, parse OUTPUT/PROFILE/WALL
                             markers, compare to PyTorch / int8-sim
                             golden, write profile.csv
```

Single-model orchestration: `modelblaster/examples/<model>/run.sh` sources
`modelblaster/examples/_run_lib.sh`. Multi-net + schedule: `multi_demo/run.sh`
and `xpurt_demo/run.sh` chain the per-model flow then run a single
harness build linking N models.

## Targets supported

Registered in `modelblaster/pipeline/backends.py::BACKENDS`. Each is a
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
| `rvv_x60` | rv64gcv, VLEN=256 | Zfh + Zvfh (SpaceMiT K1) | cross-compile; verified on the board |
| `ime_x60` | rv64gcv, VLEN=256 | K1 IME `smt.vmadot` int8, cluster 0 only | cross-compile; verified on the board |

## Quant axes supported

| `QUANT` | what it does |
|---|---|
| `fp32` | no quantization; reference and curated kernels operate on `float`. |
| `fp16` | extract path casts to fp16; needs an `_f16` backend variant. |
| `int8` | per-tensor symmetric PTQ; one calibration sample drives scale choice. |
| `int8` + `--per-channel` | per-output-channel weight scales for conv/linear (CMSIS-NN / TFLite convention). |
| **Mixed precision** | per-op overrides via `get_precision_spec()` in the model file. Walker inserts `cast_i8_to_f16` / `cast_f16_to_i8` at dtype boundaries. Most useful when one op family (e.g. ViNT's goal-encoder linear) has too-wide range for int8 but the rest of the net is fine. See `modelblaster/notes/mixed_precision_plan.md`. |

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
| `GLOBAL_CURATED_DIR` | path | unset | enables the `modelblaster/kernels/` probe; safe to leave on. |
| `RUNNER` | `spike`, `firesim` | `spike` | downstream simulator. |
| `FIRESIM_TIMEOUT` | seconds | `600` | wallclock cap for `firesim runworkload`. |
| `FIRESIM_SKIP_INFRASETUP` | `0`, `1` | `0` | skip `firesim infrasetup` (advanced — only when the bitstream + driver are known fresh). |
| `MAX_ACCURACY_CLASS` | `bit_exact`, `numeric_drift`, `approximate` | unset | tighten verify (drop curated algos with looser declared class). |
| `FIRESIM_EVAL`, `CACHE_AWARE_PROMPT` | `0`, `1` | `0` | optimize-phase FireSim re-rank + cache-aware prompt. See `notes/firesim_eval_design.md`. |

Bedrock model id is `meta.llama4-maverick-17b-instruct-v1:0` by default,
overridable via `MODEL`.

## Where artifacts land

```
modelblaster/examples/<model>/<quant>/
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
machines. `modelblaster/kernels/` is the *global* curated dir (hand-authored
kernels reusable across models); the per-model `cache/` is its
LLM-iterated cousin.

## Workflow: profiling kernels

Three depths of profile, from cheapest to most accurate:

### 1. Spike per-kernel cycle CSV (every run)

Default — no extra flags. `run_model()` brackets each kernel call with
`rdcycle()` (read of the `mcycle` CSR — 1 insn). `spike_runner` parses
the `MODELBLASTER_PROFILE_BEGIN/END` block from stdout and writes
`generated/profile.csv` with `(dispatch_id, name, op, shape, cycles)`.

```bash
QUANT=int8 TARGET=rvv BACKEND=reference \
  bash modelblaster/examples/dronet/run.sh
cat modelblaster/examples/dronet/int8/generated/profile.csv
```

### 2. FireSim per-kernel cycles (real RTL)

Same flow with `RUNNER=firesim`. Runner takes care of XDMA chmod,
runs `firesim infrasetup`, then `runworkload`, tails the uartlog
until expected `MODELBLASTER_WALL_CYCLES` count is hit.

```bash
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash modelblaster/examples/dronet/run.sh
```

The same profile.csv format is produced; the spike vs firesim
difference is the cycle counts (FireSim reflects pipeline, cache
locality, etc. that spike can't model).

### 3. IREE-shape per-dispatch profile (for scheduler ingest)

When `PROFILE_OUT_ROOT` is set, the runner additionally writes IREE-
schema `results.csv` files at
`gen/profile/<backend>/<cpu>/<model>/.../topo_<cores>/results.csv`.
XPU-RT consumes these directly. See `modelblaster/notes/profile_emission.md`.

```bash
PROFILE_OUT_ROOT=gen/profile \
PROFILE_CPU=firesim_rocket_saturn PROFILE_CORES=0,1,2,3 \
PROFILE_CLOCK_MHZ=1000.0 \
RUNNER=firesim QUANT=int8 TARGET=rvv \
  bash modelblaster/examples/dronet/run.sh
```

### Profile sweeps across pool sizes / cores (multi-model)

`modelblaster/examples/multi_demo/run.sh` builds one ELF that runs every
constituent model under each pool size in succession — useful for
amortizing FireSim infrasetup across a sweep:

```bash
MODELS=dronet,yolov8_nano TARGET=rvv QUANT=int8 \
  POOL_SIZES=1,2,4 RUNNER=firesim \
  bash modelblaster/examples/multi_demo/run.sh
```

Produces one `topo_<cores>/results.csv` per pool size, side by side.

### Optimize phase (beam-search per op)

With `BACKEND=llm OPTIMIZE=1`, each op's algorithms run through a
beam-search:

```bash
BACKEND=llm OPTIMIZE=1 TARGET=rvv \
  bash modelblaster/examples/lenet/run.sh
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
      bash modelblaster/examples/$m/run.sh
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
bash modelblaster/examples/xpurt_demo/run.sh
```

The harness links every (model × backend) object library; the dispatch
table generated from the schedule selects the right one per entry. With
`XPURT_TRACE=1`, the uartlog includes per-entry begin/end timestamps
that `modelblaster/scripts/plot_xpurt_trace.py` renders as a Gantt vs the
predicted timeline.

See `modelblaster/notes/scheduler_investigation.md` for the schedule.json
format and `modelblaster/notes/dispatch_and_cores.md` for the core-registry
and pinning model.

## Adding a new HW backend

`pipeline/backends.py` is usually the only Python file that changes.
Register a `Backend(...)` entry:

```python
NEW_TGT = Backend(
    name="new_tgt",
    description="…",
    kernel_cflags=("-march=…", "-mabi=lp64d", "-DMODELBLASTER_NEW_TGT=1"),
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
modelblaster/harness/backends/new_tgt.conf            # Kconfig overlay
modelblaster/pipeline/prompts/optimization_guide_new_tgt.md   # LLM guide (optional —
                                                          can reuse scalar/rvv)
modelblaster/cores/new_tgt/include/...                # vendored SDK headers (optional)
modelblaster/kernels/new_tgt/                         # curated kernels go here
```

If the backend has a vendored SDK (gemmini, OPU) include paths use the
`<repo_root>` placeholder — `Backend.resolved_kernel_cflags()`
substitutes at build time. If the backend needs a custom spike fork
(gemmini, rvv_opu), wire the `--spike` lookup in
`modelblaster/examples/_run_lib.sh` (mirror the existing `MODELBLASTER_GEMMINI_SPIKE`
/ `MODELBLASTER_OPU_SPIKE` env knobs).

Worked example: `modelblaster/notes/saturn_opu_backend.md` documents the
full set of changes to add the OPU backend, end to end.

## Adding a new model

1. Drop `modelblaster/models/<name>.py` with `get_model()` (returns a torch
   `nn.Module` with weights loaded) and `get_sample_input()` (returns
   the calibration / golden input tensor). For trained models, load
   weights inside `get_model()` from a checkpoint path. Optionally
   define `get_precision_spec()` for per-op mixed-precision overrides
   (`{"default": "int8", "fp16_upstream_of": ["op_name"], "fp16_ops": [...]}`).

2. Register the name in `modelblaster/pipeline/extract_graph.py`'s `--model`
   choices, OR — for models that don't FX-trace (anything with
   `nn.TransformerEncoder` internals, `len(...)`, etc.) — add a
   `torch.export` branch in `modelblaster/pipeline/extract_graph_export.py`.
   See ViNT's example for the export-path pattern.

3. Copy `modelblaster/examples/mlp_generic/run.sh` to
   `modelblaster/examples/<name>/run.sh` and change `MODEL_NAME=<name>`.

4. If the model uses ops not yet registered, add them — see below.

5. (Optional) calibration data: drop `modelblaster/datasets/<spec>.json`
   pointing at a list of input tensors; the extractor's per-channel
   activation calibration consumes it.

## Adding a new op kind

1. New `KernelSpec` in `modelblaster/pipeline/reference_kernels.py::KERNEL_SPECS`:
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
`modelblaster/kernels/<target>/<target>_<op>_<algo>.c`. The pipeline picks
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
  `modelblaster/kernels/rvv_f16/`. Ported from the canonical scalar fp16
  reference, vectorized via `vfwmacc`.
- **`gemmini_q31` tiled conv + linear** — `modelblaster/kernels/gemmini_q31/`.
  Routes through gemmini RoCC with bit-exact Q0.31 requantize.
- **`rvv_opu` outer-product matmul + linear** —
  `modelblaster/kernels/rvv_opu/`. Ported from upstream saturn
  `benchmarks/opu-gemm/kernel.h::i8_mm_bme_sq`; cited in the file
  headers. Exercises the Saturn OPU custom .insn programming model.

See `modelblaster/kernels/README.md` for the curated-vs-cache distinction
and the picker priority order.

## Curated kernel + spike correctness loop

For backends with custom instructions, you need a spike build that
decodes them. Two existing paths:

- **gemmini** — chipyard ships a `--extension=gemmini` spike fork.
  `modelblaster/examples/_run_lib.sh` finds it via `MODELBLASTER_GEMMINI_SPIKE`
  env, defaults to `/scratch2/dima/chipyard-fsim/.conda-env/...`.
- **rvv_opu** — custom spike extension at
  `hw/chipyard/toolchains/riscv-tools/riscv-isa-sim/customext/saturn_opu.cc`
  (in-repo functional model of `VOPACC` / `OPMVINBCAST` / `VMV_VR` /
  `VMV_RV`). `_run_lib.sh` finds the built spike via `MODELBLASTER_OPU_SPIKE`.
  See `notes/saturn_opu_spike_support.md` for build instructions.

For a brand-new accelerator, the path is the same: extend
`riscv-isa-sim/customext/` with a functional model, register via
`REGISTER_EXTENSION`, and point `_run_lib.sh` at the built binary.

## Notes / deep-dives

The `modelblaster/notes/` directory holds focused design notes per topic.
Highlights for this README's surface area:

| topic | note |
|---|---|
| Canonical pipeline diagram | `pipeline_overview.md` |
| int8 PTQ flow | `int8_quantization_flow.md` |
| Mixed-precision plan + experiments | `mixed_precision_plan.md`, `vint_mixed_precision_experiments.md` |
| Per-dispatch profile schema (IREE-shape) | `profile_emission.md` |
| FireSim re-rank in the optimize loop | `firesim_eval_design.md` |
| XPURT schedule format + the dispatch table | `scheduler_investigation.md`, `dispatch_and_cores.md` |
| Multi-model threading + modelblaster_pool | `multi_model_threading.md` |
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
