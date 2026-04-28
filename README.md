# agents/ — LLM-driven kernel generation + optimization for Zephyr/RISC-V

A minimal alternative to the merlin `xpurt` flow: take a PyTorch model,
extract a small IR, have an LLM write per-op C kernels, build them into a
Zephyr app, and validate + profile on `spike`. The same harness extends to
real boards once we have one.

## Status

Working end-to-end on every model × every backend × every quant axis below:

| stage                  | scalar fp32 | RVV fp32 | scalar int8 | RVV int8 |
|------------------------|-------------|----------|-------------|----------|
| extract IR             | ✅          | ✅       | ✅          | ✅       |
| LLM kernel-gen + verify| ✅          | ✅       | ✅          | ✅ (linear_s8 fully; conv2d_s8 falls back to scalar reference under `-march=rv64gcv`) |
| beam-search optimize   | ✅          | ✅       | ✅          | ✅       |
| Zephyr build + spike   | ✅          | ✅       | ✅          | ✅       |

**Models in scope** — both synthetic demos and real trained checkpoints:

| model           | source                                              | shape                          |
|-----------------|-----------------------------------------------------|--------------------------------|
| `mlp_generic`   | `agents/models/mlp_generic.py` (random init)        | 16 → 32 → 32 → 10 (ReLU)       |
| `mlp_control`   | trained rsl_rl PPO actor (steering tracking)        | 16 → 256 → 128 → 64 → 4 (ELU)  |
| `lenet`         | `agents/models/lenet.py` (random init)              | 1×28×28 LeNet-5 style          |
| `dronet`        | trained DroNet from `qnn_models/dronet.py`          | 3×112×112 → (steer, collision) |

## Quick start

From the repo root (`zephyr-chipyard-sw/`), one-time per shell:

```bash
source tools/miniforge3/etc/profile.d/conda.sh && conda activate zephyr
source scripts/set_envvars_sdk.sh
source ../set_api_keys.sh   # only needed for BACKEND=llm or --optimize
```

Then:

```bash
# 1. correctness only, hardcoded reference kernels (no LLM, no AWS)
bash agents/examples/mlp_generic/run.sh

# 2. correctness with LLM-written kernels (verified against reference)
BACKEND=llm bash agents/examples/mlp_generic/run.sh

# 3. correctness + beam-search optimization on spike (LLM, profiled)
BACKEND=llm OPTIMIZE=1 bash agents/examples/mlp_generic/run.sh

# 4. RVV variant (vector intrinsics; slower verify but bigger gains)
TARGET=rvv BACKEND=llm OPTIMIZE=1 bash agents/examples/mlp_generic/run.sh

# 5. int8 PTQ pass (per-tensor symmetric, fused linear+relu / conv+relu,
#    Q0.31 requantize, bit-exact compare against integer simulator)
QUANT=int8 bash agents/examples/mlp_generic/run.sh

# 6. RVV int8 (linear_s8 LLM-generated; conv2d_s8 falls back to scalar)
QUANT=int8 TARGET=rvv BACKEND=llm bash agents/examples/lenet/run.sh
```

LeNet, DroNet, and `mlp_control` work the same way — replace the model name
in the path. Examples live under `agents/examples/<model>/`.

## Trained models (real checkpoints)

`mlp_control` and `dronet` load real trained weights at extract time and
run them through the same flow as the synthetic demos. The PyTorch forward
of the loaded model is the golden — spike output is compared against it.

```bash
# Trained drone-control MLP policy:
# - architecture: 16 → 256 → 128 → 64 → 4 ELU (from rsl_rl_ppo_cfg.py)
# - default checkpoint: logs/rsl_rl/.../model_6998.pt (override via
#   AGENTS_MLP_CONTROL_CKPT)
bash agents/examples/mlp_control/run.sh

# Trained DroNet (steering + collision):
# - architecture: 3×112×112 input, model_size=small, dual-output head
# - default checkpoint: logs/dronet/2026-04-27_17-10-41/best.pt (override
#   via AGENTS_DRONET_CKPT)
bash agents/examples/dronet/run.sh
```

Add a new trained model by writing `agents/models/<name>.py` with `get_model()`
that returns a torch `nn.Module` (with weights loaded) and `get_sample_input()`
that returns the right input tensor — see `mlp_control.py` and `dronet.py`
for templates.

## How a run is structured

`run.sh` chains five stages. Each stage's outputs are deterministic and
inspectable on disk.

```
[1] extract_graph.py    PyTorch model -> graph.json + weights.npz + io.npz
                        (IR is target-independent; for QUANT=int8 the
                         extractor runs PTQ and emits int8 weights + per-
                         tensor scales + an integer-pipeline simulator
                         golden)
[2] generate_skeleton.py IR -> model.{c,h}, weights.{c,h}, test_io.h
                         (per-target dir generated/<target>/, instruments
                          run_model with rdcycle around each kernel call)
[3] generate_kernels.py  IR -> kernels.{c,h}
                         --backend reference|llm    where impls come from
                         --target  scalar|rvv|...   what HW they target
                         --quant   fp32|int8        which op family to use
                         --algorithms <list>        per-op algorithm filter
                         --optimize                 beam-search faster variants
[4] west build           agents/harness + generated/<target>/* -> zephyr.elf
                         (per-target build/<target>/)
[5] spike + compare      run zephyr.elf, parse output + profile blocks,
                         compare to PyTorch / integer-sim golden, write
                         profile.csv (bit-exact compare for int8, fp32
                         tolerance for fp32)
```

The harness (`agents/harness/`) is a single Zephyr sample template that
takes `-DMODEL_DIR=...`, `-DAGENTS_BACKEND=...`, and per-backend cflags
from `run.sh`. Same harness for fp32 / int8 — `model.h` typedefs
`model_input_t` and `model_output_t` so `main.c` is dtype-agnostic.

## Environment variables consumed by run.sh

| var          | values                | default     | notes |
|--------------|-----------------------|-------------|-------|
| `BACKEND`    | `reference`, `llm`    | `reference` | source of kernel implementations |
| `TARGET`     | `scalar`, `rvv`       | `scalar`    | HW backend (defined in `pipeline/backends.py`) |
| `QUANT`      | `fp32`, `int8`        | `fp32`      | quantization mode; segregates artifacts under `<model>/<quant>/` |
| `OPTIMIZE`   | `0`, `1`              | `0`         | beam-search after correctness; requires `BACKEND=llm` |
| `ALGORITHMS` | `all`, `default`, csv | `all`       | per-op algorithm filter (e.g. `direct`, `im2col_gemm` for conv2d) |
| `BEAM`       | int                   | `2`         | beam width per op |
| `EXPANSIONS` | int                   | `3`         | LLM proposals per beam member per iteration |
| `ITERATIONS` | int                   | `2`         | beam-search iterations |
| `AGENTS_MLP_CONTROL_CKPT` | path | (built-in)  | override the trained MLP policy checkpoint |
| `AGENTS_DRONET_CKPT`      | path | (built-in)  | override the trained DroNet checkpoint |

Every LLM call uses `meta.llama4-maverick-17b-instruct-v1:0` via Bedrock.
The `us.` cross-region inference profile prefix is added automatically.
`AWS_BEARER_TOKEN_BEDROCK` and `MODEL` come from `set_api_keys.sh`.

## Where artifacts land

```
agents/examples/<model>/<quant>/
  generated/                 IR (target-independent, quant-specific)
    graph.json
    weights.npz
    io.npz                   PyTorch reference input/output
    profile.csv              last spike run's per-kernel cycles
  generated/<target>/        generated C (per backend)
    model.{c,h}              run_model() driver + profile struct
    weights.{c,h}            const float / const int8 weight arrays
    kernels.{c,h}            per-op implementations (reference / LLM / optimized)
    test_io.h                model_test_input + model_test_golden
    optimize_summary.json    beam-search history (only with --optimize)
  build/<target>/            west build tree (zephyr.elf etc.)
  cache/<target>/            PASSing kernels keyed <target>_<op>_<algo>.c;
                             reused across runs to skip Bedrock dice-rolling
```

`generated/` and `build/` are regenerated by `run.sh` and gitignored.
`cache/` is intentionally NOT gitignored — successful kernels persist
across machines. The `<quant>` axis (`fp32`, `int8`) segregates everything
so different quant modes sit side by side without collision.

## Components

```
agents/
  models/                          PyTorch model defs (one .py per model)
    mlp_generic.py                 demo MLP (16 → 32 → 32 → 10, ReLU)
    mlp_control.py                 trained rsl_rl PPO actor (16 → ... → 4, ELU)
    lenet.py                       demo LeNet-5
    dronet.py                      trained DroNet (3×112×112, dual-output)
  pipeline/                        the codegen pipeline
    extract_graph.py               torch.fx symbolic_trace + ShapeProp -> IR JSON
                                   (fp32 path + int8 PTQ path with integer
                                    simulator that produces bit-exact goldens)
    reference_kernels.py           KernelSpec per op: signature, semantics,
                                   reference C, ctypes argtypes, extra_shapes,
                                   AlgorithmCandidate list
    backends.py                    Backend registry (scalar, rvv): cflags,
                                   prj.conf overlay, spike_args, verify_method
    bedrock_client.py              Converse API wrapper, transient-error retry
    generate_skeleton.py           IR -> model/weights/test_io C source;
                                   dtype-aware (float / int8_t / int32_t)
    generate_kernels.py            IR -> kernels.{c,h}; correctness loop
                                   (per-algorithm, per-spec) + beam-search
                                   optimize loop
    verify_kernel.py               host-compile + ctypes numerical compare
                                   (also used for LLM retry diagnostics);
                                   bit-exact for integer ops
    profile_kernel.py              build_and_run(): per-target Zephyr build
                                   + spike + parse profile + golden compare
    prompts/
      optimization_guide_scalar.md instruction guide for scalar codegen
      optimization_guide_rvv.md    instruction guide for RVV intrinsics
                                   (covers fp32 + int8 widening patterns)
  harness/                         Zephyr sample template
    CMakeLists.txt                 reads MODEL_DIR + AGENTS_BACKEND + cflags
    prj.conf                       baseline (FPU, fp printf, 4MB stack for
                                   im2col VLAs)
    src/main.c                     reads test_io, calls run_model, prints
                                   output + profile blocks, sys_reboots
    backends/<name>.conf           per-target Kconfig overlay
  validation/
    spike_runner.py                run spike, parse output + profile, compare
                                   (auto bit-exact tolerance for int8 goldens)
  examples/
    _run_lib.sh                    shared orchestration body (sourced)
    <model>/run.sh                 per-model launcher (3 lines)
```

## Op kinds supported

| op family      | fp32 specs    | int8 specs       |
|----------------|---------------|------------------|
| matmul         | `linear`      | `linear_s8`      |
| 2D conv        | `conv2d` (`direct`, `im2col_gemm`) | `conv2d_s8` |
| 2D max-pool    | `maxpool2d`   | `maxpool2d_s8`   |
| activations    | `relu`, `elu`, `sigmoid` | `relu_s8`, `sigmoid_s8` |
| residual add   | `add`         | `add_s8` (per-input scales + rescale) |
| BN (eval-mode) | `batchnorm2d` (pre-folded) | `batchnorm2d_s8` (per-channel float scale+bias) |
| view ops       | `view` (flatten / dropout in eval) | `view` (same) |

For int8, `linear_s8` and `conv2d_s8` follow the muRISCV-NN convention:
int8 in/out, int32 bias, Q0.31 requantize multiplier+shift, fused
`activation_min`/`activation_max` for ReLU. `linear→relu` and `conv2d→relu`
are auto-fused into a single `_s8` op with `activation_min=0`.

## Verify routing

Two paths — the backend declares which:

- **`host_ctypes`** (scalar): candidate C compiles to a host `.so`,
  invoked via ctypes against random fp32/int8 inputs at every shape from
  the IR + every shape in `KernelSpec.extra_shapes`. Compares to the
  reference C compiled the same way. Fast (~50 ms per call).

- **`spike_harness`** (rvv): candidate is cross-compiled into the full
  Zephyr binary, run on spike, model output compared to the PyTorch /
  integer-sim golden. Necessarily coarser (full-model golden, not per-op
  shapes), but the only correctness check available for backends with
  target-only intrinsics. Slow (~30 s per call).

`generate_kernels.py` chooses automatically based on `target.verify_method`.

## Adding a new model

1. Drop `agents/models/<name>.py` with `get_model()` and `get_sample_input()`.
   For trained models, load weights inside `get_model()` from a checkpoint.
2. Add `<name>` to the `--model` choices in `pipeline/extract_graph.py`.
3. Copy `agents/examples/mlp_generic/run.sh` to
   `agents/examples/<name>/run.sh` and change `MODEL_NAME=<name>`.
4. If the model uses ops not yet in `KERNEL_SPECS`, add them — see below.

## Adding a new HW backend

`pipeline/backends.py` is the only place that needs editing for the
target itself. Register a `Backend(...)` with:

- `kernel_cflags`: extra `-march=` etc. applied **only** to `kernels.c`
- `kernel_includes`: headers prepended to `kernels.c`
- `prj_conf_overlay`: name of the file under `harness/backends/`
- `spike_args`: appended to the spike command line (e.g. `--isa=...`)
- `optimization_guide`: filename under `pipeline/prompts/`
- `verify_method`: `host_ctypes` or `spike_harness`

Then add `harness/backends/<name>.conf` and
`pipeline/prompts/optimization_guide_<name>.md`. No other code changes.

## Adding a new op kind

1. New entry in `pipeline/reference_kernels.py:KERNEL_SPECS`:
   - `signature` (must match `kernels.h` byte-for-byte)
   - `semantics` (English description for the LLM prompt)
   - `reference_impl` (correct, naive scalar C — used as the verify oracle
     and as the `--backend reference` output)
   - `extra_shapes` (verify shapes beyond what's in the model IR)
   - `argtypes_factory` (ctypes signature for host verify)
   - optional `algorithms` list (alternative seeds for the LLM, e.g.
     `direct` vs `im2col_gemm`)
2. Add input gen + run-kernel branch in `pipeline/verify_kernel.py`
3. Handle the op in `pipeline/extract_graph.py` (FX node → IR entry) and
   `pipeline/generate_skeleton.py` (IR entry → kernel call site).
4. For int8 op kinds, add the matching entry to the integer simulator in
   `extract_graph.py:extract_int8` so the bit-exact golden stays in sync.

## Profiling

`run_model()` brackets each kernel call with `rdcycle()` (1-instruction
read of the `mcycle` CSR). Each call writes a record `{name, op, shape,
cycles}` into a static array. After `run_model` returns, `main.c` prints
the array as a CSV between `=== AGENTS_PROFILE_BEGIN/END ===` markers.
`spike_runner.py` parses that block, writes `profile.csv` next to
`io.npz`, and prints a per-op cycle breakdown.

The optimize loop reads the same CSV (via `profile_kernel.build_and_run`)
to gate beam-search proposals: a candidate must verify AND have lower
cycles than its parent to survive.

## Caveats / known limitations

- **`spike` is an ISA simulator with flat memory.** Cycle counts reward
  pipeline-pattern wins (multiple accumulators, breaking fp dependency
  chains, instruction-level unrolling) and are blind to memory locality
  (cache blocking, prefetch, reuse). The optimization guides intentionally
  keep both classes — they'll start producing measurable gains as soon as
  the target is a cycle-accurate simulator or real silicon. No code
  changes needed when that happens.
- **The reference impls are the trusted oracle**, not the kernels.h
  signatures. If you change a `KernelSpec.signature`, the host verify
  will keep working only if the reference impl's first line matches.
- **The Bedrock model is hardcoded** to Llama 4 Maverick. To swap, set
  `MODEL=<other model id>` before sourcing or running.
- **Stale Vitis cmake**: if `west` errors with "cmake 3.3.2", `run.sh`
  prepends `/usr/bin` to PATH to dodge it. If you invoke `west` outside
  `run.sh`, do the same.
- **int8 PTQ today**: per-tensor symmetric quantization (zero_point=0)
  with single-input calibration (one sample input). Per-channel weight
  quant + per-tensor asymmetric activations, and proper PT2E calibration,
  are next-pass adds — the IR's `quant` block already accepts the
  `input_offset`/`filter_offset`/`output_offset` fields.
- **conv2d_s8 RVV**: currently the LLM struggles with the padded/strided
  vectorize-over-OW pattern at int8, so the `dronet`/`lenet` int8 RVV
  caches fall back to the scalar reference compiled under
  `-march=rv64gcv`. A clean path forward is an int8 `im2col_gemm`
  algorithm seed mirroring the working fp32 pattern.

## Open follow-ups

- **`im2col_gemm` for `conv2d_s8`**: would unlock real RVV int8 throughput
  on LeNet/DroNet (the int8 GEMM stage is just `linear_s8` we already have).
- **Per-channel weight quant**: switch the conv/linear weight scale to a
  per-output-channel array; per-channel multiplier+shift naturally extends
  the kernel signature.
- **Optimize phase iteration over algorithms**: today the optimize loop
  starts from whichever algorithm correctness picked; a richer version
  beam-searches each algorithm independently and picks the global best.
