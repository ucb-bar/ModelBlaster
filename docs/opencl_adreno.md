# ModelBlaster ↔ KernelBlaster on Qualcomm Adreno (OpenCL)

This page covers the OpenCL target of the multi-slot flow
([kernelblaster_multislot.md](kernelblaster_multislot.md)): ModelBlaster graph
partitions exported as `mb-kb-problem/1` problems whose slots are OpenCL
kernels, optimised by KernelBlaster one slot at a time the way its existing
Qualcomm Adreno OpenCL flow works (`backends/opencl.py`, `agents/opt_opencl_rl.py`,
`docs/adreno_opencl_profiling_methodology.md` in KernelBlaster). The partition,
buffers, ops, data and checks are the CUDA design's; what changes is the slot
language, the runtime, the driver and where things build and run. fp16 follows
the CUDA fp16 contract ([fp16_cuda.md](fp16_cuda.md)). The "Design decisions"
table at the end lists each choice, the alternatives and the reason.

## What runs where

| piece | where |
|---|---|
| export (partition, torch reference walk, data.bin, seeds, driver) | any host (`kb_export --target opencl`) |
| build | on the machine with the GPU: the Adreno board itself (RB5 / QRB5165, Ubuntu arm64, its own `g++` and `libOpenCL.so` -- KernelBlaster's Adreno flow already compiles on the board, so no NDK), or a host with an OpenCL 2.0 GPU |
| run + checks + timing | the same machine; the driver needs only `data.bin` |
| agent | anywhere; it reaches the board over ssh (`AdrenoEvaluator`) or through KernelBlaster's OpenCL compile server + Adreno exec server (`OpenCLServerEvaluator`) |

The flow was first validated on the local NVIDIA TITAN RTX through its OpenCL
3.0 ICD (OpenCL C 1.2, coarse-grained SVM, no `cl_khr_fp16`) as a functional
stand-in, then on a real Qualcomm RB5 (QRB5165, Adreno 650, OpenCL 2.0, Ubuntu
20.04 aarch64, g++ 9.4) -- see "Validation on the RB5" below. Stand-in timings
are NVIDIA's; only the RB5 numbers are Adreno's.

## Problem format (target `opencl`)

The manifest is `mb-kb-problem/1` with `target: "opencl"`; slots are `.cpp`,
head/tail `walker_head.cpp` / `walker_tail.cpp`, `driver.cpp`, `data.bin`.
Buffers gain dtype `i64` (argmax/argmin outputs); fp16 problems have `f16`
buffers and the fp16 tolerances, as on CUDA.

**Slot** = one C entry point, as on CUDA (decision 1 of the CUDA design). Its
source is C++11 holding the OpenCL C programs as raw strings and the host
launcher with exactly the KernelSpec signature:

```cpp
static mb_program cl_relu = MB_CL(R"CL(
__kernel void k_relu(__global const float *input, __global float *output, int n) {
    long i = MB_TID;
    if (i < n) { float v = input[i]; output[i] = fmax(v, 0.0f); }
}
)CL");
void kernel_relu(const float* input, float* output, int n) {
    mb_launch(MB_KERNEL(cl_relu, k_relu), mb_grid(n), input, output, n);
}
```

**Runtime** (`gpu_eval/harness/opencl/mb_opencl_runtime.h`, copied into the
problem as the walker head):

- pointers in the signature are coarse-grained **SVM** pointers (`clSVMAlloc`),
  so views/aliases (`view`, `chunk2_c1`) are pointer arithmetic, as in MB's C
  ABI, and the tail is the CUDA tail (`mbslot_<id>::kernel_x((const float*)b[3], ...)`);
- a program is built on first use (`-cl-std=CL1.2` + `MB_CL_OPTS(opts, src)` +
  `$MB_CL_OPTS`) with a device prelude: `MB_THREADS`, `MB_TID`, `mb_ksum`/`mb_kadd`
  (compensated fp32 sum), `mb_idx6`, `mb_atomic_addf`;
- `mb_launch(kernel, range, args...)` sets arguments by C++ type (SVM pointer,
  scalar, `mb_local{bytes}`, `cl_mem`; `double` is a compile error) and enqueues
  with a profiling event; ranges `mb_grid(n)`, `mb_groups(g, local)`, `mb_range2/3`;
  `mb_copy`/`mb_zero` are timed device copies; `MB_SCRATCH(prog, bytes)` is a
  grow-only per-program buffer;
- each pattern of NULL pointer arguments gets its own `cl_kernel` (NVIDIA's ICD
  rejects a NULL argument after the same argument held an SVM pointer: biased
  and bias-less `linear` in one slot failed with `CL_INVALID_KERNEL_ARGS`);
- errors never abort: the first is kept and reported by the driver.

**Driver** (`driver.cpp`, generated): reads `data.bin` at start-up
(`MB_DATA_PATH`, argv[1], `./data.bin`, then the export path), runs the
partition and checks every output, runs every op alone on the reference's
inputs (attribution), then times the partition. Same lines as CUDA
(`MB_CHECK`, `MB_SLOT_CHECK`, `MB_PARTITION_US`, `MB_SLOT_US`, `MB_OP_US`,
`passed|failed`) plus `MB_DEVICE`, `MB_DEVICE_INFO`, `MB_KERNEL_US <slot>
<kernel> <us> n=<launches>` and `MB_KINFO <slot> <kernel> wg_max= pref_mult=
local_mem= private_mem=`. All times are `cl_event` COMMAND_START..END, the
Adreno flow's methodology: an op is the span from its first command's start to
its last command's end, the partition the span over all ops, per kernel the
summed busy time. An op that enqueued nothing through `mb_launch`/`mb_copy`/
`mb_zero` fails the run (and KernelBlaster rejects slot sources that call
`clEnqueue*`/`clFinish` directly), so no work escapes the timer.
`MB_MODE=profile` (or argv[2] `profile` for exec paths without environment
variables) runs every op once and prints the per-kernel lines -- the OpenCL
profile KernelBlaster uses instead of NCU.

**Build**: `g++ -O2 -std=c++11 kernels.cpp driver.cpp -lOpenCL`; kernels TU +
driver concatenated (`kb_export_opencl.single_tu`, `#define MB_SINGLE_TU`) is
one self-contained `main.cpp` for single-file paths (KernelBlaster's OpenCL
compile server, which now builds `.cpp` mains with g++ on the board or locally).

## Seeds (`pipeline/opencl_seeds.py`, `gpu_eval/harness/opencl/`)

Every fp32 KernelSpec except the int8 ViNT post-process has a seed (89 entry
points; every op kind of KernelBench Level1-3). Three families are generated from
templates, because their kernels differ in one expression and take the
launcher's parameters one-to-one: **MAP** (17 unary + 2 binary elementwise ops),
**AXIS** (11 `[outer, reduce, inner]` reductions/normalisations incl. argmax/argmin
with `int64` output) and **LOSS** (6 scalar losses: per-group compensated
partials in `MB_SCRATCH`, then one group). The rest (conv 2-D/3-D/transposed/
grouped/depthwise/N-d, pooling, the GEMM family, softmax/layer/group norm,
attention, layout, indexing, scans, LSTM/GRU) is hand-written in
`mb_opencl_kernels.cpp`, translated from the CUDA hand kernels, and a slot gets
its launcher's dependency closure (the CUDA `CudaLibrary`). Seeds are
correctness seeds (mostly one work-item per output) with two deliberate
differences from CUDA: no fp64 (dot products and long sums use compensated fp32
`mb_kadd` where CUDA used `double`), and row ops use one work-group per row (a
Level1 LayerNorm is one row of 524 288).

**fp16** (`_f16` ops): binary16 storage, fp32 math, one RNE rounding per store
-- the CUDA fp16 contract. In OpenCL that is `__global half *` with
`vload_half`/`vstore_half`, core OpenCL 1.2 (no `cl_khr_fp16`, so it also runs
on the NVIDIA stand-in). The `_f16` seeds are generated from the fp32 ones
(`opencl_seeds.to_f16`: half data pointers, `p[e]` → `vload_half(e, p)`, stores and
compound stores → `vstore_half`, float scratch kept, `float *` → `__half *` on the
host, `_f16` entry names); `softmax_f16` (input scale), `adaptive_avg_pool2d_f16`
(OH/OW), `cat2_c1_f16` and `sdpa_f16` (no fp16 accumulation) are hand variants
(`mb_opencl_kernels_f16_hand.cpp`). `__half` (the CUDA spelling of `_Float16`) is
`cl_half` in the runtime, so slot signatures and their hashes match the CUDA
fp16 problems. `scatter_dim_f16` (half atomics) has no seed.

    python -m modelblaster.pipeline.kb_export IR_DIR OUT --target opencl --build-check
    python -m modelblaster.pipeline.opencl_seeds OUT [--dtype fp16]   # every seed + one TU of all
    python -m modelblaster.pipeline.kb_export_opencl PROBLEM_DIR       # build + run here

## KernelBlaster side

See KernelBlaster `docs/multislot_optimization.md` ("OpenCL / Adreno"). In short:
`TargetProfile` per manifest target (prompt, ```` ```cpp ```` fence, Adreno technique
catalogue, Adreno roofline peaks, arch label, profiler), store/outcome/claim
records keyed by `(target, arch)` so CUDA and OpenCL winners never mix,
evaluators `OpenCLLocalEvaluator` / `AdrenoEvaluator` (board over ssh) /
`OpenCLServerEvaluator` (OpenCL compile + Adreno exec servers), and
`knowledge/opencl_profile.py` (per-kernel event times, launches, work-group
limits, private memory) in place of NCU.

## Validation (local TITAN RTX through OpenCL; 2026-10-23)

Whole-graph export (`kb_export --target opencl --build-check`, ≤ 4 in parallel):
build, partition check, and every op isolated, on the final code
(`tmp/opencl_scratch/sweep/run.sh`, the CUDA sweep's procedure):

| set | IRs | pass | failures |
|---|---|---|---|
| Level1 (`kbx_sweep2/l1_ir`) | 100 | **100** | -- (needed the Level1 walker ops / conv1d fix from `feat/kb-multislot`; before it, 28 failed in the reference walk, not in OpenCL) |
| Level2 (`level2/ir16`) | 100 | **100** | -- (two runs once hit `clCreateContext: -6` / an SVM alloc failure while the GPU was saturated; both pass on re-run) |
| Level3 (`level3/ir16`) | 50 | **50** | -- (Swin MLP / SwinV2 needed the per-NULL-pattern kernel objects) |
| Level3 fp16 (`work/fp16/ir`) | 50 | **48** | 48/49_Mamba2: `bmm_f16` outputs at the binary16 limit (65504) round to inf on one side; the fp16 doc already marks these models as not fp16-representable |

Partition times of the seeds on the stand-in (median / max): Level1 178 us / 26 ms,
Level2 17 ms / 0.42 s, Level3 115 ms / 5.7 s (41_GRUBidirectional: per-timestep
launches). They are NVIDIA's; Adreno 650 has ~1/13 of the TITAN's fp32 peak and
~1/15 of its bandwidth.

- **Seeds**: 89 fp32 and 88 fp16 entry points; every OpenCL C program builds on
  the device and every seed compiles in one TU (`pipeline/tests/test_kb_export_opencl.py`).
- **Scripted agent** (`stub:`): KernelBlaster `tests/test_multislot_opencl.py` runs
  the multi-slot agent end to end on the OpenCL LeNet fixture (float4 relu win,
  broken linear repaired by the fix loop, store/outcome records keyed
  `opencl`/`titanrtx`, OpenCL profiles in the prompts).
- **Live run** (Bedrock `us.anthropic.claude-haiku-5-5`, 1 rollout × 5 steps,
  ≤ 1 fix, `--ncu best` = OpenCL profiles): Level3 `8_ResNetBasicBlock` fp32,
  14.5 ms → 1.24 ms (11.7×) in 6 calls (25 094 prompt / 28 360 completion
  tokens, ≈ $0.17); conv2d 14.2 → 0.93 ms via work-group retiling then an
  implicit GEMM (one fix round), rel err 2.9e-7; batchnorm/relu attempts neutral
  or regressed and were not kept. Stand-in timings, not Adreno's.
- **Board path**: `AdrenoEvaluator` against `ssh localhost` (scp, g++ on the
  target, run, compile errors reported, remote dir removed; later against the
  RB5 itself, with the board lock);
  `OpenCLServerEvaluator` plumbing with the servers faked.

## Validation on the RB5 (Adreno 650; 2026-10-28)

Device as the driver reports it: `QUALCOMM Adreno(TM) | OpenCL 2.0 Adreno(TM) 650`,
3 compute units, max work-group 1024, 32 KiB local memory, SVM caps 0xb
(coarse + fine-grained buffer, atomics), `cl_khr_fp16`, **CL_DEVICE_MAX_MEM_ALLOC_SIZE
256 MiB**, CL_DEVICE_GLOBAL_MEM_SIZE 1 GiB (7.5 GB of RAM shared with the CPU). Every
problem was exported on the host (`kb_export --target opencl`), shipped as one
`main.cpp` (`single_tu`) + `data.bin`, built with the board's g++ and run there, each
build + run under `flock` of the board's shared lock.

**Seeds.** Every seed program (89 fp32, 88 fp16) was built on the device and every
kernel created; both all-seed TUs compile with the board's g++. Seed launches use
256 work-items (`MB_THREADS`) everywhere; the device's `CL_KERNEL_WORK_GROUP_SIZE` is
1024 for every kernel except `k_group_norm` (768, 276 B private memory), so no seed
exceeds its kernel's limit. One compiler bug: the Adreno OpenCL compiler **aborts the
process** (`QGPUI64Emulate: Unsupported FtoI/ItoF instructions!`, SIGABRT inside
`clBuildProgram`) on a half -> 64-bit integer conversion, and every later build in the
same process then fails (`ReplacementGlobalContext not set?`). The `_f16` seeds of
`gather_dim`, `index_select` and `topk` hit it (6 programs failed, 3 of them knock-ons);
indices now convert through `int` (`mb_wrap`, and `(long)(int)vload_half(...)` in the
fp16 transform), and all 177 programs build.

**Problems** (whole-graph, partition check + every op isolated + cl_event timing):

| set | run | pass | partition time of the seeds (median / max) | failures |
|---|---|---|---|---|
| Level1 (`kbx_sweep2/l1_ir`) | 99 (all but 100_HingeLoss: a 4 GiB buffer) | **99** | 51 ms / 345 ms | -- |
| Level2 (`level2/ir16`) | 13 | **11** | 2.2 s / 98.5 s (72_ConvTranspose3d, 771 MB) | 89, 96: one 512 / 481 MiB conv-transpose output > the 256 MiB allocation limit |
| Level3 (`level3/ir16`) | 6 | **6** | 2.9 s / 44 s (27_RegNet, 1.08 GB) | -- |
| Level3 fp16 (`work/fp16/ir`) | 5 | **5** | 0.87 s / 9.4 s (35_LSTM) | -- |

Level1 covers every Level1 op kind; Level2: 1, 6, 11, 12, 19, 28, 34, 43, 66, 72, 84, 89,
96; Level3: 8_ResNetBasicBlock, 27_RegNet, 31_VisionAttention, 32_CvT,
34_VanillaRNNHidden, 47_NetVlad; fp16: 6, 8, 17, 32, 35. Every partition check and every
isolated op passed (worst isolated rel err 1.8e-6 fp32, 6.3e-4 fp16). Memory: one
allocation is capped at 256 MiB, but the sum is not capped at GLOBAL_MEM_SIZE (RegNet
allocates 1.08 GB and passes); the driver's `MB_ERROR alloc` line now names the
device's limits. The driver also holds data.bin in host memory, so a problem needs
about twice its data.bin in RAM. Seed times are far above the stand-in's (Level1 median
51 ms vs 178 us): the seeds are naive (one work-item per output, no tiling, 256-wide
groups where the kernels report a preferred multiple of 128 and allow 1024) on a GPU
with ~1/13 of the TITAN's fp32 peak -- the agent's first steps fix exactly that.

**Live agent** (KernelBlaster `run_multislot.py --evaluator board --arch adreno650
--board-lock ...`, Bedrock `us.anthropic.claude-haiku-5-5`, 1 rollout x 4 steps, <= 1 fix,
`--ncu best`), Adreno times of the seeds vs the best variants, both measured on the board:

| problem | seed | best | speedup | winning steps |
|---|---|---|---|---|
| L1 1_Square_matmul | 56.3 ms | 2.12 ms | 26.6x | work-group retile, then native math / mad |
| L1 63_conv2d | 138.4 ms | 5.45 ms | 25.4x | thread coarsening, then implicit GEMM |
| L2 19_ConvTranspose2d_GELU_GroupNorm | 436.1 ms | 28.4 ms | 15.4x | group_norm coarsening (51.5 -> 3.4 ms), conv_transpose2d 384 -> 24.6 ms |
| L1 40_LayerNorm | 3.21 ms | 0.349 ms | 9.2x | work-group size, coarsening, native math |
| L3 32_CvT (10 slots) | 113.7 ms | 36.8 ms | 3.1x | strided_copy 45.4 -> 0.30 ms, linear 40.9 -> 9.2 ms (local tiling) |

22 LLM calls, 99 k prompt / 110 k completion tokens, about $0.65. Two of the 20 steps
produced no code (the response hit the 8192-token cap); no Adreno-specific build or
run failure.

## Design decisions

| # | Decision | Alternatives considered | Why |
|---|---|---|---|
| 1 | Slot = C++11 file with raw-string OpenCL C programs + the host launcher | `.cl` file + launcher generated by MB; launcher in the walker | One LLM edit stays in one slot and covers what matters on Adreno: work-group shape, number of kernels (two-pass reductions, per-timestep fusion), scratch, build options. A generated launcher would freeze the NDRange. The CUDA assembly rule (namespace per slot) applies unchanged. |
| 2 | Coarse-grained SVM pointers as the ABI | `cl_mem` + offset per pointer; sub-buffers for views | Keeps the KernelSpec signature (and its hash) as on CUDA; views are pointer arithmetic, while sub-buffers need `CL_DEVICE_MEM_BASE_ADDR_ALIGN`-aligned offsets that `chunk2_c1` views do not have. Adreno 5xx+ and NVIDIA support coarse-grained SVM. Cost: OpenCL 1.2-only devices (POCL builds without SVM) are not supported. |
| 3 | All launches through `mb_launch`/`mb_copy`/`mb_zero`, timed by `cl_event` | wall clock around `clFinish`; trusting the slot's own timing | Per-op/slot/kernel GPU times without a profiler, matching the Adreno flow's methodology; direct enqueues are rejected so no work runs untimed. |
| 4 | Driver reads `data.bin` at run time | `.incbin` as on CUDA | The binary is built on the board; the data travels once per problem (`AdrenoEvaluator`) or as an exec-server kernel file, and a 500 MB VGG problem does not have to be linked per candidate. |
| 5 | Build on the target with g++, OpenCL C 1.2 kernels | NDK cross-compile; OpenCL C 2.0/3.0 | The RB5 runs Ubuntu with gcc and KernelBlaster's Adreno flow already compiles there; CL 1.2 C runs on Adreno and on the NVIDIA stand-in alike. Android-only devices would need an NDK build of the same two files (not done). |
| 6 | Compensated fp32 (`mb_kadd`) instead of the CUDA seeds' fp64 | plain fp32; `cl_khr_fp64` when present | Adreno has no fp64; a plain fp32 sum over ~1e3-1e6 terms misses the 1e-3 checks the CUDA seeds were tuned against. The same code runs on both devices, so the stand-in validates what the board runs. |
| 7 | Templates for map/axis/loss ops, a hand library for the rest | 89 hand seeds; auto-porting the C references (like `ref2cuda`) | ~36 ops differ in one expression; the references are serial (one work-item for a whole conv would time out on Adreno). Hand seeds are translated from the CUDA kernels that already pass Level2/3. |
| 8 | fp16 = binary16 storage via `vload_half`/`vstore_half`, generated from fp32 | `half` arithmetic under `cl_khr_fp16`; separate fp16 seeds | Matches the CUDA fp16 contract (fp32 math, one rounding), needs no extension (validated on the NVIDIA stand-in), and keeps one source of truth; `half` math is a technique the agent may apply on the board. |
| 9 | OpenCL profile = event times + kernel resource queries | Snapdragon Profiler; none | Adreno exposes no counters to an OpenCL program; `CL_KERNEL_WORK_GROUP_SIZE` falls when a kernel needs many registers and `CL_KERNEL_PRIVATE_MEM_SIZE` shows spills -- the two register-pressure signals Adreno gives, plus launch counts for launch-bound slots. |
| 10 | Validate functionally on local OpenCL when no board is reachable | wait for the board; POCL | The NVIDIA ICD is a real OpenCL 3.0 implementation with SVM; POCL was not installed and installing system packages was out of scope. |

## Limitations / what remains

- **Allocation limit**: one buffer is at most `CL_DEVICE_MAX_MEM_ALLOC_SIZE` (256 MiB
  on Adreno 650); Level2 89/96 (and Level3 VGG16/19 activations, 392 MiB) do not fit.
  Splitting a buffer would change the slot ABI (one pointer per tensor); smaller
  shapes or a partition boundary at that tensor are the workarounds.
- **Board resources**: the RB5's `/tmp` is a 3.8 GiB tmpfs -- put `data.bin`
  (`--remote-dir`) on persistent storage; the driver needs about 2x data.bin of RAM.
- **Compiler aborts**: an OpenCL C construct that crashes the Adreno compiler kills
  the driver process (SIGABRT, no `MB_ERROR` line); the half -> 64-bit int conversion
  is the one found so far.
- The seeds are correctness seeds and slow on Adreno (see above); Adreno technique
  values (bandit statistics, claims) are re-learned per arch -- records are keyed by
  arch, so the stand-in's do not leak into Adreno's.
- Write-back (`kb_ingest`) and fusion variants (`kb_proposals`) are CUDA-only:
  OpenCL winners are exported by KernelBlaster (`--export-winners`, per target)
  but not yet ingested into an MB OpenCL kernel cache.
- No Android/NDK build; no `cl_mem`/OpenCL 1.2-only path (decision 2).
- fp16 seeds compute in fp32; Adreno's 2x `half` ALU rate is left to the agent
  (technique `3.4_half_precision` of the Adreno flow).
