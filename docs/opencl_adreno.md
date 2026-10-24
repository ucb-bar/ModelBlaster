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

No Adreno board was reachable during this work (the `qrb_cloud` QDC tunnel,
localhost:2222, was down; no adb/NDK on the box). Everything below ran on the
local NVIDIA TITAN RTX through its OpenCL 3.0 ICD (OpenCL C 1.2, coarse-grained
SVM, no `cl_khr_fp16`) as a functional stand-in: correctness results carry over,
**timings do not** (they are NVIDIA's, and the GPU was shared). The board path
(`AdrenoEvaluator`: scp, build with g++ on the target, run, `rm -rf`) was
exercised end to end against `ssh localhost`.

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
  target, run, compile errors reported, remote dir removed);
  `OpenCLServerEvaluator` plumbing with the servers faked.

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

## Limitations / what remains for real Adreno hardware

- **Run on the board**: export as above, then `scripts/run_multislot.py PROBLEMS
  --evaluator board --board-host root@<rb5> --arch adreno650` (needs the QDC
  tunnel or a lab RB5, `g++` and `libOpenCL.so` on it). Expected issues to
  check first: the Adreno compiler's limits on kernel work-group size (seeds use
  256 for `__local` reductions; a kernel whose `CL_KERNEL_WORK_GROUP_SIZE` is
  lower would fail to enqueue), SVM allocation limits for the largest Level3
  problems on 8 GB shared memory, and the 3.8 GiB `/tmp` tmpfs for `data.bin`
  (use `--remote-dir` on persistent storage for VGG-size problems).
- The live agent run and every timing above are NVIDIA's. Adreno speedups and
  technique values (bandit statistics, claims) must be re-learned on the board;
  records are keyed by arch so the stand-in's do not leak into Adreno's.
- Write-back (`kb_ingest`) and fusion variants (`kb_proposals`) are CUDA-only:
  OpenCL winners are exported by KernelBlaster (`--export-winners`, per target)
  but not yet ingested into an MB OpenCL kernel cache.
- No Android/NDK build; no `cl_mem`/OpenCL 1.2-only path (decision 2).
- fp16 seeds compute in fp32; Adreno's 2× `half` ALU rate is left to the agent
  (technique `3.4_half_precision` of the Adreno flow).
