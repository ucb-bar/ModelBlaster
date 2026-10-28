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

The export, seeds and per-problem agent below were first validated with no
Adreno board reachable, on the local NVIDIA TITAN RTX through its OpenCL 3.0
ICD (OpenCL C 1.2, coarse-grained SVM, no `cl_khr_fp16`) as a functional
stand-in: correctness results carry over, **timings do not**. The model-level
graph optimiser ("Graph optimiser on OpenCL" below) was then run on a real
RB5 (QRB5165, Adreno 650, OpenCL 2.0, Ubuntu arm64 with g++).

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
After the timing the driver replays the whole forward pass on the warm context
(programs built, SVM buffers resident) on its one in-order queue, events not
read, and prints the host wall clock from the first enqueue to `clFinish` as
`MB_E2E_US <us> basis=replay` (median of at least 5 passes, more within
`MB_E2E_MS`, default 200 ms), the kernels' summed event time per pass as
`MB_ATTR_SUM_US ... basis=events launches=<commands per pass>`, and the
difference as `MB_LAUNCH_OVERHEAD_US`: the model-level time as deployed, the
OpenCL counterpart of the CUDA driver's graph replay (see "Graph optimiser on
OpenCL"). `MB_MODE=profile` (or argv[2] `profile` for exec paths without environment
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

## Graph optimiser on OpenCL (`kb_graph_opt --target opencl`)

The model-level flow of [graph_optimizer.md](graph_optimizer.md) -- rounds of profile → propose →
plan → export → KernelBlaster → ingest → decide, to a fixpoint, then `verify_ir` and
`kb_model_run` -- runs unchanged in structure on the OpenCL target. Every timed step runs on the
board:

```bash
python -m modelblaster.pipeline.kb_graph_opt $L2/57_Conv2d_ReLU_HardSwish --work W --kb $KB --python $PY \
    --target opencl --arch adreno650 --board-host root@rb5 --remote-dir /root/mb \
    [--board-lock /root/mb_board.lock] [--board-env NAME=value] \
    --model us.anthropic.claude-haiku-5-5 --rounds 3 --rollouts 1 --steps 4 --variant-steps 2 --max-variants 2
# without --board-host: this host's OpenCL GPU (functional stand-in, --arch names it, e.g. l40s)
```

| step | CUDA | OpenCL |
|---|---|---|
| profile, verification, re-measurement | builtin (default GPU) or KernelBlaster's evaluator | always KernelBlaster's evaluator (`kb_eval.KBEval(evaluator="board")` → `time_problems.py --evaluator board`); builtin is refused |
| KernelBlaster runs | `--evaluator local|ssh` | `--evaluator board --board-host ... --remote-dir ... [--board-lock] [--board-env]` |
| model-level time | `MB_E2E_US basis=graph` (CUDA graph replay, persistent host) | `MB_E2E_US basis=replay` (one driver process per run, the pass replayed on its warm context) |
| kernel cache | `W/cache`, `.cu`, arch `sm_<N>` | `W/cache`, `target: opencl`, `.cpp`, arch = device label (`adreno650`) |
| fusion variants | fused seeds from the best stage kernels | the same, generated for OpenCL (`OpenCLSeeds.fused_seed`) |
| hints | `target: cuda` | `target: opencl` (same rewrites; decided on the board) |
| baseline | the seed model + PyTorch on the same GPU | the seed model on the board; PyTorch `skipped` (none there) |
| GPU model | `NVIDIA ...` from nvidia-smi | the driver's `MB_DEVICE` (`QUALCOMM Adreno(TM) \| OpenCL 2.0 Adreno(TM) 650`), checked on every step as on CUDA |

**The board.** KernelBlaster's `AdrenoEvaluator` copies one `main.cpp` per candidate, builds it
with the board's g++ and runs it, holding the board's lock (`flock`, default `~/mb_board.lock`,
i.e. `/root/mb_board.lock`) for the build and the run of one candidate, so other users of a shared
board wait instead of timing at the same time; the run has a board-side `timeout`, so a cut ssh
session frees the lock. Each evaluator process works in its own subdirectory of `--remote-dir`
(`kb_<pid>_<id>`) and removes only that, because a wave's KernelBlaster run and ModelBlaster's
re-measurements share the remote dir. Results record the device (`gpu`: host, name = `MB_DEVICE`,
arch), so `FlowHardware` pins and checks it as it does a GPU model.

**Ingest.** `--export-winners` is per target; `ingest` takes only the cache's target (`other-target`
otherwise), and the cache's adds and lookups only see its own target's entries -- an OpenCL and a
CUDA winner of the same op, signature and shape never stand in for each other, even in one index.

**Fusion.** A fused slot's OpenCL seed calls its stages back to back, each a complete slot source
(the current best stage kernel from the cache, else the stage's seed) in its own namespace, for
chains of any length; KernelBlaster then optimises the fused slot like any other. The variant's
accepted kernels are cached under `adreno650`.

### Results on the RB5 (Adreno 650; Bedrock Haiku)

`kb_graph_opt --target opencl --arch adreno650 --board-host root@<rb5> --remote-dir /root/mb_oclflow/<name>
--board-lock /root/mb_board.lock --model us.anthropic.claude-haiku-5-5 --rounds 3 --rollouts 1 --steps 4
--variant-steps 2 --max-variants 2 --max-fix 1 --top-k 3 --remeasure 3 --profile-runs 1`, one problem per
work dir, two runs at a time sharing the board through its lock (2026-10-28). Model time = the whole model
composed from the final cache and hints, checked against the reference, `MB_E2E_US basis=replay`; seed =
the hand-seeded model timed by the same evaluator in the same rounds. Single LLM runs.

| IR (KernelBench) | seed (ms) | optimised (ms) | speedup | rounds | hints decided (re-measured gain) | ingested OpenCL winners (speedup vs seed) | LLM calls | cost |
|---|---|---|---|---|---|---|---|---|
| L2 `57_Conv2d_ReLU_HardSwish` | 585.9 | 13.1 | 44.7× | 3 | `fused__ewise_scalar__ewise_scalar` accepted (+42 %); `fused__ewise1__mul__b` rejected (+1.8 %), `fused__conv2d__relu` rejected (+1.8 %), `fused__ewise_scalar__ewise_scalar__ewise1` rejected (−57 %) | conv2d 57×, the accepted fused slot | 31 | $0.76 |
| L2 `73_Conv2d_BatchNorm_Scaling` | 843.3 | 25.8 | 32.7× | 3 | `fused__conv2d__batchnorm2d` rejected (−47 %); `fused__batchnorm2d__mul_scalar` accepted (+25 %) | conv2d 23×, batchnorm2d 8×, the accepted fused slot | 17 | $0.38 |
| L2 `12_Gemm_Multiply_LeakyReLU` | 3825.9 | 36.3 | 105.5× | 2 (fixpoint) | none: the GEMM is 98.6 % of the time, no fusion clears the 2 % bar | linear 108× | 4 | $0.10 |
| L2 `25_Conv2d_Min_Tanh_Tanh` | 2420.7 | 179.9 | 13.5× | 3 (fixpoint) | none proposed (the conv stays dominant) | conv2d 14×, min_dim 11× | 8 | $0.22 |
| L3 `4_LeNet5` | 4348.8 | 90.8 | 47.9× | 3 | `split:conv2d` (KB, two conv shapes) rejected (−0.1 %); `fused__conv2d__relu` rejected (+1.7 %) | conv2d 44×, maxpool2d 257×, two linear shapes 11× / 25× | 20 | $0.65 |

Before the board, the same flow ran with a scripted `stub:` LLM on an NVIDIA L40S through its OpenCL
ICD (firesim2, GPU 2 only, its KernelBlaster GPU lock as the board lock, `--board-env
CUDA_VISIBLE_DEVICES=2`) and on the board: rounds, waves, a fusion chain grown to seven stages,
OpenCL winners and fused kernels in the cache, verification and the model run. Unit tests:
`pipeline/tests/test_kb_graph_opt_opencl.py` (12, no device) and KernelBlaster
`tests/test_multislot_opencl.py` (board lock, run env, per-process remote dir, device record,
`time_problems` on OpenCL; `test_board_evaluator` also ran against the RB5).

Total 80 LLM calls, $2.12 (Haiku at $1 / $5 per M tokens, from the runs' token counts). Every final model
passed its checks (outputs and every op in isolation) on the board, and every step recorded the same
device. The seeds are correctness seeds (one work-item per output, compensated sums, 64-bit index
arithmetic), so most of the speedup is the first kernel the agent writes; the graph-level decisions
come in once the dominant kernel is fast: e.g. on L2 57 the elementwise pair became worth fusing only
after the conv went from 581 ms to 12 ms; on L2 73 conv→bn lost by 47 % in round 0 (same KB budget on
both sides, two steps) while bn→scale won in round 2. Timings are single measurements on a board another
user's processes keep at load ~2; the re-measurement interleaves the two sides of each decision.

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
| 11 | Model-level time = the whole pass replayed in the one-shot driver on its warm context and in-order queue, host wall clock first enqueue → `clFinish`, median (`MB_E2E_US basis=replay`) | a persistent OpenCL host (`-DMB_HOST`, one `.so` per candidate, process kept across candidates); the sum of cl_event spans; a queue of N passes divided by N | It is what one inference costs on the board, enqueue and gaps included, which event spans hide; context, programs and SVM buffers are already resident in the driver, so a persistent host would only save process start-up and the `data.bin` read -- small next to the board's g++ build per candidate, which it would not save (the `.so` is built on the board too). Per-pass sync measures latency, not a pipelined throughput no deployment of one model sees. |
| 12 | One kernel-cache class per slot target (`CudaCache(target=)`): entries carry `target`, `.cpp` sources, arch = device label; ingest and lookups see only their target | a separate OpenCL cache format/dir; arch strings like `opencl:adreno650` | One index and one `cache_plan` (coverage splits, stage seeds) for both; CUDA entries are written exactly as before (no `target` field), and the arch already separates them; the explicit target makes the separation hold even when arch labels collide. |
| 13 | Fused OpenCL ops get generated seeds: the stages' current-best kernels back to back, each in its own namespace, any chain length | refuse fused variants on OpenCL at first; hand-written fused seeds | Fusion decisions must start from the unfused optimum (graph-optimiser decision 6) or they measure optimisation progress; namespacing whole slot sources is target-neutral, so the same mechanism as CUDA's `cuda_seed` works without per-op code. |
| 14 | Baseline on the board = the seed model, timed by the same driver right after the composed model in each round; PyTorch `skipped` with the reason | `pip install --user` a CPU torch (Python 3.8, aarch64) and time it on the A77 cores; none | The board has no torch; a CPU-eager number would measure another device under the board's background load (load average ~2 from other users), and installing a ~400 MB unsupported-Python torch into a shared root system is not cheap. The seed is the same device, harness and data, so the reported speedup is exactly what the flow bought. |
| 15 | Every timed OpenCL step goes through KernelBlaster's evaluator (`time_problems.py --evaluator board`); ModelBlaster's builtin runs are refused | build and run OpenCL locally in ModelBlaster (`kb_export_opencl.build_and_run`) | The board is the only device whose numbers count; one evaluator for the KernelBlaster runs and ModelBlaster's timings keeps them on the same device and under the same lock. |
| 16 | The board's lock is taken by the evaluator around each candidate's build + run, with a board-side `timeout`; one remote subdirectory per evaluator process | lock the whole KernelBlaster run; one shared remote dir | Short holds let another user's runs interleave; the timeout frees the lock when ssh drops; a wave's KernelBlaster run and the re-measurements run concurrently, and a shared dir let one process's `close()` delete the other's `data.bin`. |
| 17 | Hints produced by an OpenCL flow are `target: opencl`, applied only on OpenCL (and CUDA hints only on CUDA) | one target-less hint per pattern | A fusion is decided by measurement on one device; Adreno and NVIDIA disagree on launch cost and bandwidth, so a decision must not travel. The rewrite mechanics are shared. |
| 18 | fp16 / tolerances follow the existing OpenCL contract; bf16 and mixed-dtype partitions stay refused | extend them for the flow | Out of scope; the export already refuses them with a reason. |

## Limitations / what remains

- **Board limits**: the Adreno compiler's kernel work-group limits, SVM allocation limits for the
  largest Level3 problems on 8 GB shared memory, and the 3.8 GiB `/tmp` tmpfs (use `--remote-dir`
  on persistent storage) still bound which models fit; only small Level2/3 models were run.
- The per-problem live agent run in "Validation" is NVIDIA's; the board results are under
  "Graph optimiser on OpenCL". Technique values (bandit statistics, claims) are
  re-learned on the board; records are keyed by arch so the stand-in's do not leak into Adreno's.
- The graph optimiser's OpenCL runs are single LLM runs per problem on one board; decisions near
  the 3 % bar are as noisy as on CUDA. Only small models were run; the flow's resume and parallel-wave
  machinery are the CUDA code paths and were exercised on OpenCL only through these runs.
- No persistent OpenCL host (decision 11): each candidate is one process and one g++ build on the
  board (about 1.5 s for a Level2 problem); re-measurements rebuild per repetition.
- No Android/NDK build; no `cl_mem`/OpenCL 1.2-only path (decision 2).
- fp16 seeds compute in fp32; Adreno's 2× `half` ALU rate is left to the agent
  (technique `3.4_half_precision` of the Adreno flow).
