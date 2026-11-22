# GPU harness and multi-slot driver: performance (CUDA)

This covers the CUDA side of the ModelBlaster ↔ KernelBlaster loop: the generated multi-slot
driver (`pipeline/kb_export.py`, `_DRIVER_TEMPLATE`), the hand seed kernels
(`gpu_eval/harness/cuda/mb_cuda_kernels.cu` and the generated `_f16` file), and the
KernelBlaster evaluators that run the driver (`knowledge/slots.py`). It characterises where
time goes, answers four questions (launch overhead, CUDA graphs, who picks grid/block, the
torch header), and records what was changed and measured.

Hardware: the local TITAN RTX (sm_75, 72 SMs, a shared desktop GPU) and one idle L40S on
firesim2 (sm_89, 142 SMs, GPU 2). The L40S numbers are the reference: the TITAN's clocks drift
with temperature, and from about 10:00 on the day of these runs another user's job held it at
100 %. Tools: nsys 2024.5 (`--trace=cuda,nvtx --cuda-graph-trace=node`, driver built with
`-DMB_NVTX` for phase ranges), the driver's own `MB_PHASE_MS` output, and wall clock. ncu is not
needed for any of this.

Problem set, all whole-graph problems exported with `kb_export`:

- Level1: `19_ReLU`, `1_Square_matrix_multiplication_`, `23_Softmax`, `47_Sum_reduction_over_a_dimension`;
- Level2: `1_Conv2D_ReLU_BiasAdd`, `12_Gemm_Multiply_LeakyReLU`, `99_Matmul_GELU_Softmax`;
- Level3: `9_ResNet18` (68 ops, 7 slots), `30_SwinTransformerV2` (339 ops, 13 slots, 2.3 GB of
  reference data), `44_MiniGPTBlock` (29 ops), `35_LSTM` (10 ops, 6170 kernel launches).

Each problem was run with its hand seed kernels. Four were also run with optimised KB winners:
ResNet18 (`kb_live/run_r18`, 1.9 ms on sm_75), MiniGPT (`kb_haiku`, 9.5 ms), and the Level2
conv and gemm problems (`knowledge_v2/A3_sm75`, `A2_sm75`).

## Answers

**How much kernel launch overhead is there?**

- **Seeds: none worth measuring.** Seed kernels run 0.3–1000 ms per forward pass, and the GPU
  is 99–100 % busy inside a forward pass. CUDA graphs change the partition time by less than
  0.5 % on the L40S.
- **Optimised winners: 1–8 %, more with many tiny kernels.**
  - CPU: one launch costs about 2.3 µs in `cudaLaunchKernel` and about 3.2 µs of issue time.
  - GPU: each kernel boundary leaves about 0.7 µs idle (0.05 µs under a graph).
  - ResNet18 winner on the L40S: 87 kernels per forward pass, 76 % of them under 5 µs (16 % of
    kernel time). The GPU is 91 % busy and idles 65 µs per forward pass. A graph replay takes
    7.6 % less time (740 → 684 µs).
  - Level2 winners save about 2 %, the MiniGPT winner 0.7 %.
  - A single 2 µs kernel (Level1 ReLU) saves 17–20 % (5.8 → 4.7 µs).
- **LSTM.** The LSTM launcher loops over timesteps on the host: 6170 launches per forward pass,
  half of them under 5 µs. With the seeds that is hidden behind 650 ms of GPU work. Once the
  kernels are optimised, 6170 launches × about 3 µs puts an 18 ms CPU floor on the forward
  pass. It is the clearest launch-bound and graph or persistent-kernel candidate in the set.
- **Per-op timing, a side finding.** The per-op timing (events around each op, the source of
  `MB_SLOT_US` and so of the KB reward and slot shares) charges each op about 2–3 µs of event
  and launch gap. Against nsys kernel durations on the L40S ResNet18 winner:

  | slot | `MB_SLOT_US` | nsys kernel time |
  |---|---|---|
  | relu | 73 | 26 |
  | batchnorm2d | 92 | 33 |
  | add | 36 | 13 |
  | conv2d | 621 | 543 |
  | linear | 65 | 62 |

  So tiny slots look 2–3× bigger than they are (recommendation 2).

**How effectively are we using CUDA graphs?** Not at all before this change. The driver now has
`MB_GRAPH=1`:

- it captures the partition once, checks one replay against the reference, and times graph
  replays interleaved rep-by-rep with eager reps;
- it needs the kernels built with `--default-stream per-thread`, because the legacy default
  stream cannot be captured. Otherwise, or when a slot is not capturable (a host sync, an
  allocation, a copy to the host), it prints `MB_GRAPH_SKIP <reason>`.

Measured gains are the numbers above: up to 7.6 % on the most launch-bound winner, 0–2 %
elsewhere. Run-to-run noise is no better with graphs: on the idle L40S both have a CV of about
0.2 %, and on the TITAN both are 1–13 %. The reward stays eager (see "CUDA graphs").

**Who decides grid/block size?**

- **Seeds: the hand kernels.** Every seed is one thread per output element, launched as
  `<<<MB_GRID(n), MB_THREADS>>>` with `MB_THREADS` fixed at 256. Reductions use their own fixed
  grids (`mb_rblocks`, one block per row). Since this change the seeds use
  `<<<mb_blocks(n), mb_threads(n)>>>`: 256 threads, halved (down to 32) while a 256-thread
  launch would leave an SM without a block.
- **LLM-written slots: the LLM.** A rewritten slot's launcher picks its own configuration;
  KernelBlaster's `launch_configuration_tuning` technique asks for exactly that. Nothing tunes
  it without an LLM call.
- **Neither the driver nor KB** chooses or sweeps a launch configuration.

**Why would we still compile the torch header?**

- **The multi-slot path never includes it.** MB's driver is torch-free C++, and KB's fast
  compile path (`servers/cuda_fast.py`, `needs_torch`) builds and links it plain, without
  LibTorch.
- **It is compiled only on the legacy CMake path**, when `KERNELBLASTER_FAST_COMPILE=0` or the
  toolchain cannot be derived. That path forces `torch/torch.h` into the host TU and links
  LibTorch, even for MB drivers: about 9 s more per driver object and a LibTorch-linked binary.
- **Native KernelBench drivers** (`data/kernelbench-cuda/*/driver.cpp`) include it themselves
  and pay about 9.4 s per driver object on any path.
- **Inside MB's harness, only `gpu_eval/harness/kb_bridge.py`** (the H4 gate) writes LibTorch
  drivers: per-op-type KB problems in KernelBench's native format. `walk_cuda.py` loads a plain
  nvcc-built `.so` through ctypes and uses PyTorch only on the Python side, for buffers and
  references. Retiring H4 in favour of `kb_export --strategy per_op` would remove LibTorch from
  the MB harness altogether.

Measured on this box (CPU-bound, shared host, so ±20 %):

| build | native `019_ReLU` (LibTorch driver) | MB ResNet18 problem | MB LeNet fixture |
|---|---|---|---|
| server, legacy CMake: cold (configure + make) / make again | 42.7 s / 15.4 s, LibTorch | 42.9 s / 13.1 s, **LibTorch** | 38.0 s / 8.3 s, **LibTorch** |
| server, fast path: first build / new kernel only | 9.8 s / 1.6 s, LibTorch | **1.8 s / 1.6 s, no LibTorch** | **1.3 s / 1.3 s, no LibTorch** |
| driver object, plain / with `torch/torch.h` forced in | 9.4 s / 9.3 s (includes it itself) | **0.9 s** / 8.7 s | **0.3 s** / 8.4 s |
| local `nvcc` one-shot (`build_and_run`) | – | 4.6 s, no LibTorch | 6.6 s, no LibTorch |

The fast path's one-time toolchain configure is 8.2 s per server and SM version. Locally, with
plain `nvcc`, the assembled kernel object takes 1.6 s and the driver object 0.6–1.4 s. The
`.incbin`'d data dominates the driver object and the 2–3.5 s link. `LocalEvaluator` builds
slots as separate, cached units, so a candidate recompiles one slot. The only fix needed is to
keep `KERNELBLASTER_FAST_COMPILE` on.

## Characterisation

### Inside the timed forward pass (L40S)

Statistics over the eager timing reps of the full-mode driver, from nsys:

- "busy" is the union of kernel intervals over the window from the first kernel start to the
  last kernel end;
- "idle" is the rest of that window;
- "PTDS eager / graph" are the interleaved eager and graph times of the
  `--default-stream per-thread` build under nsys. nsys inflates `cudaGraphLaunch` to about
  100 µs, so the graph savings are taken from runs without nsys (next table).

| problem | eager µs | kernels / fwd | < 5 µs kernels: % of kernels / % of time | median kernel µs | GPU busy % | idle µs / fwd |
|---|---|---|---|---|---|---|
| seed L1 relu | 7.0 | 1 | 100 / 100 | 2.2 | 100 | 0 |
| seed L1 matmul | 482 | 1 | 0 / 0 | 477 | 100 | 0 |
| seed L1 softmax | 33 435 | 1 | 0 / 0 | 33 428 | 100 | 0 |
| seed L1 sum | 308 | 1 | 0 / 0 | 303 | 100 | 0 |
| seed L2 conv_relu | 3 792 | 3 | 0 / 0 | 16.8 | 100 | 1.4 |
| seed L2 gemm | 7 818 | 3 | 45 / 0.06 | 7.4 | 100 | 1.6 |
| seed L2 mm_gelu_softmax | 23 650 | 3 | 0 / 0 | 5 947 | 100 | 2.2 |
| seed L3 ResNet18 | 13 887 | 68 | 65 / 0.6 | 2.2 | 99.7 | 47 |
| seed L3 MiniGPT | 53 923 | 29 | 0 / 0 | 31.5 | 100 | 20 |
| seed L3 SwinV2 | 167 414 | 348 | 19 / 0.1 | 13.9 | 99.8 | 253 |
| seed L3 LSTM | 689 278 | 6170 | 50 / 0.5 | 1.2 | 99.3 | 4 923 |
| winner L2 conv_relu | 97.5 | 3 | 0 / 0 | 6.0 | 98.5 | 1.4 |
| winner L2 gemm | 153 | 4 | 75 / 6.5 | 3.4 | 98.6 | 2.1 |
| winner L3 MiniGPT | 2 852 | 29 | 0 / 0 | 30.2 | 99.3 | 21 |
| winner L3 ResNet18 | 753 | 87 | 76 / 15.7 | 1.5 | 91.3 | 65 |

The TITAN shows the same picture (table in the scratch logs). There the ResNet18 winner is
96.5 % busy and idles 59 µs per forward pass.

Common to every problem:

- **CPU launch cost.** About 2.3 µs per `cudaLaunchKernel`, 3.2–4 µs per launch of issue
  interval on the L40S (2.5–2.8 µs on the TITAN).
- **Timed loop.** No memcpy, memset or allocation inside the timed reps. The lazily allocated
  reduction scratch (`mb_partials`) is allocated on the first, checked run.
- **Sync points.** One `cudaEventSynchronize` per rep, by design: each rep is a latency
  measurement from an idle GPU. The checks synchronise once per op (`ok_cuda`), plus the
  blocking copies.

### Eager vs CUDA graph, without nsys (L40S, 7 runs each, interleaved reps)

| problem | eager µs (CV %) | graph µs (CV %) | graph saving |
|---|---|---|---|
| winner L3 ResNet18 (87 kernels) | 740.4 (0.20) | 684.2 (0.17) | **7.6 %** |
| winner L2 gemm (4) | 154.4 (0.22) | 150.8 (0.27) | 2.3 % |
| winner L2 conv_relu (3) | 96.4 (0.38) | 94.3 (0.32) | 2.1 % |
| winner L3 MiniGPT (29) | 2454.6 (0.14) | 2437.9 (0.14) | 0.7 % |
| seed L3 ResNet18 | 13 879 (4.3) | 13 840 (4.4) | 0.3 % |
| seed L2 gemm | 7 484 (7.7) | 7 453 (7.9) | 0.4 % |
| seed L1 relu (1 kernel) | 5.81 (9.4) | 4.67 (3.7) | 19.6 % |

On the TITAN, with interleaving, the ResNet18, MiniGPT and Level2 winners are within ±1 % and the
L1 ReLU kernel saves 13 %. Before interleaving, graph timing ran after the eager reps and
looked up to 13 % *slower*: that was the GPU warming up, not the graph.

### Where a driver run's wall time goes

Phases from `MB_PHASE_MS`, the same seed kernels, old driver against new, L40S:

| SwinV2 (339 ops, 2.3 GB data) | old driver | new, full | new, `MB_ONLY_SLOTS=linear` | new, + `MB_SCREEN=1` | new, `MB_ONLY_SLOTS=gelu` + `MB_SCREEN=1` |
|---|---|---|---|---|---|
| data load (`fread` → `mmap`) + CUDA init | 1 553 + ctx | 0 + 188 | 0 + 194 | 0 + 183 | 0 + 181 |
| buffer alloc | 140 | 29 | 29 | 26 | 26 |
| partition check | 203 | 213 | 213 | 208 | 209 |
| isolated-op checks | **17 373** | 1 929 | 636 | 627 | 206 |
| partition timing | 805 | 813 | 824 | 354 | 354 |
| per-op timing | 473 | 473 | 473 | 158 | 158 |
| **wall time** | **20.8 s** (25.6 s in an earlier run) | **3.8 s** | 2.5 s | 1.7 s | **1.2 s** |

| problem (L40S, s) | old | new, full | new, step run (`MB_ONLY_SLOTS=<slot> MB_SCREEN=1`) |
|---|---|---|---|
| SwinV2 | 20.8 | 3.8 | 1.2–1.7 |
| ResNet18 | 1.70 | 0.87 | 0.41–0.43 |
| MiniGPT | 3.06 | 1.27 | 0.54–0.61 |
| LSTM | 7.03 | 6.96 | 3.59 |
| L2 gemm | 0.71 | 0.65 | 0.35 |

LSTM's seeds run 650 ms per forward pass and the driver runs it 5–10 times, so only the
screening budget helps there.

On the TITAN, measured before the other user's job started, the old driver took 19.6 s on
SwinV2, 9.9 s of it in the isolated checks. An intermediate build of the new driver took 10.9 s.

Where the old driver's isolated checks went:

- **Re-uploads.** Before every op, `upload_inputs()` re-uploaded every graph input, constant
  and weight. On SwinV2 that is 176 MB × 339 ops, 59.5 GB of H2D per run; on ResNet18
  48 MB × 68 ops, 3.3 GB.
- **CPU scans.** Single-threaded `max_finite`/`max_err` scans re-read every input reference per
  op.

The remaining floor per run is about 185 ms of CUDA context creation plus about 80 ms of
process start and exit (occasionally 1+ s on the loaded dev box). Both are per process.

## What changed

### Driver (`pipeline/kb_export.py`)

The default run checks and times what it did before. Inside it:

- **Isolated checks reset only the op's own inputs**, weights included, to the reference before
  running it, instead of every input, constant and weight of the partition. The check is the
  same: every input of the op holds reference values.
- **Faster comparisons.** `max|finite ref|` is cached per buffer; f32 references are compared
  in place, without a copy; device values come back through one pinned staging buffer; the
  scans are split across up to 8 threads. Max is order-independent, so the results are
  identical.
- **Big data files are mmapped** instead of `fread` (the > 1 GiB `MB_DATA_PATH` case). Such a
  driver is not self-contained, but it is not local-only either. KernelBlaster's
  `SSHEvaluator` copies `data.bin` to each GPU host once per data sha1, before leasing a GPU,
  and sets `MB_DATA_PATH` for the run or the persistent host. On SwinV2 (2.27 GB) to firesim2,
  the copy takes 23.5 s once, a step candidate on the host 3.9 s, and the timings match one
  process per run (KernelBlaster `docs/fast_paths.md`).
- **Timing events are recorded on `cudaStreamLegacy`.** With legacy-stream kernels this is
  stream 0, as before. In a per-thread-default-stream build the event still waits for every
  blocking stream, so work a slot puts on its own streams stays inside the measurement.
- **New output:** `MB_PARTITION_DIST min= med= max=` (spread of the timing reps) and
  `MB_PHASE_MS <phase> <ms>` (wall time per phase, plus `total`). `-DMB_NVTX` adds NVTX ranges
  per phase, for nsys.

New run modes, set by environment:

| variable | effect |
|---|---|
| `MB_ONLY_SLOTS=a,b` | Isolated-op checks only for these slots' ops. The partition outputs are always checked; the partition and every op are always timed. Per-op timing always runs every op in order, because timing a subset changes what is in cache: on the L40S a subset run measured MiniGPT's layer_norm at 0.52× its in-place time. An unknown slot fails the run. |
| `MB_SCREEN=1` | Screening budgets: 20 ms of partition timing, 1 warm-up, at least 1 rep; per-op passes until they add up to 20 ms (1–5). |
| `MB_TIME_MS`, `MB_OP_REPS` | Override the partition budget / per-op rep cap (as before / new). |
| `MB_GRAPH=1` | Also capture the partition into a CUDA graph (see below). |

Every mode keeps the correctness gates. A slot broken outside `MB_ONLY_SLOTS` still fails the
run on the partition output check (`test_gates_kept_with_only_slots`).

Timing agreement with a full run, L40S, median ratio of `MB_SLOT_US` over 4 paired rounds:

| | ResNet18 winner (conv2d, bn, relu, add, linear) | MiniGPT winner (linear, bmm, softmax, layer_norm, add) |
|---|---|---|
| full run again (noise floor) | 0.99–1.01 | 1.00–1.00 |
| `MB_ONLY_SLOTS=<slot>` | 0.99–1.02 | 1.00–1.01 |
| `MB_ONLY_SLOTS=<slot>` + `MB_SCREEN=1` | 0.97–1.00 | 0.99–1.00 |
| `MB_SCREEN=1` | 1.00–1.02 | 0.99–1.02 |

`build_and_run(..., env=, graph=)` and `run_built(..., env=)` pass driver variables, and
`--graph` builds with `PTDS_FLAGS`. `parse_driver_output` adds `partition_graph_us`,
`graph_skip`, `partition_dist`, `partition_graph_dist` and `phase_ms`.

### KernelBlaster (`knowledge/slots.py`, `agents/opt_multislot.py`; branch `feat/harness-perf`)

- `Evaluator.evaluate(problem, sources, tag, *, only_slots=None, screen=False)`. `driver_env()`
  maps the options to `MB_ONLY_SLOTS` / `MB_SCREEN`.
  - `LocalEvaluator` and `SSHEvaluator` pass them as the run's environment.
  - `ServerEvaluator` passes them as `prefix_command` of `run_gpu_executable`.
  - The OpenCL evaluators accept and ignore them.
  - Drivers exported before this change ignore the variables and run in full.
- The agent's step evaluates a candidate with `step_eval_options(sid)`: `only_slots=[sid]` and
  `screen=True`. `KERNELBLASTER_MULTISLOT_ONLY_SLOTS=0` and `KERNELBLASTER_MULTISLOT_SCREEN=0`
  turn these off. The baseline, store-seed, confirm and final-composite runs stay full, so a
  > 3 % win is still re-measured with every check before it counts.
- `LocalEvaluator(graph=True)` or `KERNELBLASTER_MULTISLOT_GRAPH=1` builds with
  `--default-stream per-thread` and runs with `MB_GRAPH=1`; `EvalResult.partition_graph_us`
  carries the graph time. The server compile path keeps its CMake flags, so there the driver
  reports `MB_GRAPH_SKIP`.
- The `mb_lenet` and `mb_lenet_f16` fixture drivers were regenerated with the new template.

### CUDA graphs (`MB_GRAPH=1`)

1. After the checks, the partition is captured once on the per-thread default stream
   (`cudaStreamCaptureModeThreadLocal`), instantiated, and replayed once on zeroed
   intermediates and outputs. The replay's outputs must pass the same bound as the eager run.
2. Graph and eager reps are then timed alternately, so both see the same clocks and caches:
   `MB_PARTITION_GRAPH_US`, `MB_PARTITION_GRAPH_DIST`, `nodes=`.
3. Fallbacks print `MB_GRAPH_SKIP <reason>` and leave the eager numbers and the pass/fail
   unchanged:
   - the kernels are on the legacy stream ("operation failed due to a previous error during
     capture", or an empty capture);
   - a launcher synchronises, allocates, frees or copies to the host
     (`test_graph_fallback_on_host_sync`);
   - the replay does not reproduce the outputs, which catches host-side state and work left
     out of the capture.

**Decision (superseded): the KB reward stayed eager here.** Graph replay could not attribute
time to slots, and per-slot event times were the reward. The persistent runtime below makes
the whole-partition graph replay the reward, with slot attribution from CUPTI. The eager time
and the eager − graph gap stay as diagnostics.

### Seed launch configuration (`gpu_eval/harness/cuda/mb_cuda_kernels.cu`)

A block-size sweep of the seeds on the L40S showed the following.

**Fixed `MB_THREADS` from 64 to 1024.**

- **Partitions:** 0.70–1.10× of the 256 time, with no single best value. 1024 is up to 4×
  slower (Level1 sum).
- **Row-parallel slots:** smaller blocks win, and win big:

  | slot | best block | speedup |
  |---|---|---|
  | softmax over 128 rows | 64 | 3.4× |
  | layer_norm | 64 | 2.8× |
  | ResNet18's fc GEMV | 64 | 2.1× |
  | sum_dim | 64 | 1.4× |

  These slots have few outputs, so 256-thread blocks leave most of the 142 SMs empty.

**What was implemented: `mb_threads(n)`.**

- 256 threads, halved (down to 32) while `256 × SMs > n`; the SM count is read once.
- `mb_blocks(n)` is `MB_GRID` with that block size. Every seed launch uses the pair; the f16
  file was regenerated with `f16_kernels.py` and the 5 hand f16 launches were changed.
- Each thread does the same work as before, so outputs are unchanged.
- It is a rule, not a sweep: no extra build or run, and the same on every GPU apart from the
  SM count.
- `MB_GRID` and `MB_THREADS` keep their meaning for LLM-written slots.

**Interleaved A/B on the L40S, 4 rounds each:**

| problem | partition before → after | slots that changed |
|---|---|---|
| L1 sum | 305 → 207 µs (1.48×) | sum_dim 1.48× |
| L2 mm_gelu_softmax | 9.51 → 7.64 ms (1.25×) | softmax 3.73×, gelu 1.16× |
| L3 MiniGPT | 62.3 → 59.8 ms (1.05×) | layer_norm 3.72×, softmax 1.18× |
| L3 ResNet18 | 14.10 → 13.91 ms (1.01×) | linear 2.17×, adaptive_avg_pool2d 1.28× |
| L2 conv_relu, L2 gemm, L1 relu, L1 softmax | ±2–5 % (noise) | – |

No slot got slower beyond noise, so the rule is adopted.

**Not adopted:**

- **A driver-side block-size sweep.** One-thread-per-output seeds would need one build per
  block size, and the rule already captures the measured gain.
- **A tunable launch parameter for LLM slots.** Their kernels' internal tiling is tied to their
  block size, so such a parameter is not generic.

**Effect on recorded numbers.** Seed baselines get faster for row-parallel ops, so the
"speedup vs seed" of later KB runs for those ops will be lower than in earlier runs.

## Tests and gates

- **MB:** `pipeline/tests` passes: 179 passed, 6 skipped. New:
  `pipeline/tests/test_kb_driver_modes.py`, which covers the parser, full run and phases,
  `MB_ONLY_SLOTS`, unknown slot, screening, gates kept with a broken unscreened slot, graph
  capture, the legacy-stream fallback and the host-sync fallback. The worktree has to be run
  from a directory named `modelblaster`, because the end-to-end tests extract through
  `python -m modelblaster...`. The known `benchmarks/tests/test_profile_db.py::test_hetero_runs_are_skipped`
  failure is outside `pipeline/tests`.
- **KB:** the suite passes, xilinx stripped from `PATH`: 198 passed, 1 skipped (193 + 5 new
  in `tests/test_driver_modes.py`: options, parsing, server prefix, local
  only/screen/broken/graph, agent step options).
- **Export gate** with the new driver and seeds (TITAN, full mode, `MB_TIME_MS=50`): every
  IR exported as one whole-graph problem with `kb_export --build-check` (build, partition
  check, every op isolated).
  - fp32: Level1 100/100, Level2 100/100, Level3 50/50.
  - fp16 Level3: 49/50. The miss is `48_Mamba2ReturnY`, the known fp16-range overflow
    (`fp16_cuda.md`).
- **H3 hand-CUDA walk** (`walk_cuda.py`, golden plus every intermediate isolated) on Level3
  with the new seeds: 49/50. The miss is `26_ShuffleNet`, out of memory on the shared 24 GB
  GPU, as before.

## Persistent runtime and the end-to-end reward

This section covers the CUDA multi-slot flow on the `feat/persistent-runtime` branches:

- the persistent runtime;
- the end-to-end reward;
- the whole-model result.

It implements recommendations 1–3 of the list below.

### What the deployed runtime is, and what is timed

ModelBlaster's deployed GPU runtime for a partition (or a whole model):

- preallocated buffers;
- weights and inputs resident on the device;
- one stream, no host syncs or allocations between ops;
- the forward pass captured once as a CUDA graph and replayed.

The driver's timed path is exactly that:

- `MB_E2E_US` is the median graph replay of the whole partition: `basis=graph`, captured on
  the per-thread default stream and checked against the reference once.
- It falls back to the eager launch sequence (`basis=eager`) with `MB_GRAPH_SKIP <reason>`
  when a slot syncs, allocates, copies to the host or launches on the legacy stream.
- The eager time (`MB_PARTITION_US`) and `eager − graph` stay as diagnostics.

### The persistent host (`driver.cpp -DMB_HOST`)

**What it is.**

- The same `driver.cpp` built with `-DMB_HOST` becomes a long-lived process. It keeps the
  CUDA context, the device buffers and the mmapped/linked reference data.
- Each request names a candidate's **shared object**: the slot units + `walker_tail.cu` built
  `-shared -fPIC --default-stream per-thread`. Its C entry points are `mb_so_abi`,
  `mb_so_run_op`, `mb_so_launch` and `mb_so_num_ops`, which `kb_export` writes into the tail.
- The host dlopens it for the request and dlcloses it after.
- The checks and timing are the one-process driver's own code (`mb_eval()`). A request prints
  exactly what `MB_*=… ./driver` prints, minus the per-process `cuda_init`.
- `driver.cpp` still builds and runs on its own for debugging, as before.

**Protocol.** Stdin/stdout, one tab-separated line per request:
`EVAL <nonce> so=<path> [ref=<path>] [lock=<file>] [lock_wait=<s>] [MB_*=<v> ...]`. The
reply is the driver lines, then `MB_HOST_DONE <nonce> rc= poisoned= recycle=`.

**`ref=`, the paired comparison.** It loads the incumbent's object too, and times candidate
and incumbent alternately, rep by rep, in the same process (`MB_REF_E2E_US`,
`MB_E2E_RATIO … basis=`).

- Both sides are measured the same way: graph against graph when both capture, else eager
  against eager.
- The incumbent's own isolated checks are not repeated (it passed them when it was accepted).

**Failures** never reach the next candidate:

- **Crash.** A candidate that crashes the process ends the stream. The client reports a
  runtime error and starts a new host.
- **Hang.** Killed at the timeout, same outcome.
- **Sticky CUDA error** (illegal address, launch failure). The host probes the context after
  every request. A broken context replies `poisoned=1` and exits without dlclose.
- **Leaks and old hosts.** The host recycles itself after `MB_HOST_MAX_EVALS` (500) requests
  or `MB_HOST_LEAK_MB` (1 GiB) of leaked device memory.
- **Lost client.** The host exits when its stdin closes (a killed client, a dropped ssh
  channel), even with a hung kernel.
- **State between candidates is a fresh process's.** Intermediates and outputs are zeroed and
  the inputs re-uploaded per request.

**GPU lock.** The host takes the GPU's flock (the same file as KernelBlaster's
`gpu_lock_wrap`) for the request only, so an idle host holds memory but never the GPU.

**dlopen vs a process per N candidates.** Measured on the TITAN with a static-cudart object,
200 load / run / graph-capture / dlclose cycles:

- correct results;
- no device memory growth from the modules themselves;
- 0.02 ms per load.

A candidate's own leaks, such as a never-freed static scratch buffer, stay. That is what the
leak recycle is for. So the host dlcloses per candidate and recycles rarely.

**Python side.** `kb_export.build_host` / `build_so` / `DriverHost`;
`build_and_run(..., runtime="host")`; `kb_export --build-check --runtime host`.

**KernelBlaster side.** `LocalEvaluator` and `SSHEvaluator` use the host by default
(`knowledge/runtime_host.py`, `--runtime auto|host|process`). The server evaluator stays one
process per run.

### Per-op attribution without the event inflation (CUPTI)

The driver measures every op's kernels, memsets and copies with CUPTI activity records (the
durations nsys reports):

- ops run back to back as in the partition, with an external correlation id per op and no
  events between ops;
- output `MB_ATTR_SLOT_US <slot> <us> launches=<n>`, `MB_ATTR_OP_US`, `MB_ATTR_SUM_US` and
  `MB_LAUNCH_OVERHEAD_US eager= graph= frac_eager=`;
- libcupti is dlopen'ed at run time. A binary built without CUPTI's headers, or run without
  the library, prints `MB_ATTR_SKIP` and keeps the event times.

The ResNet18 winner on the L40S (µs per forward pass; nsys kernel sums from the earlier
profile):

| slot | `MB_SLOT_US` (events) | `MB_ATTR_SLOT_US` (CUPTI) | nsys | launches |
|---|---|---|---|---|
| conv2d | 612 | 540.4 | 541 | 39 |
| batchnorm2d | 95 | 34.3 | 32.6 | 20 |
| relu | 79 | 26.7 | 26.6 | 17 |
| add | 36 | 12.6 | 12.6 | 8 |
| linear | 64.8 | 61.9 | 61.8 | 1 |
| maxpool2d | 8.8 | 5.7 | 5.6 | 1 |
| adaptive_avg_pool2d | 5.9 | 3.1 | 3.1 | 1 |
| **sum** | 902 | **684.7** | | 87 |

- The CUPTI sum equals the graph replay (684 µs): under a graph the partition has no launch
  overhead left.
- Eager, it has 53 µs (7.2 %).
- **CUPTI must be detached after each pass.** Left attached, it added about 1 % to every
  later graph replay (686.7 → 694 µs), which the host's next request would have timed. The
  driver calls `cuptiFinalize` after the pass and re-attaches next time (686.7 → 687.5 µs,
  noise).

### Measurements

**Per-candidate evaluation wall time.**

- Setup: idle L40S (firesim2 GPU 2) through `--evaluator ssh`; builds on the dev box; one
  slot changed per candidate. Medians in seconds.
- "Step" is the agent's step run: `MB_ONLY_SLOTS` + `MB_SCREEN`, and on the host paired with
  the incumbent. "Full" is the confirm run, with every check.
- Process = one driver process per run, as before. Process-graph = the same, built and run
  for the graph time too.
- Tool: `scripts/bench_runtime.py` (KernelBlaster).

| problem | host step (of it: request) | process step | process-graph step | host full | process full |
|---|---|---|---|---|---|
| L2 `12_Gemm_Multiply_LeakyReLU`, winner | **1.62** (0.12) | 3.55 | 3.51 | 1.60 | 3.56 |
| L3 ResNet18, winner (68 ops, 87 launches) | **1.61** (0.14) | 4.03 | 4.02 | 2.09 | 4.13 |
| L3 MiniGPT, winner (29 ops) | **1.73** (0.18) | 6.25 | 8.08 | 2.50 | 8.85 |
| L3 LSTM, seeds (6170 launches, 220 ms forward) | **6.23** (2.57) | 13.05 | 8.14 | 14.05 | 10.16 |

What changed in each step:

- The host step is a slot compile (about 1 s), a `-shared` link of small objects, an scp of
  about 1 MB, and the request.
- The process step relinks the binary with the incbin'd data (110–480 MB), rsyncs it and pays
  CUDA init.
- 60 host requests needed 4 host starts, one per problem: 0.9 s each, paid once.
- Under a loaded dev box (a second measurement, while the test suites ran) the ratios held:
  host 1.7–5.1 s against 5.9–14 s.
- The LSTM's full run is slower on the host. A paired run times two 220 ms forward passes per
  rep. For GPU-bound problems that long, pairing costs GPU time while the noise it removes is
  small.
- On the shared TITAN at the same time, the host's runs waited on the GPU lock behind another
  agent's runs. Those numbers say nothing about the runtime.

**Timing agreement**, L40S, 5 full evaluations each, interleaved: median µs (CV %).

| problem | host graph (`MB_E2E_US`) | process-graph graph | host eager | process eager | CUPTI kernel sum |
|---|---|---|---|---|---|
| L2 gemm winner | 134.1 (0.0) | 134.1 (0.3) | 136.6 | 135.6 | 134.8 |
| ResNet18 winner | 686.1 (0.06) | 684.0 (0.0) | 741.2 | 736.4 | 688.0 |
| MiniGPT winner | 2508.8 (0.03) | 2505.7 (0.07) | 2522.1 | 2517.8 | 2501.6 |
| LSTM seeds | 218 838 (0.0) | 218 808 (0.01) | 222 836 | 222 642 | 219 286 |

- The host and the one-process driver agree within 0.3 %. A rerun of the same evaluations
  differs by up to 0.4 %.
- The paired ratio of a candidate against itself was 0.999–1.002 on every problem.

### The end-to-end reward in KernelBlaster

KernelBlaster's `docs/multislot_optimization.md` ("End-to-end reward and the persistent
runtime") has the details. In short:

- **Reward and acceptance.** The fraction of the deployed partition time saved, paired with
  the incumbent. A gain of 1 % or more is re-measured with every check against the composite
  best before it joins it.
- **Not capturable** goes back to the fix loop as a defect.
- **Attribution is diagnostics.** It chooses the slot (kernel time plus its launches' share of
  the overhead), feeds the prompt (deployed time, kernel time, launches, a launch-bound note),
  the techniques (fusion-in-slot / persistent kernel when launch-bound), the v2 `launch_frac`,
  and the proposals.
- **Records.** `OutcomeRecord` gains `timing_basis`, `e2e_*` and `partition_effect`, and old
  records still load.

**Live A/B** (Haiku 5.5 over Bedrock):

- Setup: L40S through the host, 2 rollouts × 4 steps, on three KB winners as seeds. The
  ResNet18 winner is launch-bound: graph saving 7.6 %, 87 launches. Each arm used its own GPU
  and its own LLM samples. Every e2e step also logged what the per-slot rule would have
  decided on the same runs. Cost: $1.65 for 64 calls.

| problem | e2e arm: deployed speedup | slot arm | e2e-arm steps where the two rules disagree |
|---|---|---|---|
| L2 gemm winner | **1.58×** (134 → 85 µs; paired vs seeds 0.638) | 1.00× | 0 of 8 |
| MiniGPT winner | **1.55×** (2503 → 1613 µs) | 1.53× | 1 of 8: bmm, its slot 12.6 % faster (event-timed), +0.6 % end to end; the slot rule would have taken it, the e2e reward kept it neutral (below its 1 % threshold) |
| ResNet18 winner | **1.10×** (687 → 625 µs) | 1.00× | 0 of 8 |
| ResNet18 conv+bn fused variant | 1.00× | 1.00× | 0 of 6 |

What the A/B shows:

- **The speedups differ mostly by LLM luck.** With 8 steps a run, the speedup difference is
  mostly that.
- **The slot rule misjudges the size of a win.** On ResNet18 a linear win was +6.5 % by the
  slot rule and +8.5 % end to end.
- **The only disagreement is a slot-only gain.** The end-to-end reward did not take it: 0.6 % of the partition is below its threshold.
- **The launch-oriented techniques were offered.** The agent tried `kernel_fusion_in_slot`
  and `persistent_kernel` on the launch-bound ResNet18 slots.
- **No fusion win came out of the live runs.** Haiku did not produce a working epilogue
  fusion of the 200-line implicit-GEMM conv in this budget.

**A fusion win, measured deterministically** (`demo2`, L40S, 3 repeats):

- The candidate is batchnorm folded into the conv epilogue of the fused conv+bn slot, also on
  the split-K path, because batchnorm is affine. It cuts 59 launches to 39 and drops the
  intermediate.
- **The end-to-end reward accepts it:** paired graph gain +4.1–4.3 %, 625.7 → 599.0 µs.
- **The per-slot rule accepts it too but misjudges it:** +6.4 % on its slot, a reward of
  +5.0 %.
- **The old proposal estimate was 13 %.** It took the consumer's event-timed 92 µs.
  `propose_fusions` now estimates 5.5 %: the consumer's kernel time plus launch overhead.
  The graph optimiser's `propose_from_profile` and `kb_proposals.decide` now use attributed
  op times and the deployed time too.
- **A capture-breaking candidate** (a `cudaStreamSynchronize` in the relu launcher) is
  compared eager to eager: −12.5 %. The step flags it as a capture defect for the fix loop.

### The model's result (`pipeline/kb_model_run.py`)

The number reported for a model is one whole-model runtime with every partition's winners.
The steps:

1. The KernelBlaster runs' best slots go into a kernel cache, guarded by each slot's instance
   shapes, together with an existing cache if given. Each entry keeps the GPU its run recorded
   (`source.gpu`).
2. `cache_plan` puts every op of the IR, with the accepted hints applied, on the winner that
   covers its shape. One source verified on several windows' shape sets now serves all of
   them as one slot.
3. The result is exported as one problem, checked against the full reference (outputs and
   every op isolated) and timed by graph replay, with eager as a diagnostic.
4. The hand-seeded original is timed the same way: in the same requests, paired, when the
   slots match. When the composed model is the seed model (no winner applies) it is timed once
   and `identical_to_seed` is set.

Optional (`--time-partitions`): every partition timed the same way, with its best slots paired
against the hand seeds (`seed_kind: hand`; a fused or split slot the seed model lacks keeps the
problem's own seed), for the sum.

**Where it is timed.** Steps 3–4 and the partitions run through KernelBlaster's evaluators
(`pipeline/kb_eval.py` drives the checkout's `scripts/time_problems.py` as a subprocess), so the
model and its components are timed the same way and on the same hardware:

- `--evaluator local`: this host's GPU pool (`--gpus auto|all|ids`, auto = the idle GPUs) and the
  per-GPU lock every KernelBlaster run takes.
- `--evaluator ssh`: build here, run on remote GPUs (`--ssh-host`, repeatable,
  `--cuda-visible-devices`, `--remote-dir`, `--ssh-key`). Binaries go as rsync deltas, and a
  problem above 1 GiB gets one `data.bin` copy per host and data sha1, shared by the model and
  partition problems, removed at the end.
- `--runtime auto|host|process`: the persistent driver host by default. With one process per
  run, the seeds are evaluated right after the candidate in the same round (`pairing:
  interleaved`).
- `--devices N` (default 1): every timing of the run on one GPU.
- `--evaluator builtin` is ModelBlaster's own host on the default GPU, with no pool or locks.
  `auto` (the default) picks ssh with `--ssh-host`, else local when the checkout has
  `time_problems.py`, else builtin.

**Same hardware.**

- **What the runs record.** Every KernelBlaster result records the GPU it was timed on
  (`result.json` `gpu`: name, UUID, host, arch; `EvalResult.gpu` per evaluation).
- **What is required.** The model run requires the GPU model and arch those runs recorded, or
  `--gpu-model` / `--arch`, which must agree with them. With `--arch` unset it builds for the
  runs' arch: sm_89 for L40S runs, sm_75 for TITAN RTX.
- **How it is enforced.** KernelBlaster keeps only the selected GPUs of that model (`gpu_model`).
  With none, the run fails, naming the models it found. It never times on another model.
- **Inconsistent inputs are errors too:** runs on several models or archs, or a cache entry
  from another model.
- **`--allow-gpu-mismatch`** times anyway, with the warnings in `hardware.warnings` and
  `match: false`.
- **Runs that predate the field** cannot be matched: a loud warning and `verified: false`. Pass
  `--gpu-model` to pin them.

| piece | detail |
|---|---|
| output | `modelblaster.model_runtime/v1`, plus `device`, `gpu_name` / `gpu_uuid` / `gpu_host`, `hardware` (required model and arch, recorded ones, `match`, warnings), `evaluator`, `kb_time` (pool, host statistics, KernelBlaster's telemetry); per partition `device`, `same_device`, `kb_gpu` |
| wired into | `kb_loop.sh` (step 5; `GPUS`, `SSH_HOST`, `REMOTE_DIR`, `CVD`, `SSH_KEY`, `RUNTIME`, `GPU_MODEL`), `kb_graph_opt` final summary (`model_e2e_us`, `seed_e2e_us`, `model_speedup`, `model_gpu`, `model_device`, `model_hardware_match`) with its `--gpus` / `--ssh-host` pool |
| on demand | `python -m modelblaster.pipeline.kb_model_run IR --kb-runs RUN_DIR [--cache DIR] [--hints F] [--time-partitions] [--evaluator ssh --ssh-host H --remote-dir D] [--gpu-model M]` |
| telemetry | `WORK/telemetry.jsonl`: `mb.model_run` → `mb.model_run.compose`, `mb.model_run.time` → `mb.kb_time` (one per KernelBlaster call); KernelBlaster's own spans in `WORK/kb_time/model.telemetry.jsonl` |
| whole-graph export | the same path (the problem is the model) |

**ResNet18, 6 windows of at most 12 ops, TITAN** (builtin host):

- Setup: `kb_export --strategy window`, each window given the ResNet18 winner's slots as its
  best. Shared TITAN, 3 requests.

| | µs per forward pass (graph replay) |
|---|---|
| composed model (one runtime, 17 → 7 slots after merging same-code guards) | **1603.7** |
| hand-seeded model, paired | 33 248 (speedup **20.7×**, paired 20.6×) |
| sum of the 6 windows timed the same way | 1623.7 |
| boundary effect (model − sum) | −20 µs (−1.2 %) |

- The windows' sum overstates the model by 20 µs: each window pays its own graph launch and
  ramp-up, which one whole-model replay pays once.
- On launch-bound models this gap is what a per-partition number hides.

**The same 6 windows on firesim2's L40S, over ssh.**

- Setup: the windows exported seeded with the winner's slots. KernelBlaster run with 0 rollouts
  on the L40S pool (no LLM calls); each `result.json` recorded an `NVIDIA L40S`, sm_89, on GPU 2
  or 3.
- `kb_model_run --evaluator ssh --time-partitions --reps 3`, no `--arch`: it built for sm_89 and
  required the L40S. Model, seed and all six partitions ran on GPU 2 (`GPU-22350958…`, 21
  host requests). 72 s end to end, 70 s of it KernelBlaster's timing.

| | KB run (GPU) | µs, graph replay, `kb_model_run` on L40S GPU 2 | hand seeds, paired |
|---|---|---|---|
| window conv1…layer1.1.conv1 | 141.3 (GPU 2) | 141.3 | 3208.2 |
| window layer1.1.bn1…layer2.0.downsample.0 | 100.4 (GPU 2) | 100.4 | 2107.4 |
| window layer2.0.downsample.1…layer3.0.bn1 | 92.2 (GPU 2) | 93.2 | 2065.4 |
| window layer3.0.relu…layer3.1.bn2 | 115.7 (GPU 2) | 116.7 | 2503.7 |
| window add_5…layer4.1.conv1 | 123.9 (GPU 3) | 123.9 | 2764.8 |
| window layer4.1.bn1…fc | 114.7 (GPU 3) | 114.7 | 1102.8 |
| sum of the windows | 688.1 | 690.2 | 13 752 |
| **composed model** (one runtime) | | **683.0** (eager 738.2) | 13 795 (speedup **20.2×**, paired 20.2×) |
| boundary effect (model − sum) | | −7.2 µs (−1.0 %) | |

- The partitions re-timed by the model run agree with KernelBlaster's own final numbers within
  1 µs: same model, same runtime.
- The same composition with `--evaluator local` on this host's TITAN RTX fails before timing:
  `no NVIDIA L40S GPU among the selected devices (0 (NVIDIA TITAN RTX))`.

**SwinV2, whole graph, 2.27 GB of data, L40S over ssh.**

- Setup: one problem; its driver maps `data.bin` at run time. KernelBlaster ran it with 0
  rollouts on GPU 3 (151 336 µs).
- No winner applies, so the composed model is the seed model (`identical_to_seed`).
- `data.bin` was copied once (23 s) and served both the model and the partition problem.
- 93 s end to end (20 s compose and export, 72 s timing).

| | µs, graph replay, L40S GPU 2 |
|---|---|
| composed model (3 requests: 151 232, 151 194, 151 237) | **151 232.5** (eager 151 397) |
| the KernelBlaster run's problem, re-timed | 151 257.1 |
| KernelBlaster's own final (GPU 3) | 151 335.9 |

The hint re-measurement uses the same path with `--ssh-host`. Two ResNet18 window bests,
leased from the remote pool, re-timed 3 × interleaved on L40S GPU 2 gave 123.9 / 122.9 /
122.9 µs and 141.3 µs; KernelBlaster's own numbers were 123.9 and 141.3.

### Against PyTorch (`pipeline/torch_baseline.py`, `pipeline/torch_bench.py`)

After its own timing, the model run times the original PyTorch module against the composed
model on the same GPU (`kb_model_run --torch-baseline auto|on|off`, default auto). Both are
measured the same way and in the same session, so `speedup_vs_torch` is a like-for-like number.

    python -m modelblaster.pipeline.kb_model_run IR --cache C [...] \
        --torch-baseline on --torch-python /path/to/torch/python [--torch-source FILE] \
        [--torch-rounds 4] [--torch-variants eager,graph,compile_mna,compile_ro]

`kb_graph_opt` has the same flags (`--torch-baseline`, `--torch-python`, `--torch-source`,
`--torch-rounds`, `--torch-variants`), and `gpu_eval/tools/kb_loop.sh` takes them as
`TORCH_BASELINE`, `TORCH_PYTHON`, `TORCH_SOURCE`, `TORCH_ROUNDS` and `TORCH_VARIANTS`.

- With `auto`, the comparison is skipped, with the reason recorded (`torch_status: skipped`),
  when the model failed its checks, its device has no UUID, no PyTorch source is found, or there
  is no torch python for the timing host. A failure is `torch_status: failed` and the model's own
  result stands. `on` raises in both cases.
- The torch python is `--torch-python` or `$MB_TORCH_PYTHON`. Locally it falls back to the
  KernelBlaster python, then this python, if either imports torch with CUDA. Over ssh it must be
  given: it is only run on the remote host, never installed into or modified.
- The PyTorch source is `--torch-source`, `$MB_TORCH_SRC`, or `<stem>.py` under the `adapted/` or
  `src/` trees next to the IR set (`.../level2/ir16/21_X` finds `.../level2/src/**/21_X.py`).

**Same model and data.**

1. The KernelBench source is resized to the dims the extractor shrank it to. This reuses the
   extractor's own sizing helpers, and the sizing is the one whose `get_inputs()` matches the
   IR's input count.
2. The IR's `weights.npz` is loaded into the module's state dict by name. A BatchNorm folded at
   extraction becomes weight = scale, bias = folded bias, mean 0 and var 1 − eps, which is the
   same affine map. Any state key without an IR weight refuses the bundle.
3. The forward runs on the inputs from `io.npz`, and its outputs are checked against
   `io.npz`'s reference with the driver's tolerance (the problem manifest's `output_rel` /
   `output_abs`): `max|out − ref| ≤ rel·max|ref| + abs`, with non-finite values matched exactly.
   This check runs once on the CPU when the bundle is built, and again on the GPU for every
   variant. A variant that fails it is reported and not timed.
4. An fp16 IR (`extract_graph --quant fp16`, [fp16_cuda.md](fp16_cuda.md)) runs torch in half
   (`module.half()`, half inputs). That is fp16 storage like ours, with torch's own fp16 math
   (tensor cores by default). The CPU check runs in fp32 on the fp16-rounded data, the golden's
   own recipe.

The bundle (`model.py`, `recipe.json`, `state.npz`, `io.npz`) and `torch_bench.py`, which needs
only torch and numpy, are built here. For a remote GPU they are copied to
`<remote-dir>/torch_baseline/bundles/<key>`.

**Same device.**

- The torch servers run with `CUDA_VISIBLE_DEVICES=<uuid>` of the GPU the model run used.
- Our side is pinned to that device (`KBEval.for_device`).
- Every block reports its UUID, and any other device raises `DeviceMismatch`.
- `same_device`, `uuid`, `torch_device` (name, cc, SMs) and the versions (torch, CUDA, cuDNN,
  triton, driver) are recorded.
- Both sides hold KernelBlaster's per-GPU flock for each block, so no KernelBlaster run times on
  the GPU at the same time.

**Same measurement.** Each block uses the driver's `time_multi` rule on both sides:

1. W warm-up passes of every body (W = 10).
2. Round robin, one CUDA event pair on the body's stream around one forward pass, synchronised
   before the next pass, so the GPU is idle before each. This repeats until every body has had
   the 200 ms budget or 200 reps.
3. As many host wall-clock reps, round robin. One rep runs from issuing the forward pass to the
   return of the device synchronise (`steady_clock` in the driver, `perf_counter` in torch).
   Inputs stay resident and outputs stay on the device, so a rep has no H2D, D2H or allocation
   of ours.
4. The steady-state throughput: one warm pass, then 100 passes back to back with one sync, per
   pass (wall and events).

The statistic is the driver's upper median, per block and then over the blocks, together with
the within-block CV and the spread of the block medians.

- Ours runs through KernelBlaster's evaluator (`time_problems.py --driver-env MB_WALL=1
  MB_WARM=10 MB_TIME_MS=200 MB_THRU_N=100 MB_ATTR=0 MB_OP_US_PASS=0`). The persistent driver host
  prints one `MB_TIMING` line per body (eager, graph, the paired seed), which lands in
  `EvalResult.timing`. `MB_WALL` is off by default, and drivers exported before it ignore it.
- Torch runs as one long-lived `torch_bench.py` server per math mode, speaking JSON lines over
  stdin/stdout (ssh for a remote GPU). TF32 is process-wide, and torch.compile bakes it into its
  kernels, so each mode needs its own server.
- The ssh round trip and process start are outside every timed region.
- Blocks alternate ABBA over `--torch-rounds` (default 4): round *k* runs ours, torch TF32 off,
  torch TF32 on, and round *k* + 1 runs them in reverse.

**Torch variants.**

| variant | what |
|---|---|
| `eager` | the module under `torch.inference_mode()` |
| `graph` | the forward captured once with `torch.cuda.graph` on static inputs, replayed |
| `compile_mna` | `torch.compile(mode="max-autotune-no-cudagraphs")` |
| `compile_ro` | `torch.compile(mode="reduce-overhead")` (inductor + CUDA graph trees) |

Every variant runs with `cudnn.benchmark` on, warmed (3 passes at build plus the block's
warm-up). A variant that fails to build is reported in the table with its error.

**Same math.**

- `detect_math` reads our slot sources, ignoring comments:
  - `tf32` for TF32 markers (`tf32`, `cvt.rna.tf32`, cuBLAS TF32 modes);
  - `tensor_cores` for `wmma` / `mma.sync` / `wgmma` / `ldmatrix`;
  - `reduced` for `__half` / bf16;
  - `fp32` otherwise.
- Torch is timed with TF32 off (`allow_tf32` false for matmul and cuDNN, `highest` matmul
  precision) and with TF32 on.
- The headline mode follows from that:
  - `tf32_off` when ours is plain fp32: the same arithmetic;
  - `best` (the faster torch mode) when ours uses TF32 or tensor cores;
  - `fp16` for an fp16 IR, with one server, because TF32 does not apply.

**Result.**

- `model_runtime.json` gets `torch_baseline`, with these parts:
  - `rows`, one per variant and mode, with GPU and wall µs (and the block means), CV, spread, block medians,
    throughput, the check and the speedups of ours over it;
  - `headline`, `math`, `method`, `versions`, `flags`, `checks`, `bundle` (source, sizing,
    weight mapping, CPU check);
  - `raw`, every block.
- The top-level fields are `torch_status` (`ok`, `skipped`, `failed`), `torch_reason`,
  `torch_uuid`, `torch_same_device`, `torch_us` (the headline mode's fastest variant by
  GPU time), `torch_best_variant`, `torch_wall_us`, `torch_<variant>_us`,
  `torch_<variant>_wall_us`, `speedup_vs_torch` (ours graph replay vs `torch_us`, GPU),
  `speedup_vs_torch_wall` and `speedup_vs_torch_eager`.
- `kb_graph_opt`'s `summary.json` copies them per IR.
- The ablation report's "vs torch" and "vs torch wall" columns use the paired ratios
  ([ablations.md](ablations.md)).
- The table is printed to stderr (`torch_baseline.table`).
- Ours is our CUDA-graph replay; our eager pass is a row too.

**Remaining asymmetries.**

- **"GPU time" is the event span of one forward pass.** For a host-bound eager pass (torch
  eager, `compile_mna`) the span includes the gaps while the GPU waits for launches, so it is
  close to the wall time. The throughput column shows the pipelined rate. Ours eager has the
  same property, but its launches are cheaper.
- **The wall clock includes each side's own host stack.** That is the Python dispatcher and
  caching allocator for torch, and a C++ launch sequence or `cudaGraphLaunch` for ours. This is
  what each side costs per inference, so it is part of the comparison and not removed.
- **The processes differ in lifetime.** The torch server lives for the whole comparison. Our
  driver host is started per block by `time_problems.py` and warmed with the same 10 passes.
  If anything, that disadvantages ours.
- **The neighbours in the round robin differ.** Ours alternates eager, graph and the hand
  seed. Torch alternates its 2 to 4 variants. The GPU is idle before every rep on both sides,
  but the L2 contents left by the previous body differ.
- **The references differ, the tolerance does not.** Ours is checked by the driver against the
  CPU torch-walk references linked in at export. Torch is checked against `io.npz`.
- **`reduce-overhead` copies the inputs** into its graph's static buffers on every call (CUDA
  graph trees). That is torch's cost of that mode and is counted. The `graph` variant
  (static inputs) is the copy-free torch number.
- **Clocks are not locked** on either side. ABBA interleaving and the block spread bound the
  drift.
- **The event clock is coarse.** On the L40S, CUDA event times land on 1.024 µs ticks, the
  same for both sides. A median is therefore a tick (51.2 µs = 50 ticks), and a block spread
  of 0.00 means every block's median fell on the same tick. `gpu_mean_us` (the median of the
  block means, the "GPU mean" column) resolves below a tick.

**Live, firesim2 L40S GPU 2** (`GPU-22350958-ee15-b07b-0e89-77c5157f1d3b`, the GPU the model
runs timed on; every block reported that UUID). Software: torch 2.11.0+cu128, CUDA 12.8, cuDNN
9.19.0, triton 3.6.0, driver 580.126.18. Each comparison is 4 ABBA rounds with up to 200 reps
per block, and every row passed its check. In each row, "ours faster" is that row's time divided
by ours (CUDA graph). No LLM calls: the winners come from the graph-optimiser runs' caches.

Level2 #21 Conv2d_Add_Scale_Sigmoid_GroupNorm, FP32. Its winning slots use float / float4
only, so the headline is against TF32 off. Setup took 20 s and the 12 blocks 62 s.

| variant | TF32 | check | GPU µs (CV %, spread %) | GPU mean µs | wall µs (CV %) | throughput wall / GPU µs | ours faster (GPU) | ours faster (wall) |
|---|---|---|---|---|---|---|---|---|
| ModelBlaster CUDA graph | - | pass | 51.2 (1.50, 0.00) | 51.2 | 54.8 (1.21) | 49.0 / 48.9 | - | - |
| ModelBlaster eager | - | pass | 55.3 (1.28, 0.00) | 55.2 | 58.1 (0.94) | 51.7 / 51.6 | - | - |
| torch eager | off | pass | 230.4 (3.04, 2.22) | 232.4 | 234.4 (3.14) | 134.0 / 133.6 | 4.50x | 4.28x |
| torch CUDA graph | off | pass | 135.2 (1.36, 0.00) | 135.4 | 140.1 (1.55) | 127.4 / 127.2 | 2.64x | 2.56x |
| torch.compile max-autotune-no-cudagraphs | off | pass | 246.8 (4.10, 3.32) | 247.9 | 262.8 (4.45) | 193.1 / 192.8 | 4.82x | 4.80x |
| torch.compile reduce-overhead | off | pass | 163.8 (4.20, 3.12) | 166.4 | 169.6 (4.21) | 90.5 / 90.1 | 3.20x | 3.10x |
| torch eager | on | pass | 229.4 (3.74, 2.66) | 234.2 | 235.7 (2.80) | 134.2 / 133.8 | 4.48x | 4.31x |
| torch CUDA graph | on | pass | 135.2 (0.84, 0.00) | 135.4 | 140.3 (1.23) | 127.3 / 127.0 | 2.64x | 2.56x |
| torch.compile max-autotune-no-cudagraphs | on | pass | 260.1 (4.79, 3.15) | 265.9 | 278.8 (3.77) | 209.4 / 209.0 | 5.08x | 5.09x |
| torch.compile reduce-overhead | on | pass | 160.8 (4.48, 3.18) | 165.7 | 170.9 (4.66) | 90.7 / 90.3 | 3.14x | 3.12x |

headline (tf32_off): ours graph 51.2 µs vs torch best graph 135.2 µs -> 2.64x (GPU); wall 54.8 vs 140.1 µs (graph) -> 2.56x

Level2 #21, FP16 (`--quant fp16`; `--torch-source` given, since the fp16 IR set has no
`src/` tree). The conv slot uses `mma.sync` (fp16 inputs, fp32 accumulate) and the others use
`half2`. Torch runs in half. Setup took 15 s and the 8 blocks 35 s.

| variant | TF32 | check | GPU µs (CV %, spread %) | GPU mean µs | wall µs (CV %) | throughput wall / GPU µs | ours faster (GPU) | ours faster (wall) |
|---|---|---|---|---|---|---|---|---|
| ModelBlaster CUDA graph | - | pass | 38.9 (1.72, 0.00) | 39.0 | 42.4 (1.41) | 36.8 / 36.7 | - | - |
| ModelBlaster eager | - | pass | 43.0 (1.72, 2.38) | 43.0 | 45.8 (1.96) | 39.3 / 39.2 | - | - |
| torch eager fp16 | - | pass | 232.4 (3.26, 15.86) | 233.5 | 238.9 (3.03) | 127.9 / 127.5 | 5.97x | 5.63x |
| torch CUDA graph fp16 | - | pass | 128.0 (1.31, 1.60) | 128.7 | 133.6 (1.37) | 120.6 / 120.4 | 3.29x | 3.15x |
| torch.compile max-autotune-no-cudagraphs fp16 | - | pass | 252.9 (4.60, 27.94) | 253.5 | 271.0 (3.69) | 201.5 / 201.2 | 6.50x | 6.39x |
| torch.compile reduce-overhead fp16 | - | pass | 152.6 (4.76, 26.85) | 153.4 | 169.4 (4.70) | 90.0 / 89.7 | 3.92x | 3.99x |

headline (fp16): ours graph 38.9 µs vs torch best graph 128.0 µs -> 3.29x (GPU); wall 42.4 vs 133.6 µs (graph) -> 3.15x

Level3 9_ResNet18, FP32 (batch 2, 3×224×224), composed from the L40S KernelBlaster runs'
winners. The slots are plain fp32, so the headline is against TF32 off. Setup took 38 s and the
blocks 56 s.

| variant | TF32 | check | GPU µs (CV %, spread %) | GPU mean µs | wall µs (CV %) | throughput wall / GPU µs | ours faster (GPU) | ours faster (wall) |
|---|---|---|---|---|---|---|---|---|
| ModelBlaster CUDA graph | - | pass | 687.1 (0.22, 0.89) | 686.9 | 692.7 (0.21) | 686.1 / 686.0 | - | - |
| ModelBlaster eager | - | pass | 738.3 (0.25, 0.55) | 738.2 | 742.3 (0.32) | 736.3 / 736.1 | - | - |
| torch eager | off | pass | 1611.8 (2.66, 2.22) | 1614.3 | 1619.7 (2.93) | 1552.3 / 1551.9 | 2.35x | 2.34x |
| torch CUDA graph | off | pass | 726.0 (0.19, 0.00) | 726.0 | 732.6 (0.26) | 666.1 / 665.8 | 1.06x | 1.06x |
| torch.compile max-autotune-no-cudagraphs | off | pass | 1382.4 (2.44, 1.93) | 1382.2 | 1373.6 (3.49) | 1203.5 / 1203.1 | 2.01x | 1.98x |
| torch.compile reduce-overhead | off | pass | 1009.7 (0.67, 0.30) | 1010.4 | 1016.1 (0.93) | 817.6 / 817.1 | 1.47x | 1.47x |
| torch eager | on | pass | 1767.4 (2.45, 2.90) | 1790.9 | 1800.7 (2.29) | 1715.1 / 1714.7 | 2.57x | 2.60x |
| torch CUDA graph | on | pass | 440.3 (0.44, 0.00) | 440.9 | 446.5 (0.49) | 427.9 / 427.6 | 0.64x | 0.64x |
| torch.compile max-autotune-no-cudagraphs | on | pass | 1302.5 (2.68, 3.07) | 1321.0 | 1332.0 (2.62) | 1260.8 / 1260.4 | 1.90x | 1.92x |
| torch.compile reduce-overhead | on | pass | 628.7 (1.24, 0.81) | 631.3 | 635.7 (1.10) | 456.6 / 456.3 | 0.92x | 0.92x |

headline (tf32_off): ours graph 687.1 µs vs torch best graph 726.0 µs -> 1.06x (GPU); wall 692.7 vs 732.6 µs (graph) -> 1.06x

Findings:

- On all three runs, the torch CUDA graph is torch's fastest single-inference number.
  `reduce-overhead` has torch's best back-to-back throughput on #21 (90.5 µs per pass in fp32,
  vs 127.4 µs for the CUDA graph), but its single call is slower (163.8 µs).
- ResNet18 is close to parity with torch in fp32: 1.06× over torch's CUDA graph with TF32 off
  (1.03× in an earlier session: torch's graph moved 702 → 726 µs between sessions, ours 685 →
  687 µs). With TF32 on, torch's cuDNN convolutions run on tensor cores at reduced precision and
  are faster than ours (0.64×). That row is the secondary comparison, because our slots do not
  use TF32.
- #21 is 2.6× (fp32) and 3.3× (fp16) over the fastest torch variant, and 4.5× / 6.0× over
  torch eager. TF32 makes no difference to torch's cuDNN path on #21.
- The headline wall-clock speedups are within 0.15× of the GPU ones (2.56 vs 2.64, 3.15 vs
  3.29, 1.06 vs 1.06). The inputs are resident and both sides synchronise once per pass, so the
  remaining host cost per call is 3 to 7 µs on each side.

### Tests and gates (persistent runtime)

- **Export gate on the new runtime.** Setup: TITAN, `kb_export --build-check --runtime host`,
  `MB_TIME_MS=50`. Each problem is built as host + shared object and evaluated in one host
  request: partition check, every op isolated, graph replay. Results:
  - Level1 100/100, Level2 100/100, Level3 50/50;
  - every one of the 250 with `basis=graph`: the seeds capture everywhere.
- **MB tests (against PyTorch).** `pipeline/tests/test_torch_baseline.py` covers:
  - the method: ABBA order, upper medians, CV and spread, same-device enforcement;
  - the tolerance check, TF32 / tensor-core detection, and the headline modes (fp32, TF32,
    fp16);
  - the bundle: sizing, folded-BatchNorm mapping, fp16, refusals;
  - the result fields, the `--driver-env` pass-through, the auto-skip reasons, and the
    ablation's paired column;
  - with a GPU, the whole comparison on a tiny model.
- **MB tests.** `pipeline/tests/test_kb_driver_modes.py` adds:
  - the end-to-end / attribution parser;
  - host vs one-process parity (same checks, outputs and errors);
  - host crash recovery and pairing;
  - the windowed model-runtime composition.
- **KB tests.** `tests/test_runtime_host.py`: protocol and recovery against a fake host, the
  evaluator's failure mapping, real-host parity and crash recovery (segfault, illegal address,
  hang), the end-to-end reward decisions, the capture-defect fix loop, and record
  compatibility.

## Remaining recommendations

Items 1–3 are implemented ("Persistent runtime and the end-to-end reward" above); 4–6 remain.

1. **Persistent driver process.** Each run pays about 185 ms of CUDA context creation plus
   about 80 ms of process start and exit. That is 35–55 % of a ResNet18 or MiniGPT step run
   (0.4–0.6 s) and 70 % of a Level2 one (0.35 s). Keeping one process per problem and GPU,
   with kernels loaded as a `.so` per candidate (`dlopen`), the data uploaded once and only the
   changed slot's buffers reset, would remove it. It needs a new build product (a `.so`, not
   an executable) on the KB servers.
2. **Per-op GPU time.** Measure per-op time without the event and launch gap:
   - capture each op into a graph and time replays (needs PTDS), or
   - block the stream with a host function while the timed ops are enqueued.

   Then use it, or `MB_SLOT_US` minus a measured per-launch floor, for KB's slot shares, so
   tiny slots stop looking 2–3× bigger than they are (see the slot table under "Answers").
3. **Launch-bound partitions to the graph optimiser.** Two direct launch-bound signals, when a
   partition's eager time exceeds its graph time:
   - `partition_us − partition_graph_us`;
   - `MB_OP_US` of ops below about 5 µs, which on the L40S are all launch or gap.

   The graph optimiser (`kb_plan` / `graph_hints`) could rank fusion of chains of such ops, and
   timestep loops like LSTM's 6170-launch host loop (a persistent kernel, or a graph per
   sequence), from these. Not done here, because those files are being changed elsewhere.
4. **Faster full checks.** A full SwinV2 run still spends 1.9 s in isolated checks on the L40S,
   mostly H2D of 2.4 GB of references from the mmapped file plus the compare. Possible fixes:
   - compare on the GPU, with a reduction kernel in the generated walker tail;
   - pin the mmapped region (`cudaHostRegister`, read-only).

   Only baseline, confirm and final runs pay this now.
5. **Retire `kb_bridge.py` (H4)** in favour of `kb_export --strategy per_op`. It is the last
   LibTorch user in the MB harness.
6. **Bump MB's KernelBlaster submodule** to KB `feat/harness-perf` once it is merged (the pin is
   unchanged here). Until then, KB's step runs are full runs, which is correct but slower.
