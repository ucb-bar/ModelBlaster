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
- **Big data files are mmapped** instead of `fread` (the > 1 GiB `MB_DATA_PATH` case).
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

**Decision: the KB reward stays eager.**

- The reward is per slot (`MB_SLOT_US`), and a graph replay cannot attribute time to ops.
- Graph time hides exactly the launch-overhead wins that fusion and fewer launches buy, and
  deployment through MB's runtime is eager.
- Graphs do not make the measurement quieter: on the idle L40S both have a CV of about 0.2 %.
- Not every candidate is capturable.

`MB_PARTITION_GRAPH_US` is reported next to the eager time as the deployment-with-graphs
number. The gap between the two (`partition_us − partition_graph_us`) is a direct measure of
launch overhead, the signal the graph optimiser should use to rank launch-bound partitions as
fusion candidates.

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

## Remaining recommendations (not implemented)

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
