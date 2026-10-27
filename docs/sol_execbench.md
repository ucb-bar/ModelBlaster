# SOL-ExecBench on ModelBlaster → KernelBlaster

This page covers the port of NVIDIA SOL-ExecBench to ModelBlaster, made from the original
repository's problem files and run at the workloads' native shapes. It also covers the gates
the port passes and how its problems go through today's ModelBlaster → KernelBlaster flow
([kernelblaster_multislot.md](kernelblaster_multislot.md),
[graph_optimizer.md](graph_optimizer.md)).

There was an earlier port, `gpu_eval/tools/sol_adapt.py`. It started from KernelBlaster's
KernelBench-layout copy (`sol-level1`, `sol-level2`), converted every tensor to fp32 and shrank
the inputs toward 16 MB. This port replaces it:

- the shapes and the input generation are SOL's own;
- the dtype is a parameter;
- every SOL tolerance is checked.

## Sources

| what | where |
|---|---|
| SOL-ExecBench @ `834c7c1` | `data/benchmark/{L1 (94), L2 (82), Quant (33), FlashInfer-Bench (26)}/<problem>/{definition.json, reference.py, workload.jsonl}` |
| evaluator contract | `src/sol_execbench/core/bench/{io.py, correctness.py}`, `driver/templates/eval_driver.py`, `docs/{definition,workload}.md` |
| FlashInfer-Bench safetensors inputs | `data/flashinfer-trace/blob/workloads/...` (only the files of the selected workloads are needed: about 4 MB) |
| KernelBlaster's earlier port | `KernelBlasterRelease/data/KernelBench/KernelBench/sol-level{1,2}` and its CUDA results (`out/kernel_status_by_problem.csv`, `speedup_l2.csv`, `sol-level*_eager*.csv`) |

What SOL does, and what this port reproduces:

- A **definition** has named axes, which are `const`, `var` or `expr`. Its inputs and outputs
  have shapes given as axis names, and dtypes, mostly bfloat16. It also has a reference
  `run(...)` and an optional `custom_inputs_entrypoint`.
- A **workload** binds the var axes. Each input is `random`, `scalar`, `custom` (made by the
  entrypoint) or `safetensors`. The workload carries its own tolerance.
- **Inputs.** The evaluator seeds once (200). It calls the custom entrypoint, then draws every
  random input in definition order with name heuristics: norm weights are 1, norm biases 0,
  there are causal masks, binary masks, rope cos/sin, positive std/var, SSM decays,
  softmaxed weights, and 1/√fan-in weight matrices.
- **Reference.** `run` is always called in return-value style, on the GPU, in the definition's
  dtypes.
- **Correctness.** Element-wise `|out − ref| ≤ max_atol + max_rtol·|ref|` must hold on at least
  `required_matched_ratio` of the elements (default 0.99). Any inf or NaN fails, unless
  `allow_negative_inf` is set and both sides are −inf at that position. An all-zero output
  fails against a non-zero reference. `max_error_cap` is a hard bound on the maximum error.
  L2's workloads spell the ratio `required_match_ratio`. SOL's pydantic model ignores that key,
  so the default 0.99 applies; the converter records the ignored key.

## Conversion (`gpu_eval/tools/sol_upstream.py`)

```bash
python gpu_eval/tools/sol_upstream.py convert BENCH OUT --sets L1,L2,Quant,FlashInfer-Bench \
    --dtype fp32|bf16|fp16 --cap-mb 256 --pick smallest,median,largest[,index0] \
    [--problems REGEX] [--blob-root FLASHINFER_TRACE_DIR]
python gpu_eval/tools/sol_upstream.py sizes BENCH --sets ...      # io MiB per workload: min / median / max
```

Each selected workload becomes `OUT/src/<dtype>/<stem>.py`, with
`stem = <L1|L2|Q|FIB>_<problem>__<min|med|max|index0>`; a pick under a raised cap whose stem
was already taken by another workload gets the suffix `8g`. Each file is a self-contained
KernelBench-style bench: `Model`, `get_inputs()` and `get_init_inputs()`, plus
`sol_reference_outputs()` for the gates. `OUT/manifest.<dtype>.json` records, per workload, the
problem, set, pick, workload index and uuid, the resolved axes, the scalars, the io size, the
dtype mapping, the number of dtype rewrites, the tolerance and the input kinds. Per problem it
records every skipped workload with its reason.

Rules:

- **Native shapes; nothing is resized.**
  - The axes live in a `SOL_AXES` dict. The bench defines no module-level ints, so the
    shrinkers in `extract_graph --bench-target-*` and `torch_baseline` find nothing to resize.
  - The gates extract with `--bench-target-mb 0` and check that the IR's input and output
    shapes equal the workload's.
  - A workload whose io (inputs + outputs at the variant's dtype) is over `--cap-mb` is skipped,
    with the reason recorded. It is never shrunk.
- **Workload selection.**
  - The default picks are the smallest, the median and the largest workload that fit the cap.
  - `index<N>` picks workload N. `index0` is the workload KernelBlaster's port used (see
    "Mapping" below).
  - The selected uuids are in the manifest.
- **Inputs.**
  - `get_inputs()` runs SOL's `gen_inputs` on the CPU: the seed, the custom entrypoint, the
    heuristics in definition order, and `probs` softmaxed for `op_type == "sampling"`.
  - It runs the *unmodified* reference source, with every creation op forced to the CPU, so it
    also undoes `extract_graph`'s fp16→fp32 creation patch.
  - The values are therefore the same for every dtype variant and for every caller.
  - Floating tensors are then cast to the variant's dtype. Integer and bool tensors keep theirs.
  - Scalars stay Python scalars, which `extract_graph` binds as constants.
  - Safetensors inputs are read by a small reader in the bench, so the safetensors package is
    not needed.
- **Model.**
  - `forward(in0_<name>, ...)` has one named parameter per definition input; fx does not
    trace `*args`. It calls the reference `run`, renamed `_sol_run`. The custom entrypoint is
    renamed `_sol_custom_inputs`.
  - A dict return is packed in the definition's output order.
  - Per variant, the reference's dtype tokens are rewritten with an AST pass:

    | variant | `torch.bfloat16` | `torch.float16` | float32 inputs | rewrites |
    |---|---|---|---|---|
    | `fp32` | float32 | float32 | float32 | `torch.bfloat16`/`float16`/`half` → `float32`, `.bfloat16()`/`.half()` → `.float()` |
    | `bf16` (native) | bfloat16 | float16 | float32 | none: the reference byte for byte |
    | `fp16` | float16 | float16 | float32 | `torch.bfloat16` → `float16`, `.bfloat16()` → `.half()` |

  - A hard-coded `'cuda'` device inside `run` becomes `'cpu'`, because the bench is traced on
    the CPU.
- **Refusals** are named, per workload:
  - `unsupported dtype: fp8 (e4m3) / fp4 (e2m1 x2) / int8 tensor storage`;
  - `safetensors inputs (no --blob-root)` or `safetensors blob missing`;
  - `no tensor inputs` (forward takes scalars only);
  - `io N MiB > cap`.

## Precision

The gates extract the fp32 bench files (`--dtype fp32`; the fp16 diagnostic below is the one
exception): the bf16 inputs widened exactly to fp32, the reference's bf16/fp16 tokens rewritten
to fp32. ModelBlaster's storage dtype is then chosen
at extraction, as for any model ([fp16_cuda.md](fp16_cuda.md), `--quant bf16` from `feat/bf16`):

- `--quant bf16` rounds the inputs to bfloat16 (exact for bf16-native inputs), stores every
  activation and constant in bf16 and computes in fp32; the golden is the fp32 model on the
  rounded inputs, stored in bf16.
- `--quant native` (gates) picks, per problem, **bf16** when any of its tensors is bfloat16,
  else **fp16** when any is float16, else **fp32**. An IR has one float storage dtype, so a
  bf16 problem's fp32 intermediates (the reference's explicit `.to(torch.float32)` sections)
  are stored in bf16 too.
- `--quant fp32` is the secondary variant: fp32 math and storage on the native inputs.
- `--dtype bf16` / `fp16` bench files keep the reference's own dtypes (bf16: byte for byte).
  They serve as the torch baseline's source (`--torch-source`, the true bf16 reference in
  torch) and the fp16 diagnostic (fp16 bench, `--quant fp16`, outputs cast to bf16 before the
  SOL check: `gates run --score-cast bfloat16`).

SOL's tolerances were set for kernels in the problem's own dtype, checked against its own
(bf16) reference. Two floors are reported separately from the walk:

- "golden (SOL tol)": the MB golden against SOL's reference. A miss here is the variant's math
  (fp32 math, rounded once) against a reference that rounds at intermediate steps; it is not a
  wrong IR or kernel.
- the bf16 walk can also miss where the golden passes: the IR stores in bf16 intermediates that
  the reference keeps in fp32.

The CPU gate counts the reference values above fp16's maximum (65504) and in fp16's subnormal
range (`range_risk`), for the fp16 mapping.

## Gates (`gpu_eval/tools/sol_upstream_gates.py`)

```bash
source gpu_eval/tools/env.sh
# CPU gates here, each workload in its own memory-capped scope
CUDA_VISIBLE_DEVICES= python gpu_eval/tools/sol_upstream_gates.py run OUT/manifest.fp32.json GATES \
    --quant native --jobs 2 --jobs-large 2 --mem-large-gb 40 [--stems REGEX]
# GPU gates on an ssh GPU host, one queue per GPU (the host's torch python is used, never modified)
python gpu_eval/tools/sol_upstream_gates.py remote GATES --host U@H --ssh-key K --python PY \
    --remote-dir D2 --gpu 2 --until-done [--stems REGEX]
python gpu_eval/tools/sol_upstream_gates.py summary GATES [GATES2 ...] [--native] [--by-quant]
```

On a GPU host with enough memory, `run` can also do everything there
(`--gates extract,shapes,cpu,cuda,export` with `CUDA_VISIBLE_DEVICES=<gpu>`, `--specs`/`--hand`
from `gpu_eval/tools/dump_specs.py` and
`nvcc -O3 -arch=sm_89 -Xcompiler -fPIC -shared cuda/mb_cuda_kernels{,_f16,_bf16}.cu`). The
walk uses the hand kernels of the IR's storage dtype.

- **Host memory.** Every workload runs under `systemd-run --user --scope -p MemoryMax=…`
  (no swap). It is launched only while the host has that much memory available, plus a reserve.
  At most `--jobs-large` large workloads run at once. A workload the cap kills is recorded as
  "too large for host memory" and is not retried. The extractor's peak is about 40-50× the
  workload's io (tracer values, golden, lowering), so at a 60 GB cap workloads above ~1.2 GiB of
  io are skipped.
- **GPU sharing.** Every GPU step holds KernelBlaster's per-GPU flock
  (`/dev/shm/kernelblaster_gpu_<uuid>.lock`), so it takes turns with KernelBlaster runs of any
  process. It first waits for free GPU memory (1.5× the IR's tensors). An out-of-memory is
  recorded as "skipped: GPU memory".
- **Disk.** A worker waits for free disk before it extracts. io.npz is deleted once the GPU
  gates have consumed it. Results go to `GATES/results/<stem>.json` and logs to `GATES/logs/`.

| gate | what |
|---|---|
| extract | `extract_graph --bench-file F --bench-target-mb 0 --quant Q`; a failure is classified (value-dependent op, FFT, in-graph fp8/fp4 quantisation, unsupported lowering, host memory, timeout) |
| shapes | IR graph inputs/outputs = the workload's shapes; io.npz inputs = the bench's `get_inputs()` (rounded to the storage dtype) bit for bit |
| cpu (MB bound) | `walk_torch` on the CPU vs the MB golden, with `mbir.compare`'s dtype bound |
| cpu (SOL tol) | the same walk vs the **SOL ground truth**: the original reference on the native inputs (on the CPU; SOL runs it on its GPU), with the workload's SOL tolerance per output |
| golden (SOL tol) | the MB golden vs the SOL ground truth: the variant's precision floor |
| cuda | `walk_cuda.py --impl hand`: the whole model on the hand CUDA kernels through MB's C ABI, vs the golden, and every intermediate isolated |
| export | `kb_export --strategy whole --build-check --runtime host`: the multi-slot problem built and checked in the persistent driver host (every output, every op isolated); skipped when data.bin would exceed `--export-cap-mb` |

When the walk misses SOL's tolerance and the reference ran on a GPU, the reference is also run
on the CPU. This separates the taxonomy's classes:

- a reference that is not reproducible on another device at SOL's tolerance;
- a precision floor;
- bf16/fp16 intermediate storage;
- a walk-only miss.

## Coverage (2026-10-27)

The GPU gates ran on the firesim2 L40S (sm_89, GPUs 2/3). The CPU gates and extraction ran on
the dev box, and the 8 GiB add-on's large workloads on firesim2 too. In the tables, `n (p)` is
the number of workloads (problems) that pass.

The selection: per problem, the smallest, median and largest workload under the cap, plus
workload #0 (KB's) and, for problems with nothing under 256 MiB, the smallest under 1 GiB.

- **256 MiB cap**: 569 workloads.
- **8 GiB add-on** (raised later; io = inputs + outputs, no weights: SOL problems have none):
  the same picks at 8 GiB, minus what was already selected. That is 334 more. Quant (31) is
  left out, because every Quant reference quantises in-graph. The add-on is 303 workloads,
  and 35 problems still have none (19 fp8/fp4 dtypes, 14 over 8 GiB, 2 scalar-only).

**Native dtype** (`--quant native`: bf16 / fp16 / fp32 per problem), 256 MiB selection:

| set | workloads | problems | extract | shapes | cpu (MB bound) | cpu (SOL tol) | golden (SOL tol) | cuda | export |
|---|---|---|---|---|---|---|---|---|---|
| L1 bf16 | 139 | 42 | 131 (41) | 131 (41) | 129 (41) | 82 (29) | 90 (32) | 127 (41) | 125 (40) |
| L1 fp16 | 6 | 2 | 6 (2) | 6 (2) | 6 (2) | 5 (2) | 5 (2) | 6 (2) | 6 (2) |
| L1 fp32 | 143 | 40 | 121 (34) | 121 (34) | 121 (34) | 121 (34) | 121 (34) | 119 (34) | 111 (34) |
| L2 bf16 | 68 | 22 | 61 (20) | 61 (20) | 44 (16) | 20 (8) | 41 (15) | 42 (16) | 49 (20) |
| L2 fp32 | 110 | 36 | 95 (31) | 95 (31) | 95 (31) | 90 (29) | 95 (31) | 95 (31) | 60 (25) |
| Quant bf16 | 36 | 14 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| FlashInfer-Bench bf16 | 45 | 15 | 27 (9) | 27 (9) | 27 (9) | 27 (9) | 27 (9) | 27 (9) | 27 (9) |
| FlashInfer-Bench fp16 | 22 | 8 | 22 (8) | 22 (8) | 22 (8) | 22 (8) | 22 (8) | 22 (8) | 22 (8) |
| all | 569 | 179 | 463 (145) | 463 (145) | 444 (141) | 367 (119) | 401 (131) | 438 (141) | 400 (138) |

8 GiB add-on, native dtype:

| set | workloads | problems | extract | shapes | cpu (MB bound) | cpu (SOL tol) | golden (SOL tol) | cuda | export |
|---|---|---|---|---|---|---|---|---|---|
| L1 bf16 | 86 | 38 | 68 (33) | 68 (33) | 68 (33) | 46 (22) | 52 (25) | 61 (32) | 65 (33) |
| L1 fp16 | 4 | 2 | 3 (2) | 3 (2) | 3 (2) | 3 (2) | 3 (2) | 1 (1) | 2 (1) |
| L1 fp32 | 40 | 19 | 32 (16) | 32 (16) | 32 (16) | 32 (16) | 32 (16) | 29 (16) | 30 (16) |
| L2 bf16 | 73 | 27 | 48 (22) | 47 (22) | 36 (16) | 24 (9) | 29 (13) | 31 (16) | 44 (22) |
| L2 fp32 | 52 | 24 | 37 (19) | 37 (19) | 35 (19) | 35 (19) | 37 (19) | 20 (13) | 29 (16) |
| FlashInfer-Bench bf16 | 28 | 11 | 21 (9) | 21 (9) | 21 (9) | 21 (9) | 21 (9) | 21 (9) | 21 (9) |
| FlashInfer-Bench fp16 | 20 | 8 | 20 (8) | 20 (8) | 20 (8) | 20 (8) | 20 (8) | 20 (8) | 20 (8) |
| all | 303 | 129 | 229 (109) | 228 (109) | 215 (103) | 181 (85) | 194 (92) | 183 (95) | 211 (105) |

Both selections at native dtype, by set:

| set | workloads | problems | extract | shapes | cpu (MB bound) | cpu (SOL tol) | golden (SOL tol) | cuda | export |
|---|---|---|---|---|---|---|---|---|---|
| L1 | 418 | 90 | 361 (82) | 361 (82) | 359 (82) | 289 (68) | 303 (71) | 343 (81) | 339 (80) |
| L2 | 303 | 70 | 241 (57) | 240 (57) | 210 (51) | 169 (42) | 202 (49) | 188 (51) | 182 (55) |
| Quant | 36 | 14 | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) | 0 (0) |
| FlashInfer-Bench | 115 | 25 | 90 (17) | 90 (17) | 90 (17) | 90 (17) | 90 (17) | 90 (17) | 90 (17) |
| all | 872 | 199 | 692 (156) | 691 (156) | 659 (150) | 548 (127) | 595 (137) | 621 (149) | 611 (152) |

Notes on the export column:

- In the 256 MiB runs, the export was skipped when data.bin exceeded 1 GiB (60 workloads). The
  L2-export fix (`fix/sol-l2-export`: exact data.bin size, condition-aware output bound,
  streamed data.bin, 16 GiB cap) re-ran 31 of these at their dtype, and all of them pass. With
  those, export is 346/361 L1 and 206/241 L2 of the extracted workloads.
- That fix also takes fp32 L2 export (`run/g_fp32`) from 45/82 to 82/82, and bf16 L2 from 57/70
  to 70/70. Its per-workload results are in `analysis/l2_export/after_all.tsv`.

Failure taxonomy, both selections at native dtype:

| gate | class | workloads | problems | examples |
|---|---|---|---|---|---|---|---|---|---|
| cpu | SOL tolerance: precision floor: the bf16 golden misses too | 60 | 19 | L1/005, L1/008, L1/009, L1/011, L1/012, L1/013 |
| cpu | SOL tolerance: bf16 intermediate storage (the golden passes; the reference keeps these in fp32) | 45 | 14 | L1/001, L1/009, L1/052, L1/056, L1/060, L1/064 |
| cpu | bf16 golden bound | 31 | 8 | L1/011, L2/028, L2/034, L2/039, L2/041, L2/043 |
| cpu | SOL tolerance: walk only (the golden passes) | 5 | 2 | L2/033, L2/069 |
| cpu | fp32 golden bound | 2 | 1 | L2/069 |
| cpu | SOL tolerance: precision floor: the fp16 golden misses too | 1 | 1 | L1/073 |
| cuda | golden bound | 33 | 10 | L1/009, L1/011, L2/028, L2/034, L2/039, L2/041 |
| cuda | skipped: GPU memory (out of memory) | 31 | 25 | L1/002, L1/015, L1/026, L1/027, L1/047, L1/066 |
| cuda | out of memory | 4 | 3 | L1/070, L2/015, L2/054 |
| cuda | skipped: GPU memory (not enough free on the GPU) | 3 | 3 | L2/019, L2/023, L2/077 |
| export | skipped: data.bin ~N MiB > N MiB | 73 | 47 | L1/002, L1/015, L1/022, L1/025, L1/026, L1/027 |
| export | skipped: GPU memory (not enough free on the GPU) | 3 | 3 | L2/019, L2/023, L2/077 |
| export | stage run | 2 | 1 | L2/021 |
| export | slot check | 2 | 1 | L1/009 |
| export | skipped: no ops (the workload's output is an input: nothing to optimise) | 1 | 1 | L1/053 |
| extract | value-dependent: .item() | 49 | 14 | FlashInfer-Bench/012, FlashInfer-Bench/013, FlashInfer-Bench/014, FlashInfer-Bench/015, FlashInfer-Bench/016, FlashInfer-Bench/017 |
| extract | in-graph fp8/fp4 quantisation | 36 | 14 | Quant/001, Quant/002, Quant/004, Quant/006, Quant/007, Quant/010 |
| extract | skipped: too large for host memory (MemoryMax=NG, rc -N) | 22 | 18 | L1/013, L1/026, L1/031, L1/047, L1/060, L1/067 |
| extract | value-dependent: nonzero() | 22 | 6 | L2/008, L2/009, L2/010, L2/013, L2/016, L2/061 |
| extract | complex / FFT | 14 | 3 | L1/007, L2/051, L2/052 |
| extract | other extract error | 11 | 4 | L1/008, L1/009, L1/087, L2/032 |
| extract | value-dependent: tolist() | 11 | 3 | L2/014, L2/017, L2/018 |
| extract | value-dependent: boolean-mask indexing | 9 | 2 | L1/053, L2/006 |
| extract | value-dependent: bincount() | 3 | 1 | L1/058 |
| extract | timeout | 3 | 3 | L1/094, L2/036, L2/058 |
| shapes | fail | 1 | 1 | L2/056 |

Reading the taxonomy:

- **Value-dependent ops** (`nonzero`, `.item()`, `tolist`, `bincount`, boolean masks) and
  complex FFT are the earlier port's 33 intentional refusals. They are now also counted over
  FlashInfer's paged/ragged attention and the MoE dispatch at the 8 GiB picks.
- **Host memory.** The extractor's peak is about 40-50× the workload's io, so most add-on
  workloads above about 1.2 GiB of io are skipped at the 60 GB cap.
- **GPU memory.** The hand-CUDA walk keeps every intermediate on the GPU, so large add-on
  workloads skip it. The export, which streams data.bin, still runs for most of them.
- **`bf16 golden bound` / `cuda golden bound`** (8 problems: L1/011 rope, L2/028, L2/034,
  L2/039, L2/041, L2/043, L2/049, ...). The bf16 IR misses MB's own provisional bf16 bound
  against the fp32-math golden: bf16 storage of an intermediate the problem needs in higher
  precision. For example, rope angles in the thousands, rounded to bf16, lose the phase.
  Per-tensor storage dtypes would address this.
- **SOL tolerance misses at bf16** (92 workloads of the 256 MiB bf16 run) are analysed per
  workload in `analysis/bf16_tol/failures.csv` (`feat/sol-bf16-tol`). They split as follows:

  | cause | workloads |
  |---|---|
  | intermediate-rounding: the IR stores in bf16 an intermediate the reference keeps in fp32 | 29 |
  | fp32-output: a float32 output stored in bf16 | 28 |
  | int-input-rounded: integer inputs (indices, positions) carried as bf16 | 12 |
  | tol-unattainable: not even an fp64 reference rounded to bf16 meets SOL's tolerance | 11 |
  | fp32-input-rounded: a float32 input rounded to bf16 | 11 |
  | sol-ref-outlier | 1 |

  All of them come from the IR's single storage dtype (or from SOL's tolerance itself), not
  from wrong lowering or kernels. The fp32 variant of the same workloads meets MB's bound, and
  per-tensor storage dtypes are the fix for the first four classes.

fp32 variant (secondary) of the bf16/fp16 problems, 8 GiB add-on:

| set | workloads | problems | extract | shapes | cpu (MB bound) | cpu (SOL tol) | cpu (SOL tol, fp64 ref) | golden (SOL tol) | SOL tol attainable | cuda | export |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 90 | 40 | 75 (35) | 75 (35) | 75 (35) | 69 (32) | 26 (14) | 69 (32) | 24 (13) | 67 (34) | 74 (35) |
| L2 | 73 | 27 | 49 (22) | 48 (22) | 49 (22) | 42 (18) | 27 (13) | 42 (18) | 22 (10) | 38 (22) | 42 (21) |
| FlashInfer-Bench | 48 | 19 | 41 (17) | 41 (17) | 41 (17) | 41 (17) | 2 (2) | 41 (17) | 2 (2) | 41 (17) | 41 (17) |
| all | 211 | 86 | 165 (74) | 164 (74) | 165 (74) | 152 (67) | 55 (29) | 152 (67) | 48 (25) | 146 (73) | 157 (73) |

`cpu (SOL tol, fp64 ref)` and `SOL tol attainable` come from the `feat/sol-bf16-tol`
diagnostics, which were merged before this run. They take the reference in float64, round it to
the output dtype, and check it against SOL's own reference with SOL's tolerance. They are
recorded for 55 of the 165 extracted
workloads.

- Where they are recorded, SOL's tolerance is attainable on 48 and unattainable on 7.
- The fp32 walk meets SOL's tolerance on 152 of the 165 extracted workloads. Its 13 misses are
  those 7 unattainable ones and 6 precision floors.

The 256 MiB fp32 runs:

- `run/g_fp32` holds 338 workloads, run before the 2026-10-26 host crash. Its CUDA gates ran
  on the local TITAN RTX, not the L40S. Extraction passes 276, the CPU walk 271 (MB bound) and
  245 (SOL tolerance), CUDA 265 and export 222.
- L2's export there is superseded by `fix/sol-l2-export` (82/82).
- Its 3 Quant IRs extracted before the fp8-cast refusal are stale, and are refused now.

**fp16 diagnostic of the bf16 problems** (`run/g_fp16_of_bf16`: fp16 bench, `--quant fp16`,
outputs cast to bf16 before SOL's check; this is fp16, not bf16):

252 workloads (79 problems; Quant left out). This is the same selection as the bf16 run, so
each workload has a bf16 result and an fp16 result.

| bf16 `sol_ok` → fp16 `sol_ok` | workloads |
|---|---|
| pass → pass | 116 |
| miss → pass | 38 |
| pass → miss | 9 |
| miss → miss | 48 |
| not extracted in one or both (value-dependent ops; 5 hit the 20 GB cap in fp16) | 41 |

- **fp16 meets SOL's bf16 tolerance more often: 156 workloads vs 120 for bf16.** fp16 has 3
  more mantissa bits, which covers most intermediate-rounding and fp32-output misses: of
  those, 13 + 13 workloads pass in fp16. The tol-unattainable class stays at 11 misses.
- **fp16 range.** L2/015 (audio sinusoidal embedding + conv) has reference values above 65504
  (342 to 157k elements per workload). All four of its workloads pass in bf16 and miss or fail
  in fp16.
  - The other 7 of the 11 known overflow workloads (L1/059, L2/033) are fp32-native problems,
    so they are not in this set.
  - L1/093 (4 → 0) and L2/054 and L2/060 (4 → 3) also lose workloads in fp16.
- Of the extracted fp16 IRs, MB's own fp16 bound passes on 190 workloads, the hand-CUDA walk
  on 189 and the multi-slot export on 211.
- The per-workload table is `analysis/fp16/bf16_vs_fp16.tsv`. It gives bf16 / fp16 `sol_ok`,
  the matched ratios (fp16 cast to bf16, and raw), fp16's MB bound, CUDA and export, the range
  counts, and the bf16-tol cause.

## Quant and FlashInfer-Bench: what is not supported

**Quant (33 problems): none is supported.** Every reference quantises inside `run()`, with
fp8 e4m3 (`x.to(torch.float8_e4m3fn)` with per-tensor, row or 128×128 block scales) or NVFP4
(e2m1 values packed two per byte, `float4_e2m1fn_x2`, plus fp8 block scales).

- **18 problems are refused at conversion**: they have fp8 or fp4 tensors among their inputs
  or outputs.
- **14 are refused at extraction** (`in-graph fp8/fp4 quantisation`). Before this port, a
  cast to fp8 was lowered as a view: every float cast was one, so the IR silently dropped the
  rounding, and the CPU walk then missed the golden. `generic_lowering._f_cast` now refuses a
  cast to any 1-byte float or fp4 dtype.
- **Quant/023** has no workload under 256 MiB, and the 8 GiB add-on leaves out Quant.

What would be needed:

- fp8 (e4m3/e5m2) and fp4 (e2m1, packed) storage dtypes in the IR, its kernels and the
  manifest;
- scale tensors with block layouts, and quantise/dequantise ops;
- fp8 tensor-core slots (`mma` e4m3, sm_89+) in KernelBlaster, with their prompt rules.

**FlashInfer-Bench (26 problems): 17 pass every gate.** These are rmsnorm, fused add+rmsnorm
and every gemm, at their native bf16/fp16 dtypes.

- **8 are refused at extraction as value-dependent** (`.item()`): the paged decode/prefill
  GQA, the ragged prefill GQA and the paged MLA problems. Their page tables and
  `qo_indptr`/`kv_indptr` arrays come from SOL's safetensors blobs, and `run()` loops over
  sequences with `indptr[i].item()` bounds. A static IR cannot hold this. It would need
  dynamic shapes, or a dense rewrite: gather the pages into a padded KV and mask it, which
  changes the problem.
- **020_moe_fp8_block_scale** is refused at conversion: it has fp8 inputs.



## Mapping to KernelBlaster's earlier port (`gpu_eval/tools/sol_kb_map.py`)

```bash
python gpu_eval/tools/sol_kb_map.py BENCH KB_SRC KB_RESULTS --manifest OUT/manifest.fp32.json \
    [--manifest OUT_KB/manifest.fp32.json] --out map.json --csv map.csv
```

For every L1/L2 problem, the mapping records:

- the KB-layout id (`sol-level1/<name>`, `sol-level2/<name>`);
- which SOL workload KB's `get_inputs()` reproduces, by input shapes evaluated on the meta
  device (on the CPU when the generator needs values);
- whether KB's dtypes are SOL's;
- KB's result row (`kernel_status_by_problem.csv` for sol-level1, `speedup_l2.csv` for
  sol-level2, the eager timings);
- the stems this port converted, and which of them is KB's own workload.

Result (2026-10-26):

- All 176 L1/L2 problems have a KB-layout id: 94 in `sol-level1` and 82 in `sol-level2`.
  `sol-level1` also holds a copy of L2's `001_fused_vision_multihead_attention_with_norms_backward`.
- **KB's `get_inputs()` is SOL's workload #0 in every case.** All 176 match their shapes, and
  all 176 use SOL's dtypes (bf16/fp16/fp32). KB passes SOL's Python scalars as 0-D tensors;
  the mapping compares the tensors that are not 0-D.
- KB has a result row for 110 of them. That is 76 `sol-level1` rows in
  `kernel_status_by_problem.csv`, which hold the init and best cycles, the speedup vs init and
  vs PyTorch eager, the model and the kernel path. It is also 34 `sol-level2` rows in
  `speedup_l2.csv`.
- With the 8 GiB add-on, this port has workload #0 itself (`__index0`, `__index08g`, or a
  min/med/max pick that is #0) for 159 problems: 89 L1 and 70 L2. 100 of them have a KB result
  row.
  - Of those 100, 73 pass extract + CUDA + export at their native dtype: 55 L1 and 18 L2. These
    are the pairs ready for an old-KB vs new-flow comparison.
  - 9 extract but miss a later gate, and 18 do not extract (value-dependent ops, or too large
    for host memory).
- For 17 problems, workload #0 is above 8 GiB, or the problem is refused at conversion.

## Through the flow (`kb_graph_opt`, firesim2 L40S)

`kb_graph_opt` on two converted problems at KB's own workload #0, on the firesim2 L40S (sm_89):

- one round, 1 rollout × 4 steps, `--max-variants 1`;
- the KernelBlaster checkout is `feat/multislot` plus its `feat/bf16`;
- `--torch-baseline auto`, with the torch python on firesim2. `--torch-source` is the native
  bf16 bench, so torch runs SOL's own reference in bfloat16; for the fp32 IR it is the fp32
  bench.

First with a scripted `stub:echo` LLM ($0): it answers every slot with its current source. This
checks the mechanics: profile, plan, KB runs, the hint decision, re-measurement, the final
model run and the torch comparison, all on one GPU UUID. Then with
`us.anthropic.claude-haiku-5-5`.

| problem (workload #0) | IR | LLM | seed → final (CUDA graph) | vs torch (best: CUDA graph) | vs torch eager | decision | calls | cost |
|---|---|---|---|---|---|---|---|---|
| L1/046 attention softmax + softcapping + dropout (4 ops) | fp32 | stub | 267 → 267 µs | 0.34× (91 µs) | 0.36× | fusion rejected | 6 | $0 |
| L2/035 ConvNeXtV2 block + GRN (29 ops) | fp32 | stub | 2392 → 2392 µs | 0.07× (166 µs) | – | – | – | $0 |
| L1/046 | fp32 | Haiku | 266 → 28.7 µs (9.3×) | **3.25×** (93 µs) | 3.29× | `fuse ewise_scalar→tanh` accepted | 13 | $0.22 |
| L1/046 | bf16 | stub | 223 → 223 µs | 0.40× (91 µs) | – | fusion rejected | 6 | $0 |
| L1/046 | bf16 | Haiku | 217 → 25.6 µs (8.5×) | **3.52×** (90 µs, torch bf16) | 4.36× | `fuse ewise_scalar_bf16→tanh_bf16` accepted | 16 | $0.30 |
| L2/035 | bf16 | Haiku | 2433 → 914 µs (2.7×) | 0.12× (106 µs, torch bf16) | 0.33× | `split matmul_tb_bf16` rejected | 13 | $0.47 |

- The total LLM spend is $1.09, recorded in `gpu_eval/work/llm_budget/ledger.jsonl` as
  who="sol-port". It includes about $0.10 for an L2/035 fp32 run that the 2026-10-26 host crash
  interrupted.
- KernelBlaster's earlier port reported, for the same workloads on an L40S, cycle-based
  numbers: L1/046 at 5.54× vs PyTorch eager (best 33,298 vs 184,407 cycles per SM, eager
  74.5 µs), and L2/035 at 1.45× vs eager (pt baseline 354 µs).
- In this flow torch eager is 112 µs for L1/046 (bf16) and about 300 µs for L2/035, measured
  in-session on the same GPU against our CUDA-graph replay. The KB numbers are a different
  method (NCU cycles), not a paired measurement.
- L2/035 stays far behind torch in one round. Its conv is MB's generic `convnd` seed, and
  `matmul_tb` takes 1.5 ms of the 2.1 ms seed. A longer search or the cuBLAS-class matmul slot
  techniques are needed; this is a smoke run, not a tuning run.


## Not done / next

- **Per-tensor storage dtypes.** An IR has one float storage dtype. A bf16 problem's fp32
  sections, fp32 outputs and integer indices therefore become bf16, and this accounts for most
  of the SOL-tolerance misses at bf16 (another branch is adding this).
- **Value-dependent shapes.** MoE dispatch (`nonzero`), varlen and paged attention (`.item()`,
  `tolist()` on indptr / page tables), bincount and boolean masks would need dynamic shapes,
  or dense rewrites that change the problem.
- **fp8 / fp4** storage, block scales and quantise ops (all of Quant, and FlashInfer's fp8
  MoE).
- **Extractor memory.** The peak is about 40-50× the workload's io, which skips most workloads
  above about 1.2 GiB at a 60 GB cap. Streaming the golden and not keeping every traced value
  would admit the 8 GiB workloads.
- **The hand-CUDA walk keeps every intermediate on the GPU.** It could free tensors after
  their last use, as the export's driver already does.
- **The SOL reference runs on the CPU here** (bf16 on the CPU). SOL runs it on the GPU, and
  the gate can do that too (`--ref-device cuda` on a GPU host).
- The smoke runs are single runs with one round. Tuning (more rounds and steps, the matmul
  techniques for L2/035) and the old-KB vs new-flow comparison over the 73 workload-#0 stems
  that pass every gate and have a KB result are the next experiments: `kb_map.csv` has the
  pairs.
