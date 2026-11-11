# gpu_eval — ModelBlaster IRs on CUDA

Runs an extracted IR (`graph.json` + `weights.npz` + `io.npz`) on an NVIDIA GPU
one kernel per op, through the same per-op C ABI (`KernelSpec.signature`) the
generated C uses. The same signature is KernelBlaster's
`launch_gpu_implementation`, so any op can be turned into a KernelBlaster CUDA
problem.

## harness/

| File | What it does |
|---|---|
| `walk_torch.py` | walks an IR with PyTorch ops and checks it against the `io.npz` golden (`--device cpu` for the reference gate) |
| `walk_cuda.py` | walks an IR on CUDA kernels via ctypes; checks the golden and every kernel on the reference's own inputs |
| `cuda/mb_cuda_kernels.cu` | hand-written CUDA kernel per op, with MB's C signatures |
| `cuda/mb_cuda_kernels_f16.cu` | the `_f16` kernels (fp16 storage, fp32 math), **generated** by `f16_kernels.py` from the fp32 file + `cuda/mb_cuda_kernels_f16_hand.cu`; see `docs/fp16_cuda.md` |
| `ref2cuda.py` | ports each MB reference C kernel to a single-thread CUDA seed |
| `kb_targets.py`, `kb_bridge.py` | one KernelBlaster problem per (op type) in an IR set, built and checked with KB's own split + `cuda_env` CMake |
| `mbir.py`, `mbabi.py` | IR loading/comparison; ctypes binding over `pipeline/kernel_abi.py` |
| `run_level3_gpu.sh` | the GPU gates (CUDA walk + KB problems) over a directory of IRs |
| `run_all.sh` | the original feasibility suite (walks, seeds, KB problems with NCU) |

## tools/

| File | What it does |
|---|---|
| `env.sh` | source to get a shell that runs the pipeline (see its header for the variables) |
| `extract_many.sh` | `extract_graph` over many KernelBench-style files in parallel |
| `host_verify.sh`, `host_verify_all.sh` | builds an IR's C model for x86 (`harness_linux`) and runs the verify |
| `dump_specs.py` | writes `specs.json` (kernel signatures + reference C) for the harness |
| `sol_upstream.py` | converts SOL-ExecBench problems from the original repo (definition.json + workload.jsonl) to bench files at native workload shapes ([docs/sol_execbench.md](../docs/sol_execbench.md)) |
| `sol_upstream_gates.py` | the SOL port's gates: extract, shapes, CPU walk vs MB golden and vs SOL's reference + tolerance, CUDA walk and multi-slot export (memory-capped; GPU gates on an ssh host) |
| `sol_kb_map.py` | maps the SOL problems to KernelBlaster's earlier `sol-level1/2` port and its results |
| `sol_adapt.py` | the earlier SOL port (KernelBlaster's copy): fp32, input dims shrunk toward a memory target |
| `sol_survey.py`, `sol_aten_census.py` | which problems trace, and which aten ops they use |
| `sol_gates.py`, `level3_table.py`, `sol_walk_summary.py` | per-problem gate tables from the run outputs |
| `fp64_tiebreak.py` | for an IR that misses the fp32 bound: walks it in fp64 to tell a wrong IR from an ill-conditioned problem |

## Running

```bash
source gpu_eval/tools/env.sh            # MB_ROOT defaults to this checkout
MB_TARGET_MB=16 JOBS=6 KEEP_NPZ=1 bash gpu_eval/tools/extract_many.sh IR_ROOT run problems/*.py
(cd gpu_eval/harness && python walk_torch.py IR_ROOT/*/ --device cpu --json walk.json)
bash gpu_eval/tools/host_verify_all.sh IR_ROOT HOST_OUT 3

# on the GPU box: a torch+CUDA python, nvcc on PATH
PY=python bash gpu_eval/harness/run_level3_gpu.sh IR_ROOT GPU_OUT 89   # sm_89 = L40S
```

`run_level3_gpu.sh` uses `third_party/KernelBlaster` unless `KB` is set, and
generates `specs.json` unless `SPECS` is set.

fp16: extract with `MB_QUANT=fp16`; the walkers, `host_verify.sh` and
`run_level3_gpu.sh` follow the IR's `quant` (fp16 bounds from
`pipeline/cuda_dtypes.py`). After editing `cuda/mb_cuda_kernels.cu`, run
`python gpu_eval/harness/f16_kernels.py` (a test fails while the generated file
is stale).
