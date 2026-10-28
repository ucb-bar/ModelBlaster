# Per-tensor dtypes (`--quant native`)

`extract_graph --quant fp32|fp16|bf16` gives an IR one float storage dtype:
every stored tensor (inputs, intermediates, folded constants, outputs) is
rounded to it, and integer inputs are packed into the float io buffer.
`--quant native` instead gives **every IR tensor its own dtype, the one the
source model uses**. It is what the SOL-ExecBench gates run
(`sol_upstream_gates.py run --quant native`): SOL's references are mixed
precision (bf16 tensors, fp32 math and some fp32 outputs and constants,
int64 / int32 indices), and SOL's check compares against exactly those
dtypes. The single-dtype modes are unchanged and stay the presets; the
KernelBench defaults (`--quant fp32`) are byte-identical. Native IRs are
CUDA-only, like bf16 ([bf16_cuda.md](bf16_cuda.md)).

## Contract

| tensor | dtype in a native IR (graph.json `dtype`) |
|---|---|
| graph inputs and outputs | the source's: `f32` / `bf16` / `f16`, or a real integer `i64` / `i32` / `b8` (bool, 1 byte) / `u8` |
| tensors of a traced node (intermediates) | the node's float dtype; an integer-valued one (indices, masks, `position_ids + 1`) is an exact `f32` inside the graph, as in an fp32 IR (exact below 2^24) |
| argmax / argmin outputs | `i64` (their kernels write `int64_t`), as before |
| tensors the lowering creates (no traced node) | an alias' or a data movement's keeps its input's dtype; a computed one is `f32` (a torch op on bf16 computes in fp32 and rounds once, at its output) |
| folded float constants (RoPE `inv_freq`, position tables) | their traced dtype; `weights.npz` holds them at it, named in graph.json `weight_dtypes` |
| op weights (`weight` / `bias` / `gamma_key` / `beta_key`) | the storage dtype of the op using them |
| integer constants (gather indices) | their array (`int32`), as before |

- **Ops have one storage dtype.** An op whose float tensors are all bf16
  (fp16) is the `_bf16` (`_f16`) op of its kind, with the existing contract:
  storage in that type, fp32 math and accumulation, one rounding per store,
  bf16 tensor cores allowed on sm_80+. Every other op (fp32 tensors, an
  integer-valued operand, or a mix) is the fp32 op.
- **Casts are explicit ops.** Where a tensor's dtype differs from the dtype
  its op reads or writes, extraction inserts `cast_<src>_to_<dst>`: a bf16 /
  fp16 input of an fp32 op is widened (exactly), an fp32 op's result is
  written to an f32 temporary and cast once to its output's dtype, an integer
  graph input is read through one f32 copy, and an integer output is cast
  from f32 (truncation toward zero; to `b8`: `!= 0`). A traced `.to()` /
  `.float()` / `.bfloat16()` (which lowers to a view) becomes a cast. Float to
  float rounds once (round-to-nearest-even). The casts are the 16 pairs in
  `cuda_dtypes.CASTS` (every dtype to and from f32, and between f16 and bf16);
  a pair outside them goes through f32. Specs: `reference_kernels._cast_spec`
  (excluded from the `_f16` / `_bf16` derivation); CUDA:
  `gpu_eval/harness/cuda/mb_cuda_kernels_cast.cu`.
- **Op classes**: casts are `elementwise`. **Fusion** (`kb_proposals.fuse_pair`)
  never fuses a cast or two stages of different dtypes: a fused seed calls its
  stages from one hand-kernel library.
- `cuda_dtypes.ir_dtype` of a native IR is its narrowest float storage (bf16,
  else fp16, else fp32): the bound a check through all of its tensors needs.
  `tensor_dtype`, `weight_dtype`, `cone_dtypes` and `np_storage` give each
  tensor's own.

## Extraction

1. The bench's `NativeModel` on `get_native_inputs()` (else `Model` on
   `get_inputs()`) is traced at its own dtypes (`concrete_trace` +
   `ShapeProp`); every node's dtype is recorded. The SOL port
   (`gpu_eval/tools/sol_upstream.py`) generates both: the original reference
   `run` and SOL's inputs at the definition's dtypes. The creation-op patch
   keeps dtypes in this mode.
2. The traced graph becomes its **fp32 twin** (`_fp32_twin`): bf16 / fp16 /
   fp64 dtype arguments become float32, `.bfloat16()` / `.half()` become
   `.float()`, float parameters, buffers and constants are widened exactly.
   This is the SOL port's fp32 source rewrite done on the graph, so the
   lowering, shapes and folding are the fp32 extraction's, and values that
   depend on the dtype at trace time (`torch.finfo(x.dtype)`, data-dependent
   branches) are the native model's.
3. After lowering, `_native_storage` assigns every tensor's dtype,
   `_native_views` turns dtype-changing aliases into casts (before output
   views are materialised), and `_native_ops` gives each op its storage dtype
   and inserts the other casts.
4. **io.npz** holds every input / output at its own dtype as `in.<name>` /
   `out.<name>` (int64 arrays for int64 tensors, float32 with bf16-exact
   values for bf16), next to the flat float32 `input` / `output` (the same
   values; integers exact below 2^24) for readers that ignore dtypes.
5. **Golden**: the fp32 twin on the inputs (bf16 values exact), each output
   stored at its dtype. A dtype that MB cannot store is refused: fp8 / fp4
   (`no fp8/fp4 storage`, the class the SOL gates already use); float64 is
   stored f32.

## Walkers, export, driver

- `mbir` reads the per-tensor inputs (integers at their dtype); `walk_torch`
  rounds each op to its own kind's dtype and runs the casts (one rounding,
  truncation); `walk_cuda` puts every tensor in a buffer of its dtype
  (int64 / int32 / bool / uint8 included), narrows each weight by its
  `weight_dtype`, and resolves each entry in whichever `--hand` library defines
  it (comma-separated: `lib_hand.so`, `_f16`, `_bf16`, `lib_hand_cast.so`).
- **kb_export** (`mb-kb-problem/1`):
  - buffers carry each tensor's dtype (`f32`, `f16`, `bf16`, `i64`, `i32`,
    `b8`, `u8`); `data.bin` holds each at its own element size (bf16 as its
    2-byte patterns, `b8` as one byte 0 / 1);
  - every slot states its `dtype` (a cast's is its destination's float dtype,
    fp32 for an integer destination) and its isolated bound `tolerance`;
  - the manifest `dtype` is `mixed` when the slots have more than one, with the
    narrowest dtype's bounds as `tolerance`; otherwise exactly as before;
  - slot seeds come from the hand library of each slot's dtype (plus the cast
    file); the shared head is the union of their preprocessor lines (one
    library: unchanged);
  - each output buffer has its own `output_rel` / `output_abs`: its dtype's, or a
    narrower one it was computed through (`cuda_dtypes.cone_dtypes`: an fp32
    output of bf16 intermediates carries their rounding);
  - OpenCL refuses mixed partitions and casts (no bf16, one storage dtype per seed set).
- **driver.cpp** gains per-buffer tables `B_I32`, `B_BYTE` (1-byte elements),
  `B_OUT_REL` / `B_OUT_ABS` and per-op `OP_SLOT_REL` / `OP_SLOT_ABS`; the
  checks widen i32 / bool / u8 buffers numerically. For a single-dtype
  problem every entry equals the old constants. The one-process driver and
  the persistent host (`-DMB_HOST`) build the same tables.
- **KernelBlaster** (KB `docs/multislot_optimization.md`, "Storage dtypes"):
  each slot's dtype keys its records, impl-store lookups, knowledge, roofline
  peaks and bandit arm (fp32 slots keep their old keys); a mixed problem's
  system prompt has every present dtype's rule with its own bound and a rule
  on keeping each signature's types.
- `kb_ingest` / `kb_model_run` store the slot's dtype on cache entries, never
  `mixed`. Host C (`generate_skeleton`) refuses native IRs.

## Tolerances

| check | bound |
|---|---|
| isolated op (walk_cuda, driver `MB_SLOT_CHECK`) | the op's storage dtype's `slot_rel` / `slot_abs` (fp32 1e-3; bf16 1.6e-2) |
| partition outputs (driver `MB_CHECK`) | per output: the narrowest storage in its cone's `output_*` |
| golden (walkers vs io.npz) | the IR's narrowest storage's `golden_*` |
| SOL scoring | SOL's own rule (`sol_check`), unchanged |

## SOL gates

`sol_upstream_gates.py run --quant native` extracts with per-tensor dtypes
(`--quant preset` is the old one-dtype-per-problem choice). `io_rounding` and
`out_dtype_mismatch` now read the IR's input / output dtypes, so a native IR
reports neither. `summary --native` keeps native records. The GPU host builds
`lib_hand_cast.so` too; `remote --kb-flock` holds KernelBlaster's per-GPU
flock (`/dev/shm/kernelblaster_gpu_<uuid>.lock`) around each IR's GPU steps
(`MB_GPU_FLOCK`).

## Validation (2026-10-27)

**SOL-ExecBench**, the 222 native-bf16 workloads with cpu results, before
(`--quant bf16`, `run/g2_bf16`) and after (`--quant native`); scored with SOL's
own rule (`sol_ok`) against SOL's reference on the CPU. Classes are the
failure causes of the bf16 run.

| class (bf16 run) | n | sol_ok bf16 | sol_ok native |
|---|---|---|---|
| fp32 output stored bf16 | 28 | 0 | 25 |
| bf16 storage of values the reference keeps fp32 | 29 | 0 | 28 |
| integer input corrupted by the bf16 io | 12 | 0 | 11 |
| fp32 input rounded | 11 | 0 | 11 |
| unattainable tolerance | 11 | 0 | 10 |
| reference outlier | 1 | 0 | 1 |
| passed already | 130 | 129 | 128 |
| **all** | **222** | **129** | **214** |

The 8 misses: 3 workloads quantise to fp8 inside `run()` (refused, as the fp32
lowering already refuses them); the 3 group-limited top-k routings return
`topk(sorted=False)` indices in an order no sorted answer reproduces (the walk
equals the float64 reference exactly); one tolerance no correctly rounded
answer meets; one extraction ran out of its 40 GB cap (the native trace holds
the bf16 trace and its fp32 twin).

GPU gates (firesim2 L40S): hand-CUDA walk 214/222 (bf16: 196), multi-slot
export with the persistent host 215 of the 216 exported (bf16: 201 of 201,
21 more skipped by the then 1 GiB data cap); the misses are out-of-memory on the
shared GPU. The export's end-to-end time (CUDA graph) is unchanged at the
median (1.00×, geomean 1.07× over 201 workloads); 23 are more than 1.2×
slower, up to 3.5×: where the reference computes in fp32 the native IR does
too, with fp32 intermediates and outputs that the bf16 IR stored (wrongly) in
half the bytes.

**KernelBench** (L1 19, 23, 1; L2 12, 21; fp32 / fp16 / bf16): graph.json is
byte-identical, data.bin, slots and walker files are identical, the driver
differs only by the new tables (every entry the old constant); build-check on
the L40S passes with both trees, one-process and host, at the same times.

A KernelBlaster smoke run on a mixed problem (SOL L2/046, bf16 matmuls and
gathers with fp32 math and casts; Haiku, 1 rollout × 2 steps plus a fusion
variant) passed its checks: model 10.2 → 2.2 ms (4.6×).

## Limitations

- Integer intermediates are exact f32: integer values past 2^24 inside the
  graph are not exact (graph inputs and outputs are).
- No float64 storage (stored f32), no fp8 / fp4 (refused).
- An argmax that is a graph output keeps `i64` even if the source asks for `i32`.
- Extraction holds the native trace and its fp32 twin: more host memory than
  the single-dtype modes.
- Casts add kernels (and their memory traffic) where the source changes
  dtype; the torch baseline (`torch_baseline`) still runs a native IR's module
  in fp32.
