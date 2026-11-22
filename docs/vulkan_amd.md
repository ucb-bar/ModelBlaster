# Vulkan backend: ModelBlaster IRs on AMD GPUs without ROCm (Steam Deck)

`gpu_eval/harness/vulkan` is a GPU backend in the shape of the CUDA one
(`gpu_eval/harness/cuda`): a shared library whose `kernel_<op>` entry points
have exactly the op's `KernelSpec.signature`, and a walker
(`walk_vulkan.py`) that binds every IR op to them through
`pipeline/kernel_abi.py` and applies the CUDA walker's gates. It targets
Vulkan 1.2 compute, so it runs on RADV (AMD RDNA, e.g. the Steam Deck's Van
Gogh APU, gfx1033), and on any other Vulkan GPU (it was developed on an
Intel Meteor Lake iGPU as well).

## Why Vulkan (and not HIP or OpenCL) on the Deck

Probed on the Deck (SteamOS 3.6.24, Mesa 24.1, kernel amdgpu), 2026-11-21:

| path | probe | result |
|---|---|---|
| Vulkan (RADV) | `libmb_vk.so` init + relu + 256^3 matmul | **works**: `AMD Radeon Graphics (RADV VANGOGH)`, Vulkan 1.3.278; buffer-device-address, int64, fp16, int8, fp64 all present |
| OpenCL | `clinfo` | **0 platforms**: SteamOS ships the ICD loader only (no `/etc/OpenCL/vendors`, no rusticl / clover). Needs a Mesa with rusticl built or containerised under `$HOME` first |
| HIP / ROCm | `/dev/kfd` present; KFD topology `gfx_target_version 100303` | **not run**: gfx1033 is not a ROCm target (`HSA_OVERRIDE_GFX_VERSION=10.3.0` runs gfx1030 code on it; HSA asserts and GPU hangs are the documented failure modes), ROCm userspace is a 5-10+ GB download, and an amdgpu reset would take down gamescope and the live ground-station kiosk |

Vulkan was the only path that ran a kernel on the stock OS, needs nothing
installed system-wide (`libvulkan` is `dlopen`ed; SPIR-V is embedded in the
library), and cannot wedge the GPU in the ways ROCm-on-unsupported-APU does
(a hung submit times out: `MB_VK_TIMEOUT_S`, default 120 s, and is reported).

## Runtime (`mb_vk_runtime.{h,cpp}`)

- **Pointers.** `mb_vk_alloc` returns host-mapped, GPU-visible memory
  (device-local + host-visible first, else system memory; `MB_VK_MEM`). A
  kernel argument is a host address inside such a buffer; the launcher turns it
  into the GPU address of that byte (`VK_KHR_buffer_device_address`) and
  passes it in push constants, and shaders dereference it with
  `GL_EXT_buffer_reference`. So views and offsets are pointer arithmetic, as in
  MB's C ABI, the CUDA device pointers and the OpenCL harness's SVM pointers,
  and the walker reads and writes tensors as numpy arrays over the same memory.
- **Arena.** `mb_vk_mark` / `mb_vk_release` (stack discipline, 256 MiB chunks);
  launcher scratch (`mb_vk_scratch`: split-K partials, im2col) is outside it.
- **Ordering.** Launches are recorded into one command buffer, with a
  compute-to-compute barrier between them (CUDA default-stream order), and
  submitted by `mb_vk_sync` (or every `MB_VK_FLUSH` = 4096 launches).
- **Timing.** `mb_vk_ts()` writes a GPU timestamp (per-op times in the walker).
- **Captured graphs** (the CUDA-graph analogue): launches between
  `mb_vk_capture_begin` / `_end` are recorded, not run; `mb_vk_replay(g)`
  resubmits them. Push constants hold GPU addresses, so a replay reads whatever
  the host last wrote into the same buffers: no per-op Python or launch work.
- Low queue priority (a compositor sharing the GPU is not starved).
- **Specialization constants** (`mb_vk_pipeline_spec(name, consts, n)`): one
  pipeline per value set, cached. Used for the skinny GEMM's row count, the
  elementwise kernels' opcode and vec4 width.
- **Packed weights** (`mb_vk_set_packed(w, p, f16)`): registers a copy `p` of
  the `[N, K]` weight `w` in the skinny GEMM's layout (fp32, or fp16); a linear
  of <= 64 rows whose weight has one runs the skinny GEMM on it. A
  registration is dropped when `mb_vk_release` frees either buffer.
- `VK_KHR_shader_integer_dot_product` is enabled when the device has it
  (`mb_vk_has_int_dot`; `int_dot=` in `mb_vk_info`).

## Kernels (`mb_vk_kernels.cpp`, `shaders/*.comp`)

fp32, the CUDA hand-kernel set's semantics (opcode tables, layouts, output-size
conventions): `ewise1` (all opcodes) and the standalone activations
(relu / relu6 / sigmoid / silu / swish / elu / tanh / gelu / gelu_exact /
leaky_relu / hardtanh / softplus / selu / hardsigmoid / softsign / log),
`ewise2` (6-d broadcast), `ewise_scalar`, `add` / `mul` / `mul_scalar`,
`reduce` and `{sum,mean,max,min}_dim`, `softmax` / `log_softmax` /
`softmax_dim`, `layer_norm`, `rms_norm` / `mean_abs_norm` / `l2_norm`,
`group_norm`, `batchnorm2d`, `cumsum` / `cumprod`, `linear`,
`matmul{,_ta,_tb,_tatb}`, `bmm`, `sdpa`, `conv2d` / `conv2d_grouped` /
`conv2d_dw`, `maxpool2d{,_ex}` / `avgpool2d` / `adaptive_avg_pool2d`,
`upsample_nearest`, `strided_copy`, `gather`, `index_select`, `gather_dim`,
`catn`, `cat{2,3,4}_c1`, `triu` / `tril` / `diag_matmul` / `flip`.

- Row reductions (softmax, norms, reduce) use one 256-wide workgroup per row
  with an fp32 tree.
- Every dense product is one tiled GEMM (`gemm_body.glsl`): 256 invocations per
  workgroup, 5 tile shapes (16x256, 32x128, 64x64, 64x128, 128x128) picked by M,
  N and K (`MB_VK_GEMM` forces one), vec4 loads when the operand is 16-byte
  aligned, a register-prefetched K loop (32 per step, the loop over a tile
  unrolled by 8 so the LDS reads are issued ahead), and split-K when the
  output has few tiles over a long K (partials summed by `splitk_reduce`) --
  except a linear of <= 64 rows on a packed weight, which takes the skinny GEMM.
- **Skinny GEMM** (`skinny_body.glsl`; `skinny.comp`, `skinny_h.comp` for fp16
  weights): the prefix's 31-row and the action expert's 50-row linears. One
  invocation per output column holding all M rows in registers (M is a
  specialization constant); the activation's k-slice is the same for the
  whole wave, so it is read with scalar loads (RDNA `s_load`, no VGPRs, no
  LDS) and enters the FMAs as a scalar operand; each weight is loaded once,
  one k-step (32) ahead, and used M times. The weight is pre-packed
  (`walk_vulkan.pack_weight`: per 64-column block and k-quad, the 64 columns'
  4 values contiguous), so one wave-wide vec4 load is one contiguous 1 KiB:
  at M = 1 it streams weights at 82 GB/s (the Deck's LPDDR5 peak is ~88). Two
  waves per workgroup take alternate k-steps (summed through LDS), and K is
  split over workgroups until ~64 waves run. `MB_VK_SKINNY=0` turns it off;
  `MB_VK_SKINNY_KW` / `_WG` / `_KU` are the tuning knobs (defaults measured on
  the Deck). `walk_vulkan.Weights` packs every such weight by default
  (`MB_VK_PACK=0`: don't); `Weights(f16=True)` / `smolvla_vulkan.py
  --f16-weights` packs them as fp16 (fp32 activations and accumulation:
  half the weight bytes; an approximation, off by default and outside the
  fp32 gates).
- Elementwise / data movement (`ewise1`, `ewise2`, `ewise_scalar`,
  `strided_copy`, `cat_slab`): the opcode is a specialization constant (a
  run-time `switch` compiled to a chain of scalar compares per element), a
  vec4 variant (one vec4 per invocation: twice the bytes in flight; a
  one-float-per-invocation stream reached only ~40 GB/s) when sizes and
  16-byte alignment allow, and `ewise2` / `strided_copy` coalesce their 6-d
  iteration space on the host (drop size-1 dims, merge contiguous ones) so
  the shader divides only by the active dims.
- A patchify `conv2d` (stride == kernel, no padding, N == 1: the ViT patch
  embedding) is im2col (one `strided_copy`) + GEMM; other convs are direct.

int8, the SmolVLA `_s8` family, **bit-exact with the reference C**
(`KernelSpec.reference_impl`): `linear_s8` (int32 accumulation, Q0.31 rounding
multiply, rounding shift), `matmul_s8`, `softmax_s8`, `layernorm_s8`,
`rmsnorm_s8`, `mul_s8`, `gelu_s8`, `sin_s8`, `cos_s8`. Float steps are done in
the reference's precision and order: fp32 divisions through fp64 (the
correctly rounded result), C `roundf` / `round` implemented exactly, the norms'
fp64 arithmetic in the reference's summation order. An op whose output is a
function of one int8 input (gelu / sin / cos) is a 256-entry table that the
launcher fills with the reference's own expression and libm; `softmax_s8`'s
`expf` of the 256 possible differences comes from the same libm.

`linear_s8` and `matmul_s8` (with `transpose_b`) run on packed int8 dot
products when the device has `VK_KHR_shader_integer_dot_product`, K % 4 == 0
and the operands are 4-byte aligned (`dot_s8_body.glsl`: a 32x64 output tile
per workgroup, K through LDS 64 bytes at a time, one `dotPacked4x8EXT` per 4
multiply-adds; `MB_VK_INT_DOT=0` forces the scalar kernels). The input /
filter offsets are folded in exactly, `sum (x+io)(w+fo) = sum xw + fo sum x +
io sum w + K io fo` in int32 with the reference accumulator's wraparound, so
the results stay bit-exact. Deck: `linear_s8` 50x960x320 2245 -> 90 us,
31x960x2560 22.9 -> 0.21 ms, 256x768x768 40.5 -> 0.25 ms (1.2 Tops).

## Gates (`walk_vulkan.py`)

    python gpu_eval/tools/dump_specs.py specs.json
    python gpu_eval/harness/vulkan/build.py           # -> vulkan/build/libmb_vk.so
    cd gpu_eval/harness && python walk_vulkan.py IR_DIR... --specs specs.json [--json out.json]

- fp32 IRs: the io.npz golden (cuda_dtypes' fp32 bound), and every kernel in
  isolation on walk_torch's own inputs, against walk_torch run on the CPU (no
  GPU torch is needed: the Deck has none), with the CUDA walker's per-slot
  bound. Then wall time per walk, the captured graph's replay time and output,
  and per-op-type GPU time from timestamps.
- int8 IRs: the output must equal the io.npz golden, and every kernel's output
  must equal the reference C on the same inputs (built with the host C
  compiler, or `--ref-lib` for a host without one, such as SteamOS).

`libmb_vk.so` links libstdc++ statically and needs glibc >= 2.38, so a library
built on a current Linux host runs on SteamOS (glibc 2.39): no compiler is
needed on the Deck.

## SmolVLA

`models/smolvla_split.py` restates lerobot 0.4.4's KV-cached
`VLAFlowMatching.sample_actions` as three FX-traceable graphs, which the fp32
FX extractor lowers to the generic ops above (`extract_graph --model
smolvla_split --quant fp32`, `MB_SMOLVLA_PART=vision|prefix|step`,
`MB_SMOLVLA_REAL=1` for the checkpoint):

| graph | inputs -> output | ops (kernel launches) | weights |
|---|---|---|---|
| vision | pixels [1,3,256,256] -> image tokens [1,16,960] | 288 (236) | 374 MB |
| prefix | image tokens, prompt embeddings [1,L,960], state [1,32] -> KV cache [32,P,320] | 973 (774) | 568 MB |
| step | KV cache, x_t [1,50,32], time embedding [1,720] -> v_t [1,50,32] | 990 (749) | 384 MB |

The host keeps what has no tensor compute: the token-embedding gather, the
720-wide time sinusoid and the Euler update (`smolvla_vulkan.py`). Masks are
additive -inf constants, RoPE tables and the SigLIP position rows for the image
size are constants, grouped-query attention puts each KV head's 3 query heads
into the GEMM rows. The prompt is the unpadded token sequence (lerobot pads to
48 with masked tokens; dropping them is exact, since padded tokens are never
attended and positions are cumulative over real tokens).

Results: see the tables below (Deck, 1 camera at 256 px, 5 flow steps, the
`smolvla_bench.py` drone configuration).

Measured on the Deck, 2026-11-21 (RADV VANGOGH; CPU baseline = `smolvla_bench.py
--cams 1 --img 256 --steps 5 --dtype fp32 --threads 3`, lerobot 0.4.4 / torch 2.10;
GPU = `smolvla_vulkan.py --reps 20`, median of 20 chunks, replayed graphs):

| component | PyTorch CPU (3 threads) | Vulkan, first version | Vulkan, now (fp32) | now, `--f16-weights` |
|---|---|---|---|---|
| vision (SigLIP + connector) | 471 ms (`embed_prefix`) | 84.6 ms | 78.2 ms | 78.1 ms |
| VLM prefix (KV cache) | ~300 ms | 29.7 ms | 18.2 ms | 15.7 ms |
| denoise, 5 flow steps | 818 ms | 169.6 ms (33.9 / step) | 107.1 ms (21.4 / step) | 94.0 ms (18.8 / step) |
| **chunk of 50 actions** | **1597 ms** | **283.9 ms** (5.6x) | **203.5 ms** (7.8x) | **187.7 ms** (8.5x) |

Action chunk vs lerobot's own `sample_actions` (same observation, same noise):
fp32 cosine 0.999999999999606, max |err| 1.07e-6 (|actions| <= 1.01; the
first version: 0.99999999999953 / 1.3e-6); with fp16 weights cosine
0.99999996, max |err| 3.0e-4. Per graph, the walker's fp32 gates pass with the
skinny GEMM and the new elementwise kernels: golden rel. error 9.5e-7 / 4.8e-7
/ 5.5e-7 (vision / prefix / step), every kernel within the per-slot bound
(worst 1.26e-6), int8 proxies bit-exact. Host suite: 1134 passed (29 new),
the 5 known KernelBlaster-submodule failures, 97 skipped.

Where the 203.5 ms go (fp32; by dropping op kinds from the replayed graphs):

| | vision | prefix | 5 steps | chunk | first version |
|---|---|---|---|---|---|
| linears | 59.2 | 14.0 | 78.7 | 151.9 ms | ~214 ms |
| bmm (attention) | 6.9 | 0.6 | 5.5 | 12.3 ms | |
| everything else (small ops: ~600 in the prefix and in each step) | 12.2 | 3.5 | 22.9 | 39.4 ms | 70 ms (incl. bmm) |

What changed, measured one at a time on the Deck (chunk, median of 20):
skinny GEMM on packed weights 283 -> 231 ms (prefix linears 1.6x, expert
linears 1.6x); vec4 elementwise 231 -> 211 ms; opcode as a specialization
constant 211 -> 205 ms; tiled-GEMM k-loop unroll 205 -> 203.5 ms. Tried and
dropped: two columns per invocation in the skinny GEMM (slower), waves on
different column blocks instead of alternate k-steps (no gain), wave32 for every compute shader (`RADV_PERFTEST=cswave32`: 2% slower),
a compute-only barrier between dispatches (no change: a barrier costs ~2.5 us,
the kernel time dominates).

Linear microbenchmarks (Deck, 4-32 launches over distinct weights per replay,
warm clocks; includes the split-K reduce; weight GB/s = weight bytes / time):

| M x K x N (count per chunk) | tiled GEMM | skinny, fp32 | skinny, fp16 weights |
|---|---|---|---|
| 31x960x2560 (30, prefix MLP) | 443 us, 344 GFLOPS | 302 us, 505 GFLOPS | 292 us |
| 31x2560x960 (15, prefix MLP) | 709 us, 215 | 291 us, 523 | 280 us |
| 31x960x960 (30, prefix attention) | 184 us, 311 | 123 us, 465 | 116 us |
| 50x720x2048 (160, expert MLP) | 436 us, 338 | 321 us, 460 | 302 us |
| 50x2048x720 (80, expert MLP) | 462 us, 319 | 312 us, 473 | 302 us |
| 50x720x960 (80, expert attention) | 238 us, 290 | 192 us, 361 | 133 us |
| 16x12288x960 (1, connector) | 1505 us, 251 (31 GB/s) | 907 us, 416 (52 GB/s) | 720 us (33 GB/s at 2 B) |

At M = 1 the skinny GEMM moves weights at 82 GB/s, so at 31-50 rows it is
compute-bound: ~460-520 GFLOPS against the 1.6 TFLOPS fp32 peak at 1.6 GHz
(the GPU clock sits at 1.0-1.6 GHz under the Deck's 15 W APU power cap), and
fp16 weights save only 12-14% end to end (prefix and steps).

## int8 SmolVLA (int8 IRs with fp32 islands)

`extract_graph --model smolvla_split --quant int8` (the model's
`get_int8_mixed_policy()`; `--int8-mixed POLICY` for any model) lowers each
graph with the fp32 FX path -- so the `get_attr` constants (additive -inf
masks, RoPE tables, SigLIP position rows) are fp32 constant tensors and
`expand` is a zero-stride `strided_copy`, exactly as in the fp32 IR -- and
then `pipeline/int8_mixed.py` moves the linears to int8 (`quant: int8_mixed`:
f32 and i8 tensors side by side):

- `quantize_rows_s8`: the linear's fp32 input, times a per-column SmoothQuant
  factor (`col_inv`, a constant), quantized **per row at run time** (row scale
  = max |x'| / 127, round half away, +-127). q / k / v (and gate / up) share
  one quantized tensor.
- `linear_dq_s8`: int8 x int8 (weights per output channel, `W * sm`: the
  smoothing moves into the weight exactly), int32 accumulation, **fp32
  output** `(float)((double)acc * row_scale[m] * w_scale[n]) + bias`. Nothing
  downstream sees an int8 grid: residuals, RoPE, attention, softmax (so the
  -inf masks keep exp(-inf) = 0 exactly) and the norms are the fp32 ops.
- Calibration (MB's rule, `--num-calibration N`, `MB_SMOLVLA_CALIB` = a
  directory of observations) sets the smoothing factors: per-column max |x|
  over the set, sm = max|x|^a / max|W|^(1-a). The step graph is calibrated on
  every flow step's (x_t, t) of each observation's own trajectory.
- The golden is the IR walked on the CPU (`int8_mixed.walk`: fp32 ops as
  `walk_torch`, the int8 kinds by CPU rules that are bit-exact with their
  reference C, tested). `walk_vulkan` gates an int8_mixed IR with both
  regimes: in isolation on that walk's inputs, every int8 kernel equal to the
  reference C and every fp32 kernel within the fp32 per-slot bound; the
  end-to-end output within 2% of the golden's range (or the IR's own
  int8-vs-fp32 error if larger: fp32 reassociation on the device flips int8
  roundings, and 16 layers amplify them).

Kernels: `quantize_rows_s8` (one 64-wide workgroup per row, `subgroupMax`,
vec4 variant); `linear_dq_s8` on the dot4 tile (vision, 256 rows) or, for
<= 64 rows on a weight registered with `mb_vk_set_packed(w, p, 2)`
(`walk_vulkan.pack_weight_s8`: per 64 columns, 16-byte k-chunks contiguous),
`skinny_s8.comp`: one column per invocation and every row in registers, the
activation's 16 bytes as uniform loads into `dotPacked4x8EXT`, K split over
workgroups with exact int32 partials (`linear_dq_s8_epi.comp`). All
bit-exact with the reference C on RADV and Intel (tests).

The quantization mix as first chosen on 13 held-out observations (the 193-observation validation
below revises it: SigLIP layer 0 now has outlier columns) (real HM01B0 drone
frames and synthetic images, 14-token prompts; calibration: 8 other frames
and prompts), action chunk vs lerobot's `sample_actions` (same observation and
noise; |actions| ~1), CPU reference walk of the IRs (bit-exact int8):

| mix (`models/smolvla_split.py`) | int8 linears (vision / prefix / step) | cosine mean (min) | max abs err mean (max) |
|---|---|---|---|
| fp32 | 0 / 0 / 0 | 1.0 (to 1e-12) | < 2e-6 |
| **default** `INT8_POLICY`: vision a = 0.8 with `fc2` fp32, prefix fp32, step a = 0.65 | 61 / 0 / 116 | **0.99950 (0.99901)** | **0.042 (0.074)** |
| every linear (`MB_SMOLVLA_INT8=all`; a = 0.8 / 0.65 / 0.65) | 73 / 108 / 116 | 0.99607 (0.96661) | 0.077 (0.147) |
| one graph int8 at a time (a = 0.5): vision / prefix / step | | 0.99834 / 0.99770 / 0.99969 (min 0.985 / 0.983 / 0.9991) | 0.054 / 0.075 / 0.026 |

What the sweeps showed: the prefix is the fragile graph (its KV cache feeds
every layer of every flow step; no single group or layer kept fp32 rescued
it, and its linears are ~7% of the fp32 chunk, so it stays fp32); in SigLIP,
`fc2` (input: the GELU output, a few huge channels) carries most of the
vision error; a = 0.5 -> 0.8 helps vision, hurts the prefix; the step graph is
robust (cosine 0.9997 alone). Per-tensor static activation scales (MB's
`extract_int8` scheme) were far worse than per-row + smoothing on these
graphs (fake-quant study on the fp32 IRs: vision cosine 0.969 vs 0.999).

Deck (same session, `smolvla_vulkan.py --reps 20`, observation = the fp32
table's; fp32 measured 213.0 ms in this session vs 203.5 ms earlier -- clock /
thermal state):

| | vision | prefix | 5 steps | chunk | vs lerobot: cosine, max abs err |
|---|---|---|---|---|---|
| fp32 | 79.0 | 20.6 | 113.4 (22.7 / step) | **213.0 ms** | 0.999999999999606, 1.1e-6 |
| int8, default mix | 56.4 | 20.2 | 65.9 (13.2 / step) | **142.5 ms** (1.50x) | 0.99978, 0.019 |
| int8, every linear | 48.2 | 10.7 | 65.1 (13.0 / step) | **123.8 ms** (1.72x) | 0.99864, 0.074 |

Walker gates on the Deck (`walk_vulkan.py --ref-lib`): every
`quantize_rows_s8` / `linear_dq_s8` equal to the reference C (0 mismatches),
every fp32 op within its slot bound, golden 0.9% / 0.9% of range (vision /
step, default mix; 2.2% / 1.5% / 0.9% for every-linear), replay equal. Step
graph per-op GPU time: `linear_dq_s8` 9.5 ms (fp32 linears ~15.7), the dot4
tile kernel alone 17.6 ms -- the skinny int8 path is what makes int8 pay at 50
rows; `quantize_rows_s8` ~10-18 us each.

### Optimizing the int8 chunk (Deck)

Measured one change at a time on the Deck (`smolvla_vulkan.py --reps 20`, median
chunk; differences under ~2 ms are within run-to-run clock noise; every change
keeps the int8 kernels bit-exact with the reference C and the fp32 ops within
their bound -- walker gates and tests re-run):

| change | chunk (default mix) |
|---|---|
| baseline (the table above, same session as the rows below) | 141.0 ms |
| `linear_dq_s8` dot4 tile 32x64 -> 128x64 for >= 128 rows (vision: 0.84-0.91 -> 1.18-1.35 Tops) | 133.2 (vision 56.4 -> 50.4) |
| barrier elision (walker data flow: an op whose inputs were complete at the last barrier skips its own; `MB_VK_ELIDE=0` off) | 131.8 |
| RMSNorm chain -> `rms_norm` with the norm weight folded into `quantize_rows_s8`'s column factor (6 launches -> 1), RoPE -> `rope_half` (9 -> 1): step 749 -> 476 launches, prefix 774 -> 402 | 125.2 (step 12.3 -> 11.2) |
| attention permutes folded into `bmm_strided` (layout_fold; Vulkan launcher; a non-fitting operand keeps its copy, so grouped-query K / V still fold) + subgroup reductions in softmax / layer_norm / rms_norm | **119.2** (vision 46.3) |
| `--f16-weights` (the prefix's fp32 linears read fp16 weights) | 116.4 (prefix -2 ms; opt-in) |
| `MB_SMOLVLA_INT8=fast`: SigLIP `fc2` int8 too, its 256 largest input columns as an fp32 outlier linear (`outliers`), smoothing 0.5 there | **108.5** (vision 36.2) |

Final, same session: fp32 208.6 ms; **default int8 119.2 ms (1.75x fp32)**,
action chunk vs lerobot cosine 0.99968, max abs err 0.025 on this observation;
**fast 108.5 ms (1.92x)**, 0.99966 / 0.028. On the 13 held-out observations (CPU
reference walk, bit-exact int8): default cosine mean 0.99948 (min 0.99868), max
abs err mean 0.042 (max 0.080); fast 0.99921 (min 0.99682), 0.055 (0.119). The
fusions move fp32 values by rounding only (default before them: 0.99950 /
0.99901).

Tried and dropped: 128-bit LDS reads with register prefetch in the dot4 tile
(0.7-0.95 Tops: slower), 64 x 128 tiles, two columns per lane in `skinny_s8`
(register pressure: slower), a flash-style fused fp32 `attention` for SigLIP
(`fuse_attention`, kept as an option and tested: +8 ms on the vision graph --
160 live registers per invocation on RDNA2), more split-K waves for the
skinny path. Accuracy experiments that did not make the cut (13 observations):
the prefix in int8 with 8-96 fp32 outlier columns per linear (cosine mean
0.9978-0.9991, min 0.992-0.997), step outlier columns (+0.0001 cosine for ~190
more launches per step), SigLIP fc2 smoothing 0.95.

Where the time is now (walker timestamps, slower clocks than the replay; step
graph): `linear_dq_s8` ~66%, strided bmm ~10%, quantize ~7%, softmax ~5%. The
int8 GEMMs run at ~1.0-1.6 Tops (skinny, 31-50 rows) and ~1.2-1.35 Tops (tile,
256 rows) against an int8 dot4 peak of ~6.5 Tops at 1.6 GHz; at the Deck's
15 W cap the clock sits lower (1.04 GHz measured under this load). The cross-attention K / V
hoist and the launch fusions of round 3 (below) came next.

### Numerics validation on 193 held-out observations (2026-11-22)

The 13-observation numbers above were optimistic. `scripts/smolvla_int8/` holds the validation:
`make_heldout.py` builds a larger held-out set (prompts that tokenize to the IRs' 14 tokens; lerobot
`sample_actions` reference actions for each), `validate.py` runs int8 mixes on it with the CPU
reference walk (int8 bit-exact, fp32 ops as walk_torch) -- action metrics (`metrics`), per-linear
local error, accumulator and saturation headroom (`local`), keep-fp32 group attribution (`groups`),
extracted IRs as they are (`irs`) -- and `summarize.py` tabulates. 193 observations: the 13
before, the 11 riskybird frames not used for calibration (each also mirrored, darkened to 0.55 and
brightened to 1.4: 33 augmented), 120 HM01B0 frames of the PULP-DroNet v3 set (other drones, other
rooms, grayscale) and 16 synthetic scenes, each with its own state and noise. Actions in the
model's normalized space (|actions| ~1); per-joint = action dims 0-5.

| mix (Deck chunk, ms) | cosine mean | median | p5 | min | # < 0.999 | max abs err mean / p95 / max | per-joint max err, mean over obs (dims 0-5) |
|---|---|---|---|---|---|---|---|
| fp32 IRs (209.6) | 1.0 | 1.0 | 1.0 | 1.0 (1e-12) | 0 | < 2e-6 | -- |
| old default (119.5) | 0.99917 | 0.99963 | 0.99794 | 0.96716 | 27 | 0.053 / 0.107 / 0.368 | .032 .036 .037 .028 .025 .038 |
| old fast (109.2) | 0.99889 | 0.99953 | 0.99729 | 0.95419 | 41 | 0.061 / 0.134 / 0.431 | .037 .039 .042 .031 .028 .041 |
| **new default** (110.9) | **0.99951** | 0.99971 | 0.99857 | **0.99391** | 19 | 0.045 / 0.097 / 0.148 | .027 .030 .031 .022 .022 .031 |
| **new fast** (100.5) | 0.99932 | 0.99964 | 0.99791 | 0.99425 | 40 | 0.052 / 0.110 / 0.176 | .032 .037 .036 .025 .025 .034 |
| `MB_SMOLVLA_INT8=accurate` (118.2) | 0.99966 | 0.99977 | 0.99907 | 0.99420 | 7 | 0.036 / 0.062 / 0.139 | .024 .025 .026 .019 .019 .026 |
| SigLIP fp32, step int8 (~139) | 0.99979 | 0.99983 | 0.99955 | 0.99920 | 0 | 0.030 / 0.054 / 0.088 | .017 .021 .022 .017 .016 .022 |

(CPU reference walk of the IRs. New = the IRs deployed on the Deck: hoisted cross K / V, merged siblings, fused epilogues, below;
SigLIP layer 0 with 64 fp32 outlier columns; fast also has fc2 in int8; accurate gives every SigLIP
layer 64 outlier columns. Chunk times: one Deck session, median of 20, interleaved.) By source, the
new default: HM01B0 frames mean 0.99951 (min 0.99391), riskybird frames 0.99965 (0.99907),
augmented 0.99950 (0.99801), synthetic 0.99939 (0.99596). Only SigLIP in fp32 keeps every
observation >= 0.999, at +28 ms.

What drives the error (worst 30 observations of the old default, one group kept fp32 at a time,
cosine mean / min; base 0.99725 / 0.978): SigLIP fp32 0.99972 / 0.99926; the step graph fp32
0.99749 / 0.978 (the expert is not the problem); SigLIP layers 0-3 fp32 0.99954 / 0.99895, layers
4-7 or 8-11 no change; layer 0 alone 0.99902 / 0.99687, layer 1, 2 or 3 alone no change. Layer 0's
int8 error is what 11 more layers and 5 flow steps amplify. Its 64 largest input columns as fp32
outlier linears recover most of keeping it fp32 for +0.9 ms (keeping it fp32: +2 ms); smoothing
alpha 0.65 / 0.9 / 1.0 for SigLIP, or a broader calibration set (36 observations incl. darkened /
brightened frames and HM01B0 frames) did not. The worst observations are sensitive, not broken:
the same int8 IRs walked on the CPU with 10 instead of 12 threads (fp32 reassociation, a few int8
roundings flip) move the worst one between 0.967 and 0.978.

Local error per int8 linear (13 observations, its int8 output vs the fp32 linear on the same input,
relative L2): SigLIP median 1.1% (fc1 0.6%, k / v / out_proj 1.2%), expert median 0.9%, the MLP
down_proj worst (1.5-1.9%); flat over depth. Accumulators and saturation: max |int32 acc| over all
linears and observations 996,030 (action_time_mlp_in) -- 11.1 bits below int32 overflow, 23% of the
K * 127 * 127 bound, which itself stays < 2^31 for every K here (largest K = 12288: 2^27.6);
quantize_rows_s8 never clamps (0 of 3.5e8 values beyond +-127.5 before the clamp; 0.11% land on
+-127, the row maxima). The dequantization is fp32 (no int8 requant), so nothing saturates there.

On the Deck (`smolvla_eval_set.py`, the new default IRs, all 193 observations, 3 runs each): every
action chunk bit-identical across runs (and graph replays); vs lerobot cosine mean 0.99946, median
0.99971, min 0.99027, 22 below 0.999, max abs err mean 0.047 / max 0.150 -- statistically the CPU
golden's (0.99951 / 0.99971 / 0.99391 / 19); device vs the CPU golden of the same IRs: cosine mean
0.99974, min 0.99901, max abs err mean 0.032 (the device's fp32 reassociation flips a few int8
roundings, which the layers amplify; bit-identical end to end is not expected, as the walker's
golden bound says). Bit-exactness: the walker gates on the Deck (int8 IRs
of all three graphs): every quantize_rows_s8 / linear_dq_s8 / linear3_dq_s8 / linear_dq_add_s8
equal to the reference C (0 mismatches), every fp32 op within its slot bound, replay equal; the
Vulkan test file on the Deck (RADV) and on garden's NVIDIA TITAN RTX (Vulkan): all int8 kernels
bit-exact against the reference C on every path (skinny + split-K, dot4 tiles, scalar);
the CPU rules equal the reference C (test_int8_mixed).

### Optimizing the int8 chunk, round 3 (Deck)

Measured one change at a time against the previous step, interleaved in one session (median of
20 chunks, 3 rounds; the Deck's clock moves the absolute numbers by ~4% between sessions, so
each row is its own A/B). All bit-exact: the action chunk of the reference observation is
unchanged by every row except the policy change.

| change | effect (Deck) |
|---|---|
| hoist the expert's cross-attention K / V projections into the prefix (`hoist_cross_kv`, models/smolvla_split.py): 16 linears per step run once per chunk, in fp32 | denoise -1.4, prefix +0.2: chunk -1.2 ms |
| sibling linears as one product (`linear3_dq_s8`: q / k / v, gate / up; no split-K at N = 4096) | denoise 54.0 -> 48.9: -5.1 ms |
| residual adds in the linear epilogue (`linear_dq_add_s8`) | -1.2 ms |
| SwiGLU silu + mul -> `silu_mul` (fp32) | prefix -0.4, denoise -0.9: -1.3 ms |
| attention's 1/8 score scale folded into the query projection (`fold_attn_scale`) | vision -1.0, denoise -0.5: -1.5 ms |
| step graph | 444 -> 316 kernel launches (prefix 372 -> 358, vision 189 -> 177) |
| SigLIP layer 0 with 64 fp32 outlier columns (policy; accuracy above) | +0.9 ms |
| **new default vs old default, same session** | **119.5 -> 110.9 ms** (fp32 209.6: 1.89x) |
| **new fast vs old fast** | **109.2 -> 100.5 ms** (2.09x) |

Tried and dropped (bench_dq-style microbenchmarks with a bit-exactness check, then the chunk):
- The skinny int8 GEMM's instruction stream is not its limit. Repeating the dot loop 16x on the
  same registers costs ~44 us per pass for 50x720x2048 against ~150 us for the whole launch, so
  ~2/3 of a launch is fixed cost (launch, ramp, split-K partials, epilogue). Variants that change
  the loop measured no better or worse: K split over 2-4 waves of a workgroup with an LDS sum
  (2-4x slower), 64-byte activation loads (`s_load_dwordx16`), interleaving 2-8 rows' dot
  products (no back-to-back accumulator dependency), 2-4-deep weight prefetch (slower), more
  split-K waves.
- The dequantization without fp64 (fp32 error-free products with an exact fallback near rounding
  midpoints; verified bit-exact on 8e7 random and adversarial cases and on the Deck): no faster --
  the fp64 epilogue is not the bottleneck either, and the larger code was slower in small launches.
- An in-kernel split-K fixup (the last split of a column block sums the others' partials, via
  int32 atomics and an arrival counter; bit-exact) instead of the epilogue launch: +1 ms per chunk.
- Microbenchmarks of single shapes on the Deck move by up to 1.5x between runs (clock); only the
  interleaved chunk A/B resolves ~0.3 ms.

Where the time is now (new default, Deck): vision ~45 ms (fp32 fc2 ~half of it), prefix ~17 ms
(fp32), 5 flow steps ~45 ms (~9 ms per step, 316 launches). The levers left: SigLIP fc2 (int8 with
outliers is the fast mix, at an accuracy cost), the remaining small launches (KV concat, permute
copies before quantize, rms_norm + quantize), and the per-launch fixed cost of the skinny GEMM.

## Not done / limits

- int8 SmolVLA: only the linears are int8 (above). Attention (`bmm`), softmax, the
  norms and the residual stream stay fp32: the existing `matmul_s8` /
  `softmax_s8` / `rmsnorm_s8` kernels would need static per-tensor scales on
  exactly the tensors that measured fragile, and the time is not there (bmm is
  ~6% of the chunk). The all-int8 extractor (`extract_int8`) still rejects
  these graphs (`get_attr`, `expand`); its errors now point to `--int8-mixed`.
  The prefix is fp32 in the default mix (accuracy): int8 there with up to 96 fp32
  outlier columns per linear did not reach a 0.999 chunk cosine.
- fp16 (`_f16`) IRs are not implemented. fp16 is used only as an opt-in
  weight format of the skinny GEMM (`--f16-weights`); RDNA2's 2x-rate packed
  fp16 math (`v_dot2_f32_f16`) is not used: it would need fp16 activations as
  well (accuracy) and a way to reach the instruction from SPIR-V.
- No KernelBlaster integration (`kb_export --target vulkan`): the backend is
  the walker path, like `walk_cuda.py`, not the multi-slot problem export.
- The GEMMs reach 30-45% of the Deck's fp32 peak (vision linears ~550-730
  GFLOPS, skinny ~460-520 GFLOPS); linears are still ~75% of the fp32 chunk.
  The int8 path fuses the RMSNorm and RoPE chains and folds attention's
  permutes (above); the fp32 FX lowering itself still emits them as generic
  op chains.
