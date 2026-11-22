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

The quantization mix, chosen on 13 held-out observations (real HM01B0 drone
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
15 W cap the clock sits lower. The next levers: hoisting the 8 cross-attention
layers' K / V projections of the (constant) prefix cache out of the flow loop
(~0.5 ms per step), and a GEMM kernel that keeps more of the dot4 rate.

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
