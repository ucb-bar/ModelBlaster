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
  aligned, a register-prefetched K loop (32 per step), and split-K when the
  output has few tiles over a long K (a linear over <= 64 tokens; partials
  summed by `splitk_reduce`).
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

Results: see the table below (Deck, 1 camera at 256 px, 5 flow steps, the
`smolvla_bench.py` drone configuration).

Measured on the Deck, 2026-11-21 (RADV VANGOGH; CPU baseline = `smolvla_bench.py
--cams 1 --img 256 --steps 5 --dtype fp32 --threads 3`, lerobot 0.4.4 / torch 2.10, same session):

| component | PyTorch CPU (3 threads) | Vulkan GPU (replayed graphs) | speedup |
|---|---|---|---|
| vision (SigLIP + connector) | 471 ms (`embed_prefix`) | 85 ms | 5.5x |
| VLM prefix (KV cache) | ~300 ms | 30 ms | 10x |
| denoise, 5 flow steps | 818 ms | 170 ms (34 ms / step) | 4.8x |
| **chunk of 50 actions** | **1597 ms** (median of 5) | **284 ms** (median of 20) | **5.6x** |

Action chunk vs lerobot's own `sample_actions` (same observation, same noise):
cosine 0.99999999999953, max |err| 1.3e-6 (|actions| <= 1.01). Per graph, the
walker's gates pass: golden rel. error 1e-6 / 7e-7 / 4e-7 (vision / prefix /
step), every kernel within the per-slot bound (worst 1.3e-6). GPU time is about
85% linears (GEMM).

## Not done / limits

- **int8 SmolVLA end to end**: the int8 kernels are there and bit-exact, but
  MB's FX int8 extractor does not lower these graphs (`get_attr` constants:
  "int8 extract: get_attr _tensor_constant0 not supported"; `expand`:
  "unsupported call_method 'expand'"), and the export (torch.export) path has
  no fp32/int8 lowering of the generic ops. The int8 kernels are validated on
  MB's int8 SmolVLA proxies (`models/attn_block.py`, `models/norm_block.py`)
  and at SmolVLA sizes against the reference C.
- fp16 (`_f16`) IRs are not implemented (RDNA2 has packed fp16; the shaders
  would read 16-bit storage).
- No KernelBlaster integration (`kb_export --target vulkan`): the backend is
  the walker path, like `walk_cuda.py`, not the multi-slot problem export.
- The GEMM reaches roughly a third of the Deck's 1.6 TFLOPS fp32 peak on the
  vision encoder's shapes and less on the 31- to 50-row linears of the prefix
  and the expert, which dominate the flow loop.
