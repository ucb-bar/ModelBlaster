# int8 PTQ flow — reference

Captures the int8 quantization flow as it stands today. Code: every
function name in this doc is grep-able from the repo root.

## What this is

**Symmetric per-tensor int8 PTQ**, weights and activations both. No
calibration set — a single sample input drives activation-range
observation. Bias is int32. Inference math is plain integer with a
Q0.31 fixed-point requantize tail (CMSIS-NN / muRISCV-NN convention).

## What this isn't

- **Not per-channel** for weights. All weights of an op share one
  scale, derived from `max(abs(weight))`. Depthwise convs would suffer
  badly under this — accept the loss or extend `_scale_from_max_abs`
  to a per-channel variant before adding `conv2d_dw_s8`.
- **Not asymmetric** — `input_offset`, `filter_offset`, `output_offset`
  are 0 throughout. The `_s8` kernel signatures keep them as parameters
  (matches CMSIS-NN ABI) but the extractor sets them to 0.
- **Not QAT.** The model's training was done in fp32; we just calibrate
  a fp32→int8 conversion at extract time. Networks that need QAT to
  preserve accuracy (transformer attention, anything with batchnorm
  in non-eval mode) won't survive this path.
- **Not a calibration set** — one forward-pass on the existing
  `model_mod.get_sample_input()` is all that informs scale choice.
  Crude but reproducible.

## Quantization scheme

### Symmetric per-tensor scale

```python
# agents/pipeline/extract_graph.py:460
def _scale_from_max_abs(t):
    m = float(t.detach().abs().max().item())
    return max(m, 1e-8) / 127.0   # _INT8_RANGE
```

`scale` maps `[-max_abs, max_abs] → [-127, 127]`. The `-128` slot is
deliberately given up so the multiplier math doesn't have to handle
the asymmetric range.

### Tensor quantize

```python
# agents/pipeline/extract_graph.py:466
def _quantize_per_tensor_sym(t, scale):
    return torch.round(t.detach() / scale).clamp(-127, 127).to(torch.int8)
```

### Bias quant — int32, scale = in_scale × weight_scale

```python
b_q = torch.round(b_fp32 / (in_scale * w_scale)).to(torch.int32)
```

Stored as int32. Pre-scaled so it can be added to the int32
accumulator before requantize.

### Requantize: Q0.31 multiplier + shift

The "real-multiplier" `M = (in_scale * w_scale) / out_scale` is
decomposed into `(multiplier, shift)`:

```python
# agents/pipeline/extract_graph.py:490
def _requantize_multiplier_shift(real_mult):
    mantissa, exp = np.frexp(real_mult)         # mantissa ∈ [0.5, 1.0)
    multiplier = int(round(mantissa * (1 << 31)))
    if multiplier == (1 << 31):
        multiplier //= 2
        exp += 1
    shift = -exp        # +shift = right-shift after Q0.31 multiply
    return multiplier, shift
```

Kernel side runs:

```c
int64_t prod = (int64_t)acc * (int64_t)multiplier;
prod = (prod + (1LL << 30)) >> 31;       // Q0.31 multiply, round-to-+inf
int32_t scaled = (int32_t)prod;
if (shift > 0) {
    int32_t r = (1 << (shift - 1));
    scaled = (scaled + r) >> shift;      // arithmetic right shift, rounded
} else if (shift < 0) {
    scaled = scaled << (-shift);
}
scaled += output_offset;                 // 0 today
clamp(activation_min, activation_max);
output[i] = (int8_t)scaled;
```

Bit-exact across the kernel and `_requantize_int()` Python mirror
(`agents/pipeline/extract_graph.py:472`) — they're tested against
each other by the `_s8` `KernelSpec.reference_impl`s.

## Calibration

```python
# agents/pipeline/extract_graph.py:514
class _CaptureTensors(torch.fx.Interpreter):
    """FX Interpreter that records every tensor produced by every node."""
```

One forward pass through the FX graph with `sample_input`. Every
node's output tensor is captured. `_scale_from_max_abs` runs over each
captured tensor → per-node activation scale.

Trade-off vs a real calibration set:
- Reproducible: same seed, same activation ranges, same scales.
- Sensitive to outliers in the single sample. A 100×-larger pixel
  somewhere in `sample_input` skews everything.
- Skipping the percentile clip / KL-divergence tricks that real PTQ
  flows use. Acceptable for our small networks; not for ImageNet-scale.

## Activation fusion

`linear → relu` and `conv2d → relu` are detected and fused: the relu
disappears, the linear/conv's `activation_min` becomes 0 (the relu
becomes a clamp inside the requantize tail — free):

```python
# pseudocode of the fused-detect logic
fused_relu_after = set()
for node in ops:
    if isinstance(node.next, nn.ReLU):
        fused_relu_after.add(node.next)
        emit_op_with(activation_min=0)
```

Standalone `nn.ReLU` (not preceded by linear/conv) emits an explicit
`relu_s8` op.

`linear → batchnorm2d`, `conv2d → batchnorm2d` are NOT fused yet —
batchnorm is its own op kind (`batchnorm2d_s8`) with its own
requantize. Fusion would be a Stage-2 perf win.

## Op coverage

Every op below has:
1. An `extract_int8` handler in `agents/pipeline/extract_graph.py` that
   converts the fp32 nn.Module into an int8 IR node.
2. A matching `<OP>_S8 = KernelSpec(...)` entry in
   `agents/pipeline/reference_kernels.py` with a scalar reference impl
   and ctypes argtypes.
3. A dispatch case in `emit_model()` of
   `agents/pipeline/generate_skeleton.py` that wires the IR op to a
   `kernel_<op>_s8(...)` call in the generated `model.c`.

| nn.Module | IR op kind | reference_kernels.py spec |
|---|---|---|
| `nn.Linear` | `linear_s8` | `LINEAR_S8` |
| `nn.ReLU`, `torch.relu` | `relu_s8` | `RELU_S8` |
| `nn.Conv2d` (groups=1) | `conv2d_s8` | `CONV2D_S8` |
| `nn.MaxPool2d` | `maxpool2d_s8` | `MAXPOOL2D_S8` |
| `nn.BatchNorm2d` (eval) | `batchnorm2d_s8` | `BATCHNORM2D_S8` |
| `nn.Sigmoid` | `sigmoid_s8` | `SIGMOID_S8` |
| binary `+` (residual) | `add_s8` | `ADD_S8` |
| `nn.Dropout` (eval) | `view` (alias) | — |

## What's not yet wired (extending checklist)

For each missing op, the steps to add:

1. **Handler in `extract_int8`** — pattern-match the nn.Module, compute
   per-tensor scale for the output activation, derive the Q0.31
   `(multiplier, shift)` from the activation scales involved.
2. **`<OP>_S8` `KernelSpec`** — function signature (input/output/scale
   args), ctypes argtypes factory, scalar `reference_impl` C source,
   `extra_shapes` for verify, register in `KERNEL_SPECS`.
3. **Dispatch case in `emit_model()`** — read shape + quant fields out
   of the IR op, emit the kernel call with `_f32(...)` literals for
   real-valued args.
4. **Input generator in `verify_kernel.py`** — build randomized int8
   tensors of the spec's `extra_shapes`.

Currently missing (drives the gap matrix in
`agents/notes/` of-the-overall-quant-coverage discussion):

- `relu6_s8` (mobilenet_v2)
- `conv2d_dw_s8` (mobilenet_v2 depthwise — see per-channel caveat above)
- `adaptive_avg_pool2d_s8` (mobilenet_v2 classifier head)
- `silu_s8`, `cat{2,3,4}_c1_s8`, `upsample_nearest_s8` (yolov8_nano)
- `matmul_s8`, `bmm_s8` (KernelBench int8 sweeps)
- KernelBench Phase 2 ops: leaky_relu_s8, tanh_s8, gelu_s8, etc.
- And ~13 nn.Module handlers that the fp32 extractor has but
  `extract_int8` doesn't (SiLU, Upsample, LeakyReLU, Tanh, GELU, SELU,
  Hardsigmoid, Softplus, Softsign, Hardtanh, ReLU6, AdaptiveAvgPool2d,
  ELU). Each is a small handler + matching `_s8` KernelSpec.

## Validated models

End-to-end on spike, in-binary `AGENTS_VERIFY` PASS at atol=0 / rtol=0
against the PyTorch int8 golden:

| Model | Op coverage | Profile rows |
|---|---|---|
| **mlp_generic** | `linear_s8` | 3 |
| **lenet** | `conv2d_s8`, `linear_s8`, `maxpool2d_s8`, `view` | 7 |
| **dronet** | `conv2d_s8`, `linear_s8`, `relu_s8`, `sigmoid_s8`, `maxpool2d_s8`, `batchnorm2d_s8`, `add_s8`, `view` | 30 |

LLM-RVV optimized cache exists for all three under
`agents/examples/<model>/int8/cache/rvv/rvv_<op>_<algo>.c`. Dronet's
`conv2d_s8` cache includes both the default `direct` algorithm and a
hand-tuned `rvv_widening_oc` seed (see CONV2D_S8 in
`reference_kernels.py`).

`mlp_control` shares its op set with `mlp_generic` and is a 30-minute
"just run it" away from being on this list — hasn't been triggered.

## End-to-end pipeline path (int8)

```
agents/models/<model>.py           # fp32 PyTorch model
              │
              ▼
extract_graph.py --quant int8      # extract_int8(...): FX trace + ShapeProp
              │                    # + _CaptureTensors for activation calibration
              │                    # → graph.json (i8 dtypes, _s8 ops, quant fields)
              │                    # → weights.npz (i8 weights, i32 biases)
              │                    # → io.npz (test_input, test_golden — both fp32!
              │                    #          golden is what the int8 kernels should
              │                    #          dequantize to within atol)
              ▼
generate_skeleton.py               # emit_model(): generates kernel call sites
              │                    # → model.c, model.h, weights.c, weights.h
              │                    # → test_io.S + .bin (incbin)
              │                    # → buffers.c (int8 intermediate buffers)
              ▼
generate_kernels.py --backend ...  # picks + verifies kernel impls
              │   (reference|llm)  # backend=reference: spec.reference_impl
              │                    # backend=llm: LLM produces from algorithm seed
              │                    # → kernels.c, kernels.h
              ▼
west build -b spike_riscv64        # Zephyr harness compiles + links
              │
              ▼
spike + AGENTS_VERIFY              # in-binary max_abs_err vs test_golden
                                   # int8 outputs widened to float for compare;
                                   # default tolerance for int dtype is atol=0
                                   # rtol=0 (bit-exact); per-backend overrides
                                   # via Backend.atol_override (Gemmini uses 3.0
                                   # because float-scale requantize)
```

## Gotchas

- **The `view` alias for `nn.Dropout(eval)`**: Dropout is a no-op in
  eval mode. extract_int8 emits a `view` op which is treated as a
  buffer-pointer alias (no kernel). If a model relies on Dropout
  changing shape (rare), this will silently mishandle it.
- **`nn.BatchNorm2d` is eval-mode only**: extract_int8 calls `model.eval()`
  before tracing. Running BN in train mode (with running-stat updates)
  isn't supported and isn't checked.
- **Bias overflow**: bias is stored as int32, with scale `in_scale *
  w_scale`. For very-wide layers with large activations, the
  pre-requantize accumulator + bias can saturate int32. Hasn't bitten
  any of dronet/lenet/mlp_generic at their current shapes; would hit
  on layers with `K * w_scale * in_scale` close to `2^31`.
- **`conv2d_s8` with `groups != 1`** raises NotImplementedError. Same
  for `dilation != (1,1)`. Both fixable, both gate mobilenet_v2 int8.

## Files cheat-sheet

| File | Role |
|---|---|
| `agents/pipeline/extract_graph.py:526` (`extract_int8`) | int8 IR extraction, calibration, weight quantize |
| `agents/pipeline/reference_kernels.py` (`*_S8` specs) | per-op signatures, scalar reference impls, ctypes argtypes |
| `agents/pipeline/generate_skeleton.py` (`emit_model`) | dispatch case per `_s8` op kind |
| `agents/pipeline/verify_kernel.py` (`_gen_inputs_*_s8`) | int8 input generators for the host-ctypes verify path |
| `agents/pipeline/generate_kernels.py` | backend=reference vs LLM kernel generation, cache, optimize loop |
| `agents/examples/<model>/int8/cache/<target>/` | persisted LLM kernel outputs |
| `agents/examples/<model>/int8/generated/` | extracted IR + per-target codegen |
