# fp16 (IEEE half) on ModelBlaster's CUDA path

This page covers `--quant fp16` IRs on the CUDA path: the hand kernels, the GPU
gates, and the ModelBlaster ↔ KernelBlaster multi-slot flow
([kernelblaster_multislot.md](kernelblaster_multislot.md)). Phase 1 of that flow
was fp32 only. The "Design decisions" table at the end lists each choice, the
alternatives and the reason.

## Contract

An `_f16` op stores its data as IEEE binary16 and does its math in fp32. Every
element is widened to fp32 on load and accumulated in fp32 or better. The
result is rounded once, round-to-nearest-even, when it is stored. This is the
rule MB already uses to derive every `_f16` KernelSpec from its fp32 spec
(`reference_kernels._make_f16_variant`): float *pointers* become `_Float16 *`,
while scalars and locals stay `float`. The CUDA kernels follow the same rule.

| tensor | fp16 IR (`--quant fp16`) | CUDA |
|---|---|---|
| activations: inputs, intermediates, outputs | binary16 (`graph.json` dtype `f16`) | `__half` |
| weights, folded float constants (masks, biases, BN scale/shift) | binary16 in `weights.npz` | `__half` |
| integer tensors (`gather` indices) | int32 | `const int *` |
| scalar parameters (eps, alpha, scales, shapes) | `float` / `int` | `float` / `int` |
| recurrent state (`lstm_seq`/`gru_seq` h, c) | binary16, rounded every step (the ABI's `hn`/`cn` are `_Float16 *`) | `__half` |
| accumulators, norm statistics, softmax sums | fp32 (the fp32 hand kernels' `double` accumulators are kept) | — |

- **CUDA spelling.** nvcc 12.6 parses `_Float16` but cannot generate code for
  it ("unsupported float variant"). The CUDA entry point of an `_f16`
  KernelSpec is therefore its signature with `_Float16` written as `__half`
  (`cuda_dtypes.cuda_signature`). Both are the same 2-byte type, so the C ABI
  is unchanged.
- **Golden.** The extract writes the fp32 model evaluated on the fp16-rounded
  inputs, then stored as fp16; this is unchanged since 8379297. The golden
  therefore asks the user-level question: does the fp16 deployment compute
  the model? See decision 3 for why this golden was kept.
- **Reference for the per-op checks.** `walk_torch` runs each `_f16` op's fp32
  rule and rounds every output to fp16, plus the recurrent state at every
  step. The result is the value an exact fp16-storage kernel stores. This
  reference is used by the CUDA walk's per-intermediate check, and its values
  go into `kb_export`'s `data.bin`.

Code: `pipeline/cuda_dtypes.py` holds the contract constants, the spelling,
the buffer dtypes and the tolerances. It has no dependencies, so the GPU
harness loads it by path.

## Tolerances

Notation: u = 2⁻¹¹ is the binary16 unit roundoff. One ulp of a value in
[1, 2) is 2u ≈ 9.8e-4. All numbers below come from all 50 Level3 fp16 IRs
on the TITAN RTX.

| bound | used by | check | measured max on Level3 | fp16 bound | fp32 bound (unchanged) |
|---|---|---|---|---|---|
| `slot_rel`, `slot_abs` | walk_cuda per-intermediate; driver `MB_SLOT_CHECK` | each op alone on the reference's fp16 inputs: `err ≤ slot_rel·max(\|inputs\|, \|ref\|) + slot_abs` | ≤ 1 ulp of the output: 7.8e-4 of the scale in the normal range (reduce); 1 subnormal ulp (2⁻²⁴) where activations vanish | 2e-3 (2 ulps), 2⁻²³ | 1e-3, 0 |
| `output_rel`, `output_abs` | driver `MB_CHECK` (partition outputs vs the fp16-storage walk) | `err ≤ output_rel·max\|ref\| + output_abs` | 2.0e-3 (46_NetVlad) | 1e-2, 1e-5 | 1e-3, 1e-5 |
| `golden_rel`, `golden_abs` | `walk_torch`, `walk_cuda` vs `io.npz` | `err ≤ golden_rel·max\|golden\| + golden_abs` | 1.3e-3 (IR walk), 1.4e-3 (hand CUDA, 32_CvT), 1.5e-3 (host C) | 1e-2, 1e-5 | 1e-3, 1e-5 |

How the bounds were set:

- **Isolated op.** The kernel and the reference round nearly the same fp32
  value once, so they can differ by one ulp of the output when the two fp32
  values straddle a rounding boundary. Measured, every op kind stays within
  one ulp.
  - The largest relative errors are linear 7.6e-4, layer_norm 7.0e-4, bmm
    6.9e-4 and lstm/gru 4.9e-4. Elementwise and data-movement kernels are
    exact.
  - The only excesses over 2⁻¹⁰ are in EfficientNetB0/B1 and MobileNetV2,
    where activations decay to ~1e-6 inside the fp16 subnormal range. Those
    are exactly one subnormal ulp (5.96e-8), checked element by element. The
    relative-only bound reached 7.4e-3 there and flagged a correct kernel, so
    the bound gained an absolute term of 2⁻²³ (two subnormal ulps).
  - The bound is two ulps: one for the straddle, one of margin.
- **Outputs.** The two output bounds share one number.
  - **Error floor.** The golden is itself stored in fp16, which costs up to u
    relative. Rounding the weights to fp16 adds little: with weights and
    inputs in fp16 but fp32 activations, the error is 3.9–6.3e-4 of the range
    on 13 problems.
  - **Activation storage.** Storing activations in fp16 brings the error to
    4.5e-4–1.3e-3 on those 13. Over all 48 finite problems it stays ≤ 2.7u, including
    DenseNet201's 765 ops and the 512-step LSTMs.
  - **Bound.** 1e-2 is about 5× the largest measured error, 20u. It is loose
    enough for deeper graphs and still two orders of magnitude below what a
    wrong kernel produces.
  - **No separate bound vs the fp16 walk.** The hand-CUDA output differs from
    the fp16 walk by ≤ 2.0e-3, the same order as the golden error, so no
    separate, tighter bound is used.
- **fp32 bounds are unchanged.** The fp32 manifests only gained
  `slot_abs: 0.0`.

## Kernels

`gpu_eval/harness/cuda/mb_cuda_kernels_f16.cu` is **generated** by
`gpu_eval/harness/f16_kernels.py` from the fp32 `mb_cuda_kernels.cu`. Do not
edit it by hand. A test fails if it is stale. The generator applies three
transforms:

1. `const float *` / `float *` become `const __half *` / `__half *`. The
   exception is the shared-memory reduction scratch (`KEEP_FLOAT`).
2. Every element read `p[i]` of a half pointer becomes `mb_ld(p[i])`, a
   `float`, so no expression is evaluated in fp16. A store `p[i] = <float or
   double>` rounds once through `__half`'s converting assignment. A compound
   `p[i] op= v` becomes `p[i] = mb_ld(p[i]) op (v)`. Plain copies stay
   `__half`-to-`__half`, which is exact.
3. `kernel_<op>` becomes `kernel_<op>_f16`, in both definitions and calls.

Entries whose `_f16` spec is hand-written with a different signature come from
`cuda/mb_cuda_kernels_f16_hand.cu` (~80 lines):

- `softmax_f16` (with `input_scale`; exponentials are recomputed rather than
  stored and rescaled)
- `adaptive_avg_pool2d_f16` (with `OH`, `OW`)
- `cat2_c1_f16` (with `Ca`, `Cb`)
- `sdpa_f16`: the fp32 kernel accumulates P·V in its output row, which in
  fp16 would be fp16 accumulation.

The result is 64 entry points covering every fp32 hand kernel. The 35 kinds
Level3 uses are validated below; the others compile but have not been run.

Fixes made along the way:

- The `conv2d_f16` spec had lost its DH/DW (8379297's fix was dropped by a
  merge). Every fp16 conv model failed to compile on the host build, and the
  walker would have dropped the dilation. The spec, reference, widening
  algorithm, argtypes and both curated RVV kernels take DH/DW again.
- The kernel ABI binder now binds the three divergent hand specs to what
  `generate_skeleton` passes.
- The skeleton's `adaptive_avg_pool2d_f16` branch no longer requires `OH`/`OW`.

## Multi-slot flow in fp16

- **Manifest** (`mb-kb-problem/1`, still compatible):
  - `dtype` is `fp16` when every op of the partition is `_f16`.
  - Buffers have dtype `f16`, and `data.bin` holds their fp16 values (2 bytes
    each).
  - Slot signatures are in CUDA spelling.
  - Slots are seeded from `mb_cuda_kernels_f16.cu` (the `<stem>_f16.cu`
    beside `--hand-cu`).
  - `tolerance` holds the fp16 row above, including the new `slot_abs`.
- **Driver**:
  - widens f16 buffers to float for its checks (bit-exact, including
    subnormals, inf and NaN);
  - uploads raw reference bytes for the isolated runs;
  - offsets alias views by element size;
  - applies `err ≤ slot_rel·scale + slot_abs`.
- **KernelBlaster**:
  - `SlotProblem.dtype` and `.tolerance` come from the manifest.
  - **Prompt.** The system prompt's dtype rule says: `__half` storage, widen
    on load (`__half2float` / `__half22float2`), fp32 accumulation, round once
    on store, tensor cores (`wmma` / `mma.sync`, fp32 accumulators) allowed,
    plus the manifest's isolated bound. The fp32 prompt is byte-identical to
    before.
  - **Records.** OutcomeRecords and ImplStore records carry `dtype`; old
    records read as fp32. Every retrieval tier (seed, adapt, examples) and
    `winners` match the asking slot's dtype, so an fp32 winner is never
    offered to an fp16 slot, even as a same-class example.
  - **Bandit.** fp16 arms are keyed `class|state|fp16`. The fp32 keys are
    unchanged, and the cross-key prior still seeds untried fp16 arms.
  - **Roofline.** The roofline and fusion estimates count 2 bytes per f16
    element. The peaks stay fp32, because the seeds compute in fp32.
- **Write-back** (`kb_ingest`):
  - Cache entries hash the CUDA signature (what KernelBlaster hashes from the
    manifest) and carry `dtype`.
  - `lookup` also matches the dtype that the `_f16` entry name implies.
- **Shape buckets** are unchanged.

## Level3 validation (TITAN RTX sm_75, 2026-10-23)

| gate | what it checks | result |
|---|---|---|
| extract | `extract_graph --quant fp16` (`MB_QUANT=fp16 MB_TARGET_MB=16 MB_WEIGHTS_MB=64`; 31 with `--bench-target-gflops 20`) | **50/50** |
| IR walk | `walk_torch.py --device cpu` (fp16 storage) vs golden | **48/50** |
| host C | MB's `_Float16` reference C (`host_verify.sh`, gcc 13 on x86) vs golden | **48/50** |
| hand-CUDA walk | `walk_cuda.py`: golden, plus every intermediate isolated vs the fp16 walk | **48/50** pass, 50/50 with no divergent intermediate |
| multi-slot export | every IR as one whole-graph problem (`kb_export --build-check`): build, partition check, every op isolated | **49/50** |
| scripted agent | KB `tests/test_multislot_fp16.py` on the fp16 LeNet problem | pass |
| live agent | `claude-code:sonnet`, 2 rollouts × 4 steps, on 8_ResNetBasicBlock fp16 | 29.3 → 0.59 ms (49×) in 8 calls |

The misses are two problems, and they are inherent to fp16 range:

- **48_Mamba2ReturnY / 49_Mamba2ReturnFinalState.** The fp32 model's outputs
  reach 2.4e15 and 4.6e14. binary16 tops out at 65504, so the fp16 golden is
  `inf` in 405,924 and 77,713 elements, and inf − inf cascades to NaN
  downstream. These are not fp16-representable models; FAIL is the expected
  result.
  - **Kernels are fine:** every intermediate passes its isolated check (max
    4.9e-4).
  - **Export:** 49 also passes as a multi-slot problem. 48's output check
    fails near the overflow threshold (error 1136 on values near 65504,
    where an fp16 ulp is 32).

Degenerate but passing: in 19_MobileNetV1, 20_MobileNetV2, 22_EfficientNetB0
and 23_EfficientNetB1, the activations decay to the subnormal range. The
output is then the final bias (MobileNetV2's output underflows to 0 in fp16;
it is 3e-21 in fp32). Their golden checks are weak. The isolated checks
still exercise every kernel.

The fp16 IRs have the same shapes as a fresh fp32 extract (checked on two).
The 9 problems whose shapes differ from `level3/ir16` were re-sized by MB
changes since then; the 2026-10-21 fp32 regression reported the same set.

| problem | ops | max\|golden\| | IR walk vs golden | host C vs golden | CUDA vs golden | CUDA vs fp16 walk | worst isolated op | export |
|---|---|---|---|---|---|---|---|---|
| 1_MLP | 5 | 0.264 | 9.2e-04 | 9.2e-04 | 9.2e-04 | 4.6e-04 | 3.0e-04 | pass |
| 2_ShallowWideMLP | 5 | 0.239 | 5.2e-04 | 5.3e-04 | 5.2e-04 | 5.1e-04 | 2.7e-04 | pass |
| 3_DeepNarrowMLP | 33 | 0.0773 | 7.9e-04 | 7.9e-04 | 7.9e-04 | 7.9e-04 | 7.6e-04 | pass |
| 4_LeNet5 | 11 | 0.123 | 5.0e-04 | 5.0e-04 | 5.0e-04 | 4.9e-04 | 4.9e-04 | pass |
| 5_AlexNet | 18 | 0.0244 | 6.3e-04 | 6.3e-04 | 6.3e-04 | 6.3e-04 | 3.7e-04 | pass |
| 6_GoogleNetInceptionModule | 8 | 1.67 | 5.8e-04 | 5.8e-04 | 5.8e-04 | 5.8e-04 | 5.8e-04 | pass |
| 7_GoogleNetInceptionV1 | 84 | 0.0631 | 4.8e-04 | 4.8e-04 | 4.8e-04 | 4.8e-04 | 8.1e-04 | pass |
| 8_ResNetBasicBlock | 9 | 1.42 | 6.9e-04 | 6.9e-04 | 6.9e-04 | 3.4e-04 | 6.3e-04 | pass |
| 9_ResNet18 | 68 | 0.0862 | 7.1e-04 | 7.1e-04 | 7.1e-04 | 7.1e-04 | 6.7e-04 | pass |
| 10_ResNet101 | 94 | 0.0336 | 9.1e-04 | 9.1e-04 | 9.1e-04 | 9.1e-04 | 1.2e-04 | pass |
| 11_VGG16 | 36 | 0.0251 | 6.1e-04 | 6.1e-04 | 6.1e-04 | 6.1e-04 | 6.2e-04 | pass |
| 12_VGG19 | 42 | 0.0278 | 5.5e-04 | 5.5e-04 | 5.5e-04 | 5.5e-04 | 9.0e-04 | pass |
| 13_DenseNet121TransitionLayer | 4 | 0.993 | 4.9e-04 | 4.9e-04 | 4.9e-04 | 4.9e-04 | 3.5e-04 | pass |
| 14_DenseNet121DenseBlock | 24 | 1.52 | 6.4e-04 | 6.4e-04 | 6.4e-04 | 3.2e-04 | 3.2e-04 | pass |
| 15_DenseNet121 | 325 | 0.034 | 9.0e-04 | 9.0e-04 | 9.0e-04 | 4.5e-04 | 6.3e-04 | pass |
| 16_DenseNet201 | 666 | 0.0208 | 7.3e-04 | 7.3e-04 | 7.3e-04 | 3.7e-04 | 3.7e-04 | pass |
| 17_SqueezeNetFireModule | 7 | 0.53 | 9.2e-04 | 9.2e-04 | 9.2e-04 | 4.6e-04 | 6.2e-04 | pass |
| 18_SqueezeNet | 64 | 0.109 | 5.6e-04 | 5.6e-04 | 5.6e-04 | 2.8e-04 | 8.6e-04 | pass |
| 19_MobileNetV1 | 83 | 0.0312 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 9.6e-04 | pass |
| 20_MobileNetV2 | 141 | 0 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.6e-03 | pass |
| 21_EfficientNetMBConv | 8 | 0.268 | 9.1e-04 | 9.1e-04 | 9.1e-04 | 4.6e-04 | 5.9e-04 | pass |
| 22_EfficientNetB0 | 115 | 0.0279 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 7.4e-03 | pass |
| 23_EfficientNetB1 | 64 | 0.0279 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 | 1.5e-03 | pass |
| 24_EfficientNetB2 | 73 | 0.228 | 5.3e-04 | 5.3e-04 | 5.3e-04 | 5.3e-04 | 4.7e-04 | pass |
| 25_ShuffleNetUnit | 12 | 1.54 | 6.3e-04 | 6.3e-04 | 6.3e-04 | 3.2e-04 | 3.4e-04 | pass |
| 26_ShuffleNet | 145 | 0.0757 | 8.1e-04 | 8.1e-04 | 8.1e-04 | 4.0e-04 | 6.3e-04 | pass |
| 27_RegNet | 23 | 0.0649 | 9.4e-04 | 9.4e-04 | 9.4e-04 | 5.9e-05 | 6.6e-04 | pass |
| 28_VisionTransformer | 129 | 0.41 | 8.9e-04 | 8.9e-04 | 1.2e-03 | 6.0e-04 | 4.7e-04 | pass |
| 29_SwinMLP | 178 | 0.499 | 9.8e-04 | 1.3e-03 | 1.1e-03 | 1.0e-03 | 4.6e-04 | pass |
| 30_SwinTransformerV2 | 339 | 2.03 | 1.0e-03 | 9.6e-04 | 9.6e-04 | 4.8e-04 | 7.8e-04 | pass |
| 31_VisionAttention | 18 | 3.65 | 1.1e-03 | 1.1e-03 | 1.1e-03 | 1.1e-03 | 5.9e-04 | pass |
| 32_ConvolutionalVisionTransformer | 125 | 1.99 | 1.3e-03 | 1.5e-03 | 1.4e-03 | 1.2e-03 | 3.1e-04 | pass |
| 33_VanillaRNN | 4 | 0.746 | 6.5e-04 | 6.5e-04 | 6.5e-04 | 6.5e-04 | 3.5e-04 | pass |
| 34_VanillaRNNHidden | 1061 | 0.793 | 6.2e-04 | 6.2e-04 | 6.2e-04 | 6.2e-04 | 4.9e-04 | pass |
| 35_LSTM | 10 | 0.0483 | 6.3e-04 | 6.3e-04 | 6.3e-04 | 3.2e-04 | 1.2e-04 | pass |
| 36_LSTMHn | 8 | 0.359 | 6.8e-04 | 6.8e-04 | 6.8e-04 | 6.8e-04 | 4.9e-04 | pass |
| 37_LSTMCn | 8 | 0.838 | 5.8e-04 | 5.8e-04 | 5.8e-04 | 2.9e-04 | 2.4e-04 | pass |
| 38_LSTMBidirectional | 26 | 0.208 | 5.9e-04 | 5.9e-04 | 5.9e-04 | 2.9e-04 | 4.9e-04 | pass |
| 39_GRU | 9 | 0.763 | 6.4e-04 | 6.4e-04 | 6.4e-04 | 6.4e-04 | 4.9e-04 | pass |
| 40_GRUHidden | 9 | 0.608 | 8.0e-04 | 8.0e-04 | 8.0e-04 | 4.0e-04 | 4.9e-04 | pass |
| 41_GRUBidirectional | 23 | 0.881 | 5.5e-04 | 5.5e-04 | 5.5e-04 | 5.5e-04 | 4.9e-04 | pass |
| 42_GRUBidirectionalHidden | 23 | 0.699 | 7.0e-04 | 7.0e-04 | 7.0e-04 | 7.0e-04 | 4.9e-04 | pass |
| 43_MinGPTCausalAttention | 15 | 0.681 | 7.2e-04 | 7.2e-04 | 7.2e-04 | 7.2e-04 | 5.7e-04 | pass |
| 44_MiniGPTBlock | 29 | 2.13 | 9.2e-04 | 9.2e-04 | 9.2e-04 | 9.2e-04 | 5.1e-04 | pass |
| 45_UNetSoftmax | 67 | 0.169 | 7.2e-04 | 7.2e-04 | 7.2e-04 | 4.5e-05 | 8.0e-04 | pass |
| 46_NetVladWithGhostClusters | 17 | 0.0115 | 1.3e-03 | 1.3e-03 | 1.3e-03 | 2.0e-03 | 5.3e-04 | pass |
| 47_NetVladNoGhostClusters | 16 | 0.0118 | 1.3e-03 | 1.3e-03 | 1.3e-03 | 1.3e-03 | 6.5e-04 | pass |
| 48_Mamba2ReturnY | 28 | 6.55e+04 (405924 inf) | nan FAIL | inf FAIL | nan FAIL | 1.7e-02 | 2.9e-04 | FAIL |
| 49_Mamba2ReturnFinalState | 12 | 6.55e+04 (77713 inf) | nan FAIL | inf FAIL | nan FAIL | 4.9e-04 | 0.0e+00 | pass |
| 50_ReLUSelfAttention | 15 | 138 | 4.5e-04 | 4.5e-04 | 4.5e-04 | 2.3e-04 | 6.2e-04 | pass |

Columns are errors ÷ max|golden| (CUDA vs fp16 walk: ÷ max|ref|). "Worst
isolated op" is relative to the op's scale. Values above 2e-3 are single
subnormal ulps, within `slot_abs`.

The naive fp16 kernels run at the fp32 kernels' speed (e.g. ResNetBasicBlock
24 vs 26 ms in conv2d; the GPU was shared, so timings are noisy). They
compute in fp32/double, like the fp32 kernels they are generated from. They
are correctness seeds; the live run shows the LLM moving one to a tiled conv.

## Design decisions

| # | Decision | Alternatives | Why |
|---|---|---|---|
| 1 | fp16 storage, fp32 math, one rounding per stored value; integer tensors stay int32 | fp16 arithmetic (`__hmul`/`__hadd`, or `__half` operators); bf16 | It is the existing MB `_f16` contract (C references, RVV `vfwmacc` kernels), so CUDA, C and RVV agree. fp16 accumulation over K ~ 1e3 costs several ulps and makes results order-dependent. sm_75 has no bf16. |
| 2 | CUDA signature = KernelSpec signature with `_Float16` → `__half` | `typedef` in the walker head; a custom struct type in the ABI | nvcc cannot generate code for `_Float16`, and the host compiler treats it as a keyword, so it cannot be typedef'd. `__half` is the standard CUDA type an LLM knows and is layout-identical. |
| 3 | Golden = fp32 model on fp16-rounded inputs (unchanged), with a separate fp16-storage reference walk for the per-op and driver checks | `model.half()` golden; golden on fp16-rounded weights | `model.half()` encodes one fp16 accumulation choice, lacks CPU half kernels for some ops, and drifted 0.1 on cross_entropy (8379297). Rounded weights move the golden by ≤ u (measured), which is below the output bound. Two references split two questions: "is this a faithful fp16 model?" (golden, loose) and "is this kernel exact for the contract?" (fp16 walk, ≤ 2 ulps). |
| 4 | Tolerances measured on Level3, in ulps of binary16, with an absolute subnormal term for the isolated check | scale the fp32 bounds; a per-op table | The isolated error is bounded by the rounding itself (≤ 1 ulp), so ulps are the natural unit. Subnormal ulps are absolute, as the EfficientNet/MobileNet cases showed. One number per check keeps the manifest and the prompt simple. |
| 5 | fp16 hand kernels generated from the fp32 file, plus ~80 lines of hand overrides | copy-paste 60 kernels; templating the fp32 file on the element type | The generator uses the same rule MB applies to the C references, so one kernel edit updates both precisions. A staleness test enforces it. Templating would rewrite the fp32 file, which conflicts with concurrent work (graph-opt), and the kernel library slicer keys definitions by name. |
| 6 | One hand library per problem, chosen by the partition's dtype (fp16 if all ops are `_f16`) | merge both libraries; per-slot library | Level3 fp16 IRs are all-`_f16`. Mixed int8/fp16 IRs have no CUDA int8 kernels yet. Fused fp16 ops then pick the fp16 library too: a fused kind's dtype is its stages' (`cuda_dtypes.op_dtype`), also when its name ends in the param that reads the intermediate (`fused__ewise2_f16__ewise2_f16__a`). |
| 7 | ImplStore retrieval and winners restricted to the slot's dtype; claims and technique priors shared across dtypes | share examples across dtypes; separate knowledge dirs | fp16 code and timings (half2, tensor cores, half the bytes) say nothing reliable about the fp32 op, and vice versa. Technique-level lessons ("tile before register blocking") do transfer. fp16 bandit arms get their own keys but still start from the fp32 prior. |
| 8 | fp32 manifests, prompts and cache keys unchanged (only `slot_abs: 0.0` added) | version bump `mb-kb-problem/2` | Old problems, records and caches keep working. A missing `dtype` reads as fp32 everywhere. |

## Limitations

- **Only what Level3 uses is validated:** the 35 `_f16` op kinds it
  exercises, plus the LeNet ops. The other generated kernels compile but have
  not been run; `sdpa_f16`'s hand kernel is naive.
- **fp16 range.** Models whose activations exceed 65504 (Mamba2 here) or
  decay into subnormals (deep randomly initialised CNNs) are inherently
  outside fp16. The gates report them; nothing rescales.
- **Not run in fp16:**
  - KernelBlaster's libtorch per-op problems (`kb_bridge`, H4) are fp32
    only; fp16 uses the multi-slot export instead.
  - Fusion proposals for fp16 partitions: the fused seed reuses the fp16
    library, but a fused spec's seed text still carries `_Float16` in its
    entry signature.
- **Coverage gaps:**
  - No RVV/spike fp16 run of the restored dilated `conv2d_f16` (compile-checked
    only).
  - No bf16 or int8 slots.
- **Weak accumulation check.** At K = 64 a slot that accumulates in fp16
  still passes the 2-ulp bound (1.1e-3; tested). The bound judges the result,
  not the method; the prompt forbids fp16 accumulation.
- The timings are from a shared GPU.

## Repro

```bash
source gpu_eval/tools/env.sh
MB_QUANT=fp16 MB_TARGET_MB=16 MB_WEIGHTS_MB=64 KEEP_NPZ=1 JOBS=6 \
  bash gpu_eval/tools/extract_many.sh IR16 fp16 level3/*.py        # 31: add --bench-target-gflops 20
(cd gpu_eval/harness && python walk_torch.py IR16/*/ --device cpu --json walk.json)
bash gpu_eval/tools/host_verify_all.sh IR16 HOST 3
PY=python bash gpu_eval/harness/run_level3_gpu.sh IR16 GPU 75     # builds lib from mb_cuda_kernels_f16.cu
python -m modelblaster.pipeline.kb_export IR16/<p> PROB --build-check --arch 75
```
