# REPORT: hardware-aware Adreno 650 seeds and prompts

Files: `CHANGES.md` (every file, flags), `qrb_hw.mb.patch`, `qrb_hw.kb.patch`. Board workspaces: `l3_farm/qrb_a6/cc_runs/hw_*` (`_0` = switch off / legacy seeds, `_1` = default seeds). 15 board evaluations in total (cc_eval.py, no other board access), all with the flow's own checks.

## Validation on the board (Adreno 650, fp16, exported with `kb_export --target opencl --arch adreno650`)
partition_us is the sum over the model's launches; all rows `ok=True`, per-slot `rel_err` and output bounds of the flow unchanged.
| problem | old seeds (switch off) | new seeds (default) | per op, old -> new (ms) | rel_err new (gate 2e-3 / output bound) |
|---|---|---|---|---|
| L3 8_ResNetBasicBlock | 3987.8 ms (1 run, the identical slots from `cc_resnetblock/seed.out.json`; my own re-run was not made, budget) | 58.3 / 57.9 / 58.5 ms (3 reps) = 68x | conv1 883 -> 2.29, conv2 1010 -> 52.9, downsample 873 -> 0.50, bn 406 -> 0.46, relu 0.74 -> 0.39, add 0.80 -> 0.60 | conv 4.7e-5, bn 1.0e-4, out err 4.9e-4 (bound 1.6e-2) |
| L2 1_Conv2D_ReLU_BiasAdd | 1689.7 ms | 32.2 ms (3 reps: 32.16/32.12/32.26) = 52x | conv 579.5 -> 30.3, relu 0.47 -> 0.25, add (ewise2) 1109.8 -> 1.66 | conv 3.7e-5, ewise2 0.0, out err 6.1e-5 (bound 3.2e-2) |
| L1 1_Square_matrix_multiplication_ (1024^3) | 298.6 ms | 43.9 / 43.6 / 42.9 ms = 6.9x (about 50 GFLOP/s) | matmul | 0.0 (old seed 8.7e-4); bound 2.87 |
| L1 19_ReLU (1.57 M) | 349 us | 185 us | relu | 6.1e-5 |
Reference points (their REPORTs, not re-measured by me): flow-final resnet 188.6 ms, QNN execute 14.5 ms; best hand-directed session: resnet 77 ms, conv-relu-bias 18.0 ms (conv 17.2 ms, ewise 0.57 ms), conv-div-leaky 3.3 ms. So the new seeds start at 58 / 32 ms WITHOUT any LLM step (below the flow's final 188.6 / 97.4 ms), but my conv is slower than the hand-tuned ones on conv2 (52.9 vs 36 ms) and conv-relu-bias (30.3 vs 17.2 ms) and faster on the small-IC convs of resnet (conv1 2.3 vs 19.4 ms, downsample 0.5 vs 19.1 ms). The conv reaches ~70 GFLOP/s (resnet conv2: 3.7 GFLOP in 52.9 ms), ~6 % of the fp32 peak, 5-13x below QNN. This is a starting point, not a finished kernel.

Experiments (same board):
- fp16 vs fp32 accumulation, resnet conv2 (sliding-window version of the seed): fp32 acc 65.3 ms, rel_err 4.7e-5; half8 acc flushed every 16 taps 71.4 ms, rel_err 1.48e-3 (passes 2e-3 narrowly). Not faster -> fp32 accumulate stays the default.
- sliding input window / interior fast path vs 4 independent bounds-checked loads: 65 vs 57 ms (one sample each) -> reverted to the simple loop (the final 3-rep run: conv2 52.9 ms).
- `MB_OT=4` vs 8, conv-relu-bias (IC=64, OC=128): 46.3 vs 30.1 ms -> 8 stays default; 4 was 1.8x better at IC=8 in `cc_conv_div_leaky`.
- BatchNorm with a half8 `fma()`: 38.3 ms; with an fp32 `mad` on widened data: 0.46 ms (same board state, 3.2 M elements) -> a half fma is emulated; avoid.
- bias add `ewise2`: first version (opcode `switch` with pow(), 32-bit index) 25.9 ms and rel_err 6.1e-4 (its scalar tail cast `(half)float` is not round-to-nearest-even on Adreno); per-opcode kernels + `convert_half`: 1.66 ms, rel_err 0.0 (hand-tuned session: 0.57 ms). `(half)x` casts were replaced by `convert_half` everywhere in the new seeds.
- `MB_SCRATCH_IMAGE` (image2d from a slot-owned buffer, read with `read_imageh`): works on the board, correct (relu through the image path 369 us vs 185 us direct: the staging pass costs).
What the evidence says about the original hypothesis: the order-of-magnitude improvement over the old seeds comes from native half vector I/O, register tiling, weight re-layout, 32-bit index math and avoiding the opcode switch, NOT from half ALU arithmetic (which was never faster here). Half arithmetic stays available (relu/relu6/add/mul are pure half8) and the prompt says so, with the measurements.

## What was changed, defaults and switches
See CHANGES.md. In short: default ON for adreno650 (device label gate), `MB_OPENCL_F16_MATH=0` restores today's storage-only seeds (slot files byte-identical to the base tree, checked by diff of all 88 fp16 seeds, and by a unit test); the NVIDIA stand-in path (no label) keeps the old seeds and the old prompt byte for byte. Prompt: hardware block (switch `MB_OPENCL_HW_INFO`), fp16 rule with the per-slot bound (the old OpenCL prompt told the LLM `fp32 in/out` even for fp16 problems and never gave the bound), OpenCL C 2.0, build options, images, calibration, 12 new technique-catalogue entries + 7 knowledge-seed techniques.

## Restrictions found and changed (C)
1. `-cl-std=CL1.2` hard-coded in `mb_kernel()` -> device's OpenCL C level (CL2.0 on Adreno), `MB_CL_STD` overrides; the legacy export (`MB_OPENCL_F16_MATH=0`) keeps CL1.2. Reason: OpenCL C 2.0 (generic address space, cl_khr_subgroups) was invisible to the agent. Not measured: whether CL2.0 changes the old seeds' speed (the old-seed rows above used the legacy CL1.2 head).
2. cl_khr_fp16 never enabled -> prelude enables it when the device lists it (`MB_HAS_FP16`).
3. Prompt: "OpenCL C 1.2 kernels", "fp32 in/out" (also for fp16), "-cl-mad-enable" as the only option hint -> corrected; `-cl-fast-relaxed-math`, `-cl-unsafe-math-optimizations`, `-cl-finite-math-only`, `-cl-denorms-are-zero` offered via `MB_CL_OPTS` (nothing was forced before either; the restriction was the text). Caveat stated: fast-math reassociates mb_kadd.
4. Images / cl_mem: `cl_mem` arguments were already accepted by `mb_launch`, but a slot could not obtain one (SVM ABI, clCreate* banned) -> `MB_SCRATCH_IMAGE` runtime helper (slot-owned buffer + image2d view); the ban on clCreate*/clEnqueue*/clFinish in slots and "all launches through mb_launch" are unchanged and stated in the prompt. Zero-copy `cl_qcom_ext_host_ptr` is described but not exposed.
5. Roofline peaks for fp16 problems still use the fp32 peak (`OPENCL_PEAKS`); not changed (state classification only).
No correctness check was weakened (per-slot bound, output bound, partition check untouched).

## QNN GPU backend, from the QAIRT 2.45.0.260326 docs (garden-aws, read-only)
Files (`QAIRT-Docs/QNN/`): `general/gpu/gpu_backend.html`, `OpDef/SupportedOps.html` (GPU column), `OpDef/GpuOpDefSupplement.html`, also seen: `general/gpu/gpu_tuning_mode_tutorial.html`, `gpu_qnnmem_api_tutorial.html`.
- Precision modes: FP32 (fp32 data + fp32 accumulator), FP16 (fp16 data, FP16 accumulator "where possible", best speed, lowest accuracy), HYBRID (fp16 data, FP32 accumulator), USER_PROVIDED (default, no conversion). Context perf hint HIGH (default, max clocks).
- Node optimisations: fusion of compatible ops into one op (disable with disableNodeOptimizations / `qnn-net-run --debug`), shared tensor memory, queue recording (`cl_qcom_recordable_queues`), kernel registry / on-disk kernel repository, weight sharing, optional tuning mode (kernels profiled iteratively, best kept; beta), QnnMem API for user OpenCL buffers (no host copies). Variable (dynamic) input/output dims are not supported.
- GPU-supported ops (101 of 160 in the table; fp32 and fp16 tensors): Argmax/Argmin, Batchnorm, Cast, Concat, Conv2d, DepthWiseConv2d, TransposeConv2d, FullyConnected, MatMul (transpose_in0 must be false), GroupNorm, InstanceNorm, LayerNorm, RmsNorm, Softmax/LogSoftmax, L2Norm, PoolAvg2d/PoolMax2d, Relu/Relu1/Relu6/ReluMinMax, Prelu, Elu, Gelu, HardSwish, Sigmoid, Tanh, the ElementWise binary/unary family, Reduce{Sum,Mean,Max,Min,Prod}, CumulativeSum, Pad, Resize/ResizeBilinear/ResizeNearestNeighbor, Transpose, Gather/GatherNd, TopK, Tile, Split/Pack/UnPack, StridedSlice, DepthToSpace/SpaceToDepth, Lstm, GridSample, RoiAlign, Dequantize. Not on GPU: Conv1d, Conv3d, DepthWiseConv1d, Gru, PoolAvg3d/PoolMax3d, TransposeConv1d/3d, Quantize, ElementWiseMod/Fmod/Sign/Ceil, OneHot, NonZero, ScatterNd.
- The docs do NOT say which OpenCL extensions or kernel designs QNN uses internally. The prompt therefore states only the documented facts above as fact, and says that image / texture paths and subgroup operations in QNN are inferred (from the extensions the device exposes and QNN's measured speed), not documented.

## Not verified
- `linear_f16` (fp16-math) and `MB_ACC_HALF` with `MB_OT=4`, `ewise2` non-bias paths (contiguous v8 kernel, generic kernel), `ewise1/ewise_scalar` math templates, `batchnorm` HW%8!=0 path, ops other than those in the table: syntax-checked only (clang `-x cl`), never run on the board. The final resnet / matmul rows ran before three small edits (coalesced weight-pack kernel, `convert_half` tails, ewise2 rewrite); the conv-relu-bias row (new3) ran after them, and confirms the new weight-pack and ewise2.
- No LLM run: whether the new prompt/seeds change what the agent finds is not measured. 
- The host-side runtime header was only compiled by the board's g++ (no OpenCL 2.0 headers on this Mac); local check was `-fsyntax-only` with the macOS 1.2 headers, whose only errors are the pre-existing SVM calls.
- Timings are single/3-rep runs on a shared board; the old-seed resnet number is from another session's run of the identical sources. Peak numbers are public specs.
- Whether OpenCL C 2.0 vs 1.2 changes the speed of the old seeds; the effect of the fp16 default on the full graph-optimiser flow.
