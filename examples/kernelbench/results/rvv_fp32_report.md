# KernelBench level1 on RVV / fp32 — Phase 3 report (reference backend, spike)

**34 / 34 PASS** — every level1 bench that extracts cleanly on fp32 runs
correctly on RVV via the reference kernels compiled with `-march=rv64gcv`
(spike `--isa=rv64gcv_zicntr`, board `spike_riscv64`). `rvv_fp32.md` is the
machine-generated PASS/err table; this file adds per-op cycles + triage.

Reproduce (sequential): `bash run_all.sh` with `BENCHES=<comma-list>`.
Parallel (48-core host): prepend `JOBS=8` — benches fan out (isolated
example/build/run dirs). Cycles are summed hardware-counter cycles from spike's
`MODELBLASTER_PROFILE` (one op per ELF). Inputs are shrunk to ≤65536 elements
(`--bench-max-elements`) to fit spike memory, so cycles are indicative, not the
stock-shape KernelBench numbers.

| bench | op | max_abs_err | cycles |
|---|---|---|---|
| 19_ReLU | relu | 0 | 639,244 |
| 20_LeakyReLU | leaky_relu | 0 | 540,945 |
| 21_Sigmoid | sigmoid | 1.19e-07 | 3,951,419 |
| 22_Tanh | tanh | 1.19e-07 | 6,229,051 |
| 25_Swish | swish | 1.19e-07 | 4,001,542 |
| 26_GELU | gelu | 1.79e-07 | 4,423,994 |
| 27_SELU | selu | 0 | 590,139 |
| 28_HardSigmoid | hardsigmoid | 5.96e-08 | 786,712 |
| 29_Softplus | softplus | 1.19e-07 | 8,437,242 |
| 30_Softsign | softsign | 0 | 737,550 |
| 31_ELU | elu | 0 | 540,971 |
| 32_HardTanh | hardtanh | 0 | 639,250 |
| 33_BatchNorm | batchnorm2d | 0 | 560,561 |
| 37_FrobeniusNorm | frobenius_norm | 3.26e-08 | 1,032,467 |
| 39_L2Norm | l2_norm | 3.73e-08 | 919,572 |
| 42_Max_Pooling_2D | maxpool2d | 0 | 16,867,209 |
| 47_Sum_reduction | sum_dim | 5.72e-06 | 343,376 |
| 48_Mean_reduction | mean_dim | 1.79e-07 | 345,396 |
| 49_Max_reduction | max_dim | 0 | 416,009 |
| 50_conv_std_2D_sq_sq | conv2d | 1.07e-06 | 2,117,279,370 |
| 51_Argmax | argmax_dim | 0 | 492,823 |
| 52_Argmin | argmin_dim | 0 | 492,645 |
| 53_Min_reduction | min_dim | 0 | 416,040 |
| 55_conv_std_2D_asym_sq | conv2d | 8.94e-07 | 354,010,343 |
| 56_conv_std_2D_asym_asym | conv2d | 1.04e-06 | 472,166,250 |
| 62_conv_std_2D_sq_asym | conv2d | 1.16e-06 | 426,193,172 |
| 63_conv_std_2D_sq_sq | conv2d | 5.36e-07 | 177,383,692 |
| 82_conv_dw_2D_sq_sq | conv2d_dw | 0 | 11,445,104 |
| 83_conv_dw_2D_sq_asym | conv2d_dw | 0 | 6,999,280 |
| 84_conv_dw_2D_asym_sq | conv2d_dw | 0 | 11,445,104 |
| 85_conv_dw_2D_asym_asym | conv2d_dw | 0 | 16,362,353 |
| 86_conv_dw_separable_2D | conv2d (+dw) | 0 | 61,260,197 |
| 87_conv_pointwise_2D | conv2d (1×1) | 0 | 47,118,857 |
| 88_MinGPTNewGelu | gelu_exact | 1.19e-07 | 8,551,376 |

## Distinct ops exercised (25)
relu, leaky_relu, sigmoid, tanh, swish/silu, gelu, gelu_exact, selu,
hardsigmoid, softplus, softsign, elu, hardtanh, batchnorm2d, frobenius_norm,
l2_norm, maxpool2d, sum_dim, mean_dim, max_dim, min_dim, argmax_dim,
argmin_dim, conv2d, conv2d_dw.

The reference conv is a naïve direct loop, so the big standard convs dominate
cycles (50 = 2.1B cyc) — this is exactly the op the future `BACKEND=llm`/
KernelBlaster path is meant to optimize (Phase 4).

## Fixes required for these to pass (this session)
- **fp32 `conv2d` weight layout** — the rvv backend packs conv weights IHWO
  (`_backend_pack_weight`, perm 1,2,3,0) but the fp32 `conv2d` reference impl
  read OIHW → wrong result. Added the `MODELBLASTER_RVV_IHWOC_WEIGHTS` /
  `GEMMINI_HWIO_WEIGHTS` IHWO branch (mirrors the int8 conv2d_s8 fix).
- **fp32 `conv2d_dw` weight layout** — same override hits depthwise
  `[C,1,KH,KW]` weights; added the IHWO branch (`weight[(kh*KW+kw)*C+c]`).
- **submodule env plumbing** — `_run_lib.sh` now exports `PYTHONPATH=<parent
  repo>` before the REPO_ROOT redirect so `python -m modelblaster.pipeline.*`
  resolves from the nested submodule layout.

## Triage of the other 66 level1 benches (not run)
- **Matmul family (1–18, ~15 benches)** — `matmul`/`bmm`/`linear` reference ops
  already exist; blocked only by the loader rejecting the 2-input
  `forward(A, B)` signature. Binding extra forward args as constant inputs in
  `_load_kernelbench` would unlock these (all one op: `matmul`).
- **Unsupported nn modules** — Conv1d/Conv3d, ConvTranspose{1,2,3}d,
  AvgPool{1,2,3}d, InstanceNorm2d, GroupNorm, LayerNorm(fp32), RMSNorm. Need
  extract handlers (+ fp32 reference impls where an `_s8`/`_f16` variant exists,
  e.g. layer_norm).
- **Unsupported functions** — softmax / log_softmax (fp32; `_s8`/`_f16` exist),
  abs (→ l1_norm), cumsum / cumprod, pow (RMSNorm).
- **Losses (94–100)** — MSE/CrossEntropy/Huber/KLDiv/Hinge/TripletMargin;
  multi-input + scalar output, out of first-cut scope.
- **97_ScaledDotProductAttention** — hard-codes `device='cuda'`; needs a CPU
  patch to extract.
