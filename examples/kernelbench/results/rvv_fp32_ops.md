# KernelBench level1 op-coverage additions (RVV / fp32, reference, spike)

New fp32 ops added to reach broader level1 coverage. Each is a fresh KernelSpec
(reference impl) + `KERNEL_SPECS` entry + extract handler + skeleton emission,
following the existing `_s8`/`_f16` variants. All verified on spike.

| bench | new op | max_abs_err | cycles |
|---|---|---|---|
| 23_Softmax | softmax | 4.66e-09 | 4,532,605 |
| 24_LogSoftmax | log_softmax | 9.54e-07 | 4,510,536 |
| 40_LayerNorm | layer_norm | 1.35e-05 | 1,507,943 |
| 45_Average_Pooling_2D | avgpool2d | 5.96e-08 | 407,110 |

- **softmax / log_softmax** — last-axis only (`M`=leading dims, `K`=last);
  numerically stable max-subtract / log-sum-exp. `torch.softmax` /
  `F.log_softmax` call_function handlers.
- **avgpool2d** — `nn.AvgPool2d`, count_include_pad=True (divisor KH*KW), no
  dilation. Handles `stride=None` → kernel_size.
- **layer_norm** — `nn.LayerNorm`; K = prod(normalized_shape), M = leading;
  gamma/beta flattened to K (ones/zeros when affine off).

## Corpus status
Extractable level1 benches: **34 → 51 / 100** (this batch +4, matmul family
+13 net counting 10_3D which extracts but fails verification). Full-suite run:
**50 / 51 PASS** on RVV/spike fp32 (`results/rvv_fp32.md`).

`10_3D_tensor_matrix_multiplication` extracts but fails verification
(max_abs_err=22): it's a 3D×2D broadcast matmul that the plain `matmul` op
computes incorrectly — needs a batched-broadcast matmul handler (deferred).

## Remaining not-extractable (49) — next targets
Conv1d/Conv3d, ConvTranspose{1,2,3}d (large family, ~20 benches), InstanceNorm2d,
GroupNorm, RMSNorm, AvgPool1d/3d, MaxPool1d/3d, cumsum/cumprod, diag/triu/tril/
einsum, abs (L1Norm compound), pow, and the loss family (94–100, out of scope).
