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

This was the first of several op batches. Later batches added: group_norm +
rms_norm (34/35/36), conv_transpose2d (7 benches), and 1D conv/pool (6 benches)
— see notes/kernelbench_rvv_port_plan.md Phase 3.5.

## Corpus status (2026-07-30) — COMPLETE
Extractable level1 benches: **34 → 100 / 100**. Full-suite spike run:
**100 / 100 PASS** on RVV/spike fp32 (`results/rvv_fp32.md`). 3D family in
`results/rvv_fp32_3d.md`, matmul in `results/rvv_fp32_matmul.md`.

Every KernelBench level1 problem now extracts and runs correctly on RVV/spike
via the reference kernels. Later batches beyond this doc: matmul variants
(triu/tril/diag_matmul/einsum + N-D matmul), cumulative (cumsum/cumprod/flip/
exclusive_cumsum) + elementwise mul, L1Norm, the loss family (mse/huber/hinge/
cross_entropy/kldiv/triplet), dilated conv2d, mul_scalar, and sdpa. See
notes/kernelbench_rvv_port_plan.md Phase 3.5 for the per-batch commit list.
