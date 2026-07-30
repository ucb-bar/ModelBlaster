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

## Corpus status (2026-07-30)
Extractable level1 benches: **34 → 80 / 100**. Full-suite spike run:
**79 / 80 PASS** on RVV/spike fp32 (`results/rvv_fp32.md`). 3D family in
`results/rvv_fp32_3d.md`, matmul in `results/rvv_fp32_matmul.md`.

Only `10_3D_tensor_matrix_multiplication` fails (max_abs_err=22): a 3D×2D
broadcast matmul the plain `matmul` op computes wrong — needs a
batched-broadcast handler (deferred).

## Remaining not-extractable (20) — next targets
- **Losses (7)** — CrossEntropy/MSE/Huber/KLDiv/Hinge/TripletMargin (scalar
  output, out of scope).
- **Cumulative (5)** — cumsum/cumprod/flip/select.
- **Matmul variants (4)** — diag/triu/tril/einsum.
- **Misc** — abs (L1Norm 38), dilated conv2d (76/80), SDPA (97, hardcodes cuda),
  10_3D broadcast matmul.
