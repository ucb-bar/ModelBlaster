# KernelBench level1 3D conv/pool family on RVV / fp32 (reference, spike)

**13 / 13 PASS.** Four new 5D-tensor (NCDHW) reference ops: `conv3d`,
`conv_transpose3d`, `maxpool3d`, `avgpool3d`. 5D conv weights are left in
natural OIDHW order (the backend's IHWO repack is 4D-only), so no weight-layout
branch is needed. output_padding is absorbed by reading OD/OH/OW from the
traced output shape.

| bench | op | max_abs_err |
|---|---|---|
| 54_conv_standard_3D_sq_sq | conv3d | 0 |
| 59_conv_standard_3D_asym_sq | conv3d | 0 |
| 60_conv_standard_3D_sq_asym | conv3d | 0 |
| 66_conv_standard_3D_asym_asym | conv3d | 0 |
| 58_conv_transposed_3D_asym_asym | conv_transpose3d | 0 |
| 61_conv_transposed_3D_sq_sq | conv_transpose3d | 9.54e-07 |
| 68_conv_transposed_3D_sq_asym | conv_transpose3d | 0 |
| 70_conv_transposed_3D_asym_sq | conv_transpose3d | 0 |
| 72_conv_transposed_3D_strided_padded_grouped | conv_transpose3d | 1.49e-07 |
| 73_conv_transposed_3D_strided_padded_grouped | conv_transpose3d | 0 |
| 77_conv_transposed_3D_padded_dilated_strided | conv_transpose3d | 5.96e-07 |
| 43_Max_Pooling_3D | maxpool3d | 0 |
| 46_Average_Pooling_3D | avgpool3d | 5.96e-08 |

_13 PASS / 0 FAIL_

Representative cycles (indicative, shrunk shapes): conv3d 54 = 117.1M,
conv_transpose3d 58 = 828.2M (the naïve gather is the slow path — a `BACKEND=llm`
target in Phase 4), maxpool3d 43 = 0.61M, avgpool3d 46 = 2.81M.

Corpus after this batch: **80 / 100 extractable, 79 / 80 PASS** (only 10_3D
broadcast matmul fails).
