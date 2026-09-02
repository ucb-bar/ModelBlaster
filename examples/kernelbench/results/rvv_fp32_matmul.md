# KernelBench level1 matmul family on RVV / fp32 (reference backend, spike)

**12 / 18 PASS.** Unblocked by relaxing `_load_kernelbench` to return multi-input
forwards (`forward(A, B)`), which extract() already threads through its
`packed_inputs` path (io.npz packs A‖B into one flat buffer; the skeleton reads
`input + offset` per operand). No new ops were needed — the `matmul`,
`matmul_ta`, `matmul_tb`, `matmul_tatb`, and `bmm` reference impls already
existed. Shapes are auto-shrunk to ≤65536 total input elements.

| bench | op | max_abs_err | cycles |
|---|---|---|---|
| 1_Square_matrix_multiplication | matmul | 0 | 14,878,613 |
| 2_Standard_matrix_multiplication | matmul | 0 | 29,756,096 |
| 3_Batched_matrix_multiplication | bmm | 0 | 7,716,526 |
| 4_Matrix_vector_multiplication | matmul | 3.05e-05 | 232,853 |
| 6_Matmul_with_large_K | matmul | 6.10e-05 | 29,559,488 |
| 7_Matmul_with_small_K | matmul | 0 | 120,598,039 |
| 8_Matmul_with_irregular_shapes | matmul | 0 | 30,786,752 |
| 9_Tall_skinny_matrix_multiplication | matmul | 0 | 247,487,257 |
| 13_Matmul_for_symmetric_matrices | matmul | 0 | 14,878,613 |
| 16_Matmul_with_transposed_A | matmul_ta | 0 | 29,756,097 |
| 17_Matmul_with_transposed_B | matmul_tb | 0 | 29,755,967 |
| 18_Matmul_with_transposed_both | matmul_tatb | 0 | 29,756,096 |

_12 PASS / 6 FAIL_

## The 6 that need new handlers (deferred — niche matmul variants)
| bench | blocker | note |
|---|---|---|
| 5_Matrix_scalar_multiplication | `'float' has no .shape` | 2nd "input" is a Python scalar → really `mul` by scalar; needs scalar-broadcast handling, not matmul |
| 10_3D_tensor_matrix_multiplication | matmul over 3D×2D | `torch.matmul` broadcasting a 3D batch against a 2D matrix; bmm handler expects 3D×3D |
| 11_4D_tensor_matrix_multiplication | `einsum` unsupported | uses `torch.einsum` |
| 12_Matmul_with_diagonal_matrices | `diag` unsupported | needs a `diag` extract handler + op |
| 14_Matmul_for_upper_triangular | `triu` unsupported | needs a `triu` masking op |
| 15_Matmul_for_lower_triangular | `tril` unsupported | needs a `tril` masking op |

New distinct op coverage from this batch: **matmul, matmul_ta, matmul_tb,
matmul_tatb, bmm** (5 ops).
