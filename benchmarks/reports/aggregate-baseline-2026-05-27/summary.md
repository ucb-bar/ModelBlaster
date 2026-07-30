# Benchmark dashboard

One section per workload, split into two phases:

- **pre-kernel**: graph-side metrics (fusion / fold pass fires,
  IR op counts, static cross-tile traffic). Deterministic across
  arms — A and B should agree here; divergence is a bug.
- **kernel synthesis**: properties of the compiled artifact
  (cycles, accuracy, makespan, LLM token cost, beam trajectory).
  Diverges per arm because the synthesis strategy differs.

Each per-workload section is followed by a top-op breakdown
and (when hetero) a per-op-x-tile rollup.


## `mlp_generic_scalar_smoke`
model `mlp_generic`, target `scalar`, quant `int8`, runner `spike`

## `dronet_rvv_smoke`
model `dronet`, target `rvv`, quant `int8`, runner `spike`

### pre-kernel — graph compilation

| metric | A | unit |
| --- | --- | --- |
| passes_fired_total | 0 (N=3) | count |
| linear_relu_fuse_fired | 0 (N=3) | count |
| conv2d_relu_fuse_fired | 0 (N=3) | count |
| ir_op_count | 32 (N=3) | count |
| n_input_nodes | 34 (N=3) | count |
| lowering_ratio | 0.9412 (N=3) | fraction |
| n_dispatches_graph | 32 (N=3) | count |
| n_distinct_op_kinds | 8 (N=3) | count |
| n_distinct_shapes | 25 (N=3) | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | unit |
| --- | --- | --- |
| cycles_spike | 27,563,408 (N=3) | cycles |
| wall_cycles_spike | 275,700 (N=3) | cycles |
| accuracy_linf | 0 (N=3) | dimensionless |
| verify_pass | yes (N=3) | bool |
| bit_exact | yes (N=3) | bool |
| latency_ms_spike | 27.56 (N=3) | ms |
| zephyr_elf_bytes | 895,872 (N=3) | bytes |
| kernels_c_bytes | 19,251 (N=3) | bytes |
| kernels_c_loc | 396 (N=3) | count |
| weights_npz_bytes | 326,642 (N=3) | bytes |
| extract_s | 0.0010 (N=3) | s |
| generate_skeleton_s | 0.1853 ± 0.0040 (N=3) | s |
| generate_kernels_s | 29.41 ± 0.3747 (N=3) | s |
| build_s | 5.775 ± 0.0527 (N=3) | s |
| run_s | 0.9013 ± 0.0146 (N=3) | s |
| total_stage_s | 36.27 ± 0.3505 (N=3) | s |
| n_kernels_curated | 4 (N=3) | count |
| n_kernels_reference | 3 (N=3) | count |
| n_kernels_total | 7 (N=3) | count |
| algorithms_distinct_count | 2 (N=3) | count |
| compile_wall_clock_s | 36.34 ± 0.3502 (N=3) | s |
| compile_peak_rss_mb | 137.5 (N=3) | MB |
| n_ops_profiled | 30 (N=3) | count |
| dominant_op_share | 0.8819 (N=3) | fraction |
| mean_cycles_per_dispatch | 918,780 (N=3) | cycles |
| stddev_cycles_per_dispatch | 1,492,867 (N=3) | cycles |
| op_kind_p95_max_cycles | 5,035,721 (N=3) | cycles |
| op_kind_median_max_cycles | 2,526,845 (N=3) | cycles |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 88.2% | 24,306,893 |
| A | batchnorm2d_s8 | 9.2% | 2,540,697 |
| A | add_s8 | 1.8% | 490,603 |
| A | maxpool2d_s8 | 0.8% | 218,279 |
| A | relu_s8 | 0.0% | 3,714 |

## `dronet_scalar_smoke`
model `dronet`, target `scalar`, quant `int8`, runner `spike`

### pre-kernel — graph compilation

| metric | A | unit |
| --- | --- | --- |
| passes_fired_total | 0 (N=3) | count |
| linear_relu_fuse_fired | 0 (N=3) | count |
| conv2d_relu_fuse_fired | 0 (N=3) | count |
| ir_op_count | 32 (N=3) | count |
| n_input_nodes | 34 (N=3) | count |
| lowering_ratio | 0.9412 (N=3) | fraction |
| n_dispatches_graph | 32 (N=3) | count |
| n_distinct_op_kinds | 8 (N=3) | count |
| n_distinct_shapes | 25 (N=3) | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | unit |
| --- | --- | --- |
| cycles_spike | 454,542,207 (N=3) | cycles |
| wall_cycles_spike | 4,546,550 (N=3) | cycles |
| accuracy_linf | 0 (N=3) | dimensionless |
| verify_pass | yes (N=3) | bool |
| bit_exact | yes (N=3) | bool |
| latency_ms_spike | 454.5 (N=3) | ms |
| zephyr_elf_bytes | 879,944 (N=3) | bytes |
| kernels_c_bytes | 9,222 (N=3) | bytes |
| kernels_c_loc | 202 (N=3) | count |
| weights_npz_bytes | 326,642 (N=3) | bytes |
| extract_s | 2.382 ± 4.125 (N=3) | s |
| generate_skeleton_s | 0.1987 ± 0.0058 (N=3) | s |
| generate_kernels_s | 0.1837 ± 0.0064 (N=3) | s |
| build_s | 6.534 ± 0.0229 (N=3) | s |
| run_s | 1.638 ± 0.0170 (N=3) | s |
| total_stage_s | 10.94 ± 4.153 (N=3) | s |
| n_kernels_curated | 0 (N=3) | count |
| n_kernels_reference | 7 (N=3) | count |
| n_kernels_total | 7 (N=3) | count |
| compile_wall_clock_s | 11.01 ± 4.152 (N=3) | s |
| compile_peak_rss_mb | 297.7 ± 343.8 (N=3) | MB |
| n_ops_profiled | 30 (N=3) | count |
| dominant_op_share | 0.9806 (N=3) | fraction |
| mean_cycles_per_dispatch | 15,151,407 (N=3) | cycles |
| stddev_cycles_per_dispatch | 27,446,276 (N=3) | cycles |
| op_kind_p95_max_cycles | 87,330,573 (N=3) | cycles |
| op_kind_median_max_cycles | 48,397,734 (N=3) | cycles |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 98.1% | 445,731,438 |
| A | maxpool2d_s8 | 1.1% | 5,146,711 |
| A | batchnorm2d_s8 | 0.6% | 2,540,589 |
| A | relu_s8 | 0.1% | 571,054 |
| A | add_s8 | 0.1% | 490,603 |

## `dronet_rvv_opu_int8`
model `dronet`, target `rvv_opu`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | 0 (N=3) | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| ir_op_count | 32 (N=3) | 32 (N=3) | — | — | count |
| n_input_nodes | 34 (N=3) | 34 (N=3) | — | — | count |
| lowering_ratio | 0.9412 (N=3) | 0.9412 (N=3) | — | — | fraction |
| n_dispatches_graph | 32 (N=3) | 32 (N=3) | — | — | count |
| n_distinct_op_kinds | 8 (N=3) | 8 (N=3) | — | — | count |
| n_distinct_shapes | 25 (N=3) | 25 (N=3) | — | — | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on rvv_opu)* | 454,543,545 (N=3) | 15,334,283 ± 10,827 (N=3) | — | — | cycles |
| wall_cycles_spike | 4,546,550 (N=3) | 153,367 ± 125.8 (N=3) | — | — | cycles |
| accuracy_linf | 0 (N=3) | 0 (N=3) | — | — | dimensionless |
| verify_pass | yes (N=3) | yes (N=3) | — | — | bool |
| bit_exact | yes (N=3) | yes (N=3) | — | — | bool |
| zephyr_elf_bytes | 888,784 (N=3) | 900,555 ± 391 (N=3) | — | — | bytes |
| kernels_c_bytes | 9,273 (N=3) | 19,863 ± 330.5 (N=3) | — | — | bytes |
| kernels_c_loc | 204 (N=3) | 424.7 ± 3.055 (N=3) | — | — | count |
| weights_npz_bytes | 326,642 (N=3) | 326,642 (N=3) | — | — | bytes |
| extract_s | 0.0010 (N=3) | 0.0017 ± 5.774e-04 (N=3) | — | — | s |
| generate_skeleton_s | 0.2033 ± 0.0166 (N=3) | 0.1753 ± 0.0181 (N=3) | — | — | s |
| generate_kernels_s | 410.9 ± 347.2 (N=3) | 1,347 ± 252.3 (N=3) | — | — | s |
| build_s | 6.279 ± 0.4694 (N=3) | 6.509 ± 0.0354 (N=3) | — | — | s |
| run_s | 1.489 ± 0.0339 (N=3) | 0.7843 ± 0.0616 (N=3) | — | — | s |
| total_stage_s | 418.8 ± 347.7 (N=3) | 1,355 ± 252.3 (N=3) | — | — | s |
| n_kernels_curated | 0 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_reference | 7 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_cached | — | 0 (N=3) | — | — | count |
| n_kernels_llm | — | 7 (N=3) | — | — | count |
| n_kernels_total | 7 (N=3) | 7 (N=3) | — | — | count |
| algorithms_distinct_count | — | 3 (N=3) | — | — | count |
| compile_wall_clock_s | 418.9 ± 347.7 (N=3) | 1,355 ± 252.3 (N=3) | — | — | s |
| compile_peak_rss_mb | 133.7 ± 2.491 (N=3) | 140.2 (N=3) | — | — | MB |
| n_ops_profiled | 30 (N=3) | 30 (N=3) | — | — | count |
| dominant_op_share | 0.9806 (N=3) | 0.9840 ± 6.947e-04 (N=3) | — | — | fraction |
| mean_cycles_per_dispatch | 15,151,452 (N=3) | 511,143 ± 360.9 (N=3) | — | — | cycles |
| stddev_cycles_per_dispatch | 27,446,353 (N=3) | 1,063,359 ± 126.5 (N=3) | — | — | cycles |
| op_kind_p95_max_cycles | 87,330,830 (N=3) | 3,721,125 (N=3) | — | — | cycles |
| op_kind_median_max_cycles | 48,397,872 (N=3) | 1,495,454 (N=3) | — | — | cycles |
| beam_n_candidates_total | — | 34 ± 4.583 (N=3) | — | — | count |
| beam_n_candidates_viable | — | 19.33 ± 4.619 (N=3) | — | — | count |
| beam_n_candidates_build_fail | — | 3.667 ± 1.528 (N=3) | — | — | count |
| beam_n_candidates_verify_fail | — | 1.667 ± 0.5774 (N=3) | — | — | count |
| beam_n_candidates_duplicate | — | 9.333 ± 1.528 (N=3) | — | — | count |
| beam_tokens_per_candidate_mean | — | 18,755 ± 174.2 (N=3) | — | — | tokens |
| beam_best_improvement_pct | — | 49.72 ± 40.34 (N=3) | — | — | percent |
| beam_iter_to_best | — | 1.333 ± 0.5774 (N=3) | — | — | iteration |
| tokens_input_cached | — | 0 (N=3) | — | — | tokens |
| tokens_input_uncached | — | 667,138 ± 84,604 (N=3) | — | — | tokens |
| tokens_output | — | 46,453 ± 7,676 (N=3) | — | — | tokens |
| dollars_equivalent | — | 2.698 ± 0.3669 (N=3) | — | — | USD |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 98.1% | 445,732,661 |
| A | maxpool2d_s8 | 1.1% | 5,146,770 |
| A | batchnorm2d_s8 | 0.6% | 2,540,645 |
| A | relu_s8 | 0.1% | 571,054 |
| A | add_s8 | 0.1% | 490,603 |
| B-bedrock | conv2d_s8 | 98.5% | 15,089,334 |
| B-bedrock | maxpool2d_s8 | 1.2% | 177,669 |
| B-bedrock | batchnorm2d_s8 | 0.3% | 42,782 |
| B-bedrock | add_s8 | 0.1% | 9,383 |
| B-bedrock | relu_s8 | 0.0% | 3,366 |

## `dronet_gemmini_int8`
model `dronet`, target `gemmini`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | 0 (N=3) | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| ir_op_count | 32 (N=3) | 32 (N=3) | — | — | count |
| n_input_nodes | 34 (N=3) | 34 (N=3) | — | — | count |
| lowering_ratio | 0.9412 (N=3) | 0.9412 (N=3) | — | — | fraction |
| n_dispatches_graph | 32 (N=3) | 32 (N=3) | — | — | count |
| n_distinct_op_kinds | 8 (N=3) | 8 (N=3) | — | — | count |
| n_distinct_shapes | 25 (N=3) | 25 (N=3) | — | — | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on gemmini)* | 18,808,376 (N=3) | 15,531,641 ± 801,686 (N=3) | — | — | cycles |
| wall_cycles_spike | 188,150 (N=3) | 155,333 ± 8,019 (N=3) | — | — | cycles |
| accuracy_linf | 0 (N=3) | 0 (N=3) | — | — | dimensionless |
| verify_pass | yes (N=3) | yes (N=3) | — | — | bool |
| bit_exact | yes (N=3) | yes (N=3) | — | — | bool |
| zephyr_elf_bytes | 923,056 (N=3) | 930,387 ± 2,046 (N=3) | — | — | bytes |
| kernels_c_bytes | 16,337 (N=3) | 15,088 ± 1,524 (N=3) | — | — | bytes |
| kernels_c_loc | 361 (N=3) | 359.3 ± 26.69 (N=3) | — | — | count |
| weights_npz_bytes | 326,642 (N=3) | 326,642 (N=3) | — | — | bytes |
| extract_s | 0.0013 ± 5.774e-04 (N=3) | 0.0010 (N=3) | — | — | s |
| generate_skeleton_s | 0.1903 ± 0.0055 (N=3) | 0.2023 ± 0.0112 (N=3) | — | — | s |
| generate_kernels_s | 11.34 ± 0.1735 (N=3) | 351.7 ± 32.71 (N=3) | — | — | s |
| build_s | 6.163 ± 0.3864 (N=3) | 6.545 ± 0.0638 (N=3) | — | — | s |
| run_s | 0.4180 ± 0.0249 (N=3) | 0.4237 ± 0.0130 (N=3) | — | — | s |
| total_stage_s | 18.11 ± 0.5204 (N=3) | 358.9 ± 32.7 (N=3) | — | — | s |
| n_kernels_curated | 1 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_reference | 6 (N=3) | 5 (N=3) | — | — | count |
| n_kernels_cached | — | 0 (N=3) | — | — | count |
| n_kernels_llm | — | 2 (N=3) | — | — | count |
| n_kernels_total | 7 (N=3) | 7 (N=3) | — | — | count |
| algorithms_distinct_count | 1 (N=3) | 2 (N=3) | — | — | count |
| compile_wall_clock_s | 18.16 ± 0.5226 (N=3) | 359 ± 32.7 (N=3) | — | — | s |
| compile_peak_rss_mb | 99.8 (N=3) | 99.8 (N=3) | — | — | MB |
| n_ops_profiled | 30 (N=3) | 30 (N=3) | — | — | count |
| dominant_op_share | 0.5316 (N=3) | 0.4354 ± 0.0296 (N=3) | — | — | fraction |
| mean_cycles_per_dispatch | 626,946 (N=3) | 517,721 ± 26,723 (N=3) | — | — | cycles |
| stddev_cycles_per_dispatch | 1,361,828 (N=3) | 1,160,204 ± 24,869 (N=3) | — | — | cycles |
| op_kind_p95_max_cycles | 5,146,005 (N=3) | 5,146,005 (N=3) | — | — | cycles |
| op_kind_median_max_cycles | 5,146,005 (N=3) | 5,146,005 (N=3) | — | — | cycles |
| beam_n_candidates_total | — | 12 (N=3) | — | — | count |
| beam_n_candidates_viable | — | 11 ± 1 (N=3) | — | — | count |
| beam_n_candidates_build_fail | — | 0 (N=3) | — | — | count |
| beam_n_candidates_verify_fail | — | 0.3333 ± 0.5774 (N=3) | — | — | count |
| beam_n_candidates_duplicate | — | 0.6667 ± 0.5774 (N=3) | — | — | count |
| beam_tokens_per_candidate_mean | — | 9,429 ± 1,116 (N=3) | — | — | tokens |
| beam_best_improvement_pct | — | 31.36 ± 8.118 (N=3) | — | — | percent |
| beam_iter_to_best | — | 1 (N=3) | — | — | iteration |
| tokens_input_cached | — | 0 (N=3) | — | — | tokens |
| tokens_input_uncached | — | 81,502 ± 9,693 (N=3) | — | — | tokens |
| tokens_output | — | 31,646 ± 3,702 (N=3) | — | — | tokens |
| dollars_equivalent | — | 0.7192 ± 0.0846 (N=3) | — | — | USD |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 53.2% | 9,997,760 |
| A | maxpool2d_s8 | 27.4% | 5,146,005 |
| A | batchnorm2d_s8 | 13.5% | 2,540,435 |
| A | relu_s8 | 3.0% | 571,761 |
| A | add_s8 | 2.6% | 490,603 |
| B-bedrock | conv2d_s8 | 46.1% | 7,500,450 |
| B-bedrock | maxpool2d_s8 | 31.7% | 5,146,005 |
| B-bedrock | batchnorm2d_s8 | 15.6% | 2,540,435 |
| B-bedrock | relu_s8 | 3.5% | 571,054 |
| B-bedrock | add_s8 | 3.0% | 490,603 |

## `dronet_gemmini_q31_int8`
model `dronet`, target `gemmini_q31`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | 0 (N=3) | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| ir_op_count | 32 (N=3) | 32 (N=3) | — | — | count |
| n_input_nodes | 34 (N=3) | 34 (N=3) | — | — | count |
| lowering_ratio | 0.9412 (N=3) | 0.9412 (N=3) | — | — | fraction |
| n_dispatches_graph | 32 (N=3) | 32 (N=3) | — | — | count |
| n_distinct_op_kinds | 8 (N=3) | 8 (N=3) | — | — | count |
| n_distinct_shapes | 25 (N=3) | 25 (N=3) | — | — | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on gemmini_q31)* | 6,899,785 (N=3) | — | — | — | cycles |
| wall_cycles_spike | 69,050 (N=3) | — | — | — | cycles |
| accuracy_linf | 72 (N=3) | — | — | — | dimensionless |
| verify_pass | yes (N=3) | — | — | — | bool |
| bit_exact | no (N=3) | — | — | — | bool |
| zephyr_elf_bytes | 982,328 (N=3) | 982,328 (N=3) | — | — | bytes |
| kernels_c_bytes | 31,340 (N=3) | 31,339 (N=3) | — | — | bytes |
| kernels_c_loc | 713 (N=3) | 713 (N=3) | — | — | count |
| weights_npz_bytes | 326,642 (N=3) | 326,642 (N=3) | — | — | bytes |
| extract_s | 0.0013 ± 5.774e-04 (N=3) | 0.0010 (N=3) | — | — | s |
| generate_skeleton_s | 0.2163 ± 0.0266 (N=3) | 0.2080 ± 0.0372 (N=3) | — | — | s |
| generate_kernels_s | 30.6 ± 0.7839 (N=3) | — | — | — | s |
| build_s | 6.734 ± 0.0973 (N=3) | — | — | — | s |
| run_s | 0.5287 ± 0.0065 (N=3) | — | — | — | s |
| total_stage_s | 38.08 ± 0.9099 (N=3) | 0.2090 ± 0.0372 (N=3) | — | — | s |
| n_kernels_curated | 5 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_reference | 2 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_cached | — | 0 (N=3) | — | — | count |
| n_kernels_llm | — | 7 (N=3) | — | — | count |
| n_kernels_total | 7 (N=3) | 7 (N=3) | — | — | count |
| algorithms_distinct_count | 5 (N=3) | 6 (N=3) | — | — | count |
| compile_wall_clock_s | 38.13 ± 0.9099 (N=3) | 58.8 ± 4.004 (N=3) | — | — | s |
| compile_peak_rss_mb | 99.8 (N=3) | 99.8 (N=3) | — | — | MB |
| n_ops_profiled | 30 (N=3) | — | — | — | count |
| dominant_op_share | 0.3704 (N=3) | — | — | — | fraction |
| mean_cycles_per_dispatch | 229,993 (N=3) | — | — | — | cycles |
| stddev_cycles_per_dispatch | 397,088 (N=3) | — | — | — | cycles |
| op_kind_p95_max_cycles | 1,304,755 (N=3) | — | — | — | cycles |
| op_kind_median_max_cycles | 1,304,755 (N=3) | — | — | — | cycles |
| tokens_input_cached | — | 0 (N=3) | — | — | tokens |
| tokens_input_uncached | — | 2,159 ± 3,739 (N=3) | — | — | tokens |
| tokens_output | — | 170.7 ± 295.6 (N=3) | — | — | tokens |
| dollars_equivalent | — | 0.0271 (N=3) | — | — | USD |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | batchnorm2d_s8 | 37.0% | 2,555,749 |
| A | conv2d_s8 | 36.4% | 2,514,917 |
| A | maxpool2d_s8 | 18.9% | 1,304,755 |
| A | add_s8 | 7.1% | 487,360 |
| A | relu_s8 | 0.5% | 31,871 |

## `dronet_hetero_int8`
model `dronet`, target `hetero_gemmini_opu`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | — | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | — | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | — | — | — | count |
| cross_tile_bytes | 171,553 (N=3) | — | — | — | bytes |
| ir_op_count | 32 (N=3) | — | — | — | count |
| n_input_nodes | 34 (N=3) | — | — | — | count |
| lowering_ratio | 0.9412 (N=3) | — | — | — | fraction |
| n_dispatches_graph | 32 (N=3) | — | — | — | count |
| n_distinct_op_kinds | 8 (N=3) | — | — | — | count |
| n_distinct_shapes | 25 (N=3) | — | — | — | count |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| weights_npz_bytes | 326,642 (N=3) | — | — | — | bytes |
| generate_kernels_s | 0.0643 ± 0.0032 (N=3) | — | — | — | s |
| build_s | 7.29 ± 0.1225 (N=3) | — | — | — | s |
| total_stage_s | 7.354 ± 0.1222 (N=3) | — | — | — | s |
| compile_wall_clock_s | 648.6 ± 0.4859 (N=3) | — | — | — | s |
| compile_peak_rss_mb | 166.8 ± 0.2309 (N=3) | — | — | — | MB |
| makespan_cycles | 414,650 (N=3) | — | — | — | cycles |
| accelerator_utilization_gemmini | 0.2220 (N=3) | — | — | — | fraction |
| accelerator_utilization_opu | 0.7745 (N=3) | — | — | — | fraction |
| dispatches_per_tile_gemmini | 9 (N=3) | — | — | — | count |
| dispatches_per_tile_opu | 21 (N=3) | — | — | — | count |
| tile_load_imbalance | 3.489 (N=3) | — | — | — | ratio |
| schedule_parallelism_max | 1 (N=3) | — | — | — | count |

## `yolov8_nano_scalar_smoke`
model `yolov8_nano`, target `scalar`, quant `int8`, runner `spike`

### pre-kernel — graph compilation

| metric | A | unit |
| --- | --- | --- |
| passes_fired_total | 0 (N=3) | count |
| linear_relu_fuse_fired | 0 (N=3) | count |
| conv2d_relu_fuse_fired | 0 (N=3) | count |
| ir_op_count | 212 (N=3) | count |
| n_input_nodes | 230 (N=3) | count |
| lowering_ratio | 0.9217 (N=3) | fraction |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | unit |
| --- | --- | --- |
| cycles_spike | 10,437,905,042 (N=3) | cycles |
| wall_cycles_spike | 104,405,200 (N=3) | cycles |
| accuracy_linf | 0 (N=3) | dimensionless |
| verify_pass | yes (N=3) | bool |
| bit_exact | yes (N=3) | bool |
| latency_ms_spike | 10,438 (N=3) | ms |
| zephyr_elf_bytes | 4,172,565 ± 16,831 (N=3) | bytes |
| kernels_c_bytes | 17,612 ± 9,176 (N=3) | bytes |
| kernels_c_loc | 401.7 ± 226.3 (N=3) | count |
| weights_npz_bytes | 3,277,160 (N=3) | bytes |
| extract_s | 35.23 ± 61.01 (N=3) | s |
| generate_skeleton_s | 0.7143 ± 0.0401 (N=3) | s |
| generate_kernels_s | 0.1947 ± 0.0045 (N=3) | s |
| build_s | 9.266 ± 0.2135 (N=3) | s |
| run_s | 32.94 ± 0.3846 (N=3) | s |
| total_stage_s | 78.34 ± 60.4 (N=3) | s |
| n_kernels_curated | 0 (N=3) | count |
| n_kernels_reference | 9 (N=3) | count |
| n_kernels_total | 9 (N=3) | count |
| compile_wall_clock_s | 78.42 ± 60.4 (N=3) | s |
| compile_peak_rss_mb | 377.4 ± 440.1 (N=3) | MB |
| n_ops_profiled | 204 (N=3) | count |
| dominant_op_share | 0.9824 (N=3) | fraction |
| mean_cycles_per_dispatch | 51,166,201 (N=3) | cycles |
| stddev_cycles_per_dispatch | 105,625,107 (N=3) | cycles |
| op_kind_p95_max_cycles | 484,154,120 (N=3) | cycles |
| op_kind_median_max_cycles | 126,177,040 (N=3) | cycles |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 98.2% | 10,254,324,120 |
| A | silu_s8 | 1.0% | 99,702,996 |
| A | batchnorm2d_s8 | 0.5% | 49,703,657 |
| A | cat2_c1_s8 | 0.1% | 12,253,041 |
| A | cat3_c1_s8 | 0.1% | 9,632,276 |

## `yolov8_nano_rvv_smoke`
model `yolov8_nano`, target `rvv`, quant `int8`, runner `spike`

### pre-kernel — graph compilation

| metric | A | unit |
| --- | --- | --- |
| passes_fired_total | 0 (N=3) | count |
| linear_relu_fuse_fired | 0 (N=3) | count |
| conv2d_relu_fuse_fired | 0 (N=3) | count |
| ir_op_count | 212 (N=3) | count |
| n_input_nodes | 230 (N=3) | count |
| lowering_ratio | 0.9217 (N=3) | fraction |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | unit |
| --- | --- | --- |
| cycles_spike | 631,334,979 (N=3) | cycles |
| wall_cycles_spike | 6,315,000 (N=3) | cycles |
| accuracy_linf | 0 (N=3) | dimensionless |
| verify_pass | yes (N=3) | bool |
| bit_exact | yes (N=3) | bool |
| latency_ms_spike | 631.3 (N=3) | ms |
| zephyr_elf_bytes | 4,179,160 (N=3) | bytes |
| kernels_c_bytes | 23,471 (N=3) | bytes |
| kernels_c_loc | 493 (N=3) | count |
| weights_npz_bytes | 3,277,160 (N=3) | bytes |
| extract_s | 0.0010 (N=3) | s |
| generate_skeleton_s | 0.7027 ± 0.0605 (N=3) | s |
| generate_kernels_s | 231.7 ± 2.51 (N=3) | s |
| build_s | 8.131 ± 0.1145 (N=3) | s |
| run_s | 13.59 ± 0.0357 (N=3) | s |
| total_stage_s | 254.2 ± 2.514 (N=3) | s |
| n_kernels_curated | 4 (N=3) | count |
| n_kernels_reference | 5 (N=3) | count |
| n_kernels_total | 9 (N=3) | count |
| algorithms_distinct_count | 2 (N=3) | count |
| compile_wall_clock_s | 254.2 ± 2.512 (N=3) | s |
| compile_peak_rss_mb | 137.3 ± 0.1732 (N=3) | MB |
| n_ops_profiled | 204 (N=3) | count |
| dominant_op_share | 0.8668 (N=3) | fraction |
| mean_cycles_per_dispatch | 3,094,779 (N=3) | cycles |
| stddev_cycles_per_dispatch | 5,391,471 (N=3) | cycles |
| op_kind_p95_max_cycles | 24,791,567 (N=3) | cycles |
| op_kind_median_max_cycles | 6,526,380 (N=3) | cycles |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 86.7% | 547,256,436 |
| A | batchnorm2d_s8 | 7.9% | 49,703,165 |
| A | cat2_c1_s8 | 1.9% | 12,253,855 |
| A | cat3_c1_s8 | 1.5% | 9,632,383 |
| A | cat4_c1_s8 | 0.8% | 4,979,921 |

## `yolov8n_rvv_opu_int8`
model `yolov8_nano`, target `rvv_opu`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | 0 (N=3) | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| ir_op_count | 212 (N=3) | 212 (N=3) | — | — | count |
| n_input_nodes | 230 (N=3) | 230 (N=3) | — | — | count |
| lowering_ratio | 0.9217 (N=3) | 0.9217 (N=3) | — | — | fraction |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on rvv_opu)* | 10,437,933,880 (N=3) | 626,016,151 ± 789,046 (N=3) | — | — | cycles |
| wall_cycles_spike | 104,405,500 (N=3) | 6,261,783 ± 7,895 (N=3) | — | — | cycles |
| accuracy_linf | 0 (N=3) | 0 (N=3) | — | — | dimensionless |
| verify_pass | yes (N=3) | yes (N=3) | — | — | bool |
| bit_exact | yes (N=3) | yes (N=3) | — | — | bool |
| zephyr_elf_bytes | 4,171,432 (N=3) | 4,182,765 ± 546.4 (N=3) | — | — | bytes |
| kernels_c_bytes | 12,365 (N=3) | 28,623 ± 1,434 (N=3) | — | — | bytes |
| kernels_c_loc | 273 (N=3) | 610.3 ± 34.67 (N=3) | — | — | count |
| weights_npz_bytes | 3,277,160 (N=3) | 3,277,160 (N=3) | — | — | bytes |
| extract_s | 0.0017 ± 5.774e-04 (N=3) | 0.0013 ± 5.774e-04 (N=3) | — | — | s |
| generate_skeleton_s | 0.7073 ± 0.0662 (N=3) | 0.7677 ± 0.0419 (N=3) | — | — | s |
| generate_kernels_s | 308.7 ± 0.7779 (N=3) | 1,583 ± 132.8 (N=3) | — | — | s |
| build_s | 9.261 ± 0.1234 (N=3) | 9.266 ± 0.4354 (N=3) | — | — | s |
| run_s | 29.53 ± 0.3728 (N=3) | 14.49 ± 0.5861 (N=3) | — | — | s |
| total_stage_s | 348.2 ± 0.5644 (N=3) | 1,608 ± 133.1 (N=3) | — | — | s |
| n_kernels_curated | 0 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_reference | 9 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_cached | — | 0 (N=3) | — | — | count |
| n_kernels_llm | — | 9 (N=3) | — | — | count |
| n_kernels_total | 9 (N=3) | 9 (N=3) | — | — | count |
| algorithms_distinct_count | — | 2 (N=3) | — | — | count |
| compile_wall_clock_s | 348.3 ± 0.5699 (N=3) | 1,608 ± 133.1 (N=3) | — | — | s |
| compile_peak_rss_mb | 136.4 ± 0.0577 (N=3) | 140.7 ± 0.5568 (N=3) | — | — | MB |
| n_ops_profiled | 204 (N=3) | 204 (N=3) | — | — | count |
| dominant_op_share | 0.9824 (N=3) | 0.8406 ± 0.0011 (N=3) | — | — | fraction |
| mean_cycles_per_dispatch | 51,166,343 (N=3) | 3,068,707 ± 3,868 (N=3) | — | — | cycles |
| stddev_cycles_per_dispatch | 105,625,401 (N=3) | 5,156,257 ± 2,134 (N=3) | — | — | cycles |
| op_kind_p95_max_cycles | 484,155,440 (N=3) | 23,131,837 (N=3) | — | — | cycles |
| op_kind_median_max_cycles | 126,177,370 (N=3) | 6,215,779 (N=3) | — | — | cycles |
| beam_n_candidates_total | — | 43 ± 4.583 (N=3) | — | — | count |
| beam_n_candidates_viable | — | 24 ± 3.606 (N=3) | — | — | count |
| beam_n_candidates_build_fail | — | 13.67 ± 1.155 (N=3) | — | — | count |
| beam_n_candidates_verify_fail | — | 1.333 ± 0.5774 (N=3) | — | — | count |
| beam_n_candidates_duplicate | — | 4 ± 1 (N=3) | — | — | count |
| beam_tokens_per_candidate_mean | — | 19,934 ± 62.19 (N=3) | — | — | tokens |
| beam_best_improvement_pct | — | 53.07 ± 41.77 (N=3) | — | — | percent |
| beam_iter_to_best | — | 1.667 ± 0.5774 (N=3) | — | — | iteration |
| tokens_input_cached | — | 0 (N=3) | — | — | tokens |
| tokens_input_uncached | — | 822,859 ± 55,929 (N=3) | — | — | tokens |
| tokens_output | — | 69,681 ± 8,123 (N=3) | — | — | tokens |
| dollars_equivalent | — | 3.514 ± 0.2783 (N=3) | — | — | USD |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | conv2d_s8 | 98.2% | 10,254,353,169 |
| A | silu_s8 | 1.0% | 99,703,326 |
| A | batchnorm2d_s8 | 0.5% | 49,703,061 |
| A | cat2_c1_s8 | 0.1% | 12,253,041 |
| A | cat3_c1_s8 | 0.1% | 9,632,331 |
| B-bedrock | conv2d_s8 | 84.2% | 526,251,487 |
| B-bedrock | silu_s8 | 15.6% | 97,596,167 |
| B-bedrock | batchnorm2d_s8 | 0.1% | 772,593 |
| B-bedrock | cat2_c1_s8 | 0.0% | 164,601 |
| B-bedrock | cat3_c1_s8 | 0.0% | 135,936 |

## `yolov8n_gemmini_int8`
model `yolov8_nano`, target `gemmini`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | 0 (N=3) | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | 0 (N=3) | — | — | count |
| ir_op_count | 212 (N=3) | 212 (N=3) | — | — | count |
| n_input_nodes | 230 (N=3) | 230 (N=3) | — | — | count |
| lowering_ratio | 0.9217 (N=3) | 0.9217 (N=3) | — | — | fraction |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on gemmini)* | 283,076,496 (N=3) | 249,702,937 ± 31,214,072 (N=3) | — | — | cycles |
| wall_cycles_spike | 2,831,550 (N=3) | 2,497,733 ± 312,228 (N=3) | — | — | cycles |
| accuracy_linf | 0 (N=3) | 0 (N=3) | — | — | dimensionless |
| verify_pass | yes (N=3) | yes (N=3) | — | — | bool |
| bit_exact | yes (N=3) | yes (N=3) | — | — | bool |
| zephyr_elf_bytes | 4,205,576 (N=3) | 4,237,024 ± 13,407 (N=3) | — | — | bytes |
| kernels_c_bytes | 19,429 (N=3) | 36,713 ± 11,351 (N=3) | — | — | bytes |
| kernels_c_loc | 430 (N=3) | 752.3 ± 147.8 (N=3) | — | — | count |
| weights_npz_bytes | 3,277,160 (N=3) | 3,277,160 (N=3) | — | — | bytes |
| extract_s | 0.0010 (N=3) | 0.0013 ± 5.774e-04 (N=3) | — | — | s |
| generate_skeleton_s | 0.7020 ± 0.0044 (N=3) | 0.7133 ± 0.0401 (N=3) | — | — | s |
| generate_kernels_s | 21.46 ± 0.3237 (N=3) | 1,124 ± 420 (N=3) | — | — | s |
| build_s | 9.321 ± 0.2863 (N=3) | 9.201 ± 0.2666 (N=3) | — | — | s |
| run_s | 3.855 ± 0.0906 (N=3) | 3.797 ± 0.0909 (N=3) | — | — | s |
| total_stage_s | 35.33 ± 0.1283 (N=3) | 1,138 ± 420 (N=3) | — | — | s |
| n_kernels_curated | 1 (N=3) | 0 (N=3) | — | — | count |
| n_kernels_reference | 8 (N=3) | 3 (N=3) | — | — | count |
| n_kernels_cached | — | 0 (N=3) | — | — | count |
| n_kernels_llm | — | 6 (N=3) | — | — | count |
| n_kernels_total | 9 (N=3) | 9 (N=3) | — | — | count |
| algorithms_distinct_count | 1 (N=3) | 2 (N=3) | — | — | count |
| compile_wall_clock_s | 35.39 ± 0.1287 (N=3) | 1,138 ± 420 (N=3) | — | — | s |
| compile_peak_rss_mb | 124.1 ± 0.1528 (N=3) | 124.2 ± 0.0577 (N=3) | — | — | MB |
| n_ops_profiled | 204 (N=3) | 204 (N=3) | — | — | count |
| dominant_op_share | 0.3522 (N=3) | 0.3977 ± 0.0501 (N=3) | — | — | fraction |
| mean_cycles_per_dispatch | 1,387,630 (N=3) | 1,224,034 ± 153,010 (N=3) | — | — | cycles |
| stddev_cycles_per_dispatch | 1,520,564 (N=3) | 1,421,587 ± 83,413 (N=3) | — | — | cycles |
| op_kind_p95_max_cycles | 5,816,904 (N=3) | 5,692,995 (N=3) | — | — | cycles |
| op_kind_median_max_cycles | 1,436,904 (N=3) | 1,190,221 ± 408.2 (N=3) | — | — | cycles |
| beam_n_candidates_total | — | 39 ± 13.08 (N=3) | — | — | count |
| beam_n_candidates_viable | — | 26.33 ± 6.429 (N=3) | — | — | count |
| beam_n_candidates_build_fail | — | 0 (N=3) | — | — | count |
| beam_n_candidates_verify_fail | — | 0 (N=3) | — | — | count |
| beam_n_candidates_duplicate | — | 12.67 ± 6.658 (N=3) | — | — | count |
| beam_tokens_per_candidate_mean | — | 7,463 ± 1,577 (N=3) | — | — | tokens |
| beam_best_improvement_pct | — | 56.17 ± 19.69 (N=3) | — | — | percent |
| beam_iter_to_best | — | 1.667 ± 0.5774 (N=3) | — | — | iteration |
| tokens_input_cached | — | 0 (N=3) | — | — | tokens |
| tokens_input_uncached | — | 223,999 ± 89,625 (N=3) | — | — | tokens |
| tokens_output | — | 87,065 ± 46,302 (N=3) | — | — | tokens |
| dollars_equivalent | — | 1.978 ± 0.9633 (N=3) | — | — | USD |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | silu_s8 | 35.2% | 99,703,703 |
| A | conv2d_s8 | 35.1% | 99,493,607 |
| A | batchnorm2d_s8 | 17.6% | 49,704,210 |
| A | cat2_c1_s8 | 4.3% | 12,253,041 |
| A | cat3_c1_s8 | 3.4% | 9,632,983 |
| B-bedrock | silu_s8 | 45.6% | 97,377,704 |
| B-bedrock | conv2d_s8 | 24.3% | 51,855,518 |
| B-bedrock | batchnorm2d_s8 | 23.3% | 49,704,210 |
| B-bedrock | cat2_c1_s8 | 1.9% | 4,072,193 |
| B-bedrock | maxpool2d_s8 | 1.7% | 3,577,091 |

## `yolov8n_gemmini_q31_int8`
model `yolov8_nano`, target `gemmini_q31`, quant `int8`, runner `firesim`

### pre-kernel — graph compilation

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| passes_fired_total | 0 (N=3) | — | — | — | count |
| linear_relu_fuse_fired | 0 (N=3) | — | — | — | count |
| conv2d_relu_fuse_fired | 0 (N=3) | — | — | — | count |
| ir_op_count | 212 (N=3) | — | — | — | count |
| n_input_nodes | 230 (N=3) | — | — | — | count |
| lowering_ratio | 0.9217 (N=3) | — | — | — | fraction |

### kernel synthesis — compiled-artifact + LLM loop

| metric | A | B-bedrock | B-gemini | B-claude | unit |
| --- | --- | --- | --- | --- | --- |
| cycles_spike *(not authoritative on gemmini_q31)* | 280,933,590 (N=3) | — | — | — | cycles |
| wall_cycles_spike | 2,810,150 (N=3) | — | — | — | cycles |
| accuracy_linf | 9 (N=3) | — | — | — | dimensionless |
| verify_pass | yes (N=3) | — | — | — | bool |
| bit_exact | no (N=3) | — | — | — | bool |
| zephyr_elf_bytes | 4,239,976 (N=3) | — | — | — | bytes |
| kernels_c_bytes | 29,217 (N=3) | — | — | — | bytes |
| kernels_c_loc | 659 (N=3) | — | — | — | count |
| weights_npz_bytes | 3,277,160 (N=3) | — | — | — | bytes |
| extract_s | 0.0010 (N=3) | — | — | — | s |
| generate_skeleton_s | 0.7260 ± 0.0434 (N=3) | — | — | — | s |
| generate_kernels_s | 67.4 ± 0.4062 (N=3) | — | — | — | s |
| build_s | 9.047 ± 0.1996 (N=3) | — | — | — | s |
| run_s | 3.73 ± 0.0329 (N=3) | — | — | — | s |
| total_stage_s | 80.9 ± 0.4066 (N=3) | — | — | — | s |
| n_kernels_curated | 3 (N=3) | — | — | — | count |
| n_kernels_reference | 6 (N=3) | — | — | — | count |
| n_kernels_total | 9 (N=3) | — | — | — | count |
| algorithms_distinct_count | 3 (N=3) | — | — | — | count |
| compile_wall_clock_s | 80.96 ± 0.4058 (N=3) | — | — | — | s |
| compile_peak_rss_mb | 124.1 ± 0.1000 (N=3) | — | — | — | MB |
| n_ops_profiled | 204 (N=3) | — | — | — | count |
| dominant_op_share | 0.3549 (N=3) | — | — | — | fraction |
| mean_cycles_per_dispatch | 1,377,125 (N=3) | — | — | — | cycles |
| stddev_cycles_per_dispatch | 1,527,090 (N=3) | — | — | — | cycles |
| op_kind_p95_max_cycles | 5,817,328 (N=3) | — | — | — | cycles |
| op_kind_median_max_cycles | 1,436,917 (N=3) | — | — | — | cycles |

**Top op kinds by cycle share:**

| arm | op kind | share | cycles |
| --- | --- | --- | --- |
| A | silu_s8 | 35.5% | 99,708,615 |
| A | conv2d_s8 | 35.4% | 99,495,023 |
| A | batchnorm2d_s8 | 17.7% | 49,704,539 |
| A | cat2_c1_s8 | 4.4% | 12,253,666 |
| A | cat3_c1_s8 | 3.4% | 9,631,586 |

## `yolov8n_hetero_int8`
model `yolov8_nano`, target `hetero_gemmini_opu`, quant `int8`, runner `firesim`  &nbsp;**[blocked_by: P2.1-schedule]**

## `vint_rvv_opu_fp16`
model `vint`, target `rvv_opu`, quant `fp16`, runner `firesim`

## `vint_hetero_fp16`
model `vint`, target `hetero_gemmini_opu`, quant `fp16`, runner `firesim`  &nbsp;**[blocked_by: P2.1-schedule]**

## `smolvla_scalar_fp16_step`
model `smolvla`, target `scalar`, quant `fp16`, runner `spike`, slice `single_step`  &nbsp;**[blocked_by: walker-smolvla-ops]**

## `smolvla_rvv_opu_fp16_step`
model `smolvla`, target `rvv_opu`, quant `fp16`, runner `firesim`, slice `single_step`  &nbsp;**[blocked_by: walker-smolvla-ops]**
