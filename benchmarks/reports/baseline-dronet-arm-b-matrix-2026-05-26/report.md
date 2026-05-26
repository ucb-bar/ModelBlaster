# Cost report — `baseline-dronet-arm-b-matrix-2026-05-26`

_Generated at 2026-05-26T22:32:21.135539+00:00 by_  `agustin@garden`  _on git_  `c0750c0`  _branch_  `feat/benchmark-harness`.

_Filter:_ session = `baseline-dronet-arm-b-matrix-2026-05-26`

## Totals
- **Spend:** $11.0807 across 171 calls
- This month: $11.0807 (171 calls)
- Input (uncached / cached): 2,671,958 / 0
- Output: 204,325

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 171 | 2,671,958 | 0 | 204,325 | $11.0807 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `conv2d_s8` | 59 | 1,046,723 | 97,941 | $4.6093 |
| `linear_s8` | 20 | 356,054 | 37,588 | $1.6320 |
| `maxpool2d_s8` | 28 | 338,136 | 29,123 | $1.4513 |
| `add_s8` | 20 | 316,398 | 16,143 | $1.1913 |
| `batchnorm2d_s8` | 13 | 189,989 | 15,737 | $0.8060 |
| `relu_s8` | 17 | 216,987 | 4,203 | $0.7140 |
| `sigmoid_s8` | 14 | 207,671 | 3,590 | $0.6769 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T21-15-51Z` | 43 | $3.0458 |
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T22-02-45Z` | 39 | $2.7343 |
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T21-43-14Z` | 33 | $2.3146 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-11-12Z` | 12 | $0.8643 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-06-37Z` | 12 | $0.8628 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-02-09Z` | 12 | $0.8336 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-23-36Z` | 7 | $0.1594 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-26-34Z` | 7 | $0.1249 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-25-11Z` | 4 | $0.1139 |
| `B-bedrock/dronet_gemmini_q31_int8/2026-05-26T22-28-14Z` | 2 | $0.0271 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `baseline-dronet-arm-b-matrix-2026-05-26` | Arm B-bedrock matrix: dronet × {rvv, rvv_opu, gemmini, gemmini_q31}, 3 reps each, beam=2 exp=3 iter=2 | 2026-05-26T21:01:56 | 2026-05-26T22:32:21 | 171 | $11.0807 |
| `baseline-dronet-arm-a-matrix-2026-05-26` | Arm A matrix: dronet × {rvv, rvv_opu, gemmini, gemmini_q31}, 3 reps each | 2026-05-26T20:24:12 | 2026-05-26T21:01:02 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-26T20-1` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-26T20:11:21 | 2026-05-26T20:22:01 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_q31_int8-2026-05-26T20-1` | auto-opened by arm driver A for workload 'dronet_gemmini_q31_int8' (gemmini_q31/spike) | 2026-05-26T20:10:35 | 2026-05-26T20:11:14 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_int8-2026-05-26T20-1` | auto-opened by arm driver A for workload 'dronet_gemmini_int8' (gemmini/spike) | 2026-05-26T20:10:09 | 2026-05-26T20:10:28 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-26T20-0` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-26T20:07:02 | 2026-05-26T20:07:18 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_q31_int8-2026-05-26T20-0` | auto-opened by arm driver A for workload 'dronet_gemmini_q31_int8' (gemmini_q31/spike) | 2026-05-26T20:06:25 | 2026-05-26T20:07:02 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_int8-2026-05-26T20-0` | auto-opened by arm driver A for workload 'dronet_gemmini_int8' (gemmini/spike) | 2026-05-26T20:06:07 | 2026-05-26T20:06:25 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-26T19-5` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-26T19:54:35 | 2026-05-26T19:54:52 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_q31_int8-2026-05-26T19-5` | auto-opened by arm driver A for workload 'dronet_gemmini_q31_int8' (gemmini_q31/spike) | 2026-05-26T19:53:58 | 2026-05-26T19:54:35 | 0 | $0.0000 |
| `auto-A-dronet_gemmini_int8-2026-05-26T19-5` | auto-opened by arm driver A for workload 'dronet_gemmini_int8' (gemmini/spike) | 2026-05-26T19:53:41 | 2026-05-26T19:53:58 | 0 | $0.0000 |
| `auto-A-dronet_rvv_opu_int8-2026-05-26T19-5` | auto-opened by arm driver A for workload 'dronet_rvv_opu_int8' (rvv_opu/spike) | 2026-05-26T19:53:22 | 2026-05-26T19:53:40 | 0 | $0.0000 |
| `auto-A-dronet_rvv_smoke-2026-05-26T19-5` | auto-opened by arm driver A for workload 'dronet_rvv_smoke' (rvv/spike) | 2026-05-26T19:51:57 | 2026-05-26T19:52:46 | 0 | $0.0000 |
| `auto-A-dronet_scalar_smoke-2026-05-26T19-2` | auto-opened by arm driver A for workload 'dronet_scalar_smoke' (scalar/spike) | 2026-05-26T19:29:55 | 2026-05-26T19:30:11 | 0 | $0.0000 |
| `baseline-dronet-arm-b-2026-05-26` | Arm B-bedrock baseline: dronet_scalar_smoke beam=2 exp=3 iter=2, 3 reps, --max-usd 5 | 2026-05-26T18:12:41 | 2026-05-26T19:02:18 | 0 | $0.0000 |
| `baseline-dronet-arm-a-2026-05-26` | Arm A curated baseline: dronet_scalar_smoke + dronet_rvv_smoke, 3 reps each | 2026-05-26T18:05:37 | 2026-05-26T18:11:40 | 0 | $0.0000 |
| `auto-A-mlp_generic_scalar_smoke-2026-05-26T18-0` | auto-opened by arm driver A for workload 'mlp_generic_scalar_smoke' (scalar/spike) | 2026-05-26T18:05:02 | 2026-05-26T18:05:02 | 0 | $0.0000 |
