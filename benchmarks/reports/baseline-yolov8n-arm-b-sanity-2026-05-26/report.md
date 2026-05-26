# Cost report — `baseline-yolov8n-arm-b-sanity-2026-05-26`

_Generated at 2026-05-26T23:10:12.807490+00:00 by_  `agustin@garden`  _on git_  `6de5d40`  _branch_  `feat/benchmark-harness`.

_Filter:_ session = `baseline-yolov8n-arm-b-sanity-2026-05-26`

## Totals
- **Spend:** $0.7563 across 27 calls
- This month: $0.7563 (27 calls)
- Input (uncached / cached): 121,589 / 0
- Output: 26,101

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 27 | 121,589 | 0 | 26,101 | $0.7563 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `conv2d_s8` | 3 | 27,116 | 5,209 | $0.1595 |
| `cat3_c1_s8` | 3 | 12,081 | 4,774 | $0.1079 |
| `cat2_c1_s8` | 3 | 12,272 | 3,503 | $0.0894 |
| `maxpool2d_s8` | 3 | 13,038 | 2,646 | $0.0788 |
| `cat4_c1_s8` | 3 | 12,171 | 2,670 | $0.0766 |
| `batchnorm2d_s8` | 3 | 12,348 | 2,268 | $0.0711 |
| `silu_s8` | 3 | 10,836 | 1,925 | $0.0614 |
| `add_s8` | 3 | 10,752 | 1,905 | $0.0608 |
| `upsample_nearest_s8` | 3 | 10,975 | 1,201 | $0.0509 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-33-54Z` | 27 | $0.7563 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `baseline-yolov8n-arm-a-2026-05-26` | Arm A curated baseline: yolov8_nano_scalar_smoke, 3 reps | 2026-05-26T22:48:36 | 2026-05-26T23:10:12 | 0 | $0.0000 |
| `baseline-yolov8n-arm-b-sanity-2026-05-26` | Arm B-bedrock yolov8n 1-rep sanity (beam=1 exp=2 iter=1) | 2026-05-26T22:33:54 | 2026-05-26T22:48:36 | 27 | $0.7563 |
| `baseline-dronet-arm-b-matrix-2026-05-26` | Arm B-bedrock matrix: dronet × {rvv, rvv_opu, gemmini, gemmini_q31}, 3 reps each, beam=2 exp=3 iter=2 | 2026-05-26T21:01:56 | 2026-05-26T22:32:21 | 0 | $0.0000 |
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
