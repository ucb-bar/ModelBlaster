# Cost report — `baseline-yolov8n-arm-a-2026-05-26`

_Generated at 2026-05-26T23:10:12.626270+00:00 by_  `agustin@garden`  _on git_  `6de5d40`  _branch_  `feat/benchmark-harness`.

_Filter:_ session = `baseline-yolov8n-arm-a-2026-05-26`

## Totals
- **Spend:** $1.7095 across 36 calls
- This month: $1.7095 (36 calls)
- Input (uncached / cached): 200,320 / 0
- Output: 73,901

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 36 | 200,320 | 0 | 73,901 | $1.7095 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `cat3_c1_s8` | 4 | 25,744 | 17,220 | $0.3355 |
| `cat2_c1_s8` | 4 | 23,324 | 12,814 | $0.2622 |
| `conv2d_s8` | 4 | 41,700 | 9,077 | $0.2613 |
| `maxpool2d_s8` | 4 | 22,136 | 9,753 | $0.2127 |
| `cat4_c1_s8` | 4 | 20,496 | 8,392 | $0.1874 |
| `silu_s8` | 4 | 18,184 | 6,196 | $0.1475 |
| `batchnorm2d_s8` | 4 | 17,400 | 3,553 | $0.1055 |
| `upsample_nearest_s8` | 4 | 16,064 | 3,446 | $0.0999 |
| `add_s8` | 4 | 15,272 | 3,450 | $0.0976 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-58-55Z` | 18 | $0.9664 |
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-48-40Z` | 18 | $0.7431 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `baseline-yolov8n-arm-a-2026-05-26` | Arm A curated baseline: yolov8_nano_scalar_smoke, 3 reps | 2026-05-26T22:48:36 | 2026-05-26T23:10:12 | 36 | $1.7095 |
| `baseline-yolov8n-arm-b-sanity-2026-05-26` | Arm B-bedrock yolov8n 1-rep sanity (beam=1 exp=2 iter=1) | 2026-05-26T22:33:54 | 2026-05-26T22:48:36 | 0 | $0.0000 |
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
