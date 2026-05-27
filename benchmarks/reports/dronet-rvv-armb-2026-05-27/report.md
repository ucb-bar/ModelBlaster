# Cost report — `dronet-rvv-armb-2026-05-27`

_Generated at 2026-05-27T07:48:52.512169+00:00 by_  `agustin@garden`  _on git_  `2ac7c4e`  _branch_  `feat/benchmark-harness`.

_Filter:_ session = `rvv-armb-final-2026-05-27`

## Totals
- **Spend:** $4.3671 across 66 calls
- This month: $4.3671 (66 calls)
- Input (uncached / cached): 1,107,876 / 0
- Output: 69,566

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 66 | 1,107,876 | 0 | 69,566 | $4.3671 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `maxpool2d_s8` | 15 | 275,379 | 19,840 | $1.1237 |
| `sigmoid_s8` | 15 | 237,834 | 3,903 | $0.7720 |
| `linear_s8` | 9 | 154,845 | 16,895 | $0.7180 |
| `batchnorm2d_s8` | 9 | 149,904 | 15,751 | $0.6860 |
| `add_s8` | 9 | 148,389 | 10,018 | $0.5954 |
| `relu_s8` | 9 | 141,525 | 3,159 | $0.4720 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/dronet_rvv_smoke/2026-05-27T06-58-25Z` | 30 | $1.9748 |
| `B-bedrock/dronet_rvv_smoke/2026-05-27T07-07-04Z` | 18 | $1.2065 |
| `B-bedrock/dronet_rvv_smoke/2026-05-27T07-12-59Z` | 18 | $1.1859 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `rvv-armb-final-2026-05-27` | dronet+yolov8n × rvv Arm B with conv2d_s8 skip | 2026-05-27T06:58:25 | 2026-05-27T07:48:29 | 66 | $4.3671 |
| `dronet-rvv-armb-2026-05-27` | dronet × rvv Arm B retry after harness fixes | 2026-05-27T05:54:05 | 2026-05-27T06:12:48 | 0 | $0.0000 |
| `recapture-v3-2026-05-27` | yolov8n_rvv_opu retry after saturn_opu.h macro fix | 2026-05-27T04:31:17 | 2026-05-27T05:52:42 | 0 | $0.0000 |
| `recapture-v2-2026-05-27` | Recapture with LLM-skip extended to initial gen | 2026-05-27T03:04:28 | 2026-05-27T04:31:17 | 0 | $0.0000 |
| `recapture-2026-05-26` | Recapture unblocked cells: dronet gemmini Arm B + yolov8n {rvv_opu, gemmini} Arm B | 2026-05-27T02:42:27 | 2026-05-27T03:04:28 | 0 | $0.0000 |
| `baseline-yolov8n-arm-b-rvv-opu-2026-05-26` | Arm B-bedrock yolov8n × rvv_opu, 3 reps, beam=2 exp=3 iter=2 | 2026-05-27T02:13:20 | 2026-05-27T02:24:10 | 0 | $0.0000 |
| `baseline-yolov8n-arm-a-matrix-2026-05-26` | Arm A matrix: yolov8n × {rvv_smoke, rvv_opu_fp16, gemmini_q31_int8}, 3 reps each | 2026-05-27T01:29:08 | 2026-05-27T02:12:13 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-27T00-3` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-27T00:33:31 | 2026-05-27T00:44:20 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-27T00-2` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-27T00:21:51 | 2026-05-27T00:32:41 | 0 | $0.0000 |
| `auto-A-dronet_hetero_int8-2026-05-27T00-1` | auto-opened by arm driver A for workload 'dronet_hetero_int8' (hetero_gemmini_opu/spike) | 2026-05-27T00:10:28 | 2026-05-27T00:21:16 | 0 | $0.0000 |
| `baseline-yolov8n-arm-a-2026-05-26` | Arm A curated baseline: yolov8_nano_scalar_smoke, 3 reps | 2026-05-26T22:48:36 | 2026-05-26T23:10:12 | 0 | $0.0000 |
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
