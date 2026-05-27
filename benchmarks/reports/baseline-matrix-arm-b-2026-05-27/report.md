# Cost report — `baseline-matrix-arm-b-2026-05-27`

_Generated at 2026-05-27T05:52:42.986659+00:00 by_  `agustin@garden`  _on git_  `1fe7be5`  _branch_  `feat/benchmark-harness`.

## Totals
- **Spend:** $38.3747 across 653 calls
- This month: $38.3747 (653 calls)
- Input (uncached / cached): 7,189,104 / 0
- Output: 1,120,492

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 653 | 7,189,104 | 0 | 1,120,492 | $38.3747 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `conv2d_s8` | 128 | 2,068,914 | 249,240 | $9.9453 |
| `linear_s8` | 74 | 824,075 | 234,854 | $5.9950 |
| `maxpool2d_s8` | 63 | 758,949 | 78,961 | $3.4613 |
| `cat3_c1_s8` | 38 | 353,004 | 149,819 | $3.3063 |
| `add_s8` | 51 | 629,279 | 61,908 | $2.8165 |
| `cat2_c1_s8` | 38 | 320,585 | 99,006 | $2.4468 |
| `upsample_nearest_s8` | 58 | 519,758 | 40,710 | $2.1699 |
| `batchnorm2d_s8` | 44 | 453,575 | 50,668 | $2.1207 |
| `silu_s8` | 42 | 347,023 | 47,642 | $1.7557 |
| `cat4_c1_s8` | 35 | 270,992 | 61,643 | $1.7376 |
| `relu_s8` | 44 | 342,435 | 32,734 | $1.5183 |
| `sigmoid_s8` | 38 | 300,515 | 13,307 | $1.1012 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/yolov8n_rvv_opu_int8/2026-05-27T05-22-24Z` | 48 | $3.7837 |
| `B-bedrock/yolov8n_rvv_opu_int8/2026-05-27T04-31-17Z` | 45 | $3.5299 |
| `B-bedrock/yolov8n_rvv_opu_int8/2026-05-27T04-57-24Z` | 42 | $3.2278 |
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T21-15-51Z` | 43 | $3.0458 |
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-40-39Z` | 51 | $2.8408 |
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T22-02-45Z` | 39 | $2.7343 |
| `B-bedrock/yolov8n_gemmini_int8/2026-05-27T04-05-18Z` | 48 | $2.6020 |
| `B-bedrock/yolov8n_gemmini_int8/2026-05-27T03-43-06Z` | 45 | $2.4635 |
| `B-bedrock/dronet_rvv_opu_int8/2026-05-26T21-43-14Z` | 33 | $2.3146 |
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-25-32Z` | 39 | $1.9383 |
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-13-37Z` | 36 | $1.4161 |
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-58-55Z` | 18 | $0.9664 |
| `B-bedrock/yolov8n_gemmini_int8/2026-05-27T03-32-10Z` | 29 | $0.8685 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-11-12Z` | 12 | $0.8643 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-06-37Z` | 12 | $0.8628 |
| `B-bedrock/dronet_rvv_smoke/2026-05-26T21-02-09Z` | 12 | $0.8336 |
| `B-bedrock/dronet_gemmini_int8/2026-05-27T03-10-32Z` | 12 | $0.7979 |
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-33-54Z` | 27 | $0.7563 |
| `B-bedrock/yolov8_nano_scalar_smoke/2026-05-26T22-48-40Z` | 18 | $0.7431 |
| `B-bedrock/dronet_gemmini_int8/2026-05-27T03-04-28Z` | 12 | $0.7299 |
| `B-bedrock/dronet_gemmini_int8/2026-05-27T03-17-01Z` | 12 | $0.6298 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-23-36Z` | 7 | $0.1594 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-26-34Z` | 7 | $0.1249 |
| `B-bedrock/dronet_gemmini_int8/2026-05-26T22-25-11Z` | 4 | $0.1139 |
| `B-bedrock/dronet_gemmini_q31_int8/2026-05-26T22-28-14Z` | 2 | $0.0271 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `recapture-v3-2026-05-27` | yolov8n_rvv_opu retry after saturn_opu.h macro fix | 2026-05-27T04:31:17 | 2026-05-27T05:52:42 | 135 | $10.5414 |
| `recapture-v2-2026-05-27` | Recapture with LLM-skip extended to initial gen | 2026-05-27T03:04:28 | 2026-05-27T04:31:17 | 158 | $8.0915 |
| `baseline-yolov8n-arm-a-2026-05-26` | Arm A curated baseline: yolov8_nano_scalar_smoke, 3 reps | 2026-05-26T22:48:36 | 2026-05-26T23:10:12 | 36 | $1.7095 |
| `baseline-yolov8n-arm-b-sanity-2026-05-26` | Arm B-bedrock yolov8n 1-rep sanity (beam=1 exp=2 iter=1) | 2026-05-26T22:33:54 | 2026-05-26T22:48:36 | 27 | $0.7563 |
| `baseline-dronet-arm-b-matrix-2026-05-26` | Arm B-bedrock matrix: dronet × {rvv, rvv_opu, gemmini, gemmini_q31}, 3 reps each, beam=2 exp=3 iter=2 | 2026-05-26T21:01:56 | 2026-05-26T22:32:21 | 171 | $11.0807 |
| `baseline-dronet-arm-b-2026-05-26` | Arm B-bedrock baseline: dronet_scalar_smoke beam=2 exp=3 iter=2, 3 reps, --max-usd 5 | 2026-05-26T18:12:41 | 2026-05-26T19:02:18 | 126 | $6.1953 |
