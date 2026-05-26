# Cost report — `baseline-dronet-arm-b-2026-05-26`

_Generated at 2026-05-26T19:02:18.562322+00:00 by_  `agustin@garden`  _on git_  `265851b`  _branch_  `feat/benchmark-harness`.

_Filter:_ session = `baseline-dronet-arm-b-2026-05-26`

## Totals
- **Spend:** $6.1953 across 126 calls
- This month: $6.1953 (126 calls)
- Input (uncached / cached): 810,156 / 0
- Output: 250,987

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 126 | 810,156 | 0 | 250,987 | $6.1953 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `linear_s8` | 27 | 297,711 | 129,540 | $2.8362 |
| `conv2d_s8` | 15 | 125,898 | 37,769 | $0.9442 |
| `relu_s8` | 27 | 125,448 | 28,531 | $0.8043 |
| `batchnorm2d_s8` | 15 | 82,179 | 19,839 | $0.5441 |
| `sigmoid_s8` | 24 | 92,844 | 9,717 | $0.4243 |
| `maxpool2d_s8` | 9 | 51,813 | 17,845 | $0.4231 |
| `add_s8` | 9 | 34,263 | 7,746 | $0.2190 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-40-39Z` | 51 | $2.8408 |
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-25-32Z` | 39 | $1.9383 |
| `B-bedrock/dronet_scalar_smoke/2026-05-26T18-13-37Z` | 36 | $1.4161 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `baseline-dronet-arm-b-2026-05-26` | Arm B-bedrock baseline: dronet_scalar_smoke beam=2 exp=3 iter=2, 3 reps, --max-usd 5 | 2026-05-26T18:12:41 | 2026-05-26T19:02:18 | 126 | $6.1953 |
| `baseline-dronet-arm-a-2026-05-26` | Arm A curated baseline: dronet_scalar_smoke + dronet_rvv_smoke, 3 reps each | 2026-05-26T18:05:37 | 2026-05-26T18:11:40 | 0 | $0.0000 |
| `auto-A-mlp_generic_scalar_smoke-2026-05-26T18-0` | auto-opened by arm driver A for workload 'mlp_generic_scalar_smoke' (scalar/spike) | 2026-05-26T18:05:02 | 2026-05-26T18:05:02 | 0 | $0.0000 |
