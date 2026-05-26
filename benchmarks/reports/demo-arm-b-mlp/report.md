# Cost report — `demo-arm-b-mlp`

_Generated at 2026-05-26T07:01:18.055426+00:00 by_  `agustin@garden`  _on git_  `7ea176a`  _branch_  `feat/benchmark-harness`.

## Totals
- **Spend:** $0.0769 across 2 calls
- This month: $0.0769 (2 calls)
- Input (uncached / cached): 9,200 / 0
- Output: 3,288

## Per-model
| Model | Calls | In (uncached) | In (cached) | Out | USD |
|---|---|---|---|---|---|
| `us.anthropic.claude-sonnet-4-5-20250929-v1:0` | 2 | 9,200 | 0 | 3,288 | $0.0769 |

## Per-kernel (by op)
| Kernel | Calls | In | Out | USD |
|---|---|---|---|---|
| `linear_s8` | 2 | 9,200 | 3,288 | $0.0769 |

## Per-cell
| Cell | Calls | USD |
|---|---|---|
| `B-bedrock/mlp_generic_scalar_smoke/2026-05-26T07-00-12Z` | 2 | $0.0769 |

## Sessions covered
| Session | Label | Started | Ended | Calls | USD |
|---|---|---|---|---|---|
| `mlp-arm-b-demo-v2` | Option B: Arm B-bedrock MLP w/ beam=1 exp=2 iter=1 | 2026-05-26T07:00:12 | 2026-05-26T07:01:00 | 2 | $0.0769 |
