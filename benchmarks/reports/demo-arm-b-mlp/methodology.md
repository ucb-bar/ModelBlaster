# Methodology: demo-arm-b-mlp

Reference Arm B-bedrock bundle on the smallest viable workload
(`mlp_generic_scalar_smoke`: 3 Linear + 2 ReLU, scalar int8, spike runner).
Tutorial example showing the LLM-driven beam search end-to-end: token
ledger, beam_search_trajectory, kernel C produced by the model, and the
predicted-vs-actual cycle comparison vs the Arm A oracle.

## Approach
Arm B-bedrock asks Bedrock-served Claude to author kernel C for each op
in the graph, then runs a beam search over candidate kernels: each
candidate is built, verified against the scalar reference, and timed on
spike. The best-cycle-per-op winner is kept; siblings get scored and
either iterated on or pruned. The Arm A cell on the same workload is
included in this bundle (per-cell `A__mlp_generic_scalar_smoke__*`) so
the diff is self-contained.

## Configuration
| Knob          | Value                              |
|---------------|------------------------------------|
| arm           | B-bedrock                          |
| LLM           | Bedrock claude-sonnet-4-5          |
| workload      | `mlp_generic_scalar_smoke`         |
| target        | `scalar`                           |
| quant         | `int8`                             |
| runner        | `spike`                            |
| beam          | 2                                  |
| expansions    | 3                                  |
| iterations    | 2                                  |
| reps          | 1 (demo)                           |
| FIRESIM_EVAL  | 0                                  |
| max_usd       | (capped per session by `mb-cost --budget-usd`) |

## Reproducing this bundle
```bash
git checkout feat/benchmark-harness
source scripts/setup_benchmark_env.sh
# Bedrock credentials required: source ../set_api_keys.sh
uv run mb-cost session start demo-arm-b-mlp \
    --label "Arm B-bedrock demo bundle, mlp_generic_scalar_smoke" \
    --budget-usd 1.00
uv run python -m benchmarks.arms.arm_b_bedrock \
    --workload mlp_generic_scalar_smoke \
    --beam 2 --expansions 3 --iterations 2
uv run mb-cost session end
uv run mb-cost export --full demo-arm-b-mlp
```

## What this bundle contains
- `report.json` / `report.md` — Arm B summary + side-by-side vs Arm A
- `per-cell/B-bedrock__mlp_generic_scalar_smoke__<ts>/` — Arm B raw
  artifacts including `llm_calls.jsonl` (per-call token + cost ledger),
  `llm_tokens.json`, `beam_search_trajectory.jsonl`,
  `optimize_summary.json` (winner picks per op), plus the standard
  `run.json` / `accuracy.json` / `wall_cycles.txt` /
  `profile_spike.csv` / `cycles_per_op.json` / `stage_timings.json` /
  `binary_size.json` / `passes_applied.json` / `kernel_picks.json` /
  `graph_summary.json` / `env.txt`
- `per-cell/A__mlp_generic_scalar_smoke__<ts>/` — the Arm A cell on
  the same workload (oracle for verify + cycle comparison)
- `kernels/mlp_generic/int8/scalar/` — LLM-authored kernel C from this run

## Diffing a future change against this bundle
```bash
uv run mb-cost diff benchmarks/reports/demo-arm-b-mlp <your-new-bundle>
```
