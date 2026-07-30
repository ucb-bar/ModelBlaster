# Methodology: demo-arm-a-mlp

Reference Arm A bundle on the smallest viable workload
(`mlp_generic_scalar_smoke`: 3 Linear + 2 ReLU, scalar int8, spike runner).
Tutorial example for `mb-cost diff` and a regression oracle for the
harness itself — every metric should populate, every cell should report
`verify_pass=true` + `bit_exact=true`.

## Approach
Arm A picks curated kernels from `kernels/<target>/` when one matches the
op signature and falls back to the scalar reference oracle in
`pipeline/reference_kernels.py` otherwise. No LLM in the loop.
Deterministic across reps modulo wall-clock drift.

## Configuration
| Knob          | Value                          |
|---------------|--------------------------------|
| arm           | A (curated → scalar fallback)  |
| LLM           | none                           |
| workload      | `mlp_generic_scalar_smoke`     |
| target        | `scalar`                       |
| quant         | `int8`                         |
| runner        | `spike`                        |
| reps          | 1 (demo); set `--runs 3` for stats |
| FIRESIM_EVAL  | 0                              |
| max_usd       | n/a                            |

## Reproducing this bundle
```bash
git checkout feat/benchmark-harness
source scripts/setup_benchmark_env.sh
uv run mb-cost session start demo-arm-a-mlp \
    --label "Arm A demo bundle, mlp_generic_scalar_smoke"
uv run python -m benchmarks.arms.arm_a_curated \
    --workload mlp_generic_scalar_smoke --runs 1
uv run mb-cost session end
uv run mb-cost export --full demo-arm-a-mlp
```

## What this bundle contains
- `report.json` / `report.md` — top-level summary + per-cell breakdown
- `per-cell/A__mlp_generic_scalar_smoke__<ts>/` — per-rep raw artifacts
  (`run.json`, `accuracy.json`, `wall_cycles.txt`, `profile_spike.csv`,
  `cycles_per_op.json`, `stage_timings.json`, `binary_size.json`,
  `passes_applied.json`, `kernel_picks.json`, `graph_summary.json`, `env.txt`)
- `kernels/mlp_generic/int8/scalar/` — kernel C source from this exact run
  (`kernels.c`, `kernels.h`, `graph.json`, `kernel_picks.json`,
  `optimize_summary.json`, `beam_search_trajectory.jsonl`)

## Diffing a future change against this bundle
```bash
uv run mb-cost diff benchmarks/reports/demo-arm-a-mlp <your-new-bundle>
```
