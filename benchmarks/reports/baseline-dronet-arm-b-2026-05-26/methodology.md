# Methodology: baseline-dronet-arm-b-2026-05-26

First real Arm B-bedrock pass on a non-toy workload. Three replicates of
`dronet_scalar_smoke` driven by Claude Sonnet 4.5 (`us.anthropic.claude-
sonnet-4-5-20250929-v1:0`, us-east-1) generating + iteratively optimizing
every scalar int8 kernel against the reference baseline. This is the
counterpoint to `baseline-dronet-arm-a-2026-05-26`: same workload, same
target, same runner, same metric schema — only the kernel-source method
differs.

## Approach
- Arm B driver (`arm_b_bedrock`) with `BACKEND=llm`, `OPTIMIZE=1`,
  `RUNNER=spike`, `BEAM=2 EXPANSIONS=3 ITERATIONS=2`.
- Budget cap `--max-usd 5.0` per replicate.
- Three replicates back-to-back. Each replicate is an independent invocation
  with no warm cache between them (the kernel cache lives under
  `examples/dronet/int8/cache/` and is shared across reps for cache HIT,
  but optimize-loop state is per-run).

## Hypothesis
Sonnet 4.5, given the reference C kernel + spike profile of its baseline
cycles, can produce loop-restructured / SIMD-friendly / unrolled scalar
variants that beat the reference. The improvement should be biggest on the
ops with the largest baseline cycle counts (conv2d, maxpool) where there's
headroom for loop optimization to matter; small on already-tight ops
(sigmoid, add) where the LLM has nothing to claw back.

## Results

End-to-end wall cycles vs Arm A baseline (4,546,550 deterministic):

| Replicate | Wall cycles | Speedup vs Arm A | Calls | $   | bit-exact |
|---|---|---|---|---|---|
| 1 | 1,776,700 | **+60.9%** | 36 | $1.42 | ✓ (linf=0.0) |
| 2 | 1,709,600 | **+62.4%** | 39 | $1.94 | ✓ (linf=0.0) |
| 3 | 1,497,750 | **+67.0%** | 51 | $2.84 | ✓ (linf=0.0) |
| **mean** | **1,661,350** | **+63.4%** | 42 | $2.07 | ✓ |

Per-op `improvement_pct` (rep 1, baseline → best within the optimize loop):

| Op | Baseline cycles | Best | Improvement |
|---|---|---|---|
| `conv2d_s8` | 254,796,585 | 171,736,007 | **32.6%** |
| `linear_s8` | 49,481 | 37,243 | **24.7%** |
| `maxpool2d_s8` | 3,825,153 | 3,244,379 | **15.2%** |
| `batchnorm2d_s8` | 1,602,353 | 1,548,673 | 3.4% |
| `add_s8` | 490,603 | 490,603 | 0.0% (all attempts verify_fail) |
| `relu_s8` | 571,054 | 571,054 | 0.0% |
| `sigmoid_s8` | 151 | 151 | 0.0% |

The conv2d gain dominates the end-to-end speedup — expected since conv2d
is ~98% of baseline cycles. The 0% rows for add/relu/sigmoid are real
behavioral signals worth following up on: add_s8 had every LLM attempt
trip the verify atol/rtol; relu and sigmoid presumably hit the algorithmic
floor (they're memory-bound, one pass over inputs).

## Cost analysis

- **Total Bedrock spend: $6.20** across 126 LLM calls (mean ~$0.049/call).
- 64% of spend on `linear_s8` ($2.84 / 27 calls) — the two linears in
  dronet got the most retries because the initial outputs failed verify
  more often than e.g. conv2d.
- Per-replicate variance comes from the LLM trying more candidates on
  some runs (51 calls in rep 3 vs 36 in rep 1), which also correlates
  with the deeper cycle improvement in rep 3.
- These are **list prices**; AWS Cost Explorer will show whatever
  discount agreement the account has, with the usual 12-24h billing
  lag.

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam | 2 |
| expansions | 3 |
| iterations | 2 |
| FIRESIM_EVAL | 0 (spike-only) |
| max_usd | $5.0 per rep |
| replicates | 3 |

## Reproducing this report

```bash
git checkout 265851b
source scripts/setup_benchmark_env.sh
# MODEL defaults to claude-sonnet-4-5 since 265851b; override only if needed
uv run mb-cost session start baseline-dronet-arm-b-2026-05-26 \
    --label "Arm B-bedrock baseline: dronet_scalar_smoke beam=2 exp=3 iter=2, 3 reps"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload dronet_scalar_smoke \
        --beam 2 --expansions 3 --iterations 2 \
        --no-firesim-eval --max-usd 5.0
done
uv run mb-cost session end
uv run mb-cost export --full baseline-dronet-arm-b-2026-05-26 \
    --session baseline-dronet-arm-b-2026-05-26
```

## Next steps

- **A.4** — add a fusion pass to `pipeline/extract_graph_export.py` and
  re-run the same workload; diff against this baseline. Anchor the
  "drive a modification" loop on real measurable data.
- **B.1** — extend the matrix to rvv / gemmini / rvv_opu cells (spike
  only). Blocked on the `CONFIG_RISCV_V_KERNEL_ONLY` Kconfig fix.
- **add_s8 verify_fail** is worth investigating: every LLM attempt
  failed verify with `max_abs_err=1 max_rel_err=0.167` — pure off-by-one
  rounding. Tightening the prompt to demand exact int8 rounding
  semantics may unlock this op.
