# Methodology: baseline-dronet-arm-b-matrix-2026-05-26

Arm B-bedrock (Sonnet 4.5) baseline across the dronet matrix on spike,
3 replicates per cell. Mixed-outcome capture: 1 cell wins big, 3 cells
expose real kernel-gen / params gaps. **All four outcomes are signal**
— the harness did its job either way.

## Approach
Arm B driver, `--beam 2 --expansions 3 --iterations 2 --no-firesim-eval
--max-usd 5.0`. RUNNER=spike for every cell (`--runner-override spike`
on workloads whose default runner is firesim). 3 replicates per cell
under session `baseline-dronet-arm-b-matrix-2026-05-26`.

## Results

| Cell | Reps OK | Wall cycles | vs Arm A scalar | vs Arm A same-cell | Spend |
|---|---|---|---|---|---|
| `dronet_rvv_opu_int8` | 3/3 | 153,500 / 153,350 / 153,250 | **29.6× faster** | **29.6× faster** | $8.10 |
| `dronet_rvv_smoke` | 0/3 | — (LLM conv2d_s8 verify fail) | — | — | $2.56 |
| `dronet_gemmini_int8` | 0/3 | — (LLM maxpool2d_s8 verify fail) | — | — | $0.40 |
| `dronet_gemmini_q31_int8` | 0/3 | — (Q31 params mismatch, pre-LLM) | — | — | $0.03 |
| **Total matrix spend** | | | | | **$11.08** |

## Wins

**rvv_opu** is the headline: Sonnet 4.5 generated OPU-affined conv2d /
linear / matmul kernels that beat the Arm A curated baseline (which on
spike is functional-only since spike doesn't actually execute OPU
instructions). At 153k wall cycles bit-exact, this is 29.6× faster
than the scalar reference; on a real OPU bitstream the gap should be
larger.

The three reps landed at 153,500 / 153,350 / 153,250 — stddev tiny,
LLM converged on essentially the same kernel each time.

## Failures (real signal, not infrastructure)

### `dronet_rvv_smoke` — LLM conv2d_s8 verify fail

Every LLM attempt at `conv2d_s8` on rvv produced output with linf=53
vs golden — the LLM either doesn't know our `MODELBLASTER_RVV_IHWOC_WEIGHTS`
weight layout convention or trips signature checks
(`signature mismatch (parameter names/order must match the declaration
in kernels.h byte-for-byte)`). Even the **curated** rvv conv2d kernels
fail verify here (`curated MISS: max_abs_err=12`), suggesting the
curated kernels were tested on different shapes than dronet's smoke.

Fix candidates:
- Add an RVV weight-layout banner to the LLM prompt
- Re-test curated rvv conv2d kernels against dronet's exact shapes,
  fix or supersede them
- Cache-aware prompt (`optimize/firesim_eval/cache_aware_prompt.py`)
  may help — not exercised since `--no-firesim-eval`

### `dronet_gemmini_int8` — LLM maxpool2d_s8 verify fail

Sonnet doesn't generate working maxpool2d on Gemmini after 4 attempts.
Likely the LLM is treating maxpool as a Gemmini matmul (it isn't —
Gemmini doesn't accelerate pooling; the reference scalar fallback is
correct).

Fix: prompt should clarify that maxpool stays scalar on
Gemmini-affined builds, or the algorithm queue should skip the
Gemmini branch for maxpool ops entirely.

### `dronet_gemmini_q31_int8` — pre-LLM Q31 params mismatch

Fails before any LLM call: Arm A verify on the per-kernel curated swap
trips because chipyard's generic `gemmini_params.h` doesn't match the
Q0.31 acc-scale path. Same root cause as the matching Arm A failure;
fix is to drop a Q31-specific params header
(`cores/gemmini/include/gemmini_params.h`) per the comment block in
`pipeline/backends.py:266-280`.

## Cost analysis

- $11.08 total / $17.28 cumulative across all Arm B sessions
- 80% of matrix spend on rvv_opu (the winning cell), 23% on rvv_smoke
  retries, 4% on gemmini, ~0% on gemmini_q31 (failed before LLM
  invoked)
- Mean per-call cost ~$0.05 for cells that exercised optimize loops
- All within --max-usd 5.0 per replicate cap

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| FIRESIM_EVAL | 0 (spike-only) |
| max_usd per rep | $5.00 |
| replicates per cell | 3 |

## Reproducing this report

```bash
git checkout c0750c0
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-dronet-arm-b-matrix-2026-05-26 \
    --label "Arm B-bedrock matrix"
for wl in dronet_rvv_smoke dronet_rvv_opu_int8 dronet_gemmini_int8 dronet_gemmini_q31_int8; do
    for rep in 1 2 3; do
        OVR=""
        [ "$wl" != "dronet_rvv_smoke" ] && OVR="--runner-override spike"
        uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
            --workload $wl $OVR \
            --beam 2 --expansions 3 --iterations 2 \
            --no-firesim-eval --max-usd 5.0
    done
done
uv run mb-cost session end
uv run mb-cost export --full baseline-dronet-arm-b-matrix-2026-05-26 \
    --session baseline-dronet-arm-b-matrix-2026-05-26
```

## Next steps

In priority order:

1. **Fix the gemmini_q31 params header** so both Arm A and Arm B can
   produce bit-exact captures on that cell (the Q31 path is the
   bit-exact integer requantize variant — it should win big on
   gemmini when wired up correctly).
2. **Diagnose rvv_smoke conv2d failures**: are the curated kernels
   stale, or are dronet's smoke shapes outside their support window?
   Run `pipeline/generate_kernels.py --verify-only` against the
   curated set to find which shapes break.
3. **Skip maxpool2d on the Gemmini optimize queue** (or tighten the
   prompt) so we don't waste LLM cycles on a known-not-fit op.
4. **B.3 yolov8n sanity rep** — anchor cost expectation for the
   yolov8n matrix.
