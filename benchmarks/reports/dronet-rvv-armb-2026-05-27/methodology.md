# Methodology: dronet-rvv-armb-2026-05-27

Closes out the dronet × rvv Arm B cell after the `conv2d_s8` skip
landed for the rvv target. **3/3 reps bit-exact**, but cycles are
effectively the Arm A baseline because the optimize loop now skips
the dominant op (conv2d) and falls through to the scalar reference
for it — matching exactly what Arm A does on the same workload.

## Approach
Arm B-bedrock, `--beam 2 --expansions 3 --iterations 2 --max-usd 8.0`,
RUNNER=spike (workloads.yaml default for the rvv smoke). With the
new LLM_SKIP_OPS_PER_TARGET["rvv"] = {"conv2d_s8"}, the BACKEND=llm
initial pass keeps the seeded reference_impl for conv2d_s8 instead
of asking Sonnet for an RVV-affined version that would trip verify.

## Results

| Rep | Wall cycles | bit_exact | linf |
|---|---|---|---|
| 1 | 4,607,500 | ✓ | 0.0 |
| 2 | 4,607,350 | ✓ | 0.0 |
| 3 | 4,607,400 | ✓ | 0.0 |
| **mean** | **4,607,417** | ✓ | 0.0 |

Arm A baseline on the same workload: 4,612,350 cycles. Arm B is
~5k cycles faster (~0.1%) which is noise — the optimize loop made
small wins on batchnorm / linear / etc. but the dominant conv2d
stays at reference.

## Why no speedup vs Arm A here

`dronet_rvv_smoke` on the rvv backend has:
- conv2d_s8: dominant cost (>90% of cycles), falls to reference
  scalar via the skip
- linear_s8, batchnorm2d_s8, maxpool2d_s8, relu_s8: curated rvv
  kernels work; LLM optimize loop runs and may improve marginally
- add_s8, sigmoid_s8: no curated rvv; scalar reference

With conv2d fixed at reference, the ceiling for Arm B improvement is
small (1-2% from incremental optimize on the smaller ops).

## What this cell really documents

This bundle is the **proof that the conv2d_s8 skip works**. Without
it, Arm B was hard-failing after 4 LLM attempts (curated kernels
tripping verify, LLM-gen retries tripping too, cell aborts). With it,
the cell completes cleanly + bit-exact, even though the cycle win
isn't there. That's the harness doing its job: surfacing that the
rvv conv2d kernel gap is the real bottleneck, not a missing LLM win.

## Real follow-up

Update the curated rvv conv2d kernels (`rvv_vsmul_vnclip`,
`rvv_oc_blocked`, `rvv_widening_oc`) to handle the per-channel scale
convention dronet uses. Once they pass verify, drop conv2d_s8 from
the LLM_SKIP_OPS list and rerun this cell — should see the actual
RVV speedup on conv2d (~3-5× over scalar based on the rvv_opu
indir_gemm result scaled down for missing OPU).

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| max_usd per rep | $8.00 (well under) |
| replicates | 3 |
| LLM_SKIP_OPS_PER_TARGET["rvv"] | conv2d_s8 |

## Reproducing this report

```bash
git checkout 2ac7c4e
source scripts/setup_benchmark_env.sh
uv run mb-cost session start rvv-armb-final-2026-05-27 \
    --label "dronet × rvv Arm B with conv2d_s8 skip"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload dronet_rvv_smoke \
        --beam 2 --expansions 3 --iterations 2 \
        --no-firesim-eval --max-usd 8.0
done
uv run mb-cost session end
uv run mb-cost export --full dronet-rvv-armb-2026-05-27
```

## Status (paired with companion bundles on this branch)

This is the last Arm B cell needed for the dronet matrix. With this,
dronet has 4 working Arm B-bedrock targets captured:

| Workload × Target | Result |
|---|---|
| dronet × scalar | +63% (baseline-dronet-arm-b-2026-05-26) |
| dronet × rvv | +0.1% (this bundle) — capped at reference-conv2d ceiling |
| dronet × rvv_opu | +96.6% / 29.6× (baseline-dronet-arm-b-matrix-2026-05-26) |
| dronet × gemmini | +17% on top of 24× Arm A (baseline-matrix-arm-b-2026-05-27) |

The yolov8n × rvv Arm B counterpart was not captured this session
(killed before reps completed). Tracked as a session-pause follow-up.
