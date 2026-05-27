# Methodology: yolov8n-rvv-armb-2026-05-27

Closes out the yolov8n × rvv Arm B cell — the last spike-runnable Arm B
target on the dronet+yolov8n matrix. 3/3 reps bit-exact. Same pattern
as `dronet-rvv-armb-2026-05-27`: with conv2d_s8 skipped on the rvv
target, Arm B can't optimize the dominant op, so cycles match Arm A's
reference-fallback baseline.

## Approach
Arm B-bedrock, `--beam 2 --expansions 3 --iterations 2 --max-usd 8.0`,
RUNNER=spike (workloads.yaml default). LLM_SKIP_OPS_PER_TARGET["rvv"]
= {"conv2d_s8"} carried over from `2ac7c4e`.

## Results

| Rep | Wall cycles | bit_exact | linf |
|---|---|---|---|
| 1 | 106,890,750 | ✓ | 0.0 |
| 2 | 106,890,550 | ✓ | 0.0 |
| 3 | 106,893,750 | ✓ | 0.0 |
| **mean** | **106,891,683** | ✓ | 0.0 |

Arm A baseline: 106,923,250 cycles. Arm B is ~32k cycles faster
(~0.03%) — noise. Cell completes cleanly; no LLM verify aborts.

## Cost

- 3 reps × ~$1.95/rep ≈ **$5.86 total**
- Cumulative branch spend: $50.93 of the $100 ceiling
- Per-rep cost is lower than the rvv_opu counterpart ($3.20/rep) because
  the conv2d skip eliminates the most expensive optimize-loop iterations

## Why no Arm B speedup on this cell

yolov8n × rvv is bottlenecked on conv2d (representative of every CNN
workload on RVV). With conv2d_s8 skipped, the optimize loop touches
only the small ops (silu, cat, add, etc.) whose total cycle contribution
is <5% of the workload. So even a 50% improvement on those ops would
only move the total by ~2%. The harness is correctly showing the
ceiling.

Compare to yolov8n × rvv_opu (`baseline-matrix-arm-b-2026-05-27`):
6.26M cycles, 17× faster — because rvv_opu has a working
`conv2d_s8_indir_gemm` curated kernel that the optimize loop CAN
improve on top of. The delta between these two Arm B captures
(106.9M vs 6.26M) is the cost of the broken rvv conv2d curated set.

## Real follow-up (same as dronet-rvv-armb-2026-05-27)

Fix the rvv conv2d kernels' per-channel scale convention. Once they
pass verify, drop conv2d_s8 from LLM_SKIP_OPS_PER_TARGET["rvv"] and
rerun. Expected: ~3-5× speedup vs scalar.

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| max_usd per rep | $8.00 |
| replicates | 3 |
| LLM_SKIP_OPS_PER_TARGET["rvv"] | conv2d_s8 |

## Reproducing this report

```bash
git checkout 0feba67
source scripts/setup_benchmark_env.sh
uv run mb-cost session start yolov8n-rvv-armb-2026-05-27 \
    --label "yolov8n × rvv Arm B (3 reps with conv2d_s8 skip)"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload yolov8_nano_rvv_smoke \
        --beam 2 --expansions 3 --iterations 2 \
        --no-firesim-eval --max-usd 8.0
done
uv run mb-cost session end
uv run mb-cost export --full yolov8n-rvv-armb-2026-05-27 \
    --session yolov8n-rvv-armb-2026-05-27
```

## Status: matrix complete for spike-runnable cells

| Workload × Target | Arm A | Arm B-bedrock |
|---|---|---|
| dronet × scalar     | ✅ 4.55M | ✅ **+63%** (1.66M) |
| dronet × rvv        | ✅ 4.61M | ✅ ~baseline (this bundle's companion; conv2d skip) |
| dronet × rvv_opu    | ✅ 4.55M | ✅ **29.6×** (153k) |
| dronet × gemmini    | ✅ **24×** (188k) | ✅ **+17%** stack (155k) |
| dronet × gemmini_q31 | ⚠ Q31 spike emul | ⚠ same (FireSim-only) |
| dronet × hetero     | ⚠ spike-hetero emul | — (FireSim-only) |
| yolov8n × scalar    | ✅ 104M | ✅ **+63%** (38.7M) |
| **yolov8n × rvv**   | ✅ 106.9M | ✅ **~baseline** (this bundle) |
| yolov8n × rvv_opu   | ✅ 104.4M | ✅ **+94% / 17×** (6.26M) |
| yolov8n × gemmini   | ✅ **37×** (2.83M) | ✅ **+12%** stack (2.50M) |
| yolov8n × gemmini_q31 | ⚠ Q31 spike emul | ⚠ same (FireSim-only) |
