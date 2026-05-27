# Methodology: rvv-layout-fix-armb-2026-05-27

The Arm B-bedrock rvv capture after the IHWOC weight-layout bug fix
(commit 199c559) + expanded LLM_SKIP_OPS_PER_TARGET["rvv"] for the
elementwise / norm / cat ops that have shape-bound issues in their
curated kernels.

## Approach
Arm B-bedrock, `--beam 2 --expansions 3 --iterations 2 --max-usd 8.0`,
RUNNER=spike. 3 reps per cell. LLM_SKIP_OPS_PER_TARGET["rvv"] =
{add_s8, batchnorm2d_s8, sigmoid_s8, cat2_c1_s8, cat3_c1_s8, cat4_c1_s8}
so the optimize loop only touches ops that have working curated
candidates: conv2d_s8, linear_s8, maxpool2d_s8, relu_s8, silu_s8,
upsample_nearest_s8.

## Results

### dronet × rvv (3/3 bit-exact)

| Rep | Wall cycles | vs Arm A 275,700 | vs scalar 4.55M |
|---|---|---|---|
| 1 | 269,050 | +2.4% | **16.9×** |
| 2 | 269,150 | +2.4% | 16.9× |
| 3 | 269,050 | +2.4% | 16.9× |
| **mean** | **269,083** | **+2.4%** | **16.9×** |

### yolov8n × rvv (2/3 captured; rep 3 hit Bedrock quota)

| Rep | Wall cycles | vs Arm A 6,315,000 | vs scalar 104.4M |
|---|---|---|---|
| 1 | 5,876,250 | +7.0% | **17.8×** |
| 2 | 6,314,600 | +0.0% (noise) | 16.5× |
| 3 | — | (Bedrock 429: "too many tokens per day") | |

User stop-criterion: "if rep 2 isn't better than rep 1, stop." Rep 2
matched the Arm A baseline (no Sonnet improvement); rep 3 would have
been a 4th attempt anyway and tripped a daily Bedrock quota.

## Headline finding: the IHWOC layout fix was the unlock

Before commit 199c559: dronet × rvv Arm A used `source=reference` for
conv2d_s8 because the curated rvv conv2d kernels tripped verify with
linf=12 on every shape. Arm B couldn't recover because its
target-affinity preference dropped the universal "direct" algorithm.

After 199c559: the same curated kernels (rvv_vsmul_vnclip) PASS verify
bit-exact and give a 16-17× wall-cycle speedup. Arm B's optimize loop
adds another 2-7% on top.

The bug was in `_LAYOUT_PERMUTATION["ihwoc"]`: declared as `(2,3,1,0)`
which gives physical HWIO ordering, but the kernel indexes as IHWO
(`weight[((ic*KH+kh)*KW+kw)*OC+oc]`). Fixed to `(1,2,3,0)`. No kernel
code changed -- just the buffer packing transpose.

## Cost analysis

| Cell | Reps | Calls | $ |
|---|---|---|---|
| dronet × rvv | 3 OK | ~21 each | ~$1.30/rep |
| yolov8n × rvv | 2 OK + 1 failed | ~30 each | ~$2.50/rep |
| **Total session** | | | **~$8** |

Cumulative branch spend: $59.15 of the $100 ceiling.

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| max_usd per rep | $8.00 |
| replicates per cell | 3 (yolov8n: 2/3 due to quota) |
| LLM_SKIP_OPS_PER_TARGET["rvv"] | {add, batchnorm, sigmoid, cat2/3/4}_s8 |

## Reproducing this report

```bash
git checkout 199c559
source scripts/setup_benchmark_env.sh
uv run mb-cost session start rvv-final-armb-2026-05-27 \
    --label "rvv Arm B after IHWOC layout fix + expanded skip"
for spec in dronet_rvv_smoke yolov8_nano_rvv_smoke; do
    for rep in 1 2 3; do
        uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
            --workload $spec \
            --beam 2 --expansions 3 --iterations 2 \
            --no-firesim-eval --max-usd 8.0
    done
done
uv run mb-cost session end
uv run mb-cost export --full rvv-layout-fix-armb-2026-05-27 \
    --session rvv-final-armb-2026-05-27
```

## Final coverage matrix (all sessions on this branch)

| Workload × Target | Arm A | Arm B-bedrock |
|---|---|---|
| dronet × scalar     | 4.55M (1×) | +63% (1.66M) |
| dronet × rvv        | **275k (16.9×)** ← fixed by layout | **+2.4% on top (269k)** ← this bundle |
| dronet × rvv_opu    | 4.55M (functional only on spike) | +96.6% / 29.6× (153k) |
| dronet × gemmini    | 24× (188k) | +17% stack (155k) |
| dronet × gemmini_q31 | ⚠ Q31 spike emul | ⚠ same (FireSim-only) |
| dronet × hetero     | ⚠ spike-hetero emul | — (FireSim-only) |
| yolov8n × scalar    | 104M (1×) | +63% (38.7M) |
| yolov8n × rvv       | **6.32M (16.5×)** ← fixed by layout | **+7.0% rep1 / 0% rep2 (5.88M / 6.31M)** ← this bundle |
| yolov8n × rvv_opu   | 104.4M (functional) | +94% / 17× (6.26M) |
| yolov8n × gemmini   | 37× (2.83M) | +12% stack (2.50M) |
| yolov8n × gemmini_q31 | ⚠ Q31 spike emul | ⚠ same (FireSim-only) |
