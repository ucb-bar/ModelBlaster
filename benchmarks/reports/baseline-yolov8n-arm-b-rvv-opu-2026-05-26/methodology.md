# Methodology: baseline-yolov8n-arm-b-rvv-opu-2026-05-26

**Negative result, captured deliberately as signal.** Arm B-bedrock on
yolov8n × rvv_opu fails on all 3 reps at the same op: `cat3_c1_s8`
(three-way channel-axis concat, int8). The LLM cannot synthesize a
working rvv_opu-affined kernel for this op after 4 attempts; the cell
gives up.

This contrasts with `baseline-dronet-arm-b-matrix-2026-05-26` where the
same target (`dronet_rvv_opu_int8`) hit 29.6× speedup bit-exact. The
delta is workload structure: dronet has one concat (or none) and 8 op
kinds; yolov8n has multiple `cat` variants (cat2, cat3, cat4 across
the C2f neck) plus depthwise / upsample / silu — none of which have
curated rvv_opu kernels yet, so every cat op falls to the LLM optimize
loop.

## Approach
Arm B driver, `--beam 2 --expansions 3 --iterations 2 --max-usd 8.0`,
RUNNER=spike, 3 replicates. Same knobs that won on dronet × rvv_opu.

## Outcome

| Rep | Calls | Status | Failure point |
|---|---|---|---|
| 1 | 10 | failed | `cat3_c1_s8` after 4 attempts |
| 2 | 4 | failed | same |
| 3 | 4 | failed | same |

Total spend: minimal (~$0.50-1, gave up before extensive optimize-loop
spend). All 3 reps hit the same op early in the dispatch sequence and
the harness correctly exited rather than burning budget on
unworkable ops.

## Why dronet rvv_opu won but yolov8n rvv_opu loses

dronet's op set on rvv_opu (8 kinds: conv2d, linear, batchnorm, relu,
add, maxpool, sigmoid, plus the concat-free residual structure) is
fully covered by curated rvv_opu kernels OR safe to fall through the
scalar reference. The LLM optimize loop got to work on conv2d / linear
where the OPU instructions actually help, and hit cycle wins.

yolov8n's C2f neck has cat2 / cat3 / cat4 splits + concats whose
curated coverage is scalar-only; on rvv_opu, the algorithm queue
gives the LLM no precedent to extend from, and Sonnet's attempts are
incorrect (likely treating the int8 concat as a memcpy-with-broadcast
when channel-axis concat needs explicit lane-by-lane copies with
zero-point handling).

## Fix candidates

1. **Add curated rvv_opu cat kernels.** Probably the cheapest path —
   the scalar versions already work, just need rvv intrinsic versions.
   `kernels/rvv_opu/rvv_opu_cat{2,3,4}_c1_s8.c` mirroring the existing
   `kernels/rvv/rvv_cat*.c` pattern.
2. **Algorithm queue: allow scalar fallback for cat ops on rvv_opu.**
   If the rvv_opu algorithm queue doesn't have a candidate, fall
   through to the scalar reference instead of demanding the LLM
   synthesize one. Code-level change in
   `pipeline/reference_kernels.py` or `pipeline/generate_kernels.py`.
3. **LLM prompt augmentation.** Add a small "rvv_opu cat" few-shot
   example to the prompt so Sonnet has a working reference. Lower
   confidence; concat semantics under int8 with per-axis layout are
   easy to get wrong from a few examples.

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| max_usd per rep | $8.00 (not approached) |
| replicates | 3 |

## Reproducing this report

```bash
git checkout 213faaa
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-yolov8n-arm-b-rvv-opu-2026-05-26 \
    --label "Arm B-bedrock yolov8n × rvv_opu, 3 reps, beam=2 exp=3 iter=2"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload yolov8n_rvv_opu_int8 \
        --runner-override spike \
        --beam 2 --expansions 3 --iterations 2 \
        --no-firesim-eval --max-usd 8.0
done
uv run mb-cost session end
uv run mb-cost export --full baseline-yolov8n-arm-b-rvv-opu-2026-05-26 \
    --session baseline-yolov8n-arm-b-rvv-opu-2026-05-26
```

## Next steps

- **Add the missing rvv_opu cat kernels** — unblocks this cell and
  every other yolov8n-derived cell that hits cat ops on accelerator
  targets. Quickest path forward.
- **Try yolov8n × gemmini Arm B** — gemmini doesn't accelerate cat
  but the algorithm queue should fall through to scalar reference
  there; might capture cleanly. The Arm A gemmini result (36.9×) is
  the headline win on yolov8n already; an Arm B improvement on
  conv2d-only via Sonnet would only stack on top.
