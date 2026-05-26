# Methodology: baseline-yolov8n-arm-b-sanity-2026-05-26

Three-rep Arm B-bedrock baseline on `yolov8_nano_scalar_smoke`. This is
the first non-dronet, non-MLP Arm B capture — it validates that the
harness, the bring-up of yolov8n (ultralytics weights + model-name
match in workloads.yaml), and the Sonnet 4.5 optimize loop all carry
through to a substantially larger workload (240 weight tensors vs
dronet's 36).

## Approach
Arm B driver at the deliberately small `--beam 1 --expansions 2
--iterations 1 --max-usd 5.0` to anchor cost expectations before
scaling to the full beam=2 / exp=3 / iter=2 settings used elsewhere.
RUNNER=spike. 3 replicates back-to-back.

## Results

| Rep | Wall cycles | Speedup vs Arm A | Calls | $   | bit-exact |
|---|---|---|---|---|---|
| 1 | 41,394,900 | **+60.3%** | 27 | $0.76 | ✓ (linf=0.0) |
| 2 | 42,200,100 | **+59.6%** | 18 | ~$0.50 | ✓ (linf=0.0) |
| 3 | 32,527,200 | **+68.8%** | 18 | ~$0.50 | ✓ (linf=0.0) |
| **mean** | **38,707,400** | **+62.9%** | 21 | ~$0.59 | ✓ |

Arm A baseline (deterministic across 3 reps): 104,405,200 cycles.

## Observations

- **Speedup tracks dronet** (+62.9% here vs +63.4% on dronet) — Sonnet 4.5's
  scalar kernel optimization generalizes from dronet's 8 ops to
  yolov8n's many more (SiLU, depthwise conv, upsample, etc.) at
  consistent ~60-65% improvement on the workload.
- **Cost much lower than estimated** ($0.59/rep vs the planning
  estimate of $4-8). Reason: yolov8n's many ops still mostly hit the
  curated kernel cache (cached or "curated HIT" paths skip the LLM
  call) so only a subset trigger Sonnet calls.
- **Lower beam settings are sufficient for win discovery** on scalar.
  The wins came at beam=1 exp=2 iter=1; beam=2 exp=3 iter=2 would
  probably push the best rep below 30M cycles but at 4x the cost.
- **Build-dir race observed** when Arm A and Arm B yolov8n runs
  overlap on `examples/yolov8_nano/int8/build/scalar/`. Documented
  here so future captures serialize per workload (or use the rotated
  build dirs flag if/when added).

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam | 1 |
| expansions | 2 |
| iterations | 1 |
| FIRESIM_EVAL | 0 (spike-only) |
| max_usd per rep | $5.00 |
| replicates | 3 |

## Reproducing this report

```bash
git checkout 6de5d40
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-yolov8n-arm-b-sanity-2026-05-26 \
    --label "Arm B-bedrock yolov8n 1-rep sanity (beam=1 exp=2 iter=1)"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
        --workload yolov8_nano_scalar_smoke \
        --beam 1 --expansions 2 --iterations 1 \
        --no-firesim-eval --max-usd 5.0
done
uv run mb-cost session end
uv run mb-cost export --full baseline-yolov8n-arm-b-sanity-2026-05-26 \
    --session baseline-yolov8n-arm-b-sanity-2026-05-26
```

## Next steps

- **Full-beam yolov8n run**: same workload, beam=2 exp=3 iter=2, 3 reps.
  Expected cost ~$6-10 based on the sanity-scaled estimate (4x knobs
  → ~$2.4 per rep × 3 = $7.2).
- **Matrix expansion**: yolov8n_rvv_opu_fp16 + yolov8n_gemmini_q31_int8
  via `--runner-override spike`. These will need different cycle
  baselines (fp16 vs int8) and may hit the same kernel-gen issues
  flagged in `baseline-dronet-arm-b-matrix-2026-05-26` (LLM struggles
  with RVV / gemmini idioms).
- **Pair with `baseline-yolov8n-arm-a-2026-05-26`** for the diff.
