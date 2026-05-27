# Methodology: baseline-matrix-arm-b-2026-05-27

The recapture pass after three harness fixes landed: LLM_SKIP_OPS_PER_TARGET
(skips Sonnet for ops where the curated/reference is already at the
floor), curated `rvv_opu_cat{2,3,4}` kernels (mirror of the rvv set),
and `saturn_opu.h` macro hygiene (#undef v0..v31 / m0..m3 at EOH so RVV
intrinsic kernels don't trip macro-substitution).

Three cells × 3 reps each. **All 9 reps land bit-exact.** This is the
first capture where every cell in the Arm B matrix passes verify.

## Results

| Cell | Wall cycles (3 reps mean) | Speedup vs Arm A | $ / rep | bit-exact |
|---|---|---|---|---|
| `dronet × gemmini`         | 155,333 (146.7k/156.8k/162.6k) | **+17.4%** on top of Arm A 188k (24×) → 29× vs scalar | $0.40 avg | ✓ |
| `yolov8n × gemmini`        | 2,497,733 (2.70M/2.66M/2.14M)  | **+11.7%** on top of Arm A 2.83M (37×) → 42× vs scalar | $0.50 avg | ✓ |
| `yolov8n × rvv_opu`        | **6,261,783** (6.27M/6.27M/6.25M) | **+94.0%** vs Arm A 104.4M → **17× speedup** | $3.20 avg | ✓ |

Per-rep cycle stddev is tiny (<1% on rvv_opu, ~10% on gemmini where
Sonnet's choices vary). Cost stddev tracks call count — yolov8n rvv_opu
exercises ~40 unique ops on the OPU-affined path, the gemmini cells
ride mostly curated kernels and only LLM-optimize conv2d / linear.

## Headline: yolov8n × rvv_opu

The biggest finding. Sonnet 4.5 generated OPU-affined kernels (silu,
upsample, depthwise conv, plus the curated cat/conv set we shipped
this session) that together push yolov8n end-to-end from 104.4M
cycles to 6.25M cycles -- 17× faster than the curated reference,
bit-exact, $3.20 per rep.

This is comparable to the dronet × rvv_opu result (29.6× speedup,
$2.66/rep mean from the earlier dronet matrix capture). yolov8n is
roughly 3× bigger in unique ops, but Sonnet handles it at similar
per-op cost because the OPU intrinsic patterns transfer.

## Why each fix landed where it did

- **LLM_SKIP_OPS_PER_TARGET on gemmini** unblocked the gemmini cells.
  Without it, Sonnet kept trying to rewrite ops Gemmini doesn't
  accelerate (relu, batchnorm, add, sigmoid) and tripping verify on
  every attempt, aborting the cell before conv2d / linear got
  optimized. Skip → curated path holds → optimize loop focuses on
  ops with actual win room.
- **rvv_opu cat kernels** unblocked yolov8n × rvv_opu's C2f neck.
  Without them, cat3 fell to the LLM with no curated precedent;
  Sonnet kept emitting intrinsics that didn't compile (after the
  next fix landed, those attempts WERE valid -- but having a working
  curated direct meant the optimize loop's seed was already good).
- **saturn_opu.h #undef v0..v31** was the load-bearing fix. RVV
  intrinsic code that names variables v16 / v8 / v0 / etc. is the
  norm; the macro `#define v16 "x16"` was string-substituting into
  the C declarations and producing
  `vint16m4_t "x16" = __riscv_v*` syntax errors. Without this fix,
  EVERY LLM attempt at any rvv_opu RVV-intrinsic kernel would fail
  to build, regardless of how correct the code was.

## Knobs

| Knob | This run |
|---|---|
| arm | B-bedrock |
| LLM model | claude-sonnet-4-5-20250929-v1:0 |
| beam / expansions / iterations | 2 / 3 / 2 |
| max_usd per rep | $8.00 (cap not approached on any rep) |
| replicates per cell | 3 |
| RUNNER | spike (--runner-override) |

## Cost summary

Total this session: **~$11** added (from $22.05 → ~$33.05 cumulative
on this branch). Within the $100 user-set ceiling.

| Cell | Total spent (3 reps) | Per-rep avg |
|---|---|---|
| dronet × gemmini | ~$1.20 | $0.40 |
| yolov8n × gemmini | ~$1.50 | $0.50 |
| yolov8n × rvv_opu | ~$9.60 | $3.20 |

## Reproducing this report

```bash
git checkout 1fe7be5
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-matrix-arm-b-2026-05-27 \
    --label "Arm B matrix recapture after harness fixes"
for spec in dronet_gemmini_int8 yolov8n_gemmini_int8 yolov8n_rvv_opu_int8; do
    for rep in 1 2 3; do
        uv run python -m modelblaster.benchmarks.arms.arm_b_bedrock \
            --workload $spec --runner-override spike \
            --beam 2 --expansions 3 --iterations 2 \
            --no-firesim-eval --max-usd 8.0
    done
done
uv run mb-cost session end
uv run mb-cost export --full baseline-matrix-arm-b-2026-05-27
```

## Final coverage matrix (after all fixes)

| Workload × Target | Arm A | Arm B-bedrock |
|---|---|---|
| dronet × scalar      | ✅ 4.55M (1×) | ✅ **+63%** (1.66M) |
| dronet × rvv         | ✅ 4.61M | ❌ LLM rvv conv2d gap |
| dronet × rvv_opu     | ✅ 4.55M (functional) | ✅ **29.6×** (153k) |
| dronet × gemmini     | ✅ **24×** (188k) | ✅ **+17%** on top (155k → 29× vs scalar) |
| dronet × gemmini_q31 | ⚠ runs, linf=72 (spike libgemmini float vs Q31 mismatch) | ⚠ same |
| dronet × hetero      | ⚠ runs, linf=52 (spike-hetero emul drift) | — |
| yolov8n × scalar     | ✅ 104M (1×) | ✅ **+63%** (38.7M) |
| yolov8n × rvv        | ✅ 106.9M | ⏸ not yet exercised |
| yolov8n × rvv_opu    | ✅ 104.4M (functional) | ✅ **+94% / 17×** (6.26M) |
| yolov8n × gemmini    | ✅ **37×** (2.83M) | ✅ **+12%** on top (2.50M → 42× vs scalar) |
| yolov8n × gemmini_q31| ⚠ same Q31 issue | ⏸ same |

The harness now runs end-to-end on every cell that has a working
backend dispatch + non-broken Q31 / hetero emulation. The remaining
yellow boxes are documented gotcha-#6 / FireSim-side issues, not
harness gaps.
