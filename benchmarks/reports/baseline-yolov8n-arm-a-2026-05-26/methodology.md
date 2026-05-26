# Methodology: baseline-yolov8n-arm-a-2026-05-26

Arm A (curated, no LLM) scalar baseline on yolov8n. Three replicates of
`yolov8_nano_scalar_smoke` give the deterministic reference floor that
every Arm B-bedrock yolov8n run is measured against.

## Approach
Arm A driver, BACKEND=reference, OPTIMIZE=0, RUNNER=spike. Three
replicates back-to-back under session
`baseline-yolov8n-arm-a-2026-05-26`. Pretrained COCO weights loaded
from ultralytics' yolov8n.pt (cached at repo root, gitignored).

## Results

| Rep | Wall cycles | bit_exact | linf |
|---|---|---|---|
| 1 | 104,405,200 | ✓ | 0.0 |
| 2 | 104,405,200 | ✓ | 0.0 |
| 3 | 104,405,200 | ✓ | 0.0 |
| **mean** | **104,405,200** | ✓ | 0.0 |

Deterministic across all 3 replicates (stddev = 0 — spike is fully
deterministic given identical inputs and identical generated C).

## Comparison anchor

yolov8n is 23× the cycles of dronet's scalar baseline
(104.4M / 4.55M ≈ 23×) -- expected given yolov8n's depth and the
backbone's per-scale (P3/P4/P5) feature pyramids vs dronet's small
classifier head.

The corresponding Arm B-bedrock sanity capture
(`baseline-yolov8n-arm-b-sanity-2026-05-26`) achieves +62.9% mean
wall-cycle reduction vs this baseline at beam=1 / exp=2 / iter=1.

## Known gaps

- **RVV / Gemmini / hetero yolov8n captures not yet attempted.** The
  workloads.yaml entries `yolov8n_rvv_opu_fp16`, `yolov8n_gemmini_q31_int8`,
  `yolov8n_hetero_int8` are wired but firesim-runner by default;
  spike-side captures need `--runner-override spike` and should be
  expected to surface the same LLM kernel-gen gaps as
  `baseline-dronet-arm-b-matrix-2026-05-26` (LLM trips on RVV ihwoc
  layout, Gemmini maxpool, Q31 params).
- **Verify uses synthetic torch.randn input** (no real RGB image
  via `MODELBLASTER_YOLOV8N_CALIB_IMAGE`). For evaluating real
  detection accuracy this isn't sufficient; for measuring whether
  the compiled kernels match the PyTorch reference on the same
  inputs, it's exact (and bit_exact=true confirms that).

## Reproducing this report

```bash
git checkout 6de5d40
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-yolov8n-arm-a-2026-05-26 \
    --label "Arm A curated baseline: yolov8_nano_scalar_smoke, 3 reps"
for rep in 1 2 3; do
    uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
        --workload yolov8_nano_scalar_smoke
done
uv run mb-cost session end
uv run mb-cost export --full baseline-yolov8n-arm-a-2026-05-26 \
    --session baseline-yolov8n-arm-a-2026-05-26
```

## Next steps

- See `baseline-yolov8n-arm-b-sanity-2026-05-26` for the matching
  Arm B-bedrock capture and the +62.9% speedup story.
- Bring up `yolov8_nano_rvv_smoke` (wired in workloads.yaml after the
  enablement commit) -- gates on RVV ihwoc weight layout being
  honored by either the curated kernel pool or the LLM.
