# Methodology: baseline-yolov8n-arm-a-matrix-2026-05-26

Arm A (curated, no LLM) baseline across the yolov8n matrix on spike,
3 replicates per cell. Pairs with `baseline-yolov8n-arm-a-2026-05-26`
(the original scalar-only baseline) — the full A matrix is the union.

## Approach
Arm A driver, BACKEND=reference, OPTIMIZE=0, RUNNER=spike (forced via
`--runner-override spike` for cells whose workload row defaults to
firesim). 3 replicates per cell under session
`baseline-yolov8n-arm-a-matrix-2026-05-26`. Pretrained COCO weights
streamed from `yolov8n.pt`.

## Cells captured

| Cell | Target / quant | Cycles | Speedup vs scalar | Verify |
|---|---|---|---|---|
| `yolov8_nano_scalar_smoke` | scalar / int8 | 104,405,200 | 1.00× (baseline) | ✓ linf=0.0 |
| `yolov8_nano_rvv_smoke`    | rvv / int8     | 106,923,250 | 0.98× | ✓ linf=0.0 |
| `yolov8n_rvv_opu_int8`     | rvv_opu / int8 | 104,405,500 | 1.00× (spike functional-only) | ✓ linf=0.0 |
| `yolov8n_gemmini_int8`     | gemmini / int8 | **2,831,550** | **36.9×** | ✓ linf=0.0 |
| `yolov8n_gemmini_q31_int8` | gemmini_q31 / int8 | 2,810,150 | 37.1× | ✗ linf=9.0 (Q31 params) |

All wall cycles deterministic across the 3 replicates per cell.

## Observations

- **Gemmini wins big on yolov8n.** 36.9× speedup vs scalar, bit-exact.
  Stronger relative win than the 24.2× on dronet because yolov8n is
  conv-heavier (40-ish unique ops, lots of 3x3 / 1x1 conv stages).
- **RVV on this smoke is 2% slower than scalar.** Same pattern as
  dronet — the smoke workload's tiny IO shapes don't amortize
  vector-setup. Real RVV gains would show on a non-smoke workload
  with bigger spatial dims.
- **RVV-OPU on spike == scalar cycles exactly** (104,405,500 vs
  104,405,200, 300-cycle delta is xpurt overhead). Spike doesn't
  execute OPU instructions usefully (gotcha #6); the real OPU
  comparison needs FireSim or Arm B-bedrock's LLM-generated kernels
  that target the OPU directly (see Arm B yolov8n rvv_opu capture
  for the LLM-side story).
- **gemmini_q31 fails verify with linf=9.0**, same root cause as the
  dronet matrix: chipyard's generic DIM=16 `gemmini_params.h` doesn't
  match the Q0.31 acc-scale path that the Q31 kernels emit. The 2.8M
  cycle count is real (kernel runs) but numerics drift. Fix is to
  vendor a Q31-specific params header per `pipeline/backends.py:266-280`.

## Known gaps in this baseline

- **No `yolov8n_hetero_int8`.** Blocked by `notes/yolov8n_architectural_divergence.md`
  cited as P2.1-schedule — the schedule fixture hasn't been generated.
  Once `scripts/gen_hetero_schedule.py` runs against yolov8n's graph.json,
  this cell can join the matrix.
- **No `rvv_opu_fp16`.** Initial workload was wired with `quant: fp16`
  but the backend dispatch lacks an `rvv_opu_f16` entry (only `rvv_opu`
  for int8 and `rvv_f16` for plain RVV exist). Switched to int8 to
  match the dronet matrix shape. Adding fp16-on-OPU would be a
  separate backend wiring task.

## Reproducing this report

```bash
git checkout fb4f293
source scripts/setup_benchmark_env.sh
uv run mb-cost session start baseline-yolov8n-arm-a-matrix-2026-05-26 \
    --label "Arm A matrix baseline"
for wl in yolov8_nano_rvv_smoke yolov8n_rvv_opu_int8 yolov8n_gemmini_int8 yolov8n_gemmini_q31_int8; do
    for rep in 1 2 3; do
        OVR=""
        case $wl in
            yolov8_nano_rvv_smoke) ;;
            *) OVR="--runner-override spike" ;;
        esac
        uv run python -m modelblaster.benchmarks.arms.arm_a_curated \
            --workload $wl $OVR
    done
done
uv run mb-cost session end
uv run mb-cost export --full baseline-yolov8n-arm-a-matrix-2026-05-26 \
    --session baseline-yolov8n-arm-a-matrix-2026-05-26
```

## Next steps

- **Arm B-bedrock on yolov8n × rvv_opu** — the high-leverage cell.
  dronet hit 29.6× speedup there. Yolov8n likely similar.
- **gemmini_q31 params fix** (vendor a Q0.31-specific params header)
  unblocks bit-exact verify on Q31 across dronet and yolov8n.
- **Yolov8n hetero schedule** via `scripts/gen_hetero_schedule.py`
  against yolov8n's IR; then `yolov8n_hetero_int8` joins the matrix.
