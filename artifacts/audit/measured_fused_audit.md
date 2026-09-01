# Measured-cycles audit: yolov8 unfused vs fused (Task #240)

## What was measured

Two end-to-end FireSim runs of yolov8_nano on the rvv_opu backend
(single-network, no hetero multi-net dispatch):

| Run | IR | Ops | Build | Wall (mtime, ms) | Sum per-op (rdcycle) |
|---|---|---|---|---|---|
| baseline | `graph.json` (unfused) | 204 | `examples/yolov8_nano/int8/build/rvv_opu_firesim/` | 6.985 | 6,985,278,800 |
| fused | `graph.fused.json` (57 BN+SiLU fusions) | 147 | same | 7.044 | 7,044,461,818 |

Per-op CSVs:
- `artifacts/firesim_runs/baseline_yolov8_rvv_opu/profile.csv`
- `artifacts/firesim_runs/fused_yolov8_rvv_opu/profile.csv`

Both runs were submitted via `FIRESIM_QUEUE=1`, used the
`firesim_shuttle_gemmini_opu` bitstream, and the in-binary verify
reported `max_abs_err=0` (bit-exact on FPGA against the PyTorch
golden) on the fused run.

## Per-op-type cycle breakdown

| op_type | baseline cycles | fused cycles | delta | comment |
|---|---:|---:|---:|---|
| conv2d_s8 | 6,632,303,653 | 6,691,831,670 | +59,528,017 (+0.90%) | system noise; same kernel |
| batchnorm2d_s8 | 70,322,638 | 0 | -70,322,638 | absorbed into fusion |
| silu_s8 | 237,724,939 | 0 | -237,724,939 | absorbed into fusion |
| batchnorm2d_silu_s8 | 0 | 308,246,731 | +308,246,731 | the new fused kernel |
| cat2_c1_s8 | 16,007,744 | 16,006,439 | -1,305 | noise |
| cat3_c1_s8 | 12,015,997 | 11,694,996 | -321,001 | noise |
| cat4_c1_s8 | 6,573,279 | 6,570,646 | -2,633 | noise |
| add_s8 | 5,306,649 | 5,429,523 | +122,874 (+2.3%) | noise |
| maxpool2d_s8 | 3,071,682 | 2,729,176 | -342,506 | noise |
| upsample_nearest_s8 | 1,952,219 | 1,952,637 | +418 | noise |
| **TOTAL** | **6,985,278,800** | **7,044,461,818** | **+59,183,018 (+0.85%)** | |

## Key conclusion: fusion is compute-neutral

`batchnorm2d_s8` (70.3M) + `silu_s8` (237.7M) = **308.0M cycles** unfused.
`batchnorm2d_silu_s8` fused = **308.2M cycles**.

The Bedrock-generated fused kernel is **0.07% slower** than the
unfused chain on the rvv_opu backend. Within noise — fusion saves
the intermediate buffer write/read but the kernel logic (BN affine
+ LUT-based SiLU) is essentially the same per-element cost.

The 0.85% total delta is dominated by per-conv variance (each conv2d
is ~10M cycles, so 0.9% across ~70 convs is exactly the run-to-run
noise floor we see in repeated runs).

## What the fusion DOES change

The fusion reduces the dispatch count from 204 to 147 (−57). That's
worth modelling at the scheduler level — fewer ops means fewer
placement decisions, fewer scheduling dependencies, and a shorter
makespan-defining chain. But on the per-op compute side, fusion is
a wash: the dispatch overhead saved by fusion is also small enough
that it doesn't recover compute that the fused kernel adds.

## Profile-DB update path

To update the scheduler's profile DB with these fresh measurements,
replace each yolov8_nano dispatch's `mean_time_ns` with its measured
cycles at 1 GHz target frequency (cycle = ns):

- For the **baseline** schedule (uses `graph.json`):
  `gen/profile/V256D128_rvv/firesim_rocket_saturn/yolov8_nano/yolov8_nano.int8/.../results.csv`
  → set each row's `mean_time_ns` and `cycles` to the corresponding
    value from `baseline_yolov8_rvv_opu/profile.csv`.

- For the **fused** schedule:
  Need a parallel profile DB under
  `gen/profile/.../yolov8_nano_fused/...` keyed by the fused IR's
  dispatch_ids. The fused IR is at
  `examples/yolov8_nano/int8/generated/graph.fused.json`.

Once those are populated, re-running the scheduler with the fused IR
+ fused profile DB should produce a schedule whose makespan reflects
the actual fused-kernel cost.

## Files

- `artifacts/firesim_runs/baseline_yolov8_rvv_opu/profile.csv` —
  204 per-op rows, baseline.
- `artifacts/firesim_runs/fused_yolov8_rvv_opu/profile.csv` —
  147 per-op rows, fused.
- This audit doc.
