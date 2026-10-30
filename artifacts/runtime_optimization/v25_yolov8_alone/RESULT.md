# v25 yolov8-alone FireSim result (2026-06-08)

## Wall + cycle totals

| Metric | v25 | Jun 4 yolov8_bit_exact baseline | Δ |
|---|---:|---:|---:|
| wall_clock_cycles (mtime) | **259,062** | 446,738 | **−42%** |
| TOTAL kernel cycles (sum rdcycle) | 279,525,434 | 462,937,319 | −40% |
| FireSim total cycles | 8,602,446,617 | 16,292,725,157 | −47% |

## Per-hart split

| Hart | Ops | Cycles | ~ms@1GHz |
|---|---:|---:|---:|
| rvv_opu (CPU_E) | 103 | 153,410,572 | 153 |
| gemmini_q31 (CPU_P) | 101 | 126,114,862 | 126 |
| **sum** | 204 | **279,525,434** | **280** |

Wall = 259 ms vs sum = 280 ms → harts ran in parallel ~50% of the time.

## Output divergence (caveat)

v25 yolo-alone OUTPUT head: `[25, -1, -4, 4, 1, -4, 1, 2]`
v20b multinet yolo OUTPUT head: `[45, 17, 13, 8, 3, 2, 1, 2]`

These differ. Possible causes:
- restored vectorized kernels (add/maxpool/upsample) — spike verified
  max_abs_err=0 on extra_shapes test set but may diverge at production
  shapes
- schedule re-solved against G3-calibrated PDB before this run
- both — orthogonal sources of drift

Need the multinet v25 run (job 268, in flight) which goes through
firesim_runner.py and emits explicit `max_abs_err` for a definitive
correctness verdict.

## Kernel picks

rvv_opu: add_s8=direct (restored), maxpool2d_s8=direct (restored),
upsample_nearest_s8=direct (restored), batchnorm2d_s8=per_channel_lut
(unchanged), cat2/3/4=per_input_lut (unchanged), conv2d_s8=
im2col_rvv_reduce (unchanged), silu_s8=rvv_lut_gather (unchanged).
