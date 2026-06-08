# G3 — PDB recalibration against v20b measured cycles

## What ran

`scripts/recalibrate_pdb_from_runlog.py` parsed v20b's run.log
per-network `MODELBLASTER_PROFILE_BEGIN..PROFILE_END` blocks and
wrote 235 rows back into the sweep_v8 profile DB:

| Net / backend                | rows | sum_cyc before | sum_cyc after | ratio |
|:-----------------------------|-----:|---------------:|--------------:|------:|
| mlp_control / gemmini_q31    |    1 |          0.05M |         0.05M | 0.975 |
| mlp_control / rvv_opu        |    – | (no PDB at sweep_v8 path; skipped) |  |  |
| dronet / gemmini_q31         |    7 |          9.04M |         7.09M | 0.784 |
| dronet / rvv_opu             |   23 |         11.84M |        13.57M | 1.146 |
| yolov8_nano / gemmini_q31    |   68 |         71.25M |        60.06M | 0.843 |
| yolov8_nano / rvv_opu        |  136 |        389.84M |        74.47M | 0.191 |

Per-row backups under `<results>.csv.prev20b_backup`. Net cycle delta
across all updated rows: **−326.79M cycles**, almost entirely from
the yolov8 rvv_opu hart (silu LUT v10 + cat per-input LUTs v18/v19 +
im2col_rvv_reduce conv v17 cumulative wins).

## Solver replay

Re-ran `hybrid_periodic_mosek_yolo` on the calibrated PDB
(`networks_1yolo_4mlp_2dronet_firesim.json`, time_limit 120s).

| Metric                         | Value         |
|:-------------------------------|--------------:|
| status                         | ok            |
| solve_wall_s                   | 193.06 s      |
| n_dispatches                   | 826           |
| n_deadline_miss                | 0             |
| n_release_viol                 | 0             |
| **predicted makespan**         | **393.08 ms** |
| measured makespan (v20b)       | 183.45 ms     |
| **predicted / measured**       | **2.15×**     |

`scripts/recalibrate_pdb_from_runlog.py` artefact, replay invocation:

```
PYTHONPATH=/scratch2/agustin/XPU-RT:/scratch2/agustin/XPU-RT/xpu-rt \
  /scratch2/agustin/miniforge3/envs/merlin-dev/bin/python -c \
  "from policies.hybrid_periodic_mosek_yolo import hybrid_periodic_mosek_yolo; \
   hybrid_periodic_mosek_yolo('/scratch2/agustin/XPU-RT/data/toplevel/networks_1yolo_4mlp_2dronet_firesim.json', time_limit=120.0)"
```

Output fixture: `/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json`.

## Why the ratio missed 1.10×

Per-machine post-solver sums vs measured kernel time:

| Hart        | predicted Σdur | measured kernel | ratio |
|:------------|---------------:|----------------:|------:|
| CPU_P (gemmini) |      94.98 ms |        88.71 ms |  1.07× |
| CPU_E (rvv_opu) |     221.42 ms |       133.00 ms |  1.66× |

Gemmini kernel sum is now well-calibrated (1.07×). The rvv_opu side
still overestimates by **66%**, with two contributing causes:

1. **Missing mlp_control rvv_opu entry in sweep_v8.** The
   recalibration warned: PDB row absent for
   `V256D128_rvv/.../mlp_control.fp32/...`. The solver's
   profile-loader falls through to a fallback estimate that is
   substantially larger than the measured 0.5 M cycles per pass on
   mlp_control rvv_opu. Across 4 mlp instances this contributes most
   of the 88 ms gap.
2. **Phased schedule serialisation.** The hybrid policy reserves
   periodic-instance bands first (Phase 1) and then packs yolov8
   into residual time (Phase 2). The walker actually executes mlp
   silu+linear *concurrently* with yolov8 gemmini work on the other
   hart — but the per-machine sum the policy reports treats them as
   sequential on CPU_E.

Both effects are *solver/PDB* shortcomings, not runtime ones — the
runtime is already extracting more parallelism than the solver
predicts. Closing them is documented as G3-followup; the highest-ROI
fix is to add an mlp_control V256D128_rvv PDB row (run mlp_control on
rvv-only once via `examples/mlp_control/run.sh` with TARGET=rvv_opu
to generate the missing CSV).

## What this unlocks

The solver now has the correct yolov8 rvv_opu cost (74M vs 390M) and
gemmini conv2d_s8 cost (60M vs 71M). Future granularity / fuse
proposals will be evaluated against the post-optimisation cycle
budget instead of the v9 baseline, so analytical hints (e.g. shard
yolov8.l6 from gemmini to rvv_opu) will land on realistic deltas.

## Files changed

- `/scratch2/agustin/XPU-RT/zephyr-chipyard-sw/gen/profile/sweep_v8/{gemmini_q31,V256D128_rvv}/firesim_rocket_saturn/{dronet,yolov8_nano,mlp_control}/.../results.csv`
  — 235 rows updated, original preserved at `<results>.csv.prev20b_backup`.
- `/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json`
  — new fixture from the calibrated replay (was the v9-cycles fixture
    before this run; overwritten).
- `scripts/recalibrate_pdb_from_runlog.py` — new (this repo).
