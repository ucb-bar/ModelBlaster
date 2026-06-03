# periodic_anchor end-to-end FireSim measurement — RESULT

## What actually happened

The hetero schedule (4 MLP + 2 Dronet + 1 Yolo on
CPU_P=gemmini_q31, CPU_E=rvv_opu) was submitted to FIRESIM_QUEUE=1
and executed end-to-end on the FPGA.

### Wall-clock measurements (mtime)

| Network | Per-instance walls | Total |
|:---|:---:|:---:|
| mlp_control | 0 × 8 (markers misfired) | 0 |
| dronet | 0 × 4 (markers misfired) | 0 |
| **yolov8_nano** | **3.018 ms** | **3.018 ms** |
| Whole-workload spike marker | 9,077,575,922 cycles | — |

### Output verification (atol=0 rtol=0)

| Network | max_abs_err | Result |
|:---|---:|:---|
| mlp_control | **125** | FAIL |
| dronet | **127** | FAIL |
| yolov8_nano | **45** | FAIL |

### Per-op profile

yolov8_nano per-op cycles captured (`yolov8_profile.csv`, 167M total
cycles in the run's counter, normalized to 3.02 ms wall). mlp_control
and dronet didn't produce a profile.csv table in the log.

## Honest interpretation

1. **The hetero schedule DOES run on FPGA.** Real FireSim execution.
2. **yolov8 ran in 3.02 ms** in the multi-network hetero schedule.
   That's faster than the 6.985 ms single-network baseline — likely
   because in hetero context yolov8 gets to use BOTH cores (per the
   scheduler's placement) while the standalone baseline used only
   rvv_opu.
3. **Output is corrupted.** max_abs_err=125-127 on int8 means
   essentially random output. The hetero multi-network execution has
   a bug: either wrong-kernel dispatch, cross-core data corruption,
   or runtime ordering bug.
4. **mlp_control and dronet per-instance walls = 0** doesn't mean they
   took zero time — the harness's MODELBLASTER_WALL_CYCLES_INST
   markers didn't fire correctly for the periodic instances. The
   instrumentation is the bug, not the measurement.

## What the user-suspected gap revealed

The scheduler-model said periodic_anchor = 75.6 ms / 0 deadline
misses. That was computed from per-op cycles measured in SINGLE-
NETWORK profile runs (each net alone, against PyTorch golden, all
passing). The MULTI-network hetero execution wasn't measured before
this session.

What we now know that we didn't before:
- The hetero data-transfer path between gemmini and rvv_opu cores is
  not producing correct outputs for mlp_control / dronet / yolov8_nano
  in a multi-network context.
- The per-instance instrumentation in the harness has issues
  capturing per-periodic-instance mtime walls.
- yolov8 in isolation runs ~6.985 ms; in hetero schedule it runs ~3.02
  ms (faster, both cores). Wall-clock advantage exists, but it's not
  useful if the outputs are wrong.

## What this means for the policy comparison

**All previously-quoted policy makespans (51-187 ms range) are
scheduler-model output, not measured end-to-end walls.** They presume
the hetero execution produces correct output. This measurement says
it doesn't.

Before any of the policy makespan numbers can be measurement-backed,
we need to fix the hetero execution path so verify passes. The fix
is upstream of the scheduler — likely in:
- `pipeline/ingest_xpurt_schedule.py` (dispatch table generation)
- `pipeline/generate_xpurt_main.py` (runtime orchestration)
- `harness_xpurt/` (cross-core data movement)

Until that's fixed, the scheduler-model numbers are the only thing
we can compare across policies, and the comparison is between
"how the schedule would behave IF the execution were correct."

## Per-FireSim-cycle data preserved

- Whole-run trace: `run.log` (full per-op breakdown captured)
- yolov8 per-op profile: `yolov8_profile.csv`
- mtime walls: in `run.log` (yolov8=3.018ms; others=0 marker misfire)
