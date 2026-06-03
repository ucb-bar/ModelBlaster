# Decision loop on yolov8_nano (the heaviest workload)

After dronet's loop run confirmed the framework works on real
conv-heavy networks, yolov8_nano is the natural scale-up: 212 ops,
63 conv2d_s8, **10,437,959,441 cycles = 104 ms total** for one
inference. That's 23× dronet, 100,000× mlp_control.

## Baseline

```bash
BACKEND=reference RUNNER=spike TARGET=rvv_opu QUANT=int8 \
    uv run bash examples/yolov8_nano/run.sh
```

→ 204 dispatches, 10.44B cycles, **verify PASS (max_abs_err=0
on 75,600-element output)**.

Top 10 heaviest convs:

| did | Op | Shape | Cycles | % of net |
|---:|:---|:---|---:|---:|
| 177 | detect.cv3_0_1.conv | OC=80 IC=80 3×3 @ 20×20 | 786,701,318 | 7.5% |
| 174 | detect.cv3_0_0.conv | OC=80 IC=64 3×3 @ 20×20 | 629,614,980 | 6.0% |
| 170 | detect.cv2_0_1.conv | OC=64 IC=64 3×3 @ 20×20 | 503,692,177 | 4.8% |
| 167 | detect.cv2_0_0.conv | OC=64 IC=64 3×3 @ 20×20 | 503,692,177 | 4.8% |
| 189 | detect.cv3_1_0.conv | OC=80 IC=128 3×3 @ 10×10 | 308,336,564 | 3.0% |
| ... | (10 more in the >200M cycle range) | | | |

The top-10 heaviest convs alone account for ~38% of total compute.
These are where the agent's split decisions matter most.

## Granularity advisor on the full Dima workload

The advisor returned 0 yolov8 candidates in its top-K. Why:

- yolov8 is non-periodic (one-shot), so it doesn't trigger the
  "too_fine" verdict that surfaces mlp_control fuses.
- The advisor's *predicted* model sees yolov8 as a serial chain
  dominating makespan; splitting one op doesn't move the
  prediction much. Predicted Δmakespan for splits ~ +3 µs (worse).

So we exercised the loop with **manual hints** targeting 3 different
heavy convs to test the build path at yolov8 scale.

## Manual split tests (all bit-exact PASS)

| did | Op | OC split | Verify | Δcycles vs baseline |
|---:|:---|:---:|:---:|---:|
| 177 | detect.cv3_0_1 | 80 → 40+40 | ✅ PASS (max_abs_err=0) | +160 |
| 46  | l5.conv        | 128 → 64+64 | ✅ PASS (max_abs_err=0) | +159 |
| 3   | l1.conv        | 32 → 16+16 | ✅ PASS (max_abs_err=0) | +160 |

Each baseline = 10,437,959,441 cycles. Per-op cycle Δ ≈ +160 (0.0000015%),
within mtime noise.

## What this confirms

The yolov8 build path works end-to-end through the agentic loop:
- `apply_split_hint --pairwise=N` rewrites the IR (212 → 213 ops).
- `generate_skeleton` now correctly allocates per-tile output buffer
  offsets (the round-002 fix) AND emits per-tile weight/bias
  pointer offsets (the round-004 fix).
- Spike build succeeds for all three split shapes (different OC
  values), profile table is complete, verify is bit-exact.
- `measure_candidate.sh` (after a pipefail fix for the longer
  ~30s yolov8 run) ingests cycles cleanly.

**Same finding as dronet:** per-op cycle gate correctly says
"no significant change," because splitting conserves total work on
a sequential simulator. The real value of yolov8 splits is at
multi-core wall-clock, which still requires either:

1. **FireSim hetero measurement** — tile_0 on Gemmini, tile_1 on
   Saturn-OPU+RVV, the 786M-cycle detect.cv3_0_1 conv halves to
   two ~393M-cycle pieces running in parallel → ~half wall-clock
   contribution. **Blocked on bitstream stability.**
2. **Wall-clock scheduler gate** with the duration-ingestion fix
   that's still outstanding (round 8 work).

## Why this run is the right "agentic loop" demonstration

The decision loop's job is to gate IR changes on measurement. On
yolov8 with the current measurement primitive:
- It correctly **does NOT accept** any of the 3 splits, because
  the per-op cycle measurement shows no improvement.
- It correctly **does NOT silently fail** — every candidate
  produces a bit-exact verified result before the cycles even matter.

The agent is doing exactly what an honest decision driver should:
refusing to commit changes that the measurement framework can't
prove helpful. The next unlock is teaching it the right metric for
splits (wall-clock multi-core, not per-op cycles).

## Files in this directory

- `baseline/` — empty hint, full yolov8 baseline measurement
  (204 dispatches, profile table, verify PASS).
- `manual_splits/hint_did<N>.json` — Contract-2 split hints for the
  three heaviest convs we probed.
- `manual_splits/did<N>_v2/` — per-candidate run dirs:
  - `spike.log` — full spike output with verify line + profile table
  - `measured_cycles.json` — ingested per-dispatch cycles
  - `yolov8_nano.{before,after}hint.graph.json` — IRs
  - `PASS` — verify gate passed
- `round_001/` — the auto-loop run that found 0 yolov8 candidates
  via granularity (granularity_result.json shows the top-K it did
  consider).

---

## Wall-clock test (honest answer after loader fix)

Ran `scripts/wallclock_split_eval.py` on the dispatch_177 split with
two iterations of the profile_loader fix:

**First attempt (round 7-style bug):** AFTER=75.32 ms (Δ -0.25 ms,
looked like accept). Inspection showed both tiles had dur=0.0 in the
fixture — same bookkeeping fiction as the prior session. The 0.25 ms
"improvement" was scheduling-rearrangement noise around zero-cost
tiles, not real parallelism.

**Loader patch v1 (committed earlier):** parent cycles / n_splits
fallback. Still didn't fire because the split rewrite REPLACED the
parent dispatch in the graph, so `dispatches[parent_name]` returned
None.

**Loader patch v2 (this round):** parse `dispatch_<int>` directly
from the parent name string. Now tile cycles correctly resolve to
parent_cycles / n_splits.

**After v2:**

| | Makespan | Tile placement |
|:---|---:|:---|
| BEFORE | 75.57 ms | dispatch_177 on CPU_P#0, dur=0.6546 ms |
| AFTER  | 75.57 ms | tile_0 on CPU_P#0 dur=0.3273, tile_1 on CPU_P#0 dur=0.3273 (sequential, NOT parallel) |
| Δ      | 0.00 ms  | — |

**The real result:** with proper cycle accounting, the decomposed
scheduler placed BOTH tiles on CPU_P#0 sequentially. tile_0 starts
at 71.5062, tile_1 starts at 71.8335 (= 71.5062 + 0.3273). Total
contribution = 0.6546 ms = same as the unsplit dispatch.

### Why both tiles stayed on the same core

yolov8's network entry in the workload JSON has `preferred_hw =
gemmini` (the bigger of the two accelerator classes for this
network). `profile_loader.py` applies a 1000× penalty multiplier
to non-preferred-hw machine combos. So even though tile_1 *could*
run on CPU_E#0 (OPU+RVV), the scheduler sees CPU_E#0 as 1000× more
expensive than CPU_P#0 and never picks it.

The scheduler is doing what its configuration says. The
configuration is locking out the parallelism win that splitting
should provide.

### The actual question this exposes

> *When you split a heavy op into 2 tiles, do you want both tiles
> to stay on the preferred hardware class (cache locality, no
> cross-cluster cost), or do you want one tile to migrate to a
> different class (parallelism win, but possibly slower per-tile
> compute)?*

Today the workload's `preferred_hw` answer is "stay on gemmini" —
and the scheduler obeys. For the agent's split decisions to ever
ACCEPT, we'd need one of:

1. **Relax preferred_hw for split tiles.** Mark tile_1 as
   "preferred_hw = OPU" or "no preference" so the scheduler
   considers both core classes. Requires a small extension in
   apply_split_hint.py to tag tiles with alternate preferred_hw.
2. **Per-tile profile data on the non-preferred hw.** The
   profile DB only has yolov8 cycles for gemmini_q31. If we had
   measured yolov8 ops on OPU/RVV too, the scheduler could compare
   honestly. Otherwise the 1000× penalty stays inviolate.
3. **Use a workload without preferred_hw pinning.** Trade-off:
   without pinning, the scheduler considers all cross-class
   placements, but cross-cluster overhead (e.g. CPU_P→CPU_E
   cache cold) might dominate the parallelism win.

This is the actual next layer of work. The IR rewrite, the
skeleton emitter, the measurement primitive, and the profile-DB
ingestion are all working correctly. The bottleneck has moved up
the stack to the scheduler's HW-pinning configuration.
