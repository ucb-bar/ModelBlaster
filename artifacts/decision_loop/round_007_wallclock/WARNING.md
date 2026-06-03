# ⚠️  REJECTED RESULT — same bookkeeping bug, different layer

`wallclock_summary.json` reports BEFORE=75.57 ms → AFTER=64.92 ms,
Δ=+10.65 ms (+14.09 %), decision **ACCEPT**.

That number is **not real**. Same root cause as
`artifacts/agentic_fuse_split/WARNING.md` (round 003 fuse from prior
session): adding tile dispatches whose cycles the profile DB doesn't
know about makes the scheduler think the work has vanished.

## What actually happened

`apply_split_to_dispatch_graph` rewrote
`zephyr-chipyard-sw/gen/vmfb/mlp_control/.../mlp_control.fp32_dispatch_graph.json`
to replace `dispatch_2` with two new dispatches: `dispatch_2.tile_0`
and `dispatch_2.tile_1`. The scheduler reads cycles from a separate
profile-DB ingest path (`firesim_rocket_saturn/gemmini_q31/...` results
CSVs). That DB has rows keyed by `dispatch_<id>`. After the rewrite,
`dispatch_2.tile_0` and `dispatch_2.tile_1` have **no profile entry**
— the scheduler defaults them to 0 cycles.

Net effect in the AFTER schedule fixture:
- All `mlp_control_*` dispatches show `duration = 0.0000 ms`.
- yolov8 finishes at 61.58 ms (vs ~75 ms in the BEFORE schedule).
- The 10.65 ms "wall-clock savings" come entirely from the scheduler
  treating mlp_control's work as free, not from real tile parallelism.

## How a real measurement would work

For an honest wall-clock split gate we'd need:

1. The per-tile measured cycles (we already produce these via
   `measure_candidate.sh` — round 5 saw tile_0 = tile_1 = 247,697 cyc
   each, half of the original 495,313).
2. A way to inject those numbers into whatever the scheduler reads as
   "cycles for this op_id". The profile-DB layer the scheduler uses
   (`firesim_rocket_saturn` results CSVs) is one option — append rows
   for `dispatch_2.tile_0/1` with cycles ≈ 247,697 before invoking the
   scheduler.
3. Then run the scheduler on the rewritten IR and compare makespans.
   THAT delta is the real parallelism win — if the scheduler can place
   the two tiles on different cores, both 247k-cycle tiles run in
   parallel and the contribution drops from 495k cyc serial to 247k
   cyc parallel.

This profile-DB injection is the next concrete piece of work. It's
the same shape as `scripts/ingest_measured_cycles.py` but on the
scheduler side instead of the harness side.

## Why we left this as a WARNING rather than fix-in-place

The fix is non-trivial (need to write to the profile DB layer the
scheduler reads from, not the per-network CSV I tried to patch in the
earlier session — that one was on the WRONG ingest path). Better to
document the gap clearly than ship another bogus speedup number.

The framework (`scripts/wallclock_split_eval.py`) is otherwise
correct: it runs the scheduler before, applies the split, runs after,
parses metrics, reports. Once the cycle injection is wired, the same
script gives honest numbers.

## What this round DID prove

- `scripts/wallclock_split_eval.py` correctly drives the
  scheduler-before-and-after-with-IR-substitution flow.
- The IR rewrite at the dispatch_graph.json layer works (7 → 8
  dispatches after split, structurally consistent).
- The scheduler restores cleanly when the script exits (the original
  dispatch_graph.json is backed up and restored, verified with a
  follow-up `decomposed` run that produced the expected 75.57 ms).

What it did NOT prove:
- Real wall-clock improvement from the split. Need profile-DB
  injection (see "How a real measurement would work" above).
