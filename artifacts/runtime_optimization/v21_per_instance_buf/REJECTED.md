# v21: per-instance output buffer (#247 fix attempt) — REVERTED

## What was tried
Modified `pipeline/generate_xpurt_main.py` to:
1. Declare `out_<mid>_inst[n_inst][OUTPUT_SIZE]` instead of `out_<mid>[OUTPUT_SIZE]`
2. Route `s.output` per-dispatch via `e_->instance` indexing in bs_select
3. Dump from `out_<mid>_inst[n_inst - 1][i]` (last instance's slot)

## What happened
- Sim PASSED (workload completed cleanly)
- HART_ACC numbers consistent with v20b (gemmini 176 ms, rvv_opu 183 ms)
- But OUTPUT dump regressed:
  - `MODELBLASTER_OUTPUT_BEGIN [mlp_control]` block missing entirely from uartlog
  - `MODELBLASTER_OUTPUT_BEGIN [dronet]` emitted only 1 of 2 expected values
  - yolov8 dump appeared normal
- Runner failed: `could not find MODELBLASTER_OUTPUT_{BEGIN,END} [mlp_control] block`

## Root cause (hypothesis)
The per-instance routing introduces a dependency: each kernel writes to
its instance's slot, but if the dispatch dependency graph doesn't
guarantee the last instance writes to slot `n_inst - 1` (the slot the
dump reads), some output bytes are stale/uninitialized.

## Decision
Reverted `pipeline/generate_xpurt_main.py`. v20b remains the production
baseline. #247 dronet correctness divergence still needs investigation
but is independent of the optimization workstream — it has been a known
issue since v8 baseline.

## Final cumulative state
v10 → v20b: **571 → 183 ms (-68%)**.
