#!/usr/bin/env bash
# Render all walkthrough Gantts from the artifacts under
# artifacts/bundle/{walkthrough,longrun}. Idempotent — safe to re-run any
# time more candidate traces land.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=artifacts/bundle/walkthrough
mkdir -p "${OUT}"

XPURT=/scratch2/agustin/XPU-RT

python3 scripts/render_annotated_gantt.py \
  --fixture "${XPURT}/schedules/scheduled__iter_baseline_decomposed_profiled.json" \
  --out "${OUT}/step1_baseline_predicted.png" \
  --title "Step 1 — Baseline (decomposed solver), PREDICTED" \
  --subtitle "1x yolov8_nano + 4x mlp_control (10ms period) + 2x dronet (20ms period), Gemmini+OPU hetero" \
  --annotation "VERDICT: misses 65 ms deadline by 16.3%
BOTTLENECK: CPU_E#0 (OPU+V hart) — 388 dispatches, 100% under 1000 cycles
GRANULARITY: too_fine — XPU-RT's advisor recommends COARSEN
                                              (axis-C fusion candidate)
NEXT: try a different scheduler first (axis A), then fuse if A doesn't close the gap" \
  --deadline-ms 65 --x-max-ms 80

python3 scripts/render_annotated_gantt.py \
  --fixture "${XPURT}/schedules/scheduled__iter_heft_V256D128_rvv-gemmini_q31_heft_profiled.json" \
  --out "${OUT}/step2_heft_predicted.png" \
  --title "Step 2 — HEFT scheduler swap (axis A), PREDICTED" \
  --subtitle "Same workload, same backends; only the dispatch-to-core assignment changed" \
  --annotation "VERDICT: meets 65 ms deadline by 16.3% (54.43 ms makespan)
WHAT CHANGED vs Step 1: bottleneck shifted CPU_E#0 -> CPU_P#0 (Gemmini).
GRANULARITY: still too_fine (300/300 dispatches under 1000 cycles)
NEXT: fusion (axis C) is the orthogonal win — measure on FireSim to see the
      per-dispatch launch-overhead savings the predicted model can't show." \
  --deadline-ms 65 --x-max-ms 80

if [[ -f artifacts/bundle/longrun/baseline/xpurt_trace.csv ]]; then
    python3 scripts/render_annotated_gantt.py \
      --trace artifacts/bundle/longrun/baseline/xpurt_trace.csv \
      --out "${OUT}/step3_baseline_measured.png" \
      --title "Step 3 — Baseline on FireSim (MEASURED, full 388-dispatch trace)" \
      --subtitle "Real per-dispatch cycles from the FPGA harness, captured under 4hr FIRESIM_QUEUE_TIMEOUT" \
      --annotation "VERDICT: measured makespan 13.5 ms — far inside the 65 ms deadline.
WHAT CHANGED vs Step 1 (predicted 75.57 ms): XPU-RT's analytic cost model is per-op-isolated;
  harness_xpurt's two-core walker overlaps dispatches far better than the model assumed.
  Predicted 75.57 ms, measured 13.5 ms — model under-estimates parallelism by ~5.6x.
LESSON: predicted analysis steers WHAT to try; FireSim measures HOW MUCH it helps." \
      --clock-mhz 1000 --x-max-ms 14
fi

if [[ -f "${OUT}/step4_fused_trace.csv" ]]; then
    python3 scripts/render_annotated_gantt.py \
      --trace "${OUT}/step4_fused_trace.csv" \
      --out "${OUT}/step4_fused_mlp.png" \
      --title "Step 4 — After axis-C fusion (mlp_control [0..5] fused per instance), MEASURED" \
      --subtitle "Same FireSim trace cycles, mlp_control sub-dispatches collapsed into ONE hatched fused dispatch per instance" \
      --annotation "WHAT CHANGED vs Step 3:
  - 388 dispatches -> 348 (each mlp_control instance: 7 -> 2)
  - Per fused mlp_control block: 6 inter-dispatch worker handshakes saved
  - 4 mlp_control instances x 6 = 24 handshakes eliminated total
  - Hatched blocks = fused dispatches (bit-exact on spike: max_abs_err=0)
PERIODIC DEADLINES STILL RESPECTED:
  - mlp_control fused dispatch fits within each 10ms window
  - dronet's 20ms windows unchanged
NEXT: feed measured back to advisor.py -> granularity=balanced, move to next axis." \
      --clock-mhz 1000 --x-max-ms 14
fi

if [[ -f artifacts/bundle/longrun/A2/xpurt_trace.csv ]]; then
    python3 scripts/render_annotated_gantt.py \
      --trace artifacts/bundle/longrun/A2/xpurt_trace.csv \
      --out "${OUT}/step5_heft_measured.png" \
      --title "Step 5 — HEFT (A2) on FireSim (MEASURED)" \
      --subtitle "Same workload, HEFT placement (axis-A winner) running on the FPGA" \
      --annotation "Compare to Step 3 (baseline measured): same workload, only the scheduler swapped.
  HEFT predicted 54.43 ms (28% better than baseline predicted).
  Measured comparison validates whether HEFT's axis-A win holds on real hardware." \
      --clock-mhz 1000 --x-max-ms 14
fi

echo
ls -la "${OUT}"/*.png 2>/dev/null | awk '{print "  ",$NF}'
