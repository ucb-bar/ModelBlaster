#!/usr/bin/env bash
# Phase 1 gap-fill — mlp_control int8 across all three accelerator targets,
# 3 reps each. Closes the 3 missing matrix entries.
#
# Goes through firesim-queue automatically (arm_a_curated picks RUNNER from
# the workload row). Each rep is ~5-15 min on FireSim — far faster than
# the yolov8n cells thanks to the tiny model size.
#
# Usage:
#   bash scripts/run_mlp_control_int8_gapfill.sh

cd "$(dirname "$0")/.."
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
export FIRESIM_QUEUE=1

LOG=/tmp/mb-matrix/mlp_control_int8_gapfill.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

CELLS=(mlp_control_rvv_opu_int8 mlp_control_gemmini_int8 mlp_control_gemmini_q31_int8)

for cell in "${CELLS[@]}"; do
  for rep in 1 2 3; do
    echo "=== $cell rep $rep start $(date +%T) ===" | tee -a "$LOG"
    uv run python -m modelblaster.benchmarks.arms.arm_a_curated --workload "$cell" >> "$LOG" 2>&1
    rc=$?
    echo "  rc=$rc at $(date +%T)" | tee -a "$LOG"
  done
done
echo "=== mlp_control int8 gap-fill done $(date +%T) ===" | tee -a "$LOG"
