#!/usr/bin/env bash
# Profile yolov8_nano_64 on each accelerator backend so the multi-net
# scheduler can use measured cycles instead of solo extrapolation.
cd "$(dirname "$0")/.."
source scripts/setup_benchmark_env.sh >/dev/null 2>&1 || true
export FIRESIM_QUEUE=1
LOG=/tmp/mb-matrix/yolov8n64_profile.log
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
for cell in yolov8_nano_64_gemmini_int8 yolov8_nano_64_rvv_opu_int8; do
  echo "=== $cell start $(date +%T) ===" | tee -a "$LOG"
  uv run python -m modelblaster.benchmarks.arms.arm_a_curated --workload "$cell" >> "$LOG" 2>&1
  echo "  rc=$? at $(date +%T)" | tee -a "$LOG"
done
echo "=== yolov8n64 profile done $(date +%T) ===" | tee -a "$LOG"
