#!/usr/bin/env bash
# Generate the 5 3-way schedule fixtures for the sensor-Hz sweep.
# Each config tags a different (dronet Hz, mlp_control Hz) combination
# anchored to realistic robotics sensor rates. See:
#   notes/baseline_2026-05-28.md  for current measurements
#   notes/known_issues.md         for cross-backend drift policy
#
# After generation, capture each via examples/xpurt_demo/run.sh with
# SCHEDULE_JSON= and SCHED_NAME= set; see scripts/run_3way_sweep.sh.
set -euo pipefail

cd "$(dirname "$0")/.."
GEN="uv run python -m scripts.gen_3way_schedule"
OUT_DIR="schedule_fixtures"

# Baseline — keeps the existing rates (low-rate dronet, 200 Hz mlp).
$GEN --output "$OUT_DIR/3way_baseline.json" \
     --dronet-instances 1 --dronet-period-ms 0 \
     --mlp-instances 4 --mlp-period-ms 5

# Conservative — light multi-instance load. dronet 5 Hz (200 ms),
# mlp 20 Hz (50 ms), fits comfortably inside one yolov8n window.
$GEN --output "$OUT_DIR/3way_conservative.json" \
     --dronet-instances 2 --dronet-period-ms 200 \
     --mlp-instances 9 --mlp-period-ms 50

# Camera-30Hz — dronet on a 30 Hz camera (14 instances per ~470 ms window),
# mlp_control at 100 Hz IMU (45 instances).
$GEN --output "$OUT_DIR/3way_camera-30hz.json" \
     --dronet-instances 14 --dronet-period-ms 33.33 \
     --mlp-instances 45 --mlp-period-ms 10

# Camera-60Hz — high-end camera feeding dronet at 60 Hz,
# mlp at 200 Hz IMU.
$GEN --output "$OUT_DIR/3way_camera-60hz.json" \
     --dronet-instances 28 --dronet-period-ms 16.67 \
     --mlp-instances 90 --mlp-period-ms 5

# IMU-only-hi — keeps dronet at 1× but mlp_control at 200 Hz x 90.
# Tests the OPU control-loop throughput in isolation.
$GEN --output "$OUT_DIR/3way_imu-only-hi.json" \
     --dronet-instances 1 --dronet-period-ms 0 \
     --mlp-instances 90 --mlp-period-ms 5

echo "----"
echo "Generated:"
ls -la "$OUT_DIR"/3way_{baseline,conservative,camera-30hz,camera-60hz,imu-only-hi}.json
