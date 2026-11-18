. /tmp/krisd/cpool/env.sh
cd /tmp/krisd/cpool/e2
nvidia-smi --query-gpu=clocks.sm,clocks.mem,utilization.gpu --format=csv,noheader
E2_REPS=5 E2_CYCLES=1 python e2_timing.py 2>&1 | tail -25
