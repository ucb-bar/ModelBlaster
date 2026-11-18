. /tmp/krisd/cpool/env.sh
cd /tmp/krisd/cpool/e2
for c in small mid real; do echo "=== case $c"; E2_CASE=$c E2_CYCLES=3 python e2_timing.py 2>&1 | tail -30; done
