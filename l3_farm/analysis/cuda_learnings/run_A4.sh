. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
echo "=== watchdog $(date)"
python gpu_tests.py runs/41_GRUBidirectional.siblings watchdog 2>&1 | grep -v log_tail | tail -20
echo "=== seedtest $(date)"
SEED_IR=runs/41_GRUBidirectional.siblings/ir bash seedtest.sh 2>&1 | tail -30
echo ALLDONE
