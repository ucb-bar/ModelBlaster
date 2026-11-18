. /tmp/krisd/ng/env.sh
cd /tmp/krisd/ng
python scripts/repro3.py 2>&1 | grep -v "INFO\|Event loop\|Traceback\|File \|self\.\|raise\|^    "
echo exit $?; nvidia-smi --query-gpu=clocks.sm,pstate --format=csv,noheader
