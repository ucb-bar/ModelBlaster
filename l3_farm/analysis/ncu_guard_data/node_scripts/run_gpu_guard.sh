. /tmp/krisd/ng/env.sh
cd /tmp/krisd/ng; mkdir -p out
python scripts/gpu_guard_test.py /tmp/krisd/ng/out/gpu_guard_${1:-8_ResNetBasicBlock}.json ${1:-8_ResNetBasicBlock}
echo exit $?
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,pstate --format=csv,noheader
