. /tmp/krisd/pa/env.sh
cd /tmp/krisd/pa; mkdir -p out
python scripts/gpu_audit_test.py /tmp/krisd/pa/irs/8_ResNetBasicBlock /tmp/krisd/pa/out/gpu_8_ResNetBasicBlock.json /tmp/krisd/pa/gpuwork_8 4000000; echo "exit8 $?"
python scripts/gpu_audit_test.py /tmp/krisd/pa/irs/19_MobileNetV1 /tmp/krisd/pa/out/gpu_19_MobileNetV1.json /tmp/krisd/pa/gpuwork_19 1000000; echo "exit19 $?"
