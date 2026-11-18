. /tmp/krisd/pa/env.sh
cd /tmp/krisd/pa; mkdir -p out
python scripts/gpu_audit_test.py /tmp/krisd/pa/irs/${IRN:-8_ResNetBasicBlock} /tmp/krisd/pa/out/gpu_${IRN:-8_ResNetBasicBlock}.json /tmp/krisd/pa/gpuwork_${IRN:-8_ResNetBasicBlock} ${DELAY:-4000000}
echo "exit $?"
