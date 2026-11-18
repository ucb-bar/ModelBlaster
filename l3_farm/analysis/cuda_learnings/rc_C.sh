. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader; echo CVD=$CUDA_VISIBLE_DEVICES
for p in 6_GoogleNetInceptionModule 8_ResNetBasicBlock 17_SqueezeNetFireModule 25_ShuffleNetUnit 7_GoogleNetInceptionV1 9_ResNet18 18_SqueezeNet 44_MiniGPTBlock 28_VisionTransformer 1_MLP; do
  echo "=== $p $(date)"
  timeout 2400 python conc_check.py $p cdata/$p cruns/$p --reps 5 2>&1 | tail -3
done
echo ALLDONE $(date)
