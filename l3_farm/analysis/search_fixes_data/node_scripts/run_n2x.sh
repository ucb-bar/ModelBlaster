. /tmp/krisd/a3/env.sh
export MB_PRUNE_DEAD=0
cd /tmp/krisd/a3; rm -rf data && tar xzf data.tar.gz 2>/dev/null && find data -name '._*' -delete; mkdir -p runs3
for P in 15_DenseNet121 17_SqueezeNetFireModule 6_GoogleNetInceptionModule 14_DenseNet121DenseBlock 50_ReLUSelfAttention 8_ResNetBasicBlock 31_VisionAttention 49_Mamba2ReturnFinalState 41_GRUBidirectional; do
  echo "=== exact $P $(date)"
  python a3_node.py exact $P data/$P runs3/$P --route --reps 5 2>&1 | tail -40
done
echo ALLDONE $(date)
