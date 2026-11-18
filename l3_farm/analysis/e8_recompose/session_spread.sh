#!/bin/bash
# E8 node-to-node session (one of the other nodes): setup, rebuild the 5 dev problems, then fixed kernels, 2 processes x
# 3 rounds of the interleaved bench per problem, the flow's own driver once per problem, fixed kernels again.
# usage: session_spread.sh JOB TAG
cd "$(dirname "$0")"
JOB=$1; TAG=$2
PS="1_MLP 11_VGG16 19_MobileNetV1 41_GRUBidirectional 43_MinGPTCausalAttention"
python3 nodes.py tools $JOB
python3 e8_run.py push $JOB $PS
python3 e8_run.py rebuild $JOB $PS
python3 e8_run.py smi $JOB > /dev/null
python3 e8_run.py fixed $JOB ${TAG}_before
for k in 0 1; do
  for p in $PS; do
    python3 e8_run.py bench $JOB $TAG 3 $p --k0 $k
  done
done
for p in $PS; do python3 e8_run.py flowtime $JOB $p --tag $TAG; done
python3 e8_run.py fixed $JOB ${TAG}_after
