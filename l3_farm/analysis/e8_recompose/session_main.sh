#!/bin/bash
# E8 main session: one dedicated node, 5 dev problems, K fresh processes x 4 interleaved rounds each (all impls in one
# process, shuffled per round), the flow's own driver (3 requests) after every process; problems interleaved in time.
# usage: session_main.sh JOB TAG K
cd "$(dirname "$0")"
JOB=$1; TAG=$2; K=${3:-4}
PS="1_MLP 11_VGG16 19_MobileNetV1 41_GRUBidirectional 43_MinGPTCausalAttention"
python3 e8_run.py fixed $JOB ${TAG}_before
for k in $(seq 0 $((K-1))); do
  for p in $PS; do
    python3 e8_run.py bench $JOB $TAG 4 $p --k0 $k
    python3 e8_run.py flowtime $JOB $p --tag ${TAG}_$k
  done
done
python3 e8_run.py fixed $JOB ${TAG}_after
