# Node side, one arm. Args: <arm>. Code root /tmp/krisd/sonnet_code (frozen tree + fix-follow patch, installed beforehand).
ARM=$1
NAME=43_MinGPTCausalAttention
FD=/tmp/krisd/sonnet_$ARM; D=$FD/$NAME
L=/tmp/krisd/farm/llm_ledger.jsonl
echo "$(wc -l < $L 2>/dev/null || echo 0) $(date +%s)" > $D/ledger_start
date +%s > $D/t_start
bash $FD/worker4.sh $NAME $FD > $D/worker.out 2>&1
date +%s > $D/t_end
echo "$(wc -l < $L 2>/dev/null || echo 0) $(date +%s)" > $D/ledger_end
