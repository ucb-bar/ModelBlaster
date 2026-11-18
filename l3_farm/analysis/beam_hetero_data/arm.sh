# Node side, one arm of the beam/hetero test. Args: <arm tag> <code mode: live|beam>
# Prepares a private code root (copy of the node's installed live tree, or the uploaded beam tree), a fresh farm-style
# dir /tmp/krisd/btest_<arm>/43_MinGPTCausalAttention (problem.py + job.env uploaded beforehand), snapshots the relay
# ledger position, then runs worker4.sh (uploaded to $FD/worker4.sh) in the foreground of this (detached) step.
ARM=$1; MODE=$2
NAME=43_MinGPTCausalAttention
FD=/tmp/krisd/btest_$ARM; D=$FD/$NAME; CR=/tmp/krisd/btcode_$ARM
mkdir -p $D $CR
if [ ! -f $CR/.ready ]; then
  rm -rf $CR/ModelBlaster
  if [ "$MODE" = beam ]; then tar xzf $FD/beam_mb.tar.gz -C $CR && find $CR/ModelBlaster -name '._*' -delete
  else cp -a /tmp/krisd/ModelBlaster $CR/; fi
  ln -sfn $CR/ModelBlaster $CR/modelblaster
  touch $CR/.ready
fi
L=/tmp/krisd/farm/llm_ledger.jsonl
echo "$(wc -l < $L 2>/dev/null || echo 0) $(date +%s)" > $D/ledger_start
date +%s > $D/t_start
bash $FD/worker4.sh $NAME $FD > $D/worker.out 2>&1
date +%s > $D/t_end
echo "$(wc -l < $L 2>/dev/null || echo 0) $(date +%s)" > $D/ledger_end
