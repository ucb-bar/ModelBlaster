. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
for p in 41_GRUBidirectional 38_LSTMBidirectional; do
  echo "=== $p siblings $(date)"
  python recompose.py $p data/$p runs/$p.siblings --mode siblings --reps 5 2>&1 | tail -60
done
echo "=== global 38_LSTMBidirectional $(date)"
python recompose.py 38_LSTMBidirectional data/38_LSTMBidirectional runs/38_LSTMBidirectional.global --mode global --gimpl data/gimpl --reps 5 2>&1 | tail -60
echo ALLDONE $(date)
