. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
for p in 42_GRUBidirectionalHidden 39_GRU; do
  echo "=== $p siblings $(date)"
  python recompose.py $p data/$p runs/$p.siblings --mode siblings --reps 5 2>&1 | tail -60
done
echo "=== global 39_GRU $(date)"
python recompose.py 39_GRU data/39_GRU runs/39_GRU.global --mode global --gimpl data/gimpl --reps 5 2>&1 | tail -60
echo ALLDONE $(date)
