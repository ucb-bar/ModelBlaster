. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
p=41_GRUBidirectional
echo "=== $p siblings $(date)"
python recompose.py $p data/$p runs/$p.siblings --mode siblings --reps 5 2>&1 | tail -80
echo "=== gpu tests $(date)"
python gpu_tests.py runs/$p.siblings 2>&1 | tail -120
echo "=== $p global $(date)"
python recompose.py $p data/$p runs/$p.global --mode global --gimpl data/gimpl --reps 5 2>&1 | tail -40
echo ALLDONE $(date)
