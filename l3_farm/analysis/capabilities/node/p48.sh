. /tmp/krisd/a2/env.sh
cd /tmp/krisd/a2
for p in 48_Mamba2ReturnY 49_Mamba2ReturnFinalState; do
  python /tmp/krisd/a2/node/p48.py /tmp/krisd/a2/problems/$p.py /tmp/krisd/a2/runs/prec_$p > /tmp/krisd/a2/logs/p48_$p.log 2>&1
  grep "@@P48@@" /tmp/krisd/a2/logs/p48_$p.log | cut -c1-3000
done
