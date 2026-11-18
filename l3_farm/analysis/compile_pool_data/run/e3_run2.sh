. /tmp/krisd/cpool/env.sh
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
export PATH=/tmp/krisd/cpool/sshbin:$PATH
cd /tmp/krisd/cpool
H=a4u8g-mil-0024,a1u1g-mil-0572
for a in A B; do for r in 2 3; do
  echo "=== ${a}r$r tiny $(date +%T)"; python /tmp/krisd/cpool/e3_pooled.py $a tiny /tmp/krisd/cpool/e3/${a}${r}_tiny $H 2>&1 | grep '^{"arm'
done; done
python /tmp/krisd/cpool/e3_report.py
