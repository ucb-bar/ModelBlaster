. /tmp/krisd/cpool/env.sh
until grep -q ALLDONE2 /tmp/krisd/cpool/logs/load_remote2.log; do sleep 5; done
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
export PATH=/tmp/krisd/cpool/sshbin:$PATH
export LT_HOSTS=a1u1g-mil-0578,a1u1g-mil-0572,a4u8g-mil-0024
export LT_RB_JOBS=6
cd /tmp/krisd/cpool
for N in 16; do for T in 0 8; do
  echo "=== remote3 (6 slots/host) N=$N think=$T $(date +%T)"; python load_test.py remote $N $T remote3_N${N}_t${T} 2>&1 | tail -2 | cut -c1-2500
done; done
echo ALLDONE3
