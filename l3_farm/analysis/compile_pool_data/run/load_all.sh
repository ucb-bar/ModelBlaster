. /tmp/krisd/cpool/env.sh
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
export PATH=/tmp/krisd/cpool/sshbin:$PATH
export LT_HOSTS=a1u1g-mil-0578,a1u1g-mil-0572,a4u8g-mil-0024
cd /tmp/krisd/cpool
rm -rf lt/smoke_*
for N in 8 16; do for T in 0 8; do for A in local shared remote; do
  echo "=== $A N=$N think=$T $(date +%T)"; python load_test.py $A $N $T ${A}_N${N}_t${T} 2>&1 | tail -2 | cut -c1-2500
done; done; done
echo ALLDONE
