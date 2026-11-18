. /tmp/krisd/cpool/env.sh
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
cd /tmp/krisd/cpool
export PATH=/tmp/krisd/cpool/sshbin:$PATH
for h in a4u8g-mil-0024 a1u1g-mil-0572; do ssh -o BatchMode=yes $h hostname; done
H=a4u8g-mil-0024,a1u1g-mil-0572
for m in tiny branches; do for a in A B; do
  echo "=== $a $m $(date +%T)"; python /tmp/krisd/cpool/e3_pooled.py $a $m /tmp/krisd/cpool/e3/${a}_$m $H 2>&1 | tail -15
done; done
