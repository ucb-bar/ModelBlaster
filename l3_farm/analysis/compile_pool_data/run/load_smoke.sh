. /tmp/krisd/cpool/env.sh
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
export PATH=/tmp/krisd/cpool/sshbin:$PATH
export LT_HOSTS=a1u1g-mil-0578,a1u1g-mil-0572,a4u8g-mil-0024
cd /tmp/krisd/cpool
for h in ${LT_HOSTS//,/ }; do ssh -o BatchMode=yes $h hostname; done
python load_test.py local 2 0 smoke_local 2>&1 | tail -5
python load_test.py remote 2 0 smoke_remote 2>&1 | tail -5
