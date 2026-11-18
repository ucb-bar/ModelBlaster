. /tmp/krisd/cpool/env.sh
until grep -q ALLDONE /tmp/krisd/cpool/logs/load_all.log; do sleep 5; done
KBS=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src/kernelblaster/knowledge
cp $KBS/remote_build.py /tmp/krisd/cpool/remote_build_v1.py
cp /tmp/krisd/cpool/remote_build_v2.py $KBS/remote_build.py
sha256sum $KBS/remote_build.py
export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:/tmp/krisd/cpool/code
export PATH=/tmp/krisd/cpool/sshbin:$PATH
export LT_HOSTS=a1u1g-mil-0578,a1u1g-mil-0572,a4u8g-mil-0024
cd /tmp/krisd/cpool
for N in 8 16; do for T in 0 8; do
  echo "=== remote2 N=$N think=$T $(date +%T)"; python load_test.py remote $N $T remote2_N${N}_t${T} 2>&1 | tail -2 | cut -c1-2500
done; done
echo ALLDONE2
