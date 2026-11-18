. /tmp/krisd/cpool/env.sh
cd /tmp/krisd/cpool/e2/units; ls | head -40 | tail -12 > /tmp/krisd/cpool/bc_list.txt
mkdir -p /tmp/krisd/cpool/bc; rm -f /tmp/krisd/cpool/bc/*
FL="-O3 -arch=sm_89 -std=c++17 -lineinfo --default-stream per-thread -Xcompiler -fPIC"
same=0; diff=0
for f in $(cat /tmp/krisd/cpool/bc_list.txt); do
  mkdir -p /tmp/krisd/cpool/bc/a /tmp/krisd/cpool/bc/b; cp $f /tmp/krisd/cpool/bc/a/x1.cu; cp $f /tmp/krisd/cpool/bc/b/other_name_$f
  nvcc $FL -c /tmp/krisd/cpool/bc/a/x1.cu -o /tmp/krisd/cpool/bc/a.o
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/krisd/cpool/kh a1u1g-mil-0578 ". /tmp/krisd/cpool/env.sh; mkdir -p /tmp/krisd/cpool/bc2; cat > /tmp/krisd/cpool/bc2/rb_xyz.cu; nvcc $FL -c /tmp/krisd/cpool/bc2/rb_xyz.cu -o /tmp/krisd/cpool/bc2/o.o; cat /tmp/krisd/cpool/bc2/o.o" < $f > /tmp/krisd/cpool/bc/b.o 2>/tmp/krisd/cpool/bc/err.txt
  if cmp -s /tmp/krisd/cpool/bc/a.o /tmp/krisd/cpool/bc/b.o; then same=$((same+1)); else diff=$((diff+1)); head -c 400 /tmp/krisd/cpool/bc/err.txt; echo "differs: $f $(stat -c %s /tmp/krisd/cpool/bc/a.o /tmp/krisd/cpool/bc/b.o | tr '\n' ' ')"; fi
done
echo "byte-identical objects (local file name x1.cu vs remote host rb_xyz.cu): same=$same diff=$diff"
