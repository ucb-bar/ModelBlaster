. /tmp/krisd/cpool/env.sh
cd /tmp/krisd/cpool/e2/units; ls | head -40 | tail -12 > /tmp/krisd/cpool/bc_list.txt
FL="-O3 -arch=sm_89 -std=c++17 -lineinfo --default-stream per-thread -Xcompiler -fPIC"
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/krisd/cpool/kh a1u1g-mil-0578"
same=0; diff=0; same2=0
mkdir -p /tmp/krisd/cpool/bc/c
for f in $(cat /tmp/krisd/cpool/bc_list.txt); do
  cp $f /tmp/krisd/cpool/bc/c/x1.cu
  nvcc $FL -c /tmp/krisd/cpool/bc/c/x1.cu -o /tmp/krisd/cpool/bc/c.o
  # remote: same dir + same name
  $SSH ". /tmp/krisd/cpool/env.sh; mkdir -p /tmp/krisd/cpool/bc/c; cat > /tmp/krisd/cpool/bc/c/x1.cu; nvcc $FL -c /tmp/krisd/cpool/bc/c/x1.cu -o /tmp/krisd/cpool/bc/cr.o; cat /tmp/krisd/cpool/bc/cr.o" < $f > /tmp/krisd/cpool/bc/cr.o 2>/dev/null
  if cmp -s /tmp/krisd/cpool/bc/c.o /tmp/krisd/cpool/bc/cr.o; then same=$((same+1)); else diff=$((diff+1)); fi
  # local twice same path (determinism)
  nvcc $FL -c /tmp/krisd/cpool/bc/c/x1.cu -o /tmp/krisd/cpool/bc/c2.o
  cmp -s /tmp/krisd/cpool/bc/c.o /tmp/krisd/cpool/bc/c2.o && same2=$((same2+1))
done
echo "same path+name: local-vs-remote identical=$same differ=$diff; local rebuild identical=$same2 of 12"
