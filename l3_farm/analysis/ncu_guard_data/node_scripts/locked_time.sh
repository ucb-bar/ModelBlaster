C=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
cd /tmp/krisd/ng/repro
Q="nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,utilization.gpu,pstate,power.draw --format=csv,noheader"
echo "locked: timing"; ./t 3000
(./t 8000 > b3.out &) ; sleep 1.5; echo "busy while locked: $($Q)"; wait; cat b3.out
sleep 3; echo "idle again: $($Q)"
