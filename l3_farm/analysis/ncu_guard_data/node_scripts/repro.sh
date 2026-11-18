# reproduce: ncu killed with SIGKILL mid-profile -> what is the clock state? (X-149)
set +e
C=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export PATH=$C/bin:$PATH
cd /tmp/krisd/ng && mkdir -p repro && cd repro
cp ../scripts/t.cu . 2>/dev/null || cp /tmp/krisd/ng/t.cu .
nvcc -O3 -arch=sm_89 t.cu -o t || exit 1
Q="nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,clocks.applications.graphics,clocks_throttle_reasons.active,utilization.gpu,pstate,power.draw --format=csv,noheader"
S() { echo "--- $1  [$(date +%T)]"; $Q; }
S idle_before
nvidia-smi -q -d CLOCK,PERFORMANCE > q_idle_before.txt
echo "== unlocked timing"; ./t 3000
(./t 6000 > busy.out &) ; sleep 1.5; S busy_unlocked; wait; sleep 1
echo "== ncu (base) killed with SIGKILL after 10 s"
setsid ncu --kernel-name-base demangled -k regex:'mbslot_a::' --launch-count 100000 --section SpeedOfLight --csv ./t 3000 > ncu.out 2>&1 &
NP=$!
sleep 10; S during_ncu; nvidia-smi -q -d CLOCK,PERFORMANCE > q_during_ncu.txt
kill -9 -- -$NP 2>/dev/null; kill -9 $NP 2>/dev/null; pkill -9 -f "ncu --kernel-name-base" ; pkill -9 -x t; pkill -9 -f ncu
sleep 1; S after_kill_1s
sleep 6; S after_kill_7s
nvidia-smi -q -d CLOCK,PERFORMANCE > q_after_kill.txt
echo "== timing after kill (idle GPU)"; ./t 3000
(./t 6000 > busy2.out &) ; sleep 1.5; S busy_after_kill; wait; cat busy2.out
sleep 20; S after_kill_30s
