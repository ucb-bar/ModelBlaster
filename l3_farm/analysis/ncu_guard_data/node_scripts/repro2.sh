set +e
C=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export PATH=$C/bin:$PATH
cd /tmp/krisd/ng/repro
Q="nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,clocks.mem,clocks.applications.graphics,clocks_throttle_reasons.active,utilization.gpu,pstate --format=csv,noheader"
S() { echo "--- $1  [$(date +%T)] $($Q)"; }
NCU="ncu --kernel-name-base demangled -k regex:mbslot_a:: --launch-count 100000 --section SpeedOfLight --csv"
echo "##### V1: SIGKILL the ncu process only (target child left alone)"
$NCU ./t 3000 > v1.out 2>&1 & NP=$!; sleep 8; S v1_during; ps -eo pid,ppid,stat,cmd | grep -E "[.]/t |ncu" | grep -v grep
kill -9 $NP; sleep 1; S v1_after1s; ps -eo pid,ppid,stat,etime,cmd | grep -E "[.]/t 3000"; sleep 5; S v1_after6s
pkill -9 -x t; pkill -9 ncu; sleep 8; S v1_cleaned
echo "##### V2: coreutils timeout -k 5 6 ncu (flow's remote path)"
timeout -k 5 6 env X=1 $NCU ./t 3000 > v2.out 2>&1; echo rc=$?; S v2_after0s; sleep 3; S v2_after3s; ps -eo pid,ppid,stat,etime,cmd | grep -E "[.]/t 3000|[n]cu"; sleep 10; S v2_after13s
pkill -9 -x t; pkill -9 ncu; sleep 8; S v2_cleaned
echo "##### V3: SIGKILL the whole group, target stopped (SIGSTOP) first"
setsid $NCU ./t 3000 > v3.out 2>&1 & NP=$!; sleep 8; pkill -STOP -x t; sleep 1; S v3_during; kill -9 -- -$NP; sleep 1; S v3_after1s; ps -eo pid,stat,cmd | grep -E "[.]/t 3000|[n]cu"; pkill -9 -x t; sleep 8; S v3_cleaned
echo "##### V4: SIGTERM only"
$NCU ./t 3000 > v4.out 2>&1 & NP=$!; sleep 8; kill -TERM $NP; sleep 2; S v4_after2s; sleep 6; S v4_after8s; ps -eo pid,stat,cmd | grep -E "[.]/t 3000|[n]cu"; pkill -9 -x t; pkill -9 ncu; sleep 8; S v4_cleaned
