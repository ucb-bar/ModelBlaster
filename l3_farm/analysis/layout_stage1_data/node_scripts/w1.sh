# walker-level check, problems given as args (default: all six), sequential
. /tmp/krisd/x142/env.sh
cd /tmp/krisd/x142; mkdir -p out logs
for p in ${@:-50_ReLUSelfAttention 46_NetVladWithGhostClusters 47_NetVladNoGhostClusters 49_Mamba2ReturnFinalState 43_MinGPTCausalAttention 44_MiniGPTBlock}; do
  echo "=== $p $(date)"
  python walker.py $p --dev ${DEV:-cpu} --runs 1 2>&1 | tail -25
done
echo ALLDONE $(date)
