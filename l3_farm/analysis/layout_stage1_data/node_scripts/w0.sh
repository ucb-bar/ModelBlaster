# extract the six real IRs (farm's command) into /tmp/krisd/x142/irs
. /tmp/krisd/x142/env.sh
cd /tmp/krisd/x142; mkdir -p irs logs
df -h /tmp | tail -1
for p in 50_ReLUSelfAttention 46_NetVladWithGhostClusters 47_NetVladNoGhostClusters 49_Mamba2ReturnFinalState 43_MinGPTCausalAttention 44_MiniGPTBlock; do
  if [ ! -f irs/$p/.done ]; then
    mkdir -p src; cp problems/$p.py src/$p.py
    ( cd /tmp/krisd/x142/code && python -m modelblaster.pipeline.extract_graph --bench-file /tmp/krisd/x142/problems/$p.py --bench-target-mb 0 --quant fp16 --out-dir /tmp/krisd/x142/irs/$p > /tmp/krisd/x142/logs/extract_$p.log 2>&1 ) && touch irs/$p/.done
    echo "$p extract rc=$? $(ls irs/$p | tr '\n' ' ')"
  fi
done
df -h /tmp | tail -1
