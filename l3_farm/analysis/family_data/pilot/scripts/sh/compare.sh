. /tmp/krisd/family/sh/env.sh; cd $W
B=$W/pilot/B/r00
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; pgrep -af "kb_graph_opt|time_problems" | grep -v guard | cut -c1-80
BS=$(ls $B/run/kb_*/best/slots/conv2d_f16.cu); TD=$(ls -d $B/family_tune/*/ | head -1)
ls -la $BS $TD/tuned.cu $TD/pool/pool.cu
python pyn/compare.py 3 A_rep=$W/pilot/A_rep.json A_pool4=$W/pilot/A_pool4.cu B=$BS C=$TD/tuned.cu C_pool=$TD/pool/pool.cu > logs/compare.log 2>&1
echo rc=$? >> logs/compare.log
