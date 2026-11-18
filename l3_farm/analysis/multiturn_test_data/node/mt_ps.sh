echo "--- compute apps"; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
echo "--- my procs"; ps -u $(id -u) -o pid,etimes,stat,pcpu,args --sort=-pcpu | grep -v "ps -u\|grep" | cut -c1-200 | head -25
