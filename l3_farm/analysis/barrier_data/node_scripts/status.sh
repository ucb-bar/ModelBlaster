pgrep -af kb_graph_opt | head -3
echo "setup.status: $(cat /tmp/krisd/farm/setup.status 2>/dev/null)"
ls /tmp/krisd /tmp/krisd/farm 2>/dev/null | head -30
ps -eo pid,etime,cmd | grep -E "setup.sh|pip " | grep -v grep | head
nvidia-smi --query-gpu=name,clocks.max.memory --format=csv,noheader
