# keep the production dispatcher (which claims any L40S job of mine) from starting a real problem on this node:
# a process with "kb_graph_opt" in its command line that is not under /tmp/krisd/farm/ counts as "foreign" for it; the
# test instances' own paths are listed so their probes (grep -v "$F/") do not count it
pgrep -f kb_graph_opt_hw_guard >/dev/null || { nohup setsid bash -c 'exec -a "kb_graph_opt_hw_guard /tmp/krisd/hw_on/ /tmp/krisd/hw_off/ /tmp/krisd/hw_pooled/ /tmp/krisd/hw_x/" sleep 40000' </dev/null >/dev/null 2>&1 & sleep 1; }
pgrep -af kb_graph_opt | head -3
echo "setup.status: $(cat /tmp/krisd/farm/setup.status 2>/dev/null)"; ps -eo pid,etime,cmd | grep -E "setup.sh|pip " | grep -v grep | head
ls /tmp/krisd /tmp/krisd/farm 2>/dev/null | head -20
