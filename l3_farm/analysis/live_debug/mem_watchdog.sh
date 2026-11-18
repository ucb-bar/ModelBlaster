#!/bin/bash
# Run ON the node whose controllers share the RAM (driver 5310063: `srun --jobid=5310063 --overlap nohup setsid bash mem_watchdog.sh &`).
# Every 10 s: any `python -m modelblaster.pipeline.kb_graph_opt` above KILL_GB RSS is killed (SIGTERM to its process group's
# python, then SIGKILL) so the node's OOM killer never picks a random victim; the farm sees exit != 0 and retries the
# problem (resume, attempts <= 3). Controllers above WARN_GB get oom_score_adj 800 so the kernel prefers them anyway.
# Usage: KILL_GB=70 WARN_GB=35 bash mem_watchdog.sh   (log: /tmp/krisd/mem_watchdog.log)
KILL_GB=${KILL_GB:-70}; WARN_GB=${WARN_GB:-35}; LOG=/tmp/krisd/mem_watchdog.log
while true; do
  for pid in $(pgrep -f '^python -m modelblaster.pipeline.kb_graph_opt'); do
    kb=$(awk '/^VmRSS:/{print $2}' /proc/$pid/status 2>/dev/null) || continue
    [ -z "$kb" ] && continue
    gb=$((kb/1048576))
    name=$(tr '\0' ' ' < /proc/$pid/cmdline | sed -n 's#.*/irs/\([^ ]*\) .*#\1#p')
    if [ "$gb" -ge "$WARN_GB" ]; then echo 800 > /proc/$pid/oom_score_adj 2>/dev/null; fi
    if [ "$gb" -ge "$KILL_GB" ]; then
      echo "$(date +%T) KILL $name pid $pid rss ${gb} GB" >> $LOG
      kill -TERM $pid; sleep 5; kill -0 $pid 2>/dev/null && kill -KILL $pid
    fi
  done
  sleep 10
done
