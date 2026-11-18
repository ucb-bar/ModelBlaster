Q="nvidia-smi --query-gpu=index,uuid,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,clocks.applications.graphics,clocks_throttle_reasons.active,utilization.gpu,pstate --format=csv,noheader"
date +%T; $Q; ps -eo pid,ppid,stat,etime,cmd | grep -E "[.]/t |[n]cu|nv-nsight" | grep -v "grep"
nvidia-smi -q -d CLOCK,PERFORMANCE | grep -iE "lock|applications|SM  |Graphics|Throttle|Slowdown|Idle|Setting|Reasons" | head -30
