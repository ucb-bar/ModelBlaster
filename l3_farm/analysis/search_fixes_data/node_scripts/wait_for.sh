#!/bin/bash
# wait_for.sh JOB LOGNAME PATTERN: poll the node log until PATTERN appears (max ~3 h)
cd ~/Documents/ModelBlaster-clone/l3_farm/analysis/search_fixes_data
S=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
printf "grep -c '%s' /tmp/krisd/a3/logs/%s.log\n" "$3" "$2" > $S/a3/w_$2.sh
for i in $(seq 1 180); do
  n=$(python3 nodes_a3.py run $1 $S/a3/w_$2.sh 120 2>/dev/null | head -1)
  [ "$n" != "" ] && [ "$n" -gt 0 ] 2>/dev/null && { echo "found after $i polls"; exit 0; }
  sleep 60
done
echo timeout
