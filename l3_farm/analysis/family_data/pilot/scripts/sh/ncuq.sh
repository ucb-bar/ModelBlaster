. /tmp/krisd/family/sh/env.sh
ncu --query-metrics --chip ad102 2>/dev/null | grep -E "^smsp__warps_issue_stalled_[a-z_]+ " | awk '{print $1}' | tr '\n' ' '
