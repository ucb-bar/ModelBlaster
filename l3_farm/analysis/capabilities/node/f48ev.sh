cd /tmp/krisd/a2/f48; grep -h "@@TORCH48\|EXTRACT_FAILED\|=== flow\|F48_DONE\|Traceback\|RuntimeError\|\"round\"" /tmp/krisd/a2/logs/flow48.log flow.log 2>/dev/null | cut -c1-300
