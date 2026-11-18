grep -h "=== \|AB5_DONE\|F48_DONE\|EXTRACT_FAILED" /tmp/krisd/a2/logs/ab5.log /tmp/krisd/a2/logs/flow48.log 2>/dev/null | cut -c1-200
grep -h "Traceback\|RuntimeError\|\"round\": [0-9]" /tmp/krisd/a2/f48/flow.log 2>/dev/null | cut -c1-200 | tail -5
