until grep -q "=== rc" /tmp/krisd/cl/logs/flow_B.log; do sleep 15; done; grep "=== rc" /tmp/krisd/cl/logs/flow_B.log
