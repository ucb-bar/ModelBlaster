until grep -q ALLDONE /tmp/krisd/cl/logs/flow_S41.log; do sleep 20; done; tail -c 300 /tmp/krisd/cl/logs/flow_S41.log
