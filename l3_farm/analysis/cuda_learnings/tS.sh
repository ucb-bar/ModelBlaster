tail -c 300 /tmp/krisd/cl/logs/flow_S41.log; tail -c 2500 /tmp/krisd/cl/flows/S41.log | grep -v "^ "| tail -8 | cut -c1-600
