until grep -q ALLDONE /tmp/krisd/cl/logs/run_A4.log; do sleep 10; done; cat /tmp/krisd/cl/logs/run_A4.log | cut -c1-400
