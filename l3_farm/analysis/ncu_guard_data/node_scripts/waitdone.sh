for i in $(seq 1 55); do grep -q "^done" /tmp/krisd/ng/logs/node_suite.log && break; sleep 10; done
tail -12 /tmp/krisd/ng/logs/node_suite.log | cut -c1-200
