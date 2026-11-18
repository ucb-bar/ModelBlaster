for f in /tmp/krisd/a2/logs/*.log; do echo "== $f"; grep -v "Warning\|warn(" $f | tail -6 | cut -c1-1500; done
