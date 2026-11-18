. /tmp/krisd/a2/env.sh
cd /tmp/krisd/a2 && rm -rf b28 && tar xzf b28.tgz 2>/dev/null
python /tmp/krisd/a2/node/view28.py /tmp/krisd/a2/b28 /tmp/krisd/a2/runs/view28 > logs/view28.log 2>&1; grep "@@\|Error" logs/view28.log | cut -c1-1500
