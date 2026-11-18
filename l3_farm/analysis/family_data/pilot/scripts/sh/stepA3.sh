. /tmp/krisd/family/sh/env.sh; cd $W; python pyn/stepA3.py > logs/stepA3.log 2>&1; echo rc=$?; grep -v "^\s*$" logs/stepA3.log | tail -6
