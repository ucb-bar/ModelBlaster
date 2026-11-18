#!/bin/bash
# waitfor.sh JOB PATTERN : poll every 120 s until node/isdone2.sh output on JOB matches PATTERN
cd "$(dirname "$0")"
until python3 a2run.py $1 node/isdone2.sh 60 2>/dev/null | grep -q "$2"; do sleep 120; done
python3 a2run.py $1 node/isdone2.sh 60
