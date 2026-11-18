#!/bin/bash
# E8 tool validation: rebuild + time (the flow's own driver, 3 requests) every given problem on one node, one at a time.
# usage: session_validate.sh JOB P [P ...]
cd "$(dirname "$0")"
JOB=$1; shift
python3 e8_run.py push $JOB "$@"
for p in "$@"; do
  python3 e8_run.py rebuild $JOB $p && python3 e8_run.py flowtime $JOB $p --tag val
done
