#!/bin/bash
# host_verify_all.sh IR_ROOT WORK_ROOT [JOBS]  -> WORK_ROOT/host_status.txt
IRR=$1; W=$2; J=${3:-12}; mkdir -p "$W"
export OMP_NUM_THREADS=1; export HV=$(cd "$(dirname "$0")" && pwd -P)/host_verify.sh
ls -d "$IRR"/*/graph.json | xargs -n1 dirname | xargs -P "$J" -I{} bash -c 'p=$(basename {}); r=$(HOST_TIMEOUT=${HOST_TIMEOUT:-1800} "$HV" {} '"$W"'/$p 2>&1 | tail -1); printf "%s\t%s\n" "$p" "$r"' | sort -V > "$W/host_status.txt"
