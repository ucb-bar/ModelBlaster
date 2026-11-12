#!/usr/bin/env bash
# Build + run a k3_*_probe.c on the K3 board on one CPU (static binary).
# usage: run_k3_probe.sh <probe-name-without-k3_-prefix-.c> <cpu> [outfile]
#   e.g. run_k3_probe.sh rvv_microcost_probe 8 artifacts/k3_probes/microcost_cpu8.txt
# Needs: MODELBLASTER_SSH_CONFIG, K3_BOARD_LOCK (flock file), CROSS (setup_spacemit_toolchain.sh).
# CPUs >=8 build with zvl1024b, else zvl256b; the A100 registration happens inside the
# probe (mb_place_current_thread), never from the shell.
set -euo pipefail
probe=$1; cpu=$2; out=${3:-/dev/stdout}
here=$(cd "$(dirname "$0")/.." && pwd)
z=256; [ "$cpu" -ge 8 ] && z=1024
bin=$(mktemp -d)/k3_${probe}_z$z
${CROSS}gcc -O2 -static -fno-tree-vectorize -march=rv64gcv_zvl${z}b_zfh_zvfh -mabi=lp64d \
  -DMODELBLASTER_PLATFORM_LINUX -DMODELBLASTER_AI_CPUS=\"8-15\" -I"$here/runtime" \
  "$here/scripts/k3_${probe}.c" "$here/runtime/mb_placement.c" -o "$bin" -pthread
SSH="ssh -F $MODELBLASTER_SSH_CONFIG -o ConnectTimeout=60"
SCP="scp -F $MODELBLASTER_SSH_CONFIG -o ConnectTimeout=60"
rd=/home/sudospace/mb_k3/probes
for i in 1 2 3 4; do
  flock "$K3_BOARD_LOCK" $SCP "$bin" spacemitk3-local:$rd/ && break || sleep 3
done
for i in 1 2 3 4; do
  flock "$K3_BOARD_LOCK" $SSH spacemitk3-local "cd $rd && taskset -c 0-7 ./$(basename "$bin") $cpu" > "$out" 2>&1 && break || sleep 3
done
cat "$out"
