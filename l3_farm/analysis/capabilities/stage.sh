#!/bin/bash
# Frozen tarball of the live ModelBlaster tree (KernelBlaster included) for A2's node runs; sha256 recorded.
set -e
H=$(cd "$(dirname "$0")" && pwd)
MB=$HOME/Documents/ModelBlaster-clone
T=$(date +%Y%m%d-%H%M%S)
mkdir -p $H/stage
COPYFILE_DISABLE=1 tar czf $H/stage/mb_a2_$T.tar.gz -C $MB --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
  --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
SHA=$(shasum -a 256 $H/stage/mb_a2_$T.tar.gz | cut -d' ' -f1)
echo "$T $SHA" >> $H/stage/stages.txt
ls -t $H/stage/*.tar.gz | tail -n +3 | xargs rm -f
echo $T $SHA
