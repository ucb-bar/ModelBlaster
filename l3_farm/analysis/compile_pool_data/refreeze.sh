#!/bin/bash
# freeze the live ModelBlaster tree into the scratch tarball, record sha256, push + install on my nodes: refreeze.sh JOB...
cd "$(dirname "$0")"
SP=/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad
COPYFILE_DISABLE=1 tar czf $SP/mb_cpool.tar.gz -C ../../../ --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
(cd $SP && shasum -a 256 mb_cpool.tar.gz) | tee frozen/mb_cpool.sha256
for j in "$@"; do
  python3 nodes_cp.py run $j node_scripts/install.sh 900 --put $SP/mb_cpool.tar.gz=/tmp/krisd/cpool/mb_cpool.tar.gz --put node_scripts/env.sh=/tmp/krisd/cpool/env.sh
done
