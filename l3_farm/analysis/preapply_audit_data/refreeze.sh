#!/bin/bash
# freeze the live ModelBlaster tree into frozen/mb_pa.tar.gz, push + install on my node: refreeze.sh JOB
cd "$(dirname "$0")"
COPYFILE_DISABLE=1 tar czf frozen/mb_pa.tar.gz -C ../../../ --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
shasum -a 256 frozen/mb_pa.tar.gz | tee frozen/mb_pa.sha256
python3 nodes_pa.py run $1 node_scripts/install.sh 900 --put frozen/mb_pa.tar.gz=/tmp/krisd/pa/mb_pa.tar.gz --put node_scripts/env.sh=/tmp/krisd/pa/env.sh
