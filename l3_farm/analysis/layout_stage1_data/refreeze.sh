#!/bin/bash
# freeze the live ModelBlaster tree into frozen/mb_x142.tar.gz, push + install on my node: refreeze.sh JOB
cd "$(dirname "$0")"
COPYFILE_DISABLE=1 tar czf frozen/mb_x142.tar.gz -C ../../../ --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
shasum -a 256 frozen/mb_x142.tar.gz | tee frozen/mb_x142.sha256
python3 nodes_x142.py run $1 node_scripts/install.sh 900 --put frozen/mb_x142.tar.gz=/tmp/krisd/x142/mb_x142.tar.gz --put node_scripts/env.sh=/tmp/krisd/x142/env.sh
