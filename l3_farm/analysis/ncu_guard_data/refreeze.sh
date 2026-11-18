#!/bin/bash
# freeze the live ModelBlaster tree into frozen/mb_ng.tar.gz, push + install on my node: refreeze.sh JOB
cd "$(dirname "$0")"
COPYFILE_DISABLE=1 tar czf frozen/mb_ng.tar.gz -C ../../../ --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
shasum -a 256 frozen/mb_ng.tar.gz | tee frozen/mb_ng.sha256
python3 nodes_ng.py run $1 node_scripts/install.sh 900 --put frozen/mb_ng.tar.gz=/tmp/krisd/ng/mb_ng.tar.gz --put node_scripts/env.sh=/tmp/krisd/ng/env.sh
