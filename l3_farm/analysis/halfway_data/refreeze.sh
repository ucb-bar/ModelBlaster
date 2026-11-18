#!/bin/bash
# freeze the live ModelBlaster tree (same excludes as farm.stage) into frozen/mb_hw.tar.gz and record its sha256
cd "$(dirname "$0")"
COPYFILE_DISABLE=1 tar czf frozen/mb_hw.tar.gz -C ../../../ --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs --exclude=ModelBlaster/gpu_eval/ablations ModelBlaster
shasum -a 256 frozen/mb_hw.tar.gz | tee frozen/mb_hw.sha256
