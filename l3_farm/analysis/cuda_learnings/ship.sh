#!/bin/bash
# Pack my live ModelBlaster tree (with KernelBlaster) for the nodes: $1 = output tarball
set -e
cd ~/Documents/ModelBlaster-clone
COPYFILE_DISABLE=1 tar --no-xattrs -czf "$1" --exclude=.git --exclude=__pycache__ --exclude='*.pyc' --exclude=ModelBlaster/gpu_eval/runs \
  --exclude=ModelBlaster/gpu_eval/ablations --exclude=ModelBlaster/experiments --exclude='ModelBlaster/third_party/KernelBlaster/runs' ModelBlaster
ls -la "$1"
