#!/bin/zsh
# PRINTS (does not run) the commands that put seed/ on a driver node without scp: tar | base64 over ssh stdin.
# usage: ship_seed.sh <ssh-host-or-"srun --jobid=J --overlap"-style prefix> [DEST=/tmp/krisd/abl_code/empty_seed]
E=${0:A:h}; DEST=${2:-/tmp/krisd/abl_code/empty_seed}; H=${1:-<driver-host>}
B64=$(cd $E && tar czf - seed | base64 | tr -d '\n')
echo "# size: ${#B64} base64 chars; sha256 of targets.json (must match the node's repo copy): $(shasum -a 256 $E/seed/targets.json | cut -d' ' -f1)"
echo "ssh $H 'mkdir -p $DEST && base64 -d | tar xzf - -C $DEST && sha256sum $DEST/seed/targets.json \\"
echo "   /tmp/krisd/abl_code/ModelBlaster/third_party/KernelBlaster/data/kernelblaster/knowledge/targets.json' <<< '$B64'"
