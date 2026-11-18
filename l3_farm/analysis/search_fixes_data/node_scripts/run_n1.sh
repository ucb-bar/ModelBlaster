. /tmp/krisd/a3/env.sh
export MB_PRUNE_DEAD=0
cd /tmp/krisd/a3; mkdir -p runs
echo "=== audit 15 $(date)"
python a3_node.py audit 15_DenseNet121 data/15_DenseNet121 runs/15.audit --route --reps 5 2>&1 | tail -40
echo "=== inject 15 $(date)"
python a3_node.py inject 15_DenseNet121 data/15_DenseNet121 runs/15.inject --op dense_blocks.3.layers.14.2 2>&1 | tail -60
echo ALLDONE $(date)
