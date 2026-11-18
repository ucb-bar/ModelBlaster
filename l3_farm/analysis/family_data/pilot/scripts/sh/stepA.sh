# A: rebuild the sweep's DenseNet121 kernel cache, re-measure the model e2e and the conv family on this node
. /tmp/krisd/family/sh/env.sh
cd $W; mkdir -p src irs logs; cp in/problem.py src/15_DenseNet121.py
IR=$W/irs/15_DenseNet121
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file in/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > logs/extract.log 2>&1 && touch $IR/.done || { echo extract failed; tail -20 logs/extract.log; exit 1; }
fi
python $W/pyn/build_cache.py > logs/build_cache.log 2>&1; echo build_cache rc=$?; tail -5 logs/build_cache.log
