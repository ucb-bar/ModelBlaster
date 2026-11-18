. /tmp/krisd/family/sh/env.sh
cd $W
python -m modelblaster.pipeline.kb_model_run $W/irs/15_DenseNet121 --cache $W/cacheA --hints $W/in/sweep/graph_hints.json --arch 89 --reps 5 \
  --work $W/pilot/modelA --out $W/pilot/modelA.json --evaluator local --kb $KB --kb-python $(which python) --gpus 0 --torch-baseline off > $W/logs/modelA.log 2>&1
tail -2 $W/logs/modelA.log
python $W/pyn/stepA2.py > $W/logs/stepA2.log 2>&1; echo rc=$?; tail -8 $W/logs/stepA2.log
