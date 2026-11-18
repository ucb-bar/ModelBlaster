# One problem end to end (worker3.sh + global knowledge): extract IR at native size (no scaling), then
# kb_graph_opt. Status in $D/status. Global knowledge: when the controller uploaded a seed
# ($D/gk_seed.tar.gz, materialized from the current global snapshot) and job.env has GK_MODE=read, a fresh
# work dir's knowledge/ starts from it (kb_graph_opt --knowledge-global read); the controller merges the run
# back after collecting it. A resumed work dir keeps its own knowledge (kb_graph_opt applies a seed once).
# Args: <name> [farm dir, default /tmp/krisd/farm]. job.env may set CODE_ROOT (the dir holding ModelBlaster/ and
# the modelblaster symlink, default /tmp/krisd) -- the smoke test runs a private tree that way -- and KB_EXTRA
# (more kb_graph_opt flags).
NAME=$1; FD=${2:-/tmp/krisd/farm}; D=$FD/$NAME; . $D/job.env; CR=${CODE_ROOT:-/tmp/krisd}
st(){ echo "$1" > $D/status; }
st running
# resumed from a pack that omits the exported problem dirs: drop the round that was in flight so it is re-exported
if [ -f $D/resume.tar.gz ] && [ -f $D/work/state.json ]; then
  python3 - $D/work <<"PY"
import json, sys, shutil, pathlib
w = pathlib.Path(sys.argv[1]); n = len(json.load(open(w / "state.json")).get("rounds", []))
for p in sorted(w.glob("r[0-9][0-9]")):
    if int(p.name[1:]) >= n:
        shutil.rmtree(p); print("dropped partial round", p.name)
PY
fi
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR MB_TORCH_SRC=$D/src
. /tmp/krisd/farm/relay.env
KB=$CR/ModelBlaster/third_party/KernelBlaster
cd $CR
IR=$D/irs/$NAME; mkdir -p $D/src; cp $D/problem.py $D/src/$NAME.py   # torch baseline finds <ir name>.py under $MB_TORCH_SRC
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $D/problem.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $D/extract.log 2>&1 \
    || { st failed:extract; exit 1; }
  touch $IR/.done
fi
GK=""
# an older staged tree without global knowledge runs unseeded (the controller then merges it as such)
if [ "${GK_MODE:-off}" != off ] && [ -f $D/gk_seed.tar.gz ] && grep -q knowledge_global_seed $CR/ModelBlaster/pipeline/kb_graph_opt.py; then
  rm -rf $D/gk_seed && mkdir -p $D/gk_seed && tar xzf $D/gk_seed.tar.gz -C $D/gk_seed \
    && GK="--knowledge-global read --knowledge-global-seed $D/gk_seed --knowledge-global-problem $NAME --knowledge-global-part ${GK_PART:-cuda:sm_${ARCH}|fp16}"
fi
timeout ${TIMEOUT_H}h python -m modelblaster.pipeline.kb_graph_opt $IR --work $D/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $MODEL --reflect-model $MODEL --upkeep-model $MODEL --knowledge v2 --ncu best --gpus 0 --arch $ARCH \
  --plan-mode all --plan-top-k $PLAN_TOPK --plan-min-share $PLAN_MINSHARE --min-gain-pct $MIN_GAIN --rounds $ROUNDS --rollouts $ROLLOUTS --steps $STEPS \
  --variant-steps $VSTEPS --max-variants $MAXVAR --parallel-hints auto $GK ${KB_EXTRA:-} > $D/run.log 2>&1
rc=$?
if [ $rc = 124 ]; then st failed:timeout; elif [ $rc = 0 ]; then st done; else st failed:exit$rc; fi
