# worker4.sh + the scheduling features (analysis/scheduling.md); worker3.sh / worker4.sh are unchanged.
# One problem end to end: extract IR at native size, then kb_graph_opt. Status in $D/status.
# Args: <name> [farm dir, default /tmp/krisd/farm]. job.env (written by farm.py start) may set:
#   CODE_ROOT   dir holding ModelBlaster/ and the modelblaster symlink (default /tmp/krisd)
#   RELAY_ENV   the node's LLM relay env (default /tmp/krisd/farm/relay.env)
#   GK_MODE / GK_PART   global knowledge seed ($D/gk_seed.tar.gz), as worker4.sh
#   GK_REFRESH=1        mid-run sharing: kb_graph_opt --knowledge-global-refresh $D/gk_refresh (the controller drops
#                       refresh packages there; imported at each round start)
#   GK_HALFWAY=1        (with GK_REFRESH) the halfway sync: kb_graph_opt --knowledge-global-halfway on (analysis/halfway_sync.md);
#                       GK_HALFWAY_WAIT_S the node's bounded wait for the controller's package (default 180)
#   POOL_HOSTS='h1 h2'  pooled mode: candidates / profiles / verification are timed on these hosts' GPUs over ssh
#                       (KernelBlaster SSHEvaluator; POOL_DIR remote dir, POOL_CVD GPU spec), not on this node's GPU
#   PARK_AFTER_CHAR=1   barrier scheduling phase 1 (analysis/barrier_scheduling.md): kb_graph_opt --park-after-char on; the run exits 78
#                       after its characterization pass and the status becomes "parked:char" (not done, not failed; the controller
#                       resumes it by starting this worker again WITHOUT the line)
#   GK_CHAR_SYNC=1      (with GK_REFRESH) the characterization sync: --knowledge-global-char-sync on (analysis/barrier_scheduling.md);
#                       GK_CHAR_SYNC_WAIT_S the bounded wait (default 180)
#   KB_EXTRA    more kb_graph_opt flags
NAME=$1; FD=${2:-/tmp/krisd/farm}; D=$FD/$NAME; . $D/job.env; CR=${CODE_ROOT:-/tmp/krisd}
st(){ echo "$1" > $D/status; }
st running
if [ -f $D/resume.tar.gz ] && [ -f $D/work/state.json ]; then
  python3 - $D/work $CR/ModelBlaster/pipeline/kb_barrier.py <<"PY"
import json, sys, shutil, pathlib
w = pathlib.Path(sys.argv[1])
if (w / "parked.json").exists() and pathlib.Path(sys.argv[2]).exists():
    # a barrier-parked round is kept, rolled back to its park point (kb_barrier.drop_partial_rounds); the rest as below
    import importlib.util
    sp = importlib.util.spec_from_file_location("kb_barrier", sys.argv[2]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    for ln in m.drop_partial_rounds(w): print(ln)
    sys.exit(0)
n = len(json.load(open(w / "state.json")).get("rounds", []))
for p in sorted(w.glob("r[0-9][0-9]")):
    if int(p.name[1:]) >= n:
        shutil.rmtree(p); print("dropped partial round", p.name)
PY
fi
export PATH=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04/bin:$PATH
export CUDA_HOME=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export MB_CUPTI_LIB=$CUDA_HOME/extras/CUPTI/lib64/libcupti.so.12
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$CR MB_TORCH_SRC=$D/src
. ${RELAY_ENV:-/tmp/krisd/farm/relay.env}
KB=$CR/ModelBlaster/third_party/KernelBlaster
cd $CR
IR=$D/irs/$NAME; mkdir -p $D/src; cp $D/problem.py $D/src/$NAME.py
if [ ! -f $IR/.done ]; then
  python -m modelblaster.pipeline.extract_graph --bench-file $D/problem.py --bench-target-mb ${BENCH_TARGET_MB:-0} --quant fp16 --out-dir $IR > $D/extract.log 2>&1 \
    || { st failed:extract; exit 1; }
  touch $IR/.done
fi
GK=""
if [ "${GK_MODE:-off}" != off ] && [ -f $D/gk_seed.tar.gz ] && grep -q knowledge_global_seed $CR/ModelBlaster/pipeline/kb_graph_opt.py; then
  rm -rf $D/gk_seed && mkdir -p $D/gk_seed && tar xzf $D/gk_seed.tar.gz -C $D/gk_seed \
    && GK="--knowledge-global read --knowledge-global-seed $D/gk_seed --knowledge-global-problem $NAME --knowledge-global-part ${GK_PART:-cuda:sm_${ARCH}|fp16}"
  if [ -n "$GK" ] && [ "${GK_REFRESH:-0}" = 1 ] && grep -q knowledge_global_refresh $CR/ModelBlaster/pipeline/kb_graph_opt.py; then
    mkdir -p $D/gk_refresh && GK="$GK --knowledge-global-refresh $D/gk_refresh"
    # halfway sync: the plan run parks at half its steps and waits (<= GK_HALFWAY_WAIT_S) for the controller's package
    if [ "${GK_HALFWAY:-0}" = 1 ] && grep -q knowledge_global_halfway $CR/ModelBlaster/pipeline/kb_graph_opt.py; then
      GK="$GK --knowledge-global-halfway on --knowledge-global-halfway-wait-s ${GK_HALFWAY_WAIT_S:-180}"
    fi
  fi
fi
CS=""      # characterization sync (X-151): wait for the controller's package after the characterization pass (not when parking)
[ "${GK_CHAR_SYNC:-0}" = 1 ] && [ -n "$GK" ] && grep -q knowledge_global_char_sync $CR/ModelBlaster/pipeline/kb_graph_opt.py \
  && CS="--knowledge-global-char-sync on --knowledge-global-char-sync-wait-s ${GK_CHAR_SYNC_WAIT_S:-180}"
PK=""
[ "${PARK_AFTER_CHAR:-0}" = 1 ] && grep -q park_after_char $CR/ModelBlaster/pipeline/kb_graph_opt.py && PK="--park-after-char on"
GPU="--gpus 0"
if [ -n "${POOL_HOSTS:-}" ]; then
  GPU="--remote-dir ${POOL_DIR:-/tmp/krisd/pool} --kb-arg=--remote-cuda-bin=$CUDA_HOME/bin"
  # KernelBlaster's ssh / rsync run "ssh" from PATH with BatchMode: a wrapper gives them a private known_hosts
  # file in the problem dir with accept-new (a new pool host's key is recorded once, a CHANGED key is refused;
  # the shared ~/.ssh/known_hosts was seen empty on the nodes)
  mkdir -p $D/.sshbin
  printf '#!/bin/sh\nexec /usr/bin/ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=%s/.known_hosts "$@"\n' $D > $D/.sshbin/ssh
  chmod +x $D/.sshbin/ssh; export PATH=$D/.sshbin:$PATH
  for h in $POOL_HOSTS; do
    ssh -o BatchMode=yes -o ConnectTimeout=15 $h true </dev/null >/dev/null 2>&1 || echo "pool host $h unreachable over ssh" >> $D/pool.log
    GPU="$GPU --ssh-host $h --cuda-visible-devices ${POOL_CVD:-all}"
  done
  # one --cuda-visible-devices for every host (kb_graph_opt / run_multislot: once, or once per host)
  # short control path: a Unix socket path is limited to ~104 chars ($D/.sshmux/<40 hex>.<16 random> is too long for long problem names)
  mkdir -p /tmp/krisd/m; export KERNELBLASTER_SSH_CONTROL_DIR=/tmp/krisd/m/$(printf %s "$NAME" | md5sum | cut -c1-8)
fi
timeout ${TIMEOUT_H}h python -m modelblaster.pipeline.kb_graph_opt $IR --work $D/work --kb $KB --python /tmp/krisd/venv/bin/python \
  --model $MODEL --reflect-model $MODEL --upkeep-model $MODEL --knowledge v2 --ncu best $GPU --arch $ARCH \
  --plan-mode all --plan-top-k $PLAN_TOPK --plan-min-share $PLAN_MINSHARE --min-gain-pct $MIN_GAIN --rounds $ROUNDS --rollouts $ROLLOUTS --steps $STEPS \
  --variant-steps $VSTEPS --max-variants $MAXVAR --parallel-hints auto $GK $CS $PK ${KB_EXTRA:-} > $D/run.log 2>&1
rc=$?
if [ $rc = 124 ]; then st failed:timeout; elif [ $rc = 0 ]; then st done; elif [ $rc = 78 ] && [ -n "$PK" ]; then st parked:char; else st failed:exit$rc; fi
