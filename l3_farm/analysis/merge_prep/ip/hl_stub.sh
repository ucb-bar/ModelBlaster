# headline flags, zero-LLM stub flow (identity LLM) on a frozen post-merge tree. Args: <tree label> <problem...>
L=$1; shift
E=/tmp/krisd/mc; CR=$E/$L; KB=$CR/ModelBlaster/third_party/KernelBlaster
. /tmp/krisd/venv/bin/activate
CUDA_N=/home/scratch.svc_compute_arch/release/cuda_toolkit/public/12.6.1/x86_64/u22.04
export PATH=$CUDA_N/bin:$PATH CUDA_HOME=$CUDA_N MB_CUPTI_LIB=$CUDA_N/extras/CUPTI/lib64/libcupti.so.12 PYTHONPATH=$CR
FLAGS=$(python3 -c "import json;print(' '.join(e['flag']+' '+e['value'] for e in json.load(open('/tmp/krisd/mc/ip/headline.json'))['flags'] if e.get('enabled',True)))")
mkdir -p $E/ip/hl; cat > $E/ip/hl/plugin.py <<'PY'
from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm
def scripted(msgs):
    user = msgs[-1]["content"]
    cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user else user.split("Current slot source:")[-1])
    return "```cuda\n" + (cur or "") + "```"
agentic_llm.register_stub("graphopt", scripted)
PY
for P in "$@"; do
  W=$E/ip/hl/$L/$P; rm -rf $W; mkdir -p $W
  IR=$E/ip/irs_hl/$P/ir
  if [ ! -f $IR/graph.json ]; then mkdir -p $E/ip/src_$P $E/ip/irs_hl/$P; cp $E/ip/probs/$P.py $E/ip/src_$P/; MB_TORCH_SRC=$E/ip/src_$P python -m modelblaster.pipeline.extract_graph --bench-file $E/ip/probs/$P.py --bench-target-mb 0 --quant fp16 --out-dir $IR > $W/extract.log 2>&1; fi
  echo "== $P flags: $FLAGS"
  timeout 3000 python -m modelblaster.pipeline.kb_graph_opt $IR --work $W/work --kb $KB --python /tmp/krisd/venv/bin/python \
    --model stub:graphopt --reflect-model stub:graphopt --upkeep-model stub:graphopt --plugin $E/ip/hl/plugin.py --knowledge v2 --ncu off --gpus 0 --arch 89 \
    --plan-mode all --plan-top-k 32 --plan-min-share 0.002 --min-gain-pct 0.5 --rounds 2 --rollouts 2 --steps 2 \
    --variant-steps 1 --max-variants 4 --parallel-hints auto --knowledge-global off --seed 1 --torch-baseline off $FLAGS > $W/run.log 2>&1
  echo "rc=$? $P"; tail -5 $W/run.log | cut -c1-300
  python3 - <<PY
import json
try:
    s=json.load(open("$W/work/summary.json"))
    print(json.dumps({k:s.get(k) for k in ("rounds","fixpoint","accepted","rejected","llm_calls")}))
    for k in ("graph_search","cuda_learnings","capabilities","search_fixes"):
        v=s.get(k)
        if v: print(k, json.dumps(v,default=str)[:600])
    for ir,x in (s.get("irs") or {}).items(): print(ir, {k:x.get(k) for k in ("model_e2e_us","seed_e2e_us","passed","speedup")})
except Exception as e: print("no summary", e)
PY
done
echo DONE
