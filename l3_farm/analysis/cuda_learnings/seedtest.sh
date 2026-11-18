# Zero-LLM GPU test of --reuse-winners seeding: a 39_GRU plan problem (the 5 GRU ops of layers 1-5, hand seeds) gets
# seed candidates from the pseudo-global impl store (the sweep winners of 38/39/41/42); KernelBlaster (0 steps)
# must try them at start-up and keep the fastest passing one.
. /tmp/krisd/cl/env.sh
cd /tmp/krisd/cl
rm -rf seedtest && mkdir -p seedtest
python - <<'PY'
import json
from pathlib import Path
from modelblaster.pipeline.kb_export import export_problem
from modelblaster.pipeline.kb_ingest import CudaCache
from modelblaster.pipeline.kb_proposals import _all_specs
from modelblaster.pipeline import kb_reuse
import os
ir = Path(os.environ.get("SEED_IR", "runs/39_GRU.siblings/ir"))
g = json.loads((ir / "graph.json").read_text())
idx = [i for i, o in enumerate(g["ops"]) if o["op"] == "gru_seq_f16" and o["shape"]["I"] != 128]
p = export_problem(ir, Path("seedtest/problem"), op_indices=idx, specs=dict(_all_specs()), name="seedtest_39_gru",
                   strategy="plan")
cache = CudaCache(Path("seedtest/cache"))      # empty: the slot starts from the hand kernel
r = kb_reuse.add_seed_candidates(p, cache, arch="89", specs=dict(_all_specs()), mode="global",
                                 impl_dirs=[Path("data/gimpl")], n=3)
print("seed candidates", json.dumps(r))
PY
KB=/tmp/krisd/cl/code/ModelBlaster/third_party/KernelBlaster
cd $KB && timeout 1800 python scripts/run_multislot.py /tmp/krisd/cl/seedtest/problem --out /tmp/krisd/cl/seedtest/run \
  --knowledge-dir /tmp/krisd/cl/seedtest/knowledge --model azure/anthropic/claude-opus-5-5 --arch 89 --rollouts 1 --steps 0 \
  --gpus 0 --knowledge v1 --ncu off > /tmp/krisd/cl/seedtest/run.log 2>&1
echo "rc $?"
grep -i "seed candidate\|baseline\|store seed" /tmp/krisd/cl/seedtest/run.log | head -20
python3 -c "
import json,glob
for f in glob.glob('/tmp/krisd/cl/seedtest/run/*/result.json'):
    r=json.load(open(f)); print({k:r.get(k) for k in ('baseline_partition_us','final_partition_us','baseline_e2e_us','final_e2e_us','llm_calls','final_ok')})
"
