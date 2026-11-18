. /tmp/krisd/cpool/env.sh
bash /tmp/krisd/cpool/ci.sh 2>&1 | grep -v Ignoring; export PYTHONPATH=/tmp/krisd/cpool/code/ModelBlaster/third_party/KernelBlaster/src:$PYTHONPATH; mkdir -p /tmp/krisd/cpool/e2/units; cd /tmp/krisd/cpool/e2
python - <<'P'
import sys, shutil
from pathlib import Path
from kernelblaster.knowledge.slots import SlotProblem
C = Path("/tmp/krisd/cpool/corpus"); out = Path("/tmp/krisd/cpool/e2/units")
import json
idx = json.loads((C/"index.json").read_text())
seen = set(); n = 0
for P in ("15_DenseNet121", "26_ShuffleNet", "44_MiniGPTBlock", "30_SwinTransformerV2"):
    for s in idx[P]:
        p = SlotProblem(C/P/s)
        for name, text in p.units(p.seed_sources()):
            h = hash(text)
            if h in seen: continue
            seen.add(h); (out/f"u{n:04d}.cu").write_text(text); n += 1
print("units", n)
P
