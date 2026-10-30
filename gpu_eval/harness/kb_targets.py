"""Pick one representative op instance per op type (the smallest output;
forward and reverse separately for the recurrent ops) across a set of IRs,
as IR_DIR:OP_NAME@INDEX targets for kb_bridge.py (one fx node can lower to
several IR ops sharing its name).

    python kb_targets.py IR_ROOT > targets.txt
"""
import json
import sys
from pathlib import Path

best = {}
for g in sorted(Path(sys.argv[1]).glob("*/graph.json")):
    ir = json.loads(g.read_text())
    for i, op in enumerate(ir["ops"]):
        if op["op"] in ("view", "chunk2_c1"):
            continue
        n = 1
        for d in ir["tensors"][op["outputs"][0]]["shape"]:
            n *= d
        key = (op["op"], op.get("shape", {}).get("reverse", 0))
        if key not in best or n < best[key][0]:
            best[key] = (n, f"{g.parent}:{op['name']}@{i}")
print("\n".join(v[1] for _, v in sorted(best.items())))
