"""Per-problem markdown table for LEVEL3_REPORT.md from the gate outputs.

    python level3_table.py IR_ROOT HOST_STATUS H3_JSON
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GENERIC = {"strided_copy", "gather", "catn", "ewise1", "ewise2", "ewise_scalar", "reduce",
           "softmax_dim", "conv2d_grouped", "maxpool2d_ex", "lstm_seq", "gru_seq"}
irr, host, h3 = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
hostv = {}
for line in open(host):
    p, r = line.rstrip("\n").split("\t")
    m = re.search(r"max_abs_err=(\S+)", r)
    hostv[p] = float(m.group(1)) if m else None
gpu = {r["ir"].removeprefix("kb_"): r for r in json.load(open(h3))}
print("| problem | IR ops | generic ops used | host C err/range | GPU err/range | GPU naive ms | torch eager ms |")
print("|---|---|---|---|---|---|---|")
for d in sorted(irr.glob("*/graph.json"), key=lambda p: int(p.parent.name.split("_")[0])):
    p = d.parent.name
    g = json.loads(d.read_text())
    c = Counter(o["op"] for o in g["ops"] if o["op"] != "view")
    gen = ", ".join(f"{k}×{v}" for k, v in sorted(c.items()) if k in GENERIC) or "—"
    rng = float(np.ptp(np.load(d.parent / "io.npz")["output"])) or 1.0
    he = hostv.get(p)
    r = gpu.get(p, {})
    ge = r.get("rel_to_range")
    ok = r.get("ok") and not r.get("first_divergent_op")
    print(f"| {p} | {sum(c.values())} | {gen} | "
          f"{'—' if he is None else f'{he / rng:.1e}'} | "
          f"{'—' if ge is None else f'{ge:.1e}'}{'' if ok else ' FAIL'} | "
          f"{r.get('model_ms', float('nan')):.1f} | {r.get('torch_eager_ms', float('nan')):.1f} |")
