"""Scan every first-sweep final IR: apply each region pattern (and layout fold) via the repo functions and simulate the
walk_torch env discipline (ops in list order, sub_ops recursed) to find ops reading a tensor not yet defined."""
import sys, json, copy, glob, os
repo = sys.argv[1]
sys.path.insert(0, repo + "/src")
from modelblaster.pipeline import graph_hints, region_slots
from modelblaster.pipeline.kb_proposals import _all_specs
specs = _all_specs()

def sim(g):
    prod = set()
    def walk_prod(o):
        if "sub_ops" in o:
            for s in o["sub_ops"]: walk_prod(s)
            prod.update(o.get("internal", []))
        prod.update(o["outputs"])
    for o in g["ops"]: walk_prod(o)
    env = {t for t in g["tensors"] if t not in prod} | {g["input"]["tensor"]}
    bad = []
    def run(o):
        if "sub_ops" in o:
            for s in o["sub_ops"]: run(s)
            return
        for t in o["inputs"]:
            if t not in env: bad.append((o["name"], t))
        env.update(o["outputs"])
    for o in g["ops"]: run(o)
    return bad

for p in sorted(glob.glob("/Users/krisd/Documents/ModelBlaster-clone/l3_farm/results*/*/work/final/*/ir/graph.json")):
    name = p.split("/")[-5]; res = p.split("/")[-6]
    try: g = json.load(open(p))
    except Exception as e: print(name, "load", e); continue
    out = [res, name, "ops=%d" % len(g["ops"]), "base=%s" % (sim(g) or "ok")]
    for pat in ("rnn_stack", "attention_block", "all"):
        h = {"id": "region:" + pat, "kind": "fuse_region", "pattern": pat, "target": "cuda", "status": "accepted"}
        try:
            g2, _, prov = graph_hints.apply_hints(g, [h], dict(specs))
        except Exception as e:
            out.append("%s=EXC %r" % (pat, e)); continue
        if not prov["applied"]: out.append("%s=none" % pat); continue
        b = sim(g2)
        out.append("%s=%s" % (pat, ("BAD" + str(b[:2])) if b else "ok"))
    print(" | ".join(out), flush=True)
