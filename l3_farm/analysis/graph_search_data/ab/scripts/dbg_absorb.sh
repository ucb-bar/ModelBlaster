# Args: <stamp> <arm>. For the accepted hints of round 0: the atoms absorbed (graph_search.absorbed) vs the plan problems' atoms.
A=/tmp/krisd/a4/$1; D=/tmp/krisd/a4/runs/$2/work
. /tmp/krisd/venv/bin/activate; export PYTHONPATH=$A
cd $A && python3 - $D <<'PY'
import json, sys, pathlib
from modelblaster.pipeline import graph_hints as gh, graph_search as gsm
from modelblaster.pipeline.kb_proposals import _all_specs
from types import SimpleNamespace
d = pathlib.Path(sys.argv[1])
doc = gh.load(d / "graph_hints.json")
prog = json.loads((d / "r00" / "progress.json").read_text())["phases"]
profs = prog.get("gs_pre_profile", prog.get("gs_char_profile", prog.get("profile")))["value"]
p = profs[0]
g = json.loads((pathlib.Path(p["ir_dir"]) / "graph.json").read_text())
print("graph ops:", [(o["name"], o["op"]) for o in g["ops"] if o["op"] != "view"])
gs = gsm.GraphSearch(SimpleNamespace(**gsm.CFG_DEFAULTS, arch="89", knowledge_global_part="cuda:sm_89|fp16"), d, [])
specs = dict(_all_specs())
for h in doc["hints"]:
    if h["status"] == "accepted":
        job = {"hint": h, "ev": {"ir": p["ir"]}, "names": []}
        print(h["id"], "chains", h.get("chains"), "tried", h.get("tried_chains"), "->", sorted(gs.absorbed(job, doc, {p["ir"]: g}, specs)))
for m in sorted((d / "r00" / "problems").glob("*/manifest.json")):
    man = json.loads(m.read_text()); print(man["name"], [o["name"] for o in man["ops"]])
PY
