"""Kind-match dry run (zero LLM): which kind-gated seed techniques become available for the slot kinds of the Level 3
sweep with KERNELBLASTER_KIND_MATCH=base. Replicates ontology._kind_match / _base_kind (KernelBlaster knowledge/v2/ontology.py)."""
import json, glob, collections, os
KB = os.path.expanduser("~/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/data/kernelblaster/knowledge/techniques")
RES = os.path.expanduser("~/Documents/ModelBlaster-clone/l3_farm/results")

def base_kind(k):
    for s in ("_f16", "_s8_pc", "_s8", "_i8"):
        if k.endswith(s): return k[:-len(s)]
    return k
def stages(k):
    st = k.split("__")[1:] if k.startswith("fused__") else [k]
    return [s for s in st if s and s != "par"] or st
def match(k, kinds, base):
    st = stages(k)
    if base: st = st + [base_kind(s) for s in st]
    return any(x in kinds for x in st)

# techniques with op_kinds gates
techs = []
for f in sorted(glob.glob(KB + "/*.json")):
    d = json.load(open(f))
    items = d if isinstance(d, list) else d.get("techniques", d.get("items", []))
    for t in items:
        if isinstance(t, dict) and t.get("op_kinds"):
            techs.append({"name": t.get("name") or t.get("id"), "kinds": t["op_kinds"], "classes": t.get("op_classes"), "file": os.path.basename(f)})
print("gated techniques:", len(techs))

# slot kinds of the sweep: ops in the final graphs
kinds = collections.Counter(); per_problem = collections.defaultdict(set)
for g in glob.glob(RES + "/*/work/final/*/ir/graph.json") + glob.glob(RES + "/*/work/final/*/graph.json") + glob.glob(RES + "/*/irs/*/graph.json"):
    p = g.split("/results/")[1].split("/")[0]
    try: G = json.load(open(g))
    except Exception: continue
    ops = G.get("ops") or G.get("nodes") or []
    for o in ops:
        k = (o.get("op") or o.get("kind")) if isinstance(o, dict) else None
        if k and k.endswith("_f16") or (k and k.startswith("fused__")):
            kinds[k] += 1; per_problem[k].add(p)
print("distinct slot kinds seen:", len(kinds))

rows = []
for k in sorted(kinds):
    new = [t["name"] for t in techs if match(k, t["kinds"], True) and not match(k, t["kinds"], False)]
    old = [t["name"] for t in techs if match(k, t["kinds"], False)]
    if new: rows.append({"kind": k, "problems": len(per_problem[k]), "instances": kinds[k], "newly_available": new, "already_available": old})
by_tech = collections.defaultdict(list)
for r in rows:
    for n in r["newly_available"]: by_tech[n].append(r["kind"])
out = {"gated_techniques": techs, "kinds_affected": rows, "technique_to_kinds": {n: sorted(v) for n, v in by_tech.items()},
       "n_kinds_seen": len(kinds), "n_kinds_affected": len(rows),
       "problems_affected": sorted({p for r in rows for p in per_problem[r["kind"]]})}
json.dump(out, open("dryrun.json", "w"), indent=1)
print("kinds affected:", len(rows), "problems affected:", len(out["problems_affected"]))
for r in sorted(rows, key=lambda r: -r["instances"])[:25]:
    print(f'{r["kind"]:55s} problems={r["problems"]:2d} inst={r["instances"]:4d} +{r["newly_available"]}')
print("techniques newly applicable:", {n: len(v) for n, v in by_tech.items()})
