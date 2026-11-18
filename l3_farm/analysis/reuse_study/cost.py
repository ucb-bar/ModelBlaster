#!/usr/bin/env python3
"""LLM spend of the finished headline problems (state.json telemetry, assumed Opus-class list prices as analysis/cost_breakdown.py; per-slot-problem
attribution by prompt+completion tokens of r*/run/*/result.json) and the part of it spent on op kinds whose final ops have an exact / near-K foreign
donor that was already available early (delivered in the problem's seed). Lower bound / rough: only the share of kinds is attributed, not per shape."""
import json, glob, re, collections
from pathlib import Path
D = Path(__file__).resolve().parent / "data"
P = json.load(open(D / "problems.json")); C = json.load(open(D / "coverage.json"))
PRICE = dict(uncached=5.0, cache_read=0.5, cache_write=6.25, output=25.0)
def cost_of(c): return sum(c[t] * PRICE[t] / 1e6 for t in PRICE)
res = {}
tot = collections.Counter()
for p, d in P.items():
    W = Path(d["dir"]) / "work"
    pc = 0.0; calls = 0
    st = json.load(open(W / "state.json"))
    for r in st.get("rounds", []):
        llm = (((r.get("telemetry") or {}).get("kb") or {}).get("llm") or {})
        for br in llm.get("by_role_model", []):
            c = collections.Counter(output=br.get("output_tokens", 0), cache_write=br.get("cache_write_input_tokens", 0),
                                    cache_read=br.get("cache_read_input_tokens", 0), uncached=br.get("uncached_input_tokens", 0))
            pc += cost_of(c); calls += br.get("calls", 0)
    kind_tok = collections.Counter(); T = 0
    for f in glob.glob(str(W / "r[0-9][0-9]" / "run" / "*" / "result.json")):
        name = f.split("/")[-2]
        m = re.match(r"kb_problem__(.+?)_\d+__r\d+$", name)
        kind = m.group(1) if m else "other:" + name.split("__")[1] if "__" in name else "other"
        try:
            u = json.load(open(f)).get("llm_usage") or {}
        except ValueError:
            continue
        t = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
        kind_tok[kind] += t; T += t
    # kind-level early coverage from the final-graph ops (share-weighted fraction of the kind's time)
    kt = collections.Counter(); ke = collections.Counter(); kn = collections.Counter()
    for o in C[p]["ops"]:
        if o["ctype"] == "region": continue
        s = o["win"]; kt[o["kind"]] += o["share"]
        a_ex = s["avail_exact"] or {}; a_nr = s["avail_near"] or {}
        if s["n_exact"] and (a_ex.get("seed") or a_ex.get("created_before_start")): ke[o["kind"]] += o["share"]
        if s["dmin"] is not None and s["dmin"] <= 2.5 and (a_nr.get("seed") or a_nr.get("created_before_start")): kn[o["kind"]] += o["share"]
    ex = near = 0.0
    for kind, tk in kind_tok.items():
        base = kind.split("__")[0] if kind.startswith("fuse") else kind
        if kt.get(kind):
            f_ex, f_nr = ke[kind] / kt[kind], kn[kind] / kt[kind]
        else:
            f_ex = f_nr = 0.0
        w = pc * tk / T if T else 0
        ex += w * f_ex; near += w * f_nr
    res[p] = {"usd": round(pc, 1), "calls": calls, "usd_kinds_exact_early": round(ex, 1), "usd_kinds_near_early": round(near, 1)}
    for k, v in res[p].items(): tot[k] += v
json.dump(res, open(D / "cost.json", "w"), indent=1)
print({k: round(v) for k, v in tot.items()})
for p, r in sorted(res.items(), key=lambda kv: -kv[1]["usd_kinds_near_early"])[:12]: print(p, r)
