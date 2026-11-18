#!/usr/bin/env python3
"""Markdown tables for sonnet_test.md from res/<arm>/analysis.json + remeasure + cross-node results."""
import json, statistics as S
from pathlib import Path
DATA = Path(__file__).resolve().parent.parent
CTRL = Path("/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad/bt/res")
ARMS = ["S3", "S5", "S0", "H5"]
A = {a: json.load(open(DATA / "res" / a / "analysis.json")) for a in ARMS}
for c in ("C2", "C3", "H"):
    A[c] = json.load(open(CTRL / c / "analysis.json"))
RM = {a: json.load(open(next((DATA / "res").glob(f"remeasure_{a}_on_*.json")))) for a in ARMS}
cross = json.load(open(DATA / "res" / "cross_one_node.json"))["results"] if (DATA / "res" / "cross_one_node.json").exists() else {}


def roles(a):
    out = {}
    for k, v in A[a]["ledger_what"].items():
        r, m = k.split("|")
        out[(r, m)] = v
    return out


print("| arm | calls (total) | by model | $ total | $ by model | wall min |")
print("|---|---|---|---|---|---|")
for a in ["C2", "C3", "H"] + ARMS:
    L = A[a]["ledger"]
    cm = ", ".join(f"{m.split('-')[-3] if False else m.split('claude-')[1][:-4] if False else m.split('/')[-1].replace('claude-','').replace('-5-5','')} {int(v['calls'])}" for m, v in L.items())
    dm = ", ".join(f"{m.split('/')[-1].replace('claude-','').replace('-5-5','')} ${v['usd']:.2f}" for m, v in L.items())
    print(f"| {a} | {A[a]['ledger_total']['calls']} | {cm} | {A[a]['ledger_total']['usd']:.2f} | {dm} | {A[a]['wall_s']/60:.1f} |")
print()
print("| arm | codegen_simple | codegen_hard | fix | reflect | hypothesis+discovery |")
print("|---|---|---|---|---|---|")
for a in ["C2", "C3", "H"] + ARMS:
    R = roles(a)
    def g(name):
        rows = [(k, v) for k, v in R.items() if k[0] == name]
        return f"{sum(v['calls'] for k, v in rows)} / ${sum(v['usd'] for k, v in rows):.2f}" + (" (" + "+".join(sorted({k[1] for k, v in rows})) + ")" if rows else "")
    up = [(k, v) for k, v in R.items() if k[0] in ("hypothesis", "discovery")]
    print(f"| {a} | {g('codegen_simple')} | {g('codegen_hard')} | {g('fix')} | {g('reflect')} | {sum(v['calls'] for k,v in up)} / ${sum(v['usd'] for k,v in up):.2f} |")
print()
print("| arm | final model us (flow) | quiet own node | quiet common node | torch graph / compile (own node) | vs graph | vs compile |")
print("|---|---|---|---|---|---|---|")
for a in ARMS:
    r = RM[a]
    cn = cross.get(a, {}).get("model_e2e_us") if a != "S3" else cross.get("S3", {}).get("model_e2e_us")
    print(f"| {a} | {A[a]['final']['model_e2e_us']:.0f} | {r['model_e2e_us']:.0f} | {cn if cn is None else round(cn)} | {r['torch_graph_us']:.0f} / {r['torch_compile_us']:.0f} | {r['torch_graph_us']/r['model_e2e_us']:.3f}x | {r['torch_compile_us']/r['model_e2e_us']:.3f}x |")
if cross:
    print("\ncommon node cross-measurements:", {k: round(v["model_e2e_us"]) for k, v in cross.items()},
          "torch graph", {k: round(v["torch_graph_us"]) for k, v in cross.items()})
