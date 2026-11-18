#!/usr/bin/env python3
"""Part 1 tables from data/coverage.json (pool tier 'win' unless noted). Writes data/tables.md and data/summary.json."""
import json, collections, statistics as st
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
C = json.load(open(D / "coverage.json"))
P = json.load(open(D / "problems.json"))
DNEAR = 2.5      # "same-signature other-shape" donors count up to this mean-|log2| shape distance
TIER = "win"
out = []
summ = {}


def lvl(s):
    if s["n_exact"]:
        return "b"
    if s["dmin"] is not None and s["dmin"] <= DNEAR:
        return "c"
    return "d"


def pct(x):
    return f"{100 * x:4.1f}"


rows = []
agg = collections.Counter()
timing = collections.Counter()
matrix = collections.defaultdict(collections.Counter)      # consumer -> donor origin -> share (top-1 donor of S ops and exact A ops)
donor_tot = collections.Counter()
seedfix = []
for p, d in C.items():
    c = collections.Counter()
    t = collections.Counter()
    for o in d["ops"]:
        sh = o["share"]
        if o["ctype"] == "region":
            c["region"] += sh
            continue
        isS = o["ctype"] == "seed" or o["weak"]
        key = "S" if isS else ("Ain" if o["ctype"] == "foreign-in" else "Aown")
        c[key] += sh
        s = o[TIER]
        L = lvl(s)
        c[key + L] += sh
        if L in "bc":
            av = (s["avail_exact"] if L == "b" else s["avail_near"]) or {}
            t[av.get("cls", "?")] += sh
            t["cov"] += sh
            if isS:
                t["S_" + av.get("cls", "?")] += sh
                t["Scov"] += sh
        if isS and L in "bc" and "top1" in o:
            tp = o["top1"]
            matrix[p][tp["origin"]] += sh
            donor_tot[tp["origin"]] += sh * 1.0
            seedfix.append({"problem": p, "op": o["name"], "kind": o["kind"], "share": round(sh, 4), "ctype": o["ctype"],
                            "weak": o["weak"], "cur_sp": o["speedup_vs_seed"], "level": L, "n_exact": s["n_exact"], "dmin": s["dmin"],
                            "top1": tp, "avail": av.get("cls"),
                            "pred_sp_exact": s["exact_best_speedup"]})
        elif (not isS) and s["n_exact"] and "top1" in o:
            matrix[p][o["top1"]["origin"]] += 0  # keep key
    rows.append((p, c, t))
    summ[p] = {"c": dict(c), "t": dict(t)}

# ---- table 1
out.append("| problem | e2e us | a own | a foreign-in | seed/weak | region | S->exact (b) | S->near (c) | S->none (d) | any foreign donor (b+c, all ops) | b-covered all ops |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for p, c, t in rows:
    sP = c["S"]
    nr = 1 - c["region"]
    out.append(f"| {p} | {C[p]['e2e_us']:.0f} | {pct(c['Aown'])} | {pct(c['Ain'])} | {pct(sP)} | {pct(c['region'])} | {pct(c['Sb'])} | {pct(c['Sc'])} | {pct(c['Sd'])} | {pct(t['cov'])} | {pct(c['Aownb'] + c['Ainb'] + c['Sb'])} |")
open(D / "table_cov.md", "w").write("\n".join(out))

# ---- availability table
out2 = ["| problem | covered share (b+c) | delivered at seed | via refresh | existed pre-start but never delivered | created during run | only after finish | S-ops covered: seed/refresh/other |",
        "|---|---|---|---|---|---|---|---|"]
tot = collections.Counter()
for p, c, t in rows:
    cv = t["cov"]
    if cv <= 0:
        continue
    out2.append(f"| {p} | {pct(cv)} | {pct(t['seed'])} | {pct(t['refresh'])} | {pct(t['pre_start_undelivered'])} | {pct(t['during_run'])} | {pct(t['after_finish'])} | {pct(t['S_seed'])}/{pct(t['S_refresh'])}/{pct(t['Scov'] - t['S_seed'] - t['S_refresh'])} |")
    for k, v in t.items():
        tot[k] += v
n = len(rows)
out2.append(f"| MEAN over {n} | {pct(tot['cov'] / n)} | {pct(tot['seed'] / n)} | {pct(tot['refresh'] / n)} | {pct(tot['pre_start_undelivered'] / n)} | {pct(tot['during_run'] / n)} | {pct(tot['after_finish'] / n)} | |")
open(D / "table_avail.md", "w").write("\n".join(out2))

# ---- seed-left fixable list
seedfix.sort(key=lambda x: -x["share"])
json.dump(seedfix, open(D / "seedfix.json", "w"), indent=1)
o3 = ["| problem | op (kind) | share | currently | exact donors | nearest donor (origin, dist, rec speedup, delivered) | availability |", "|---|---|---|---|---|---|---|"]
for x in seedfix[:40]:
    tp = x["top1"]
    o3.append(f"| {x['problem']} | {x['op']} ({x['kind']}) | {pct(x['share'])} | {'seed' if x['ctype']=='seed' else 'weak '+str(round(x['cur_sp'] or 0,2))+'x'} | {x['n_exact']} | {tp['origin']}, d={tp['dist']}, {tp['sp']:.2f}x, {tp['delivered']} | {x['avail']} |")
open(D / "table_seedfix.md", "w").write("\n".join(o3))

# ---- donor -> consumer matrix (share of consumer's S-op time covered with top-1 donor)
don = collections.Counter()
for cons, m in matrix.items():
    for dn, v in m.items():
        don[dn] += v
o4 = ["donor origin | S-share covered (sum over consumers) | consumers (share)", "---|---|---"]
for dn, v in don.most_common(15):
    cons = sorted(((m[dn], c) for c, m in matrix.items() if m.get(dn)), reverse=True)[:5]
    o4.append(f"{dn} | {v:.3f} | " + ", ".join(f"{c.split('_')[0]}:{s:.2f}" for s, c in cons))
open(D / "table_matrix.md", "w").write("\n".join(o4))
json.dump({k: {a: round(b, 4) for a, b in v.items()} for k, v in matrix.items()}, open(D / "matrix_S_top1.json", "w"), indent=1)
json.dump(summ, open(D / "summary.json", "w"), indent=1)
print("\n".join(out) + "\n\n" + "\n".join(out2) + "\n\n" + "\n".join(o3) + "\n\n" + "\n".join(o4))
