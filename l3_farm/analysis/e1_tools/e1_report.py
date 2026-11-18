#!/usr/bin/env python3
"""E1 report tables (markdown) from e1_data/e1_results.json (run e1_analyze.py first) -> e1_data/e1_tables.md"""
import json, math, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "analysis" / "e1_data"
rows = json.load(open(DATA / "e1_results.json"))
good = [r for r in rows if r.get("arms")]
AUDIT = ["15_DenseNet121", "17_SqueezeNetFireModule", "31_VisionAttention", "50_ReLUSelfAttention", "49_Mamba2ReturnFinalState",
         "6_GoogleNetInceptionModule", "14_DenseNet121DenseBlock", "8_ResNetBasicBlock"]
out = []
P = out.append


def f0(x):
    return f"{x:,.0f}" if x is not None else ""


def f1(x):
    return f"{x:,.1f}" if x is not None else ""


def pc(x):
    return f"{100 * x:+.1f}%" if x is not None else ""


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def a(r, k):
    return r["arms"].get(k, {})


def okarm(r, k):
    s = k.split("@")[0].split("+")[0]
    need = [f"{s}_seq"] + ([f"{s}_conc"] if k.endswith(("+on", "+auto")) else [])
    return not any(n in r["bad"] for n in need)


def best_of(r, keys):
    c = [(a(r, k)["med"], k) for k in keys if k in r["arms"] and a(r, k).get("med") and okarm(r, k)]
    return min(c) if c else (None, None)


def sname(k):
    return k.replace("+auto", "+auto").replace("@2", "")


# ---------------------------------------------------------------- per problem table
P("| problem | PyTorch graph us | shipped us | vs torch | conc auto | reuse sib | reuse glob | best arm | best us | vs torch | gain | wins from | orig CV |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in good:
    o = r["orig"]
    ca = a(r, "orig+auto")
    sib, sibk = best_of(r, ["sib", "sibf"])
    glob, globk = best_of(r, ["glob", "globf"])
    tag = lambda k: "f" if k and k.endswith("f") else ""
    wins = []
    if r["win_reuse"]:
        wins.append("reuse" + ("(fix)" if (r["gain"].get("sibf") or 0) > r["noise_thr"] and (r["gain"].get("sib") or 0) <= r["noise_thr"]
                                 or (r["gain"].get("globf") or 0) > r["noise_thr"] and (r["gain"].get("glob") or 0) <= r["noise_thr"] else ""))
    if r["win_conc"]:
        wins.append("conc")
    bm = r["best"]
    P(f"| {r['problem']} | {f0(r['torch_us'])} | {f0(o)} | {r['vs_torch']['orig']:.2f}x | "
      f"{pc(o / ca['med'] - 1) if ca.get('med') and abs(o / ca['med'] - 1) > .003 else '0'} | "
      f"{pc(o / sib - 1) + tag(sibk) if sib and abs(o / sib - 1) > .003 else '0'} | "
      f"{pc(o / glob - 1) + tag(globk) if glob and abs(o / glob - 1) > .003 else '0'} | {sname(r['best_arm'])} | {f0(bm)} | "
      f"{r['vs_torch_best']:.2f}x | {pc(r['gain_best']) if abs(r['gain_best']) > .003 else '0'} | {' + '.join(wins) or '-'} | "
      f"{100 * (a(r, 'orig').get('cv') or 0):.2f}% |")

# ---------------------------------------------------------------- aggregates
def agg_row(label, fn):
    xs = [fn(r) for r in good]
    xs = [x for x in xs if x]
    return f"| {label} | {st.mean(xs):.3f}x | {geo(xs):.3f}x | {len(xs)} |"


P("")
P("AGG")
P("| arm | mean speedup vs PyTorch graph | geomean | n |")
P("|---|---|---|---|")
vs = lambda r, t: r["torch_us"] / t if t and r["torch_us"] else None
P(agg_row("shipped (re-measured today)", lambda r: r["vs_torch"].get("orig")))
P(agg_row("concurrency auto only", lambda r: r["vs_torch"].get("orig+auto")))
P(agg_row("reuse siblings only (as shipped; fix where it reverted)", lambda r: vs(r, best_of(r, ["sib", "sibf"])[0])))
P(agg_row("reuse global only (as shipped; fix where it reverted)", lambda r: vs(r, best_of(r, ["glob", "globf"])[0])))
P(agg_row("R1 = reuse siblings + concurrency auto", lambda r: vs(r, best_of(r, ["sib+auto", "sibf+auto"])[0])))
P(agg_row("R2 = reuse global + concurrency auto", lambda r: vs(r, best_of(r, ["glob+auto", "globf+auto"])[0])))
P(agg_row("best arm per problem (upper bound, selection bias <1%)", lambda r: r["vs_torch_best"]))
P(agg_row("as-shipped features only: best of sib/glob (+auto), no fix", lambda r: vs(r, best_of(r, ["orig+auto", "sib+auto", "glob+auto"])[0])))
P("")
P("n problems: %d; sweep-recorded numbers (validated.md) for the same 48 problems: geomean %s" % (len(good), "see doc"))

# ---------------------------------------------------------------- regressions
P("")
P("REGRESSIONS")
P("| problem | arm | change vs shipped | threshold | note |")
P("|---|---|---|---|---|")
nreg = 0
for r in good:
    o, thr = r["orig"], r["noise_thr"]
    for k, v in r["arms"].items():
        if not v.get("med") or k == "orig":
            continue
        d = v["med"] / o - 1
        if d > thr and okarm(r, k):
            nreg += 1
            P(f"| {r['problem']} | {k} | {pc(d)} | {100 * thr:.1f}% | {'conc arm' if 'on' in k.split('+')[-1:] else ''} |")
if not nreg:
    P("| none | | | | |")

# ---------------------------------------------------------------- correctness
P("")
P("CORRECTNESS")
P("| problem | arm | what failed |")
P("|---|---|---|")
nb = 0
for r in good:
    for k, w in sorted(r["bad"].items()):
        nb += 1
        P(f"| {r['problem']} | {k} | {w} |")
    for e in r.get("errors") or []:
        nb += 1
        P(f"| {r['problem']} | (script) {e['stage']} | {e['tb'][-200:].strip().replace(chr(10), ' ')} |")
if not nb:
    P("| none | | |")

# ---------------------------------------------------------------- reuse reverts
P("")
P("REUSE")
P("| problem | pass | result | guard before -> after (us) | kinds routed (matrix before -> after us) |")
P("|---|---|---|---|---|")
for r in good:
    for t, v in r["reuse"].items():
        res = "reverted: " + (v["reverted"] or "")[:40] if v.get("reverted") else ("applied" if v.get("changed") else "no change")
        g = v.get("guard") or [None, None]
        kinds = "; ".join(f"{k['kind'][:26]} {f0(k['before_us'])}->{f0(k['after_us'])}" for k in v["kinds"] if k.get("before_us"))
        P(f"| {r['problem']} | {t} | {res} | {f0(g[0])} -> {f0(g[1])} | {kinds} |")

# ---------------------------------------------------------------- audit
P("")
P("AUDIT")
P("| problem | seed-left share before | top seed kinds | shipped us | best reuse us (arm) | gain | vs torch before -> after |")
P("|---|---|---|---|---|---|---|")
for r in good:
    if r["problem"] not in AUDIT:
        continue
    ss = (r.get("seed_share") or {}).get("orig_seq") or {}
    kinds = ", ".join(f"{k} {100 * v:.0f}%" for k, v in (ss.get("by_kind") or {}).items())
    cand = [(a(r, k)["med"], k) for k in r["arms"] if k.split("@")[0] in ("sib", "glob", "sibf", "globf", "widef") and a(r, k).get("med") and okarm(r, k)]
    bm, bk = min(cand) if cand else (None, None)
    P(f"| {r['problem']} | {100 * (ss.get('total') or 0):.1f}% | {kinds} | {f0(r['orig'])} | {f0(bm)} ({bk}) | "
      f"{pc(r['orig'] / bm - 1) if bm else ''} | {r['vs_torch']['orig']:.2f}x -> {(r['torch_us'] / bm if bm and r['torch_us'] else 0):.2f}x |")

# ---------------------------------------------------------------- noise / repeatability
P("")
P("NOISE")
cvs = [(a(r, "orig").get("cv") or 0) for r in good]
tcv = [r["torch"].get("cv_graph") for r in good if r["torch"].get("cv_graph") is not None]
P(f"orig CV over 6 fresh-process runs: median {100 * st.median(cvs):.2f}%, p90 {100 * sorted(cvs)[int(.9 * len(cvs))]:.2f}%, max {100 * max(cvs):.2f}%")
P(f"PyTorch graph CV over 3 invocations: median {100 * st.median(tcv):.2f}%, max {100 * max(tcv):.2f}%")
rp = [(r["problem"], r["orig_repeat"]["diff_pct"]) for r in good if r.get("orig_repeat") and r["orig_repeat"].get("diff_pct") is not None]
if rp:
    P("orig repeated in the second session (other node): " + ", ".join(f"{p} {d:+.1f}%" for p, d in rp))
mem = {}
for r in good:
    s = r.get("smi") or {}
    mem.setdefault((s.get("max_mem_mhz"), s.get("vbios")), []).append(r["problem"])
P("GPU memory clock / VBIOS: " + "; ".join(f"{k}: {len(v)} problems" for k, v in mem.items()))
oth = [(r["problem"], (r.get("smi") or {}).get("others")) for r in good if (r.get("smi") or {}).get("others")]
P("other compute processes seen before a problem: " + (str(oth) if oth else "none"))

# ---------------------------------------------------------------- recoveries
P("")
P("RERUN")
big = [r for r in good if (r["gain_best"] or 0) >= 0.10]
mid = [r for r in good if .03 <= (r["gain_best"] or 0) < .10]
P(f">=10%: {len(big)}: " + ", ".join(f"{r['problem']} {pc(r['gain_best'])}" for r in big))
P(f"3-10%: {len(mid)}: " + ", ".join(f"{r['problem']} {pc(r['gain_best'])}" for r in mid))
(DATA / "e1_tables.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
