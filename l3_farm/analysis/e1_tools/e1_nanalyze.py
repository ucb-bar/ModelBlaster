#!/usr/bin/env python3
"""E1 nearest-K arm analysis: e1_data/runs_nearest/<P> (own session: orig + nk2/nk4/nk4b arms, PyTorch on the same GPU)
vs the main session's sib / glob arms (e1_data/e1_results.json). -> e1_data/e1_nearest_tables.md + e1_nearest.json"""
import json, math, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_analyze as A
A.STATES = ("orig", "nk2", "nk4", "nk4b")
D = A.DATA
RN = D / "runs_nearest"
main = {r["problem"]: r for r in json.load(open(D / "e1_results.json")) if r.get("arms")}
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
rows, out = [], []
P = out.append
foreign_wins, bad_pairs, slow_pairs, guard = [], [], [], []
tot_pairs = {"nk2": 0, "nk4": 0, "nk4b": 0}
for pd in sorted(RN.iterdir(), key=lambda p: (int(p.name.split("_")[0]), p.name)):
    p = pd.name
    s = A.session(pd)
    tj = A.load_torch(pd)
    if not s or not s["arms"] or p not in main or "t1" not in s["j"]:
        continue
    gs = [t["graph_us"] for t in tj if t.get("graph_us")]
    torch = st.median(gs) if gs else None
    m = main[p]
    row = {"problem": p, "torch": torch, "orig_n": s["arms"]["orig"]["med"]}
    sp = lambda us: torch / us if us and torch else None
    row["orig"] = m["vs_torch"]["orig"]
    row["orig_session2"] = sp(row["orig_n"])
    mt = m["torch_us"]
    for X in ("sib", "glob"):
        for suf in ("", "+auto"):
            c = [mt / m["arms"][k + suf]["med"] for k in (X, X + "f") if k + suf in m["arms"] and m["arms"][k + suf].get("med")]
            if c:
                row["m_" + X + suf] = max(c)
    for k in ("nk2", "nk2+auto", "nk4", "nk4+auto", "nk4b", "nk4b+auto"):
        if k in s["arms"] and A.arm_ok(k, s["bad"]):
            # speedup vs torch measured in this session, normalised by this session's orig so only the ratio carries over
            row[k] = row["orig"] * s["arms"]["orig"]["med"] / s["arms"][k]["med"]   # ratio to this session's own orig, transported to the main session
        elif k in s["arms"]:
            row[k + "_bad"] = s["bad"].get(k.split("+")[0] + "_seq") or "bad"
    for k in ("sib+auto", "glob+auto", "nk2+auto", "nk4+auto", "nk4b+auto"):
        pass
    # reuse passes: guard result and foreign winners
    for tag in ("nk2", "nk4", "nk4b"):
        r = s["j"]["reuse"].get(tag)
        if not r:
            continue
        guard.append((p, tag, "reverted: " + str(r.get("reverted"))[:60] if r.get("reverted") else ("applied" if r.get("changed") else "no change")))
        for kr in r["kinds"]:
            pool = {c["sha"]: c["from"] for c in kr.get("pool", [])}
            tab = kr.get("table") or {}
            ch = kr.get("choice") or {}
            for si, rw in tab.items():
                inc = (rw.get("incumbent") or {}).get("us")
                fbest = None
                for sha, cv_ in rw["cands"].items():
                    src = next((f for k2, f in pool.items() if k2.startswith(sha)), "?")
                    if not str(src).startswith("impl:"):
                        continue
                    tot_pairs[tag] += 1
                    if not cv_.get("ok"):
                        bad_pairs.append((p, tag, kr["kind"], si, src))
                    elif inc and cv_.get("us") and cv_["us"] > inc:
                        slow_pairs.append((p, tag, kr["kind"], si, src))
                    if cv_.get("ok") and cv_.get("us") and (fbest is None or cv_["us"] < fbest):
                        fbest = cv_["us"]
                c = ch.get(si)
                if c:
                    src = next((f for k2, f in pool.items() if k2.startswith(c)), "?")
                    if str(src).startswith("impl:"):
                        foreign_wins.append((p, tag, kr["kind"], rw.get("n", 1), src.split(":")[2], round(100 * (1 - (rw["cands"][c]["us"] or 0) / inc), 1) if inc else None,
                                             (kr.get("share"))))
    rows.append(row)
json.dump(rows, open(D / "e1_nearest.json", "w"), indent=1)

def f(x, p=2):
    return f"{x:.{p}f}x" if x else ""
ARMS = [("orig", "shipped"), ("m_sib+auto", "sib+auto (R1)"), ("m_glob+auto", "glob+auto (R2)"), ("nk2+auto", "nearest-2+auto"),
        ("nk4+auto", "nearest-4+auto"), ("nk4b+auto", "nearest-4 impl_pool metric+auto")]
P("| problem | " + " | ".join(n for _, n in ARMS) + " | best | wins |")
P("|---|" + "---|" * (len(ARMS) + 2))
for r in rows:
    vals = {k: r.get(k) for k, _ in ARMS}
    best = max((v, k) for k, v in vals.items() if v)
    P(f"| {r['problem']} | " + " | ".join(f(vals[k]) for k, _ in ARMS) + f" | {dict(ARMS)[best[1]]} | |")
P("")
P("AGG (same problems for every column)")
P("| arm | mean | geomean | n |\n|---|---|---|---|")
common = [r for r in rows if all(r.get(k) for k, _ in ARMS[:5])]
for k, n in ARMS:
    xs = [r[k] for r in common if r.get(k)]
    P(f"| {n} | {mean(xs):.3f}x | {geo(xs):.3f}x | {len(xs)} |")
P("")
for k, n in ARMS[3:5]:
    d = [(r[k] / r["m_sib+auto"] - 1) for r in common]
    P(f"{n} vs sib+auto: wins>2% {sum(x > .02 for x in d)}, losses<-2% {sum(x < -.02 for x in d)}, mean {100*mean(d):+.1f}%")
    d = [(r[k] / r["m_glob+auto"] - 1) for r in common]
    P(f"{n} vs glob+auto: wins>2% {sum(x > .02 for x in d)}, losses<-2% {sum(x < -.02 for x in d)}, mean {100*mean(d):+.1f}%")
P("")
P("GUARD: " + "; ".join(f"{p}/{t}: {g}" for p, t, g in guard if g != "applied" and g != "no change"))
import collections
P(f"\nFOREIGN WINS: shape slots where an other-problem candidate beat the incumbent by >= 3% (before the model guard): nk2 {sum(1 for w in foreign_wins if w[1]=='nk2')}, nk4 {sum(1 for w in foreign_wins if w[1]=='nk4')}")
agg = collections.defaultdict(lambda: [0, 0, collections.Counter(), collections.Counter()])
for p_, t, kind, nops, src, gain, share in foreign_wins:
    if t != "nk4":
        continue
    a_ = agg[p_]; a_[0] += 1; a_[1] += nops; a_[2][src] += 1; a_[3][kind] += nops
P("| problem (nk4) | foreign-chosen shape slots | ops covered | source problems | op kinds (ops) | pass outcome |\n|---|---|---|---|---|---|")
gd = {(p_, t): g for p_, t, g in guard}
for p_, (n, nops, src, kinds) in agg.items():
    P(f"| {p_} | {n} | {nops} | {', '.join(k.split('_')[0] + ' x' + str(v) for k, v in src.most_common(4))} | "
      f"{', '.join(k[:28] + ' ' + str(v) for k, v in kinds.most_common(3))} | {gd.get((p_, 'nk4'), '')} |")
for t in ("nk2", "nk4"):
    P(f"\n{t}: foreign (shape, candidate) pairs failing correctness: {sum(1 for b in bad_pairs if b[1]==t)}; correct but slower than the incumbent: "
      f"{sum(1 for b in slow_pairs if b[1]==t)} (of {tot_pairs[t]})")
fc = collections.Counter((b[0], b[1], b[2][:30], b[4].split(":")[1] + "@" + b[4].split(":")[2]) for b in bad_pairs)
P("correctness failures by (problem, pass, kind, candidate@source problem): shape slots failed")
for k, v in sorted(fc.items()):
    P(f"  {k}: {v}")
(D / "e1_nearest_tables.md").write_text("\n".join(out))
print("\n".join(out))
