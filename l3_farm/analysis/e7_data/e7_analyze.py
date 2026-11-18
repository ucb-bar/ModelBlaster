#!/usr/bin/env python3
"""E7 (X-101): efficiency replay of the Level 3 sweep. Reads the CSVs written by e7_build.py and writes
  e7_results.json (all numbers quoted in e7_efficiency.md), sim_grid.csv (combined-policy replay per problem),
  and PNG plots in l3_farm/analysis/ (e7_*.png).
Run with the matplotlib venv:  <viz>/bin/python e7_analyze.py
"""
import csv, json, math, os, collections, itertools, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.dirname(HERE)
rd = lambda fn: list(csv.DictReader(open(os.path.join(HERE, fn))))
F = lambda x, d=None: (float(x) if x not in ("", None) else d)
STEPS, KB, ROUNDS, PROBS, DECS = rd("steps.csv"), rd("kbruns.csv"), rd("rounds.csv"), rd("problems.csv"), rd("decisions.csv")
JOIN = json.load(open(os.path.join(HERE, "join_check.json")))
OUT = {}
KNOW_ROLES = ("reflect", "hypothesis", "discovery")
FAMS = ["CNN", "Attention/transformer", "Recurrent", "MLP", "Other"]

# ------------------------------------------------------------------ problems / sample
prob = {}
for p in PROBS:
    if p["sample"] not in ("main",): continue
    parts = [F(p.get(f"part_r{r}")) for r in range(6) if p.get(f"part_r{r}") not in (None, "")]
    prob[p["problem"]] = dict(family=p["family"], first=F(p["first_us"]), final=F(p["final_us"]), parts=parts,
                              n_rounds=int(p["n_rounds"]))
MAIN = sorted(prob, key=lambda n: int(n.split("_")[0]))
# round 0 of these two was lost to infra (node expiry / resume without the winners cache): the measured model time after
# round 0 equals the seed, and round 1 redid the work. Kept for step-level stats, excluded from round-level and replay stats.
ROUND0_LOST = {"10_ResNet101", "22_EfficientNetB0"}
REPLAY = [p for p in MAIN if p not in ROUND0_LOST]
OUT["sample"] = dict(n_main=len(MAIN), n_replay=len(REPLAY), round0_lost=sorted(ROUND0_LOST), excluded=[p["problem"] + ":" + p["sample"] for p in PROBS if p["sample"] != "main"])

# knowledge-role spend per (problem, round) from telemetry; total telemetry spend by sample
know = collections.defaultdict(float); tel_round = collections.defaultdict(float); tel_by_sample = collections.Counter()
for r in ROUNDS:
    k = (r["problem"], int(r["round"]))
    know[k] = sum(F(r.get(f"usd_{x}"), 0) for x in KNOW_ROLES)
    tel_round[k] = F(r["usd"], 0); tel_by_sample[r["sample"]] += F(r["usd"], 0)

# ------------------------------------------------------------------ kb runs with trajectories
steps_by_kb = collections.defaultdict(list)
for s in STEPS: steps_by_kb[(s["problem"], s["round"], s["kb_run"])].append(s)
kbs = []
for k in KB:
    ss = steps_by_kb.get((k["problem"], k["round"], k["kb_run"]), [])
    real = [s for s in ss if s["status"] != "orphan"]
    real.sort(key=lambda s: int(s["order"]))
    st = []
    for s in real:
        st.append(dict(order=int(s["order"]), ro=int(s["rollout"]), st=int(s["step"]), usd=F(s["usd"], 0), fix_usd=F(s["fix_usd"], 0),
                       status=s["status"], slot=s["slot"], after=F(s["slot_us_after"]), nb=int(s["new_best"]),
                       delta=F(s["delta_best_us"], 0), tech=s["technique_group"], ts=F(s["ts"])))
    base = {}
    for s in real:
        if s["slot"] not in base: base[s["slot"]] = F(s["slot_base_us"])
    best = dict(base)
    for s in st:
        if s["nb"] and s["after"] is not None: best[s["slot"]] = min(best[s["slot"]], s["after"])
    fa = F(k["fused_away_frac"])
    kbs.append(dict(problem=k["problem"], sample=k["sample"], round=int(k["round"]), phase=k["phase"], kb=k["kb_run"], side=k["variant_side"],
                    hint=k["hint"], decision=k["decision"], share=F(k["share"], 0.0), op_class=k["op_class"], op_kind=k["op_kind"],
                    base=base, best_full=best, w=(1 - fa) if fa is not None else 1.0, fused_away=fa, steps=st,
                    orphan_usd=sum(F(s["usd"], 0) for s in ss if s["status"] == "orphan"), usd=F(k["usd"], 0),
                    base_part=F(k["baseline_partition_us"]), final_part=F(k["final_partition_us"])))
# fusion candidates: op kinds named in any fusion hint tried for the problem (hints are known before round 0: graph_hints.json)
hint_ops = collections.defaultdict(set)
for k in kbs:
    if k["hint"]:
        for part in k["hint"].split(":", 1)[1].split("__"):
            if part.endswith("_f16"): hint_ops[k["problem"]].add(part)
for k in kbs:
    k["fusion_cand"] = k["phase"] == "opt" and k["round"] == 0 and k["op_kind"] in hint_ops[k["problem"]]
def gain_full(kb):   # model-us saved by this kb run on its slot(s) (attribution level), >=0
    return sum(max(kb["base"][s] - kb["best_full"][s], 0) for s in kb["base"] if kb["base"][s] is not None)

# ------------------------------------------------------------------ (1) join validation & spend overview
sp = collections.Counter(); calls = collections.Counter()
for s in STEPS:
    sp[(s["sample"], s["phase"], "orphan" if s["status"] == "orphan" else "step")] += F(s["usd"], 0)
    calls[s["sample"]] += int(s["n_calls"] or 0)
OUT["join"] = dict(JOIN["totals"], step_rows=JOIN["step_rows"], journal_usd=JOIN["step_usd"], telemetry_usd=JOIN["tel_usd"],
                   tel_usd_by_sample=dict(tel_by_sample), journal_usd_by=dict((":".join(k), v) for k, v in sp.items()),
                   journal_calls_by_sample=dict(calls))
mism = []
for pid, v in JOIN["per_problem"].items():
    if v["tel_calls_codegen_fix"] and abs(v["journal_calls"] - v["tel_calls_codegen_fix"]) > 0.15 * v["tel_calls_codegen_fix"]:
        mism.append((pid, v["journal_calls"], v["tel_calls_codegen_fix"]))
OUT["join"]["large_mismatch"] = mism

# problem total spend (denominator): journal step+orphan spend + knowledge-role telemetry spend
def P_kbs(p): return [k for k in kbs if k["problem"] == p and k["sample"] == "main"]
ptot = {}; pround_usd = {}
for p in MAIN:
    ru = collections.Counter()
    for k in P_kbs(p): ru[k["round"]] += k["usd"]
    for r in range(prob[p]["n_rounds"]): ru[r] += know[(p, r)]
    pround_usd[p] = ru; ptot[p] = sum(ru.values())
TOT = sum(ptot.values())
OUT["spend_main_usd"] = TOT

# ------------------------------------------------------------------ (2) round value (measured)
def m_after(p, r):
    pr = prob[p]; return pr["parts"][r + 1] if r + 1 < len(pr["parts"]) else pr["final"]
round_tab = []
for r in range(4):
    usd_r = sum(pround_usd[p].get(r, 0) for p in REPLAY)
    ps = [p for p in REPLAY if prob[p]["n_rounds"] > r]
    logsh = [math.log(prob[p]["parts"][r] / m_after(p, r)) / math.log(prob[p]["first"] / prob[p]["final"]) for p in ps]
    rel = [1 - m_after(p, r) / prob[p]["parts"][r] for p in ps]
    round_tab.append(dict(round=r, n_problems=len(ps), usd=usd_r, usd_pct=usd_r / sum(ptot[p] for p in REPLAY),
                          usd_pct_all_main=sum(pround_usd[p].get(r, 0) for p in MAIN) / TOT,
                          mean_logshare=float(np.mean(logsh)), median_rel_improvement=float(np.median(rel)),
                          mean_rel_improvement=float(np.mean(rel)), share_below_1pct=float(np.mean([x < 0.01 for x in rel])),
                          share_below_3pct=float(np.mean([x < 0.03 for x in rel]))))
round_tab = [r for r in round_tab if r["n_problems"] > 0]
OUT["rounds"] = round_tab
# "after round 0, how far from final": final/part[1]
OUT["after_round0_to_final"] = dict(
    median_ratio=float(np.median([prob[p]["final"] / prob[p]["parts"][1] for p in REPLAY if len(prob[p]["parts"]) > 1])),
    per_problem={p: prob[p]["final"] / prob[p]["parts"][1] for p in REPLAY if len(prob[p]["parts"]) > 1})

# ------------------------------------------------------------------ (2) marginal value curve per problem
def curve(p):
    """cumulative spend fraction -> model time, calibrated to the measured per-round model times."""
    pr = prob[p]; pts = [(0.0, pr["first"])]; cum = 0.0
    for r in range(pr["n_rounds"]):
        ev = []
        for k in P_kbs(p):
            if k["round"] != r: continue
            last_ts = 0
            for s in k["steps"]:
                ts = s["ts"] or last_ts; last_ts = ts
                e = 0.0
                if k["phase"] == "opt" and s["nb"]: e = s["delta"]   # no fused-away weight: the round-end model still ran these kernels
                ev.append((ts, s["usd"], e))
            if k["orphan_usd"]: ev.append((last_ts, k["orphan_usd"], 0.0))
        ev.sort(key=lambda x: x[0])
        tot_u = sum(x[1] for x in ev) + know[(p, r)]; tot_e = sum(x[2] for x in ev)
        m0, m1 = pr["parts"][r], m_after(p, r)
        ku = know[(p, r)] / max(sum(x[1] for x in ev), 1e-9)
        ce = cu = 0.0
        for ts, u, e in ev:
            cu += u * (1 + ku); ce += e
            frac = (ce / tot_e) if tot_e > 0 else (cu / max(tot_u, 1e-9))
            pts.append((cum + cu, m0 - (m0 - m1) * frac))
        cum += tot_u
        pts.append((cum, m1))
    T = pts[-1][0]
    return [(x / T, m) for x, m in pts]
grid = np.linspace(0, 1, 101)
curves = {}
for p in REPLAY:
    c = curve(p); xs = np.array([x for x, _ in c]); ms = np.array([m for _, m in c])
    pr = prob[p]; L = math.log(pr["first"] / pr["final"])
    # step function: model time reached by spend fraction g
    f = []
    for g in grid:
        i = np.searchsorted(xs, g, side="right") - 1
        f.append(math.log(pr["first"] / max(ms[: i + 1].min(), 1e-9)) / L if L > 0 else 1.0)
    # time-saved fraction
    tsf = [(pr["first"] - max(pr["first"] * math.exp(-v * L), 0)) / (pr["first"] - pr["final"]) for v in f]
    curves[p] = dict(log=np.array(f), saved=np.array(tsf), final_over_m=np.array([pr["final"] / (pr["first"] * math.exp(-v * L)) for v in f]))
def at(arr_key, g, ps):
    i = int(round(g * 100)); return [curves[p][arr_key][i] for p in ps]
mv = {}
for g in (0.1, 0.25, 0.5, 0.75, 0.9):
    mv[f"{int(g*100)}pct_spend"] = dict(
        median_logshare=float(np.median(at("log", g, REPLAY))), median_timesaved=float(np.median(at("saved", g, REPLAY))),
        median_speedup_retained=float(np.median(at("final_over_m", g, REPLAY))),
        by_family={f: float(np.median(at("final_over_m", g, [p for p in REPLAY if prob[p]["family"] == f]))) for f in FAMS
                   if any(prob[p]["family"] == f for p in REPLAY)})
OUT["marginal"] = mv
# spend needed to reach 90/95/99% of final speedup (final/M >= x)
need = {}
for thr in (0.5, 0.8, 0.9, 0.95, 0.99):
    v = []
    for p in REPLAY:
        a = curves[p]["final_over_m"]; i = int(np.argmax(a >= thr - 1e-12)); v.append(grid[i])
    need[str(thr)] = dict(median=float(np.median(v)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)))
OUT["spend_to_reach_fraction_of_final_speed"] = need

# ------------------------------------------------------------------ (2) where the spend went: productive vs waste
cat = collections.Counter(); cat_calls = collections.Counter()
for k in kbs:
    if k["sample"] != "main": continue
    cat["orphan (no committed step)"] += k["orphan_usd"]
    if k["phase"] == "variant":
        cat["variant search, decision accepted" if k["decision"] == "accepted" else "variant search, decision rejected/none"] += k["usd"] - k["orphan_usd"]
        continue
    if not k["steps"]: cat["opt run without steps"] += k["usd"] - k["orphan_usd"]; continue
    g = gain_full(k)
    stp = k["usd"] - k["orphan_usd"]
    if g <= 0: cat["opt: slot never improved"] += stp; continue
    if k["fused_away"] is not None and k["fused_away"] >= 0.999: cat["opt: improved, op later fused away"] += stp; continue
    # split productive run into: steps up to the final best, and the tail after it
    last = max(i for i, s in enumerate(k["steps"]) if s["nb"])
    head = sum(s["usd"] for s in k["steps"][: last + 1]); tail = sum(s["usd"] for s in k["steps"][last + 1:])
    if k["share"] < 0.005: cat["opt: improved, seed-time share < 0.5%"] += stp; continue
    cat["opt: improved slot, steps up to final best"] += head; cat["opt: improved slot, tail after final best"] += tail
cat["knowledge roles (reflect/hypothesis/discovery)"] = sum(know[(p, r)] for p in MAIN for r in range(prob[p]["n_rounds"]))
OUT["waste"] = {k: dict(usd=v, pct=v / TOT) for k, v in sorted(cat.items(), key=lambda x: -x[1])}
# step-level productivity
opt_steps = [s for k in kbs if k["sample"] == "main" and k["phase"] == "opt" for s in k["steps"]]
n_nb = sum(s["nb"] for s in opt_steps)
final_best_steps = 0; final_best_usd = 0.0
for k in kbs:
    if k["sample"] != "main" or k["phase"] != "opt" or not k["steps"]: continue
    nbs = [i for i, s in enumerate(k["steps"]) if s["nb"]]
    if nbs: final_best_steps += 1; final_best_usd += k["steps"][nbs[-1]]["usd"]
OUT["step_productivity"] = dict(opt_steps=len(opt_steps), new_best_steps=n_nb, new_best_frac=n_nb / len(opt_steps),
                                final_best_steps=final_best_steps, final_best_frac=final_best_steps / len(opt_steps),
                                final_best_usd_frac=final_best_usd / sum(s["usd"] for s in opt_steps),
                                status=dict(collections.Counter(s["status"] for s in opt_steps)))
# never-improved / fused-away calls share (refines the ~17% estimate)
def calls_of(k): return sum(1 for _ in k["steps"])
never = [k for k in kbs if k["sample"] == "main" and k["phase"] == "opt" and k["steps"] and gain_full(k) <= 0]
fused = [k for k in kbs if k["sample"] == "main" and k["phase"] == "opt" and k["steps"] and gain_full(k) > 0 and (k["fused_away"] or 0) >= 0.999]
tiny_never = [k for k in never if k["share"] < 0.005]
OUT["never_or_fused"] = dict(
    never_improved_runs=len(never), never_improved_usd=sum(k["usd"] for k in never),
    fused_away_runs=len(fused), fused_away_usd=sum(k["usd"] for k in fused),
    partially_fused_runs=sum(1 for k in kbs if k["sample"] == "main" and k["phase"] == "opt" and 0 < (k["fused_away"] or 0) < 0.999),
    combined_pct_of_spend=(sum(k["usd"] for k in never) + sum(k["usd"] for k in fused)) / TOT,
    never_improved_share_lt_0p5pct=len(tiny_never),
    n_opt_runs=sum(1 for k in kbs if k["sample"] == "main" and k["phase"] == "opt"))
# variant value
acc = [d for d in DECS if d["sample"] == "main" and d["status"] == "accepted"]
rej = [d for d in DECS if d["sample"] == "main" and d["status"] != "accepted"]
var_usd = sum(k["usd"] for k in kbs if k["sample"] == "main" and k["phase"] == "variant")
OUT["variants"] = dict(decisions=len(acc) + len(rej), accepted=len(acc), rejected=len(rej), usd=var_usd, pct=var_usd / TOT,
                       accepted_gain_us_sum=sum(F(d["original_us"], 0) - F(d["variant_us"], 0) for d in acc))
# decision reproducibility from the variant kb runs' best times
pairs = collections.defaultdict(dict)
for k in kbs:
    if k["phase"] == "variant" and k["sample"] == "main" and k["side"]:
        pairs[(k["problem"], k["round"], k["hint"])][k["side"]] = k
agree = n_pairs = 0
for key, d in pairs.items():
    if "orig" in d and "variant" in d and d["orig"]["steps"] and d["variant"]["steps"] and d["orig"]["decision"]:
        n_pairs += 1
        pa = sum(d["variant"]["best_full"].values()) < sum(d["orig"]["best_full"].values())
        agree += int(pa == (d["orig"]["decision"] == "accepted"))
OUT["variants"]["decision_replay_agreement"] = dict(pairs=n_pairs, agree=agree)

# per family spend / speedup
fam = {}
for f in FAMS:
    ps = [p for p in REPLAY if prob[p]["family"] == f]
    if not ps: continue
    fam[f] = dict(n=len(ps), usd=sum(ptot[p] for p in ps), usd_per_problem=float(np.mean([ptot[p] for p in ps])),
                  geomean_speedup=float(np.exp(np.mean([math.log(prob[p]["first"] / prob[p]["final"]) for p in ps]))),
                  logshare_round0=float(np.mean([math.log(prob[p]["parts"][0] / m_after(p, 0)) / math.log(prob[p]["first"] / prob[p]["final"]) for p in ps])),
                  speedup_retained_at_50pct=float(np.median(at("final_over_m", 0.5, ps))))
OUT["families"] = fam

# ------------------------------------------------------------------ (3) technique x op class
def rel_gain(s, k):
    b = s.get("_bb");  return b
tech = collections.defaultdict(lambda: dict(n=0, nb=0, imp=0, fail=0, usd=0.0, lg=0.0))
cls = collections.defaultdict(lambda: dict(n=0, nb=0, imp=0, fail=0, usd=0.0, lg=0.0, fix_usd=0.0, needfix=0))
by_step = collections.defaultdict(lambda: dict(n=0, nb=0, usd=0.0, lg=0.0, imp=0))
roll_best = collections.Counter(); roll_n = 0; corr = []
same_tech_step0 = tot_step0 = 0
lead_change = 0; within5 = 0; nk = 0
for k in kbs:
    if k["sample"] != "main" or k["phase"] != "opt" or len(k["base"]) != 1 or not k["steps"]: continue
    slot = next(iter(k["base"])); b = k["base"][slot]
    if not b: continue
    cur = b
    for s in k["steps"]:
        lg = 0.0
        if s["nb"]: lg = math.log(cur / s["after"]); cur = s["after"]
        for key, D in ((s["tech"], tech), ):
            pass
        t = tech[(k["op_class"], s["tech"])]; c = cls[k["op_class"]]; q = by_step[s["st"]]
        for D in (t, c, q):
            D["n"] += 1; D["nb"] += s["nb"]; D["usd"] += s["usd"]; D["lg"] += lg; D["imp"] += int(s["status"] == "improved")
        for D in (t, c):
            D["fail"] += int(s["status"] in ("compile_error", "wrong_output", "runtime_error"))
        c["fix_usd"] += s["fix_usd"]
    # rollouts
    rb = {}
    for s in k["steps"]:
        if s["status"] == "improved" and s["after"] is not None:
            rb[s["ro"]] = min(rb.get(s["ro"], b), s["after"])
    if gain_full(k) > 0 and rb:
        best_ro = min(rb, key=rb.get); roll_best[best_ro] += 1; roll_n += 1
        vals = sorted(rb.get(r, b) for r in range(4))
        within5 += int(vals[1] <= vals[0] * 1.05); nk += 1
        # leader after step 1 vs final
        rb1 = {}
        for s in k["steps"]:
            if s["st"] <= 1 and s["status"] == "improved" and s["after"] is not None: rb1[s["ro"]] = min(rb1.get(s["ro"], b), s["after"])
        if rb1 and min(rb1, key=rb1.get) != best_ro: lead_change += 1
    s0 = [s["tech"] for s in k["steps"] if s["st"] == 0]
    if len(s0) == 4: tot_step0 += 1; same_tech_step0 += int(len(set(s0)) == 1)
tt = []
for (c, t), D in tech.items():
    if D["n"] < 40: continue
    tt.append(dict(op_class=c, technique=t, n=D["n"], new_best_rate=D["nb"] / D["n"], accept_rate=D["imp"] / D["n"], fail_rate=D["fail"] / D["n"],
                   mean_loggain_per_step=D["lg"] / D["n"], loggain_per_usd=D["lg"] / max(D["usd"], 1e-9), usd=D["usd"]))
tt.sort(key=lambda x: (x["op_class"], -x["loggain_per_usd"]))
OUT["technique_x_class"] = tt
OUT["op_class"] = {c: dict(n=D["n"], new_best_rate=D["nb"] / D["n"], accept_rate=D["imp"] / D["n"], fail_rate=D["fail"] / D["n"],
                           loggain_per_usd=D["lg"] / max(D["usd"], 1e-9), usd=D["usd"], fix_usd_share=D["fix_usd"] / max(D["usd"], 1e-9))
                   for c, D in cls.items()}
OUT["by_step"] = {st: dict(n=D["n"], new_best_rate=D["nb"] / D["n"], accept_rate=D["imp"] / D["n"], loggain_share=D["lg"],
                           loggain_per_usd=D["lg"] / max(D["usd"], 1e-9), usd=D["usd"]) for st, D in sorted(by_step.items())}
totlg = sum(D["lg"] for D in by_step.values())
for st in OUT["by_step"]: OUT["by_step"][st]["loggain_share"] /= totlg
OUT["rollouts"] = dict(best_from_rollout={str(r): roll_best[r] / roll_n for r in range(4)}, n=roll_n,
                       step0_all_same_technique=same_tech_step0 / max(tot_step0, 1),
                       second_best_within_5pct=within5 / max(nk, 1), leader_after_step1_not_final_best=lead_change / max(nk, 1))
# fix-loop overhead and truncation
allst = [s for k in kbs if k["sample"] == "main" for s in k["steps"]]
fix_usd = sum(s["fix_usd"] for s in allst); st_usd = sum(s["usd"] for s in allst)
fixed = [s for s in STEPS if s["sample"] == "main" and s["status"] != "orphan" and int(s["fix_attempts"] or 0) > 0]
OUT["fix"] = dict(fix_usd=fix_usd, fix_pct_of_step_usd=fix_usd / st_usd, steps_needing_fix=len(fixed),
                  steps_needing_fix_frac=len(fixed) / len(allst),
                  fixed_outcomes=dict(collections.Counter(s["status"] for s in fixed)),
                  fixed_new_best=sum(int(s["new_best"]) for s in fixed))
tr = [s for s in STEPS if s["sample"] == "main" and s["truncated"] not in ("", "0")]
OUT["truncation"] = dict(steps_truncated=len(tr), frac=len(tr) / len(allst), usd=sum(F(s["usd"], 0) for s in tr),
                         outcomes=dict(collections.Counter(s["status"] for s in tr)))
# hypothesis vs catalog technique
hyp = [s for s in allst if s["tech"] == "hypothesis"]
OUT["hypothesis_steps"] = dict(n=len(hyp), usd=sum(s["usd"] for s in hyp), new_best_rate=sum(s["nb"] for s in hyp) / max(len(hyp), 1))

# ------------------------------------------------------------------ (4) replay simulator
def kb_kept(k, s_cap, n_ro, patience, prune):
    st = [s for s in k["steps"] if s["st"] < s_cap and s["ro"] < n_ro]
    if prune and k["phase"] == "opt":
        j, margin = prune
        b = sum(v for v in k["base"].values() if v)
        rb = {r: b for r in range(n_ro)}
        for s in st:
            if s["st"] < j and s["status"] == "improved" and s["after"] is not None: rb[s["ro"]] = min(rb[s["ro"]], s["after"])
        lead = min(rb.values())
        alive = {r for r, v in rb.items() if v <= lead * (1 + margin)}
        st = [s for s in st if s["st"] < j or s["ro"] in alive]
    if patience:
        best = dict(k["base"]); out = []; miss = 0
        for s in st:
            out.append(s)
            if s["status"] == "improved" and s["after"] is not None and s["after"] < best.get(s["slot"], 1e18):
                best[s["slot"]] = s["after"]; miss = 0
            else:
                miss += 1
                if miss >= patience: break
        st = out
    return st
def best_of(k, st):
    best = dict(k["base"])
    for s in st:
        if s["status"] == "improved" and s["after"] is not None and best.get(s["slot"]) is not None and s["after"] < best[s["slot"]]:
            best[s["slot"]] = s["after"]
    return best

DEFAULT = dict(fc_cap=4, fc_ro=4, min_share=0.0, ms_from=0, top_k=None, s_cap=4, n_ro=4, patience=None, prune=None, round_cap=None, eps=None, v_steps=2, v_ro=4)
def simulate(prm):
    """returns {problem: (saved_usd, total_usd, loss_us, M_rule_us)} -- additive replay on the logged trajectories."""
    q = dict(DEFAULT, **prm); res = {}
    for p in REPLAY:
        pr = prob[p]; nR = pr["n_rounds"]
        R = nR if q["round_cap"] is None else min(nR, q["round_cap"])
        if q["eps"] is not None:
            for r in range(1, R - 1):   # round 0 always runs; after round r>=1, stop if it gained < eps
                if 1 - m_after(p, r) / pr["parts"][r] < q["eps"]: R = r + 1; break
        M = pr["final"] if R >= nR else pr["parts"][R]
        saved = sum(pround_usd[p].get(r, 0) for r in range(R, nR))
        loss = 0.0
        ks = [k for k in P_kbs(p) if k["round"] < R]
        rank = collections.defaultdict(list)
        for k in ks:
            if k["phase"] == "opt": rank[k["round"]].append(k)
        topset = set()
        for r, lst in rank.items():
            lst = sorted(lst, key=lambda k: -k["share"])
            for i, k in enumerate(lst):
                if q["top_k"] is None or i < q["top_k"]: topset.add(id(k))
        cg_round = collections.Counter(); cg_saved = collections.Counter()
        pairs_ = collections.defaultdict(dict)
        for k in ks:
            cg_round[k["round"]] += k["usd"]
            if k["phase"] == "opt":
                if (k["round"] >= q["ms_from"] and k["share"] < q["min_share"]) or id(k) not in topset:
                    cg_saved[k["round"]] += k["usd"]; loss += k["w"] * gain_full(k); continue
                if k["fusion_cand"]:
                    kept = kb_kept(k, min(q["s_cap"], q["fc_cap"]), min(q["n_ro"], q["fc_ro"]), q["patience"], q["prune"])
                else:
                    kept = kb_kept(k, q["s_cap"], q["n_ro"], q["patience"], q["prune"])
                cg_saved[k["round"]] += sum(s["usd"] for s in k["steps"]) - sum(s["usd"] for s in kept)
                bk = best_of(k, kept)
                loss += k["w"] * sum(max(bk[s] - k["best_full"][s], 0) for s in k["base"] if k["base"][s] is not None)
            else:
                kept = kb_kept(k, q["v_steps"], q["v_ro"], None, None)
                cg_saved[k["round"]] += sum(s["usd"] for s in k["steps"]) - sum(s["usd"] for s in kept)
                if k["side"]: pairs_[(k["round"], k["hint"])][k["side"]] = (k, kept)
        if q["v_steps"] < 2 or q["v_ro"] < 4:
            for (r, h), d in pairs_.items():
                if "orig" not in d or "variant" not in d: continue
                (ko, keo), (kv, kev) = d["orig"], d["variant"]
                full = sum(kv["best_full"].values()) < sum(ko["best_full"].values())
                new = sum(best_of(kv, kev).values()) < sum(best_of(ko, keo).values())
                if full and not new:
                    dd = next((x for x in DECS if x["problem"] == p and int(x["round"]) == r and x["hint"] == h), None)
                    if dd and dd["status"] == "accepted": loss += max(F(dd["original_us"], 0) - F(dd["variant_us"], 0), 0)
        for r in range(R):
            saved += cg_saved[r]
            if cg_round[r] > 0: saved += know[(p, r)] * cg_saved[r] / cg_round[r]
        res[p] = (saved, ptot[p], loss, M + loss)
    return res
def summarize(res, ps=None):
    ps = ps or list(res)
    med = float(np.median([1 - prob[p]["final"] / res[p][3] for p in ps]))
    s = sum(res[p][0] for p in ps); t = sum(res[p][1] for p in ps)
    ret = [prob[p]["final"] / res[p][3] for p in ps]
    return dict(saved_pct=s / t, saved_usd=s, geo_retained=float(np.exp(np.mean(np.log(ret)))),
                speedup_loss_pct=1 - float(np.exp(np.mean(np.log(ret)))), median_problem_loss=med, worst_problem_loss=1 - min(ret),
                worst_problem=ps[int(np.argmin(ret))],
                n_loss_gt_2pct=int(sum(r < 0.98 for r in ret)))
RNG = np.random.default_rng(7)
BOOT = RNG.integers(0, len(REPLAY), size=(1000, len(REPLAY)))
def boot(res):
    sv = np.array([res[p][0] for p in REPLAY]); tt_ = np.array([res[p][1] for p in REPLAY])
    lr = np.array([math.log(prob[p]["final"] / res[p][3]) for p in REPLAY])
    S = sv[BOOT].sum(1) / tt_[BOOT].sum(1); L = 1 - np.exp(lr[BOOT].mean(1))
    return dict(saved_ci=[float(np.percentile(S, 2.5)), float(np.percentile(S, 97.5))],
                loss_ci=[float(np.percentile(L, 2.5)), float(np.percentile(L, 97.5))])

assert abs(summarize(simulate({}))["speedup_loss_pct"]) < 1e-9
single = []
def run(name, prm):
    r = simulate(prm); d = dict(rule=name, params=json.dumps(prm), **summarize(r), **boot(r)); single.append(d); return r
for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05): run(f"min_share={t} (all rounds, seed-time share in round 0)", dict(min_share=t))
for t in (0.002, 0.005, 0.01, 0.02, 0.05): run(f"min_share={t} (rounds>=1 only)", dict(min_share=t, ms_from=1))
for K in (4, 8, 12, 16, 24): run(f"top_k={K}", dict(top_k=K))
for k_ in (2, 3, 4, 6, 8): run(f"patience={k_} (time-ordered steps w/o new best)", dict(patience=k_))
for s_ in (1, 2, 3): run(f"steps_per_rollout={s_}", dict(s_cap=s_))
for n_ in (1, 2, 3): run(f"rollouts={n_}", dict(n_ro=n_))
for j, m in ((1, 0.0), (1, 0.1), (1, 0.3), (2, 0.0), (2, 0.1), (2, 0.3)): run(f"prune rollouts after step {j} if > leader*(1+{m})", dict(prune=(j, m)))
for R in (1, 2, 3): run(f"round_cap={R}", dict(round_cap=R))
for e in (0.005, 0.01, 0.02, 0.05): run(f"stop problem if round gain < {e:.1%}", dict(eps=e))
for v in (1,): run("variant steps/rollout=1", dict(v_steps=1))
for c, n_ in ((2, 4), (1, 4), (2, 2), (1, 2), (1, 1)): run(f"round-0 fusion-candidate slots: {c} steps x {n_} rollouts", dict(fc_cap=c, fc_ro=n_))
for v in (2, 1): run(f"variant rollouts={v}", dict(v_ro=v))
OUT["single_rules"] = single

# combined grid
GRID = dict(fc_cap=[4, 2, 1], min_share=[0.0, 0.002, 0.005, 0.01, 0.02], ms_from=[0, 1], patience=[None, 8, 6, 4], n_ro=[4, 3, 2], s_cap=[4, 3],
            round_cap=[None, 2], eps=[None, 0.01], v_steps=[2, 1])
keys = list(GRID); combos = list(itertools.product(*[GRID[k] for k in keys]))
mat_s = np.zeros((len(combos), len(REPLAY))); mat_t = np.zeros_like(mat_s); mat_l = np.zeros_like(mat_s)
with open(os.path.join(HERE, "sim_grid.csv"), "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(keys + ["problem", "saved_usd", "total_usd", "loss_us", "model_us_rule", "final_us"])
    for i, c in enumerate(combos):
        prm = dict(zip(keys, c)); r = simulate(prm)
        for j, p in enumerate(REPLAY):
            mat_s[i, j], mat_t[i, j] = r[p][0], r[p][1]; mat_l[i, j] = math.log(prob[p]["final"] / r[p][3])
            wr.writerow(list(c) + [p, round(r[p][0], 4), round(r[p][1], 4), round(r[p][2], 4), round(r[p][3], 4), prob[p]["final"]])
def agg(idx):  # idx: problem indices
    S = mat_s[:, idx].sum(1) / mat_t[:, idx].sum(1); L = 1 - np.exp(mat_l[:, idx].mean(1)); return S, L
S_all, L_all = agg(np.arange(len(REPLAY)))
# Pareto frontier
order = np.argsort(L_all); front = []; bestS = -1
for i in order:
    if S_all[i] > bestS + 1e-9: front.append(int(i)); bestS = S_all[i]
OUT["pareto"] = [dict(params=dict(zip(keys, combos[i])), saved_pct=float(S_all[i]), loss_pct=float(L_all[i])) for i in front]
# choose recommendation: max saving subject to loss <= budget (full sample), report bootstrap CI
def pick(S, L, budget): ok = np.where(L <= budget)[0]; return int(ok[np.argmax(S[ok])])
rec = {}
for budget in (0.005, 0.01, 0.02):
    i = pick(S_all, L_all, budget); prm = dict(zip(keys, combos[i])); r = simulate(prm)
    rec[str(budget)] = dict(params=prm, **summarize(r), **boot(r))
OUT["recommended_by_budget"] = rec
# train/test split
tt_res = {}
for budget in (0.005, 0.01, 0.02):
    sv, ls, exceed, chosen = [], [], 0, collections.Counter()
    rng = np.random.default_rng(11)
    for _ in range(300):
        perm = rng.permutation(len(REPLAY)); tr_, te_ = perm[: len(REPLAY) // 2], perm[len(REPLAY) // 2:]
        Str, Ltr = agg(tr_); i = pick(Str, Ltr, budget); Ste, Lte = agg(te_)
        sv.append(Ste[i]); ls.append(Lte[i]); exceed += int(Lte[i] > budget); chosen[json.dumps(dict(zip(keys, combos[i])))] += 1
    tt_res[str(budget)] = dict(test_saved_median=float(np.median(sv)), test_saved_iqr=[float(np.percentile(sv, 25)), float(np.percentile(sv, 75))],
                               test_loss_median=float(np.median(ls)), test_loss_p90=float(np.percentile(ls, 90)),
                               frac_test_loss_over_budget=exceed / 300, top_choices=chosen.most_common(3))
OUT["train_test"] = tt_res
# single-rule train/test for min_share and patience (1-D)
def tt_1d(name, values, mk, budget=0.01):
    mats = []
    for v in values:
        r = simulate(mk(v)); mats.append(([r[p][0] for p in REPLAY], [r[p][1] for p in REPLAY], [math.log(prob[p]["final"] / r[p][3]) for p in REPLAY]))
    rng = np.random.default_rng(3); sv, ls, ch = [], [], collections.Counter()
    for _ in range(300):
        perm = rng.permutation(len(REPLAY)); tr_, te_ = perm[: len(REPLAY) // 2], perm[len(REPLAY) // 2:]
        best = None
        for vi, (s_, t_, l_) in enumerate(mats):
            S = np.array(s_)[tr_].sum() / np.array(t_)[tr_].sum(); L = 1 - np.exp(np.array(l_)[tr_].mean())
            if L <= budget and (best is None or S > best[1]): best = (vi, S)
        vi = best[0] if best else 0; s_, t_, l_ = mats[vi]
        sv.append(np.array(s_)[te_].sum() / np.array(t_)[te_].sum()); ls.append(1 - np.exp(np.array(l_)[te_].mean())); ch[str(values[vi])] += 1
    return dict(rule=name, budget=budget, test_saved_median=float(np.median(sv)), test_loss_median=float(np.median(ls)),
                test_loss_p90=float(np.percentile(ls, 90)), chosen=dict(ch))
OUT["train_test_1d"] = [tt_1d("min_share", [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05], lambda v: dict(min_share=v)),
                        tt_1d("min_share_rounds>=1", [0.0, 0.002, 0.005, 0.01, 0.02, 0.05], lambda v: dict(min_share=v, ms_from=1)),
                        tt_1d("steps_per_rollout", [4, 3, 2, 1], lambda v: dict(s_cap=v)),
                        tt_1d("rollouts", [4, 3, 2, 1], lambda v: dict(n_ro=v)),
                        tt_1d("patience", [None, 8, 6, 4, 3, 2], lambda v: dict(patience=v)),
                        tt_1d("round_cap", [None, 3, 2, 1], lambda v: dict(round_cap=v)),
                        tt_1d("eps_stop", [None, 0.005, 0.01, 0.02, 0.05], lambda v: dict(eps=v))]

json.dump(OUT, open(os.path.join(HERE, "e7_results.json"), "w"), indent=1, default=str)

# ------------------------------------------------------------------ plots
INK, SUB, GRIDC, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRIDC, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
                     "axes.facecolor": BG, "figure.facecolor": BG, "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left"})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(color=GRIDC, linewidth=0.8); ax.set_axisbelow(True)

# 1. marginal value
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
ax = axs[0]
A = np.array([curves[p]["final_over_m"] for p in REPLAY])
ax.fill_between(grid * 100, np.percentile(A, 25, 0) * 100, np.percentile(A, 75, 0) * 100, color=COL[0], alpha=0.15, linewidth=0, label="all problems, IQR")
ax.plot(grid * 100, np.median(A, 0) * 100, color=COL[0], lw=2.2, label="all problems, median")
for i, f in enumerate(FAMS):
    ps = [p for p in REPLAY if prob[p]["family"] == f]
    if not ps: continue
    ax.plot(grid * 100, np.median(np.array([curves[p]["final_over_m"] for p in ps]), 0) * 100, color=COL[i + 1], lw=1.4, label=f"{f} (n={len(ps)}), median")
ax.set_xlabel("% of the problem's LLM spend (in execution order)"); ax.set_ylabel("% of final speedup reached (final time / current time)")
ax.set_title("Speedup reached vs spend (round-calibrated replay)"); ax.set_ylim(0, 102); clean(ax); ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axs[1]
x = np.arange(len(round_tab)); wdt = 0.38
ax.bar(x - wdt / 2, [r["usd_pct"] * 100 for r in round_tab], wdt, color=COL[0], label="% of spend", edgecolor=BG, linewidth=2)
ax.bar(x + wdt / 2, [r["mean_logshare"] * 100 for r in round_tab], wdt, color=COL[2], label="% of final log-speedup (mean over problems)", edgecolor=BG, linewidth=2)
for i, r in enumerate(round_tab):
    ax.text(i - wdt / 2, r["usd_pct"] * 100 + 1, f"{r['usd_pct']:.0%}", ha="center", fontsize=9)
    ax.text(i + wdt / 2, r["mean_logshare"] * 100 + 1, f"{r['mean_logshare']:.1%}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([f"round {r['round']}\n(n={r['n_problems']})" for r in round_tab]); clean(ax)
ax.set_title("Spend vs speedup contribution by round (measured model times)"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(AN, "e7_marginal_value.png"), dpi=130); plt.close(fig)

# 2. pareto
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.scatter(L_all * 100, S_all * 100, s=10, color="#b9b8b2", label="combined-policy grid (%d)" % len(combos) + "", zorder=2)
fr = OUT["pareto"]
ax.plot([f["loss_pct"] * 100 for f in fr], [f["saved_pct"] * 100 for f in fr], color=COL[0], lw=2, marker="o", ms=5, label="Pareto frontier", zorder=3)
for d in single:
    if d["rule"].startswith(("min_share=0.01 (rounds", "patience=8", "patience=4", "round_cap=2", "rollouts=2", "steps_per_rollout=2", "stop problem if round gain < 1.0%")):
        ax.scatter(d["speedup_loss_pct"] * 100, d["saved_pct"] * 100, s=40, color=COL[1], zorder=4)
        lab = d["rule"].split(" (time")[0]
        off = (5, -11) if lab.startswith("steps_per") else (5, 3)
        ax.annotate(lab, (d["speedup_loss_pct"] * 100, d["saved_pct"] * 100), fontsize=8, xytext=off, textcoords="offset points")
ax.set_xscale("symlog", linthresh=0.5); ax.set_xlabel("geomean final speedup lost (%) - replay estimate, symlog"); ax.set_ylabel("% of LLM spend saved")
ax.set_title("Budget rules: dollars saved vs speedup lost (replay over %d problems)" % len(REPLAY) + ""); clean(ax); ax.legend(frameon=False, fontsize=9, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(AN, "e7_pareto.png"), dpi=130); plt.close(fig)

# 3. step index / rollout / op class
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
ax = axs[0]; bs = OUT["by_step"]; xs_ = sorted(bs)
ax.bar([str(s) for s in xs_], [bs[s]["new_best_rate"] * 100 for s in xs_], color=COL[0], width=0.6, edgecolor=BG)
for i, s in enumerate(xs_): ax.text(i, bs[s]["new_best_rate"] * 100 + 0.5, f"{bs[s]['new_best_rate']:.0%} new best\n{bs[s]['loggain_share']:.0%} of log-gain", ha="center", fontsize=8)
ax.set_xlabel("step index within rollout"); ax.set_ylabel("% of steps setting a new slot best"); ax.set_title("Do later steps still win?"); clean(ax)
ax.set_ylim(0, max(bs[s]["new_best_rate"] for s in xs_) * 135)
ax = axs[1]; rbv = OUT["rollouts"]["best_from_rollout"]
ax.bar(list(rbv), [v * 100 for v in rbv.values()], color=COL[6], width=0.6, edgecolor=BG)
for i, v in enumerate(rbv.values()): ax.text(i, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=9)
ax.set_xlabel("rollout"); ax.set_ylabel("% of improved slots whose best came from it"); ax.set_title("Which rollout produced the slot best"); clean(ax)
ax = axs[2]; oc = sorted(OUT["op_class"].items(), key=lambda x: -x[1]["usd"])
ax.barh([c for c, _ in oc], [d["usd"] for _, d in oc], color=COL[0], height=0.6, edgecolor=BG)
for i, (c, d) in enumerate(oc): ax.text(d["usd"] + 20, i, f"\\${d['usd']:,.0f}; {d['new_best_rate']:.0%} new best; {d['loggain_per_usd']:.3f} log-gain/$", va="center", fontsize=8)
ax.invert_yaxis(); ax.set_xlim(0, max(d["usd"] for _, d in oc) * 2.1); ax.set_title("Step spend by op class (main sample)"); clean(ax)
fig.tight_layout(); fig.savefig(os.path.join(AN, "e7_steps_rollouts_classes.png"), dpi=130); plt.close(fig)

# 4. waste breakdown
fig, ax = plt.subplots(figsize=(10, 4.8))
wl = [x for x in OUT["waste"].items() if x[1]["usd"] > 0.5]
ax.barh([k for k, _ in wl], [v["usd"] for _, v in wl], color=[COL[2] if "up to final best" in k or "accepted" in k else "#b9b8b2" if "knowledge" in k else COL[1] for k, _ in wl], height=0.6, edgecolor=BG)
for i, (k, v) in enumerate(wl): ax.text(v["usd"] + 15, i, f"\\${v['usd']:,.0f} ({v['pct']:.0%})", va="center", fontsize=8.5)
ax.invert_yaxis(); ax.set_xlim(0, max(v["usd"] for _, v in wl) * 1.25); clean(ax)
ax.set_title("Where the LLM spend went (48 problems)")
fig.text(0.01, 0.01, "green: produced a kept kernel or accepted fusion; orange: no lasting effect on the final model, or a slot under 0.5% of seed time; gray: overhead", fontsize=8, color=SUB)
fig.tight_layout(); fig.savefig(os.path.join(AN, "e7_spend_breakdown.png"), dpi=130); plt.close(fig)
print(json.dumps({k: OUT[k] for k in ("join", "rounds", "marginal", "waste", "never_or_fused", "variants", "step_productivity")}, indent=1, default=str)[:6000])
