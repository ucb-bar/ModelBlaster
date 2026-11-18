#!/usr/bin/env python3
"""Search-waste metrics for the headline Level 3 sweep (stdlib only, read-only).

Usage (from anywhere):
  python3 search_waste.py                 # all finished problems, all sections
  python3 search_waste.py -s 1,2          # sections only (1 overview, 2 rounds, 3 llm, 4 steps, 5 hints, 6 effort/gain,
                                          #   7 families, 8 missed, 9 loop-aggregates, 10 slot time split, 0 compact summary)
  python3 search_waste.py -p 19,27        # problems by number prefix
  python3 search_waste.py --json out.json # dump per-problem metrics

Finished = results_headline*/<problem>/work/model_runtime.json exists. First-sweep reference = results/<problem>/work.
Cost model (ASSUMED list prices, $/M tokens; prompt tokens are ~92% cache-write / 8% cache-read in kb telemetry):
Opus 25 out / 5.8 in (blend), Haiku 0.5 out / 0.116 in. The flow's own est_usd is 0 for Opus, so it is not used.
"""
import argparse, collections, glob, json, math, os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
PRICE = {"opus": (5.8, 25.0), "haiku": (0.116, 0.5), "sonnet": (2.0, 10.0)}
SLOT_RE = re.compile(r"^kb_problem__(.*)_(\d+)__(r\d+|char)$")
FAMS = [  # first match wins
    ("recurrent", r"lstm|gru|rnn|recur"), ("attention", r"sdpa|attention|softmax|bmm"),
    ("conv_dw", r"conv2d_dw"), ("conv_grouped", r"conv2d_grouped"), ("conv", r"conv"),
    ("linear/gemm", r"linear|matmul|gemm|mm_"), ("norm", r"batchnorm|layernorm|groupnorm|norm"),
    ("pool", r"pool"), ("reduce", r"reduce|mean|sum"), ("copy/cat/layout", r"cat|copy|transpose|permute|view"),
    ("ewise/act", r"relu|gelu|tanh|sigmoid|ewise|add|mul|silu|act|clamp|drop"),
]


def fam(kind):
    k = kind.lower()
    for f, rx in FAMS:
        if re.search(rx, k):
            return f
    return "other"


def jl(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def jsonl(path):
    out = []
    try:
        with open(path) as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def usd(model, pt, ct):
    m = "haiku" if "haiku" in model else "sonnet" if "sonnet" in model else "opus"
    i, o = PRICE[m]
    return (pt * i + ct * o) / 1e6


def active_seconds(root, name):
    """Sum of start->(park|finish|next start) intervals over run_meta events of every instance dir; parked wall separately."""
    tot, att, parked = 0.0, 0, 0.0
    ev = []
    for d in glob.glob(os.path.join(root, "results_headline*", name)):
        m = jl(os.path.join(d, "run_meta.json"), {}) or {}
        ev += [e for e in m.get("events", []) if "t" in e]
    ev.sort(key=lambda e: e["t"])
    cur, last_park = None, None
    for e in ev:
        if e["event"] == "start":
            if cur is not None:
                tot += e["t"] - cur
            if last_park is not None:
                parked += e["t"] - last_park
                last_park = None
            cur = e["t"]
            att += 1
        elif e["event"] in ("park", "finish") and cur is not None:
            tot += e["t"] - cur
            cur = None
            if e["event"] == "park":
                last_park = e["t"]
    return tot, att, parked


def find_dirs(root, only):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "results_headline*", "*", "work", "model_runtime.json"))):
        d = os.path.dirname(os.path.dirname(f))
        name = os.path.basename(d)
        if only and not any(name.split("_")[0] == o for o in only):
            continue
        out[name] = d
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("_")[0])))


def load_problem(root, name, d):
    W = os.path.join(d, "work")
    mr = (jl(os.path.join(W, "model_runtime.json")) or {}).get(name, {})
    sm = jl(os.path.join(W, "summary.json"), {}) or {}
    st = jl(os.path.join(W, "state.json"), {}) or {}
    first = (jl(os.path.join(root, "results", name, "work", "model_runtime.json")) or {}).get(name, {})
    fsum = jl(os.path.join(root, "results", name, "work", "summary.json"), {}) or {}
    P = {"name": name, "dir": d}
    irs = (sm.get("irs") or {}).get("kb_problem", {})
    P["seed_us"] = mr.get("seed_e2e_us")
    P["e2e_us"] = mr.get("model_e2e_us")
    P["first_e2e_us"] = first.get("model_e2e_us")
    P["vs_first"] = (P["first_e2e_us"] / P["e2e_us"]) if P["first_e2e_us"] and P["e2e_us"] else None
    P["speedup"] = mr.get("speedup")
    P["vs_torch"] = mr.get("speedup_vs_torch")
    P["rounds_done"] = sm.get("rounds")
    P["fixpoint"] = sm.get("fixpoint")
    P["first_part_us"], P["final_part_us"] = irs.get("first_us"), irs.get("final_us")
    P["first_rounds"] = fsum.get("rounds")
    fi = ((fsum.get("irs") or {}).get("kb_problem") or {})
    P["fused_head"] = len(irs.get("fused") or [])
    P["fused_first"] = len(fi.get("fused") or [])
    P["hints_first"] = sorted(fi.get("hints") or [])
    P["hints_head"] = sorted(irs.get("hints") or [])
    P["wall_s"], P["attempts"], P["parked_s"] = active_seconds(root, name)
    rounds = []
    for rd in sorted(glob.glob(os.path.join(W, "r[0-9]*"))):
        rn = int(os.path.basename(rd)[1:])
        rj = jl(os.path.join(rd, "round.json"), {}) or {}
        pj = jl(os.path.join(rd, "plan.json"), {}) or {}
        items = pj.get("items") or []
        sent = sum(1 for i in items if i.get("why_not") == "sent-before")
        sel = [i for i in items if i.get("selected")]
        prof = glob.glob(os.path.join(rd, "profile", "*", "profile.json"))
        pu = (jl(prof[0], {}) or {}).get("partition_us") if prof else None
        if pu is None:
            pu = (rj.get("partition_us") or {}).get("kb_problem")
        eff = rj.get("efficiency") or {}
        gs = rj.get("graph_search") or {}
        rounds.append(dict(r=rn, seconds=rj.get("seconds"), decide_s=rj.get("decide_seconds"), part_us=pu,
                           items=len(items), selected=len(sel), sel_share=pj.get("selected_share"), sent_before=sent,
                           llm_cum=rj.get("llm_calls_total"), planned=eff.get("steps_planned"),
                           committed=eff.get("steps_committed"), not_run=eff.get("steps_not_run"),
                           patience_stopped=eff.get("n_patience_stopped"), ext_stopped=len(gs.get("stopped") or []),
                           has_run=os.path.isdir(os.path.join(rd, "run"))))
    P["rounds"] = rounds
    tl = [("seed", P["first_part_us"])]
    for tag, pat in (("char", "r00/profile_char"), ("preapply", "r00/profile_pre")):
        g = glob.glob(os.path.join(W, pat, "*", "profile.json"))
        if g:
            tl.append((tag, (jl(g[0], {}) or {}).get("partition_us")))
    for r in rounds:
        tl.append(("r%02d_start" % r["r"], r["part_us"]))
    tl.append(("final", P["final_part_us"]))
    P["timeline"] = tl
    # round-0 slot runs start from the post-characterization partition (char winners enter via reuse_seed), not the naive seed
    post = [v for k, v in tl if k in ("preapply", "char") and v]
    if rounds and post:
        rounds[0]["part_naive"] = rounds[0]["part_us"]
        rounds[0]["part_us"] = post[-1]
    gd = jsonl(os.path.join(W, "graph_decisions.jsonl"))
    P["decisions"] = [e for e in gd if e.get("event") == "decision"]
    P["surv_saved"] = {e["round"]: e.get("steps_saved") for e in gd if e.get("event") == "survival"}
    P["surv_ev"] = [e for e in gd if e.get("event") == "survival"]
    hints = (jl(os.path.join(W, "graph_hints.json"), {}) or {}).get("hints", [])
    P["hints"] = [dict(id=h["id"], kind=h.get("kind"), status=h.get("status")) for h in hints]
    runs = []
    # every kb run dir (journal/state.json always exists; result.json is missing for runs killed/never started)
    for jd in glob.glob(os.path.join(W, "r[0-9]*", "run", "kb_problem__*")) + \
            glob.glob(os.path.join(W, "r[0-9]*", "char", "run", "kb_problem__*")):
        j = jl(os.path.join(jd, "journal", "state.json"))
        if j is None:
            continue
        r = jl(os.path.join(jd, "result.json"))
        sd = os.path.basename(jd)
        m = SLOT_RE.match(sd)
        if m:
            kind, ph = m.group(1), ("char" if m.group(3) == "char" else "run")
        else:
            kind, ph = re.sub(r"^kb_problem__(fuse_)?|__(orig|variant)$", "", sd), "decision"
        rn = int(jd.split("/work/r")[1].split("/")[0])
        llm = collections.defaultdict(lambda: [0, 0, 0, 0.0])
        for u in (j.get("llm_usage_by") or {}).values():
            x = llm[u.get("role", "?")]
            x[0] += u.get("calls", 0); x[1] += u.get("prompt_tokens", 0); x[2] += u.get("completion_tokens", 0)
            x[3] += usd(u.get("model", ""), u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        steps = [dict(st=s.get("status"), step=s.get("step") or 0, b=s.get("slot_us_before"), a=s.get("slot_us_after"),
                      sec=s.get("seconds") or 0, fix=s.get("fix_attempts") or 0,
                      role=(s.get("context") or {}).get("model_role")) for s in (j.get("steps_log") or [])]
        eff = (r or {}).get("efficiency") or {}
        base = (r or {}).get("baseline_partition_us") or (j.get("baseline") or {}).get("partition_us")
        fin = (r or {}).get("final_partition_us") if r else (j.get("last_full") or {}).get("partition_us")
        started = bool(steps) or any(v[0] for v in llm.values())
        tel = ((r or {}).get("telemetry") or {})
        runs.append(dict(round=rn, phase=ph, kind=kind, fam=fam(kind), base=base, final=fin,
                         has_result=r is not None, ok=bool((r or {}).get("final_ok")), infra=bool((r or {}).get("final_infra")),
                         stop=eff.get("stopped") or (None if started else "never-started"),
                         planned=eff.get("steps_planned") or (j.get("steps", 0) * j.get("rollouts", 0)),
                         committed=len(steps), started=started,
                         not_run=eff.get("steps_not_run"),
                         wall=tel.get("wall_s") or sum(s["sec"] for s in steps),
                         tsec=tel.get("time_s") or {}, llm=dict(llm), steps=steps))
    P["runs"] = runs
    prof = sorted(glob.glob(os.path.join(W, "r[0-9]*", "profile", "*", "profile.json")))
    P["final_ops"], P["final_prof_round"] = [], "-"
    if prof:
        p = jl(prof[-1], {}) or {}
        P["final_prof_round"] = prof[-1].split("/work/")[1].split("/")[0]
        P["final_ops"] = [dict(kind=o.get("op", ""), us=o.get("us") or 0,
                               seed=re.sub(r":[0-9a-f]{6,}.*", "", str(o.get("seed")))) for o in p.get("ops", [])]
    fprof = sorted(glob.glob(os.path.join(root, "results", name, "work", "r[0-9]*", "profile", "*", "profile.json")))
    P["first_ops"] = [dict(kind=o.get("op", ""), us=o.get("us") or 0) for o in
                      ((jl(fprof[-1], {}) or {}).get("ops", []) if fprof else [])]
    # first-sweep effort for the same problem (all-Opus era)
    fs = fu = 0
    for jd in glob.glob(os.path.join(root, "results", name, "work", "r[0-9]*", "run", "kb_problem__*")):
        j = jl(os.path.join(jd, "journal", "state.json")) or {}
        rr = jl(os.path.join(jd, "result.json")) or {}
        fs += len(j.get("steps_log") or rr.get("steps") or [])
        for u in (j.get("llm_usage_by") or {}).values():
            fu += usd(u.get("model", ""), u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        if not j:
            for u in rr.get("llm_usage_by_model") or []:
                fu += usd(u.get("model", ""), u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
    P["first_steps"], P["first_usd"] = fs, fu
    return P


def step_stats(P):
    c = collections.Counter()
    by_idx = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in P["runs"]:
        if r["phase"] == "decision":
            continue
        for s in r["steps"]:
            c[s["st"]] += 1
            k = min(s["step"], 3)
            by_idx[k][0] += 1
            if s["st"] == "improved" and s["b"] and s["a"] and s["a"] > 0:
                by_idx[k][1] += 1
                by_idx[k][2] += math.log(s["b"] / s["a"])
    return c, by_idx


def run_effort(P):
    """gain_frac = (slot-run baseline - final partition) / whole-model partition at the start of that phase
    (round-0 runs: post-characterization partition; char runs: naive seed; decision runs: no gain, cost only)."""
    pstart = {r["r"]: r["part_us"] for r in P["rounds"]}
    out = []
    for r in P["runs"]:
        tot = P["first_part_us"] if r["phase"] == "char" else pstart.get(r["round"])
        gain = 0.0
        if r["phase"] != "decision" and r["has_result"] and r["ok"] and r["base"] is not None and r["final"] is not None:
            gain = r["base"] - r["final"]
        share = (r["base"] / tot) if (tot and r["base"]) else 0.0
        out.append(dict(r, usd=sum(v[3] for v in r["llm"].values()), tcalls=sum(v[0] for v in r["llm"].values()),
                        gain_us=gain, gain_frac=(gain / tot if tot else 0.0), share=share,
                        own_gain=(gain / r["base"] if r["base"] else 0.0),
                        cls=("decision" if r["phase"] == "decision" else "never-started" if not r["started"] else "low-leverage(<1% share)" if share < 0.01 else
                             "stalled big slot(own gain<5%)" if gain / (r["base"] or 1) < 0.05 else "productive big slot")))
    return out


def fmt_us(x):
    return "-" if x is None else ("%.0f" % x if x >= 100 else "%.1f" % x)


def table(rows, hdr):
    w = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(hdr)]
    print("| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[i]).ljust(w[i]) for i in range(len(hdr))) + " |")
    print()


def pct(a, b):
    return "-" if not b else "%.0f%%" % (100.0 * a / b)


def sec1(PS):
    print("## 1. Overview (wall = run_meta start->park/finish intervals, parked time excluded)")
    rows = []
    for n, P in PS.items():
        rows.append([n, fmt_us(P["seed_us"]), fmt_us(P["e2e_us"]), fmt_us(P["first_e2e_us"]),
                     "%.2fx" % P["vs_first"] if P["vs_first"] else "-", "%.0fx" % P["speedup"] if P["speedup"] else "-",
                     "%.1f" % (P["wall_s"] / 3600), "%.1f" % (P["parked_s"] / 3600), P["attempts"],
                     "%s/%s" % (P["rounds_done"], P["fixpoint"]), P["first_rounds"]])
    table(rows, ["problem", "seed us", "head us", "first us", "first/head", "vs seed", "wall h", "parked h", "att", "rounds/fixpt", "first rounds"])


def sec2(PS):
    print("## 2. Per round: time, plan, partition (part us = profile at round start; gain = drop to next round start/final)")
    rows = []
    for n, P in PS.items():
        for i, r in enumerate(P["rounds"]):
            nxt = P["rounds"][i + 1]["part_us"] if i + 1 < len(P["rounds"]) else P["final_part_us"]
            gain = (1 - nxt / r["part_us"]) * 100 if r["part_us"] and nxt else None
            flag = []
            if r["items"] and r["selected"] == 0:
                flag.append("EMPTY")
            if r["sent_before"] and r["sent_before"] == r["items"]:
                flag.append("all-sent-before")
            if not r["has_run"]:
                flag.append("no-run")
            if i == len(P["rounds"]) - 1 and len(P["rounds"]) < 3:
                flag.append("<3 rounds")
            rows.append([n if i == 0 else "", r["r"], fmt_us(r["seconds"]), fmt_us(r["decide_s"]), r["items"], r["selected"],
                         "%.2f" % (r["sel_share"] or 0), r["sent_before"], r["planned"], r["committed"], r["not_run"],
                         r["patience_stopped"], r["ext_stopped"], P["surv_saved"].get(r["r"], "-"), r["llm_cum"],
                         fmt_us(r["part_us"]), "-" if gain is None else "%.0f%%" % gain, ",".join(flag)])
    table(rows, ["problem", "rd", "sec", "decide s", "items", "sel", "share", "sent-bf", "planned", "commit", "notrun", "pat",
                 "ext", "surv-saved", "llm cum", "part us", "gain", "flags"])
    print("Timeline seed -> char -> preapply -> rounds -> final (partition us):")
    for n, P in PS.items():
        print("  %-30s %s" % (n, "  ".join("%s=%s" % (k, fmt_us(v)) for k, v in P["timeline"])))
    print()


def sec3(PS):
    print("## 3. LLM by role: summed over every kb run result.json (characterize + rounds); $ = assumed list price")
    rows = []
    allr = collections.defaultdict(lambda: [0, 0, 0, 0.0])
    for n, P in PS.items():
        by = collections.defaultdict(lambda: [0, 0, 0, 0.0])
        for r in P["runs"]:
            for k, v in r["llm"].items():
                for i in range(4):
                    by[k][i] += v[i]
                    allr[k][i] += v[i]
        top = sorted(by.items(), key=lambda kv: -kv[1][3])[:4]
        last = P["rounds"][-1]["llm_cum"] if P["rounds"] else None
        rows.append([n, sum(v[0] for v in by.values()), "%.1f" % (sum(v[1] for v in by.values()) / 1e6),
                     "%.1f" % (sum(v[2] for v in by.values()) / 1e6), "%.0f" % sum(v[3] for v in by.values()), last,
                     sum(d.get("llm_calls", 0) for d in P["decisions"]),
                     ", ".join("%s %d/$%.0f" % (k, v[0], v[3]) for k, v in top)])
    table(rows, ["problem", "calls", "Min in", "Mout", "est $", "flow llm_calls_total", "decision calls", "top roles (calls/$)"])
    print("All finished: " + ", ".join("%s %d/$%.0f" % (k, v[0], v[3]) for k, v in sorted(allr.items(), key=lambda kv: -kv[1][3]))
          + "  TOTAL $%.0f" % sum(v[3] for v in allr.values()))
    print()


def tail_waste(P):
    """steps with ordinal index beyond the step at which the run's best slot value first appeared (convergence tail)."""
    tot = tail = 0
    for r in P["runs"]:
        st = [x for x in r["steps"] if x["a"] and x["b"] and r["phase"] != "decision"]
        if len(st) < 2:
            continue
        best = min(st, key=lambda x: x["a"])
        tot += len(st)
        tail += sum(1 for x in st if x["step"] > best["step"])
    return tail, tot


def sec4(PS):
    print("## 4. KB steps (char + round runs; decision runs excluded): outcomes, errors, cuts")
    rows = []
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for n, P in PS.items():
        c, byi = step_stats(P)
        for i, v in byi.items():
            for k in range(3):
                agg[i][k] += v[k]
        tot = sum(c.values())
        err = c["runtime_error"] + c["compile_error"] + c["wrong_output"]
        R = [r for r in P["runs"] if r["phase"] != "decision"]
        ok = sum(1 for r in R if r["has_result"] and r["ok"])
        infra = sum(1 for r in R if r["infra"])
        never = [r for r in R if r["stop"] == "never-started"]
        killed = [r for r in R if not r["has_result"] and r["started"]]
        ext = [r for r in R if r["stop"] == "external"]
        tl, tt = tail_waste(P)
        rows.append([n, tot, c["improved"], pct(c["improved"], tot), c["regressed"], c["neutral"], c["runtime_error"],
                     c["compile_error"], pct(err, tot), "%d/%d" % (ok, len(R)), infra, len(killed), len(never),
                     sum(r["planned"] or 0 for r in never), sum(1 for r in R if r["stop"] == "patience"), len(ext),
                     sum(r["committed"] for r in ext), sum((r["not_run"] or 0) for r in R),
                     sum(v or 0 for v in P["surv_saved"].values()), pct(tl, tt)])
    table(rows, ["problem", "steps", "improved", "%", "regr", "neut", "rt_err", "cmp_err", "err%", "ok/runs", "infra-fail",
                 "killed(no result)", "never-started", "steps of never-started", "pat-stop", "absorbed-stop", "steps in absorbed",
                 "not-run (eff)", "surv-saved", "tail steps"])
    print("Hit rate by step ordinal within a rollout (improved/steps; share of total ln-gain from improved steps):")
    tg = sum(v[2] for v in agg.values()) or 1
    table([["step %d%s" % (i, "+" if i == 3 else ""), agg[i][0], agg[i][1], pct(agg[i][1], agg[i][0]), "%.0f%%" % (100 * agg[i][2] / tg)]
           for i in sorted(agg)], ["ordinal", "steps", "improved", "hit rate", "share of ln gain"])
    # what-if: cap every rollout at k steps (slot-level, per run; weighted by run share is not available per slot)
    cap = collections.defaultdict(lambda: [0, 0, 0, 0.0, 0.0])
    for P in PS.values():
        for r in P["runs"]:
            st = [x for x in r["steps"] if x["a"] and x["b"] and r["phase"] != "decision"]
            if len(st) < 2:
                continue
            best = min(x["a"] for x in st)
            for k in (1, 2, 3):
                kept = [x for x in st if x["step"] < k]
                bk = min([x["a"] for x in kept] + [min(x["b"] for x in st)])
                c = cap[k]
                c[0] += 1
                c[1] += 1 if bk <= best * 1.02 else 0
                c[2] += len(kept)
                c[3] += len(st)
                c[4] += math.log(bk / best)
    print("What-if cap on steps per rollout (slot level, over all runs with >=2 timed steps): runs whose best stays within 2%, steps kept, mean slot loss")
    table([["cap %d" % k, c[0], pct(c[1], c[0]), pct(c[2], c[3]), "%.1f%%" % (100 * (math.exp(c[4] / c[0]) - 1))] for k, c in sorted(cap.items())],
          ["cap", "runs", "best within 2%", "steps kept", "mean slot loss"])
    rl = collections.defaultdict(lambda: collections.Counter())
    for P in PS.values():
        for r in P["runs"]:
            if r["phase"] == "decision":
                continue
            for x in r["steps"]:
                k = x["role"] or "?"
                rl[k]["n"] += 1; rl[k][x["st"]] += 1; rl[k]["fix"] += 1 if x["fix"] else 0
                if x["st"] == "improved" and x["b"] and x["a"] and x["a"] > 0:
                    rl[k]["lng"] += math.log(x["b"] / x["a"])
    print("Step outcomes by generating model role (codegen_simple = Haiku, codegen_hard = Opus):")
    table([[k, v["n"], pct(v["improved"], v["n"]), pct(v["regressed"], v["n"]), pct(v["runtime_error"] + v["compile_error"], v["n"]),
            pct(v["fix"], v["n"]), "%.2f" % (v["lng"] / v["n"])] for k, v in sorted(rl.items(), key=lambda kv: -kv[1]["n"])],
          ["role", "steps", "improved", "regressed", "error", "needed fix", "mean ln-gain/step"])
    print("'improved' = step status (slot faster + e2e accept); rollouts are parallel, so most improved steps are not on the winning path.")
    print("never-started = slot queued but ended before its first step (stopped by an accepted fusion absorbing its ops, or run killed).")
    print("absorbed-stop = run externally stopped by graph_search (ops absorbed by an accepted fusion); 'steps in absorbed' already spent.")
    print("surv-saved = steps removed by the survival head at planning; tail steps = steps after the ordinal where the run's best slot value appeared.")
    print()


def hint_type(i, kind=""):
    if i.startswith("attention:") or kind == "fuse_attention":
        return "attention:fuse_sdpa"
    if i.startswith("fuse_siblings") or kind == "fuse_siblings":
        return "fuse_siblings"
    if i.startswith("region") or kind == "fuse_region":
        return "region"
    if i.startswith("layout") or kind == "fold_layout":
        return "layout"
    return "fuse"


def sec5(PS):
    print("## 5. Decisions and hints")
    rows = []
    agg = collections.defaultdict(collections.Counter)
    gains = collections.defaultdict(list)
    for n, P in PS.items():
        sts = collections.Counter(x["status"] for x in P["decisions"] if x["phase"] != "preapply_audit")
        hs = collections.Counter(h["status"] for h in P["hints"])
        und = [h["id"] for h in P["hints"] if h["status"] == "proposed"]
        for h in P["hints"]:
            agg[hint_type(h["id"], h["kind"])][h["status"]] += 1
        for x in P["decisions"]:
            g = ((x.get("evidence") or {}).get("tests") or {}).get("gain")
            if x["status"] == "accepted" and g is not None:
                gains[hint_type(x["hint"])].append(g)
        rows.append([n, len(P["hints"]), hs["accepted"], hs["rejected"], hs["proposed"], sts["promising"], sts["undecided"],
                     sum(1 for x in P["decisions"] if x["phase"] == "preapply"),
                     "%d/$%.1f" % (sum(x.get("llm_calls", 0) for x in P["decisions"]), sum(x.get("usd", 0) for x in P["decisions"])),
                     ";".join(u[5:50] for u in und)[:90]])
    table(rows, ["problem", "hints", "accepted", "rejected", "proposed(never decided)", "promising", "undecided", "preapplied",
                 "decision calls/$ (flow est)", "never-decided hints"])
    rows = []
    for t, c in sorted(agg.items()):
        gl = gains.get(t, [])
        rows.append([t, sum(c.values()), c["accepted"], c["rejected"], c["proposed"],
                     "%.0f%% (n=%d)" % (100 * sum(gl) / len(gl), len(gl)) if gl else "-"])
    print("By hint type (statuses over finished problems; mean measured rung gain of accepted hints):")
    table(rows, ["type", "n", "accepted", "rejected", "proposed", "mean accepted gain"])
    sec5b(PS)
    c = collections.defaultdict(lambda: [0, 0, 0.0])
    for P in PS.values():
        for x in P["decisions"]:
            if x["phase"] in ("preapply", "preapply_audit"):
                continue
            k = (hint_type(x["hint"]), x["status"])
            c[k][0] += 1; c[k][1] += x.get("llm_calls", 0); c[k][2] += x.get("usd", 0)
    tot = sum(v[2] for v in c.values()) or 1
    print("Rung decisions: cost by hint type and outcome (flow's own $ estimate, all-uncached Opus pricing so higher than section 3):")
    table([[k[0], k[1], v[0], v[1], "$%.0f" % v[2], pct(v[2], tot)] for k, v in sorted(c.items())], ["type", "outcome", "decisions", "llm calls", "$ (flow est)", "share"])


def sec5b(PS):
    print("Survival-head step cuts by the pending hints that justified them (saved = max steps - granted steps per slot):")
    by = collections.defaultdict(lambda: [0, 0])   # label -> [saved steps, slots]
    tot = 0
    for n, P in PS.items():
        status = {h["id"]: h["status"] for h in P["hints"]}
        for e in P["surv_ev"]:
            mx = e.get("steps") or 4
            for v in (e.get("problems") or {}).values():
                saved = mx - (v.get("steps") or 0)
                if saved <= 0:
                    continue
                tot += saved
                hs = v.get("by") or []
                reg = [h for h in hs if hint_type(h) == "region"]
                oth = [h for h in hs if hint_type(h) != "region"]
                acc = any(status.get(h) == "accepted" for h in oth)
                if reg and not oth:
                    label = "region only (0 of 20 region hints ever accepted)"
                elif reg:
                    label = "region + fuse (fuse %s)" % ("accepted" if acc else "NOT accepted")
                else:
                    label = "fuse only (%s)" % ("accepted" if acc else "NOT accepted")
                by[label][0] += saved
                by[label][1] += 1
    table([[k, v[0], pct(v[0], tot), v[1]] for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])], ["pending-hint kind", "steps cut", "share", "slot-rounds"])


def sec6(PS):
    print("## 6. Effort vs gain (run gain = slot-run baseline-final partition drop / whole-model partition at round start)")
    rows = []
    famagg = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0, 0])
    CL = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for n, P in PS.items():
        R = run_effort(P)
        tu = sum(r["usd"] for r in R); tw = sum(r["wall"] for r in R); tc = sum(r["tcalls"] for r in R)
        lo = [r for r in R if r["gain_frac"] < 0.01]
        rr = []
        for rd in P["rounds"]:
            rs = [r for r in R if r["round"] == rd["r"] and r["phase"] == "run"]
            if rs:
                u = sum(r["usd"] for r in rs); g = sum(r["gain_us"] for r in rs)
                rr.append("r%d:$%.0f/%s" % (rd["r"], u, pct(g, rd["part_us"])))
        zero = [r for r in R if r["phase"] == "run" and r["ok"] and r["gain_us"] <= 0]
        for r in R:
            CL[r["cls"]][0] += 1; CL[r["cls"]][1] += r["usd"]; CL[r["cls"]][2] += r["wall"]; CL[r["cls"]][3] += r["tcalls"]
        rows.append([n, len(R), "$%.0f" % tu, "%.1f" % (tw / 3600), len(lo), pct(len(lo), len(R)),
                     pct(sum(r["usd"] for r in lo), tu), pct(sum(r["wall"] for r in lo), tw),
                     pct(sum(r["tcalls"] for r in lo), tc), len(zero), " ".join(rr)])
        for r in R:
            f = famagg[r["fam"]]
            f[0] += 1; f[1] += r["usd"]; f[2] += r["wall"]; f[3] += r["gain_us"]
            f[4] += sum(1 for s in r["steps"] if s["st"] == "improved"); f[5] += len(r["steps"])
    table(rows, ["problem", "slot runs", "est $", "slot-wall h", "runs <1%", "of runs", "of $", "of wall", "of calls",
                 "zero/neg-gain runs", "per round: $ / partition drop"])
    print("Leverage classes (share = slot baseline / whole-model partition at phase start; own gain = slot-run drop / slot baseline):")
    T = [sum(v[i] for v in CL.values()) or 1 for i in range(4)]
    table([[k, v[0], pct(v[0], T[0]), "$%.0f" % v[1], pct(v[1], T[1]), "%.1f" % (v[2] / 3600), pct(v[2], T[2]), pct(v[3], T[3])]
           for k, v in sorted(CL.items())], ["class", "runs", "of runs", "est $", "of $", "slot-wall h", "of wall", "of calls"])
    print("By kernel family (all finished problems):")
    rows = [[f, v[0], "$%.0f" % v[1], "%.1f" % (v[2] / 3600), "%d/%d" % (v[4], v[5]), pct(v[4], v[5]), "$%.1f" % (v[1] / v[0])]
            for f, v in sorted(famagg.items(), key=lambda kv: -kv[1][1])]
    table(rows, ["family", "runs", "est $", "slot-wall h", "improved/steps", "hit rate", "$ per run"])
    print("Stage gains: share of total log-gain (naive seed -> final partition) per stage, with $ and wall; char = 1 step on the 8 largest ops of the NAIVE seed")
    rows = []
    agg = collections.Counter()
    for n, P in PS.items():
        R = run_effort(P)
        tl = dict(P["timeline"])
        seed, ch, fin = tl.get("seed"), tl.get("char"), P["final_part_us"]
        r1s = tl.get("r01_start")
        if not (seed and ch and r1s and fin):
            continue
        lg = lambda a_, b_: math.log(a_ / b_)
        tot = lg(seed, fin)
        sh = (lg(seed, ch) / tot, lg(ch, r1s) / tot, lg(r1s, fin) / tot)
        cu = sum(r["usd"] for r in R if r["phase"] == "char"); r0u = sum(r["usd"] for r in R if r["phase"] == "run" and r["round"] == 0)
        lu = sum(r["usd"] for r in R if r["phase"] == "run" and r["round"] >= 1); du = sum(r["usd"] for r in R if r["phase"] == "decision")
        r0h = sum(r["seconds"] or 0 for r in P["rounds"] if r["r"] == 0) / 3600
        lh = sum(r["seconds"] or 0 for r in P["rounds"] if r["r"] >= 1) / 3600
        for k, v in zip(("sc", "s0", "sl", "uc", "u0", "ul", "ud", "h0", "hl"), (sh[0], sh[1], sh[2], cu, r0u, lu, du, r0h, lh)):
            agg[k] += v
        agg["n"] += 1
        rows.append([n, "%.0f%%" % (100 * sh[0]), "$%.1f" % cu, "%.0f%%" % (100 * sh[1]), "$%.0f" % r0u, "%.1f" % r0h,
                     "%.0f%%" % (100 * sh[2]), "$%.0f" % lu, "%.1f" % lh, "$%.0f" % du])
    k = agg["n"] or 1
    rows.append(["MEAN share / TOTAL $,h", "%.0f%%" % (100 * agg["sc"] / k), "$%.0f" % agg["uc"], "%.0f%%" % (100 * agg["s0"] / k),
                 "$%.0f" % agg["u0"], "%.0f" % agg["h0"], "%.0f%%" % (100 * agg["sl"] / k), "$%.0f" % agg["ul"], "%.0f" % agg["hl"], "$%.0f" % agg["ud"]])
    table(rows, ["problem", "char: gain", "char $", "round0: gain", "r0 $", "r0 h", "rounds>=1: gain", "r>=1 $", "r>=1 h", "decision runs $"])


def sec7(PS):
    print("## 7. Families in the final profile (last round profile): share, hand/fused-seed share, KB touch status")
    rows = []
    agg = collections.defaultdict(float)
    for n, P in PS.items():
        ops = P["final_ops"]
        tot = sum(o["us"] for o in ops) or 1
        by = collections.defaultdict(lambda: [0.0, 0.0])
        for o in ops:
            f = fam(o["kind"]); by[f][0] += o["us"]
            if o["seed"].startswith(("hand", "fused-seed")):
                by[f][1] += o["us"]
        touched = set(r["fam"] for r in P["runs"] if r["phase"] == "run" and (r["committed"] or r["planned"]))
        improved = set(r["fam"] for r in P["runs"] if r["phase"] == "run" and any(s["st"] == "improved" for s in r["steps"]))
        cells = []
        for f, v in sorted(by.items(), key=lambda kv: -kv[1][0])[:4]:
            tag = "" if f in improved else ("!untouched" if f not in touched else "!no-gain")
            cells.append("%s %.0f%% (hand %.0f%%)%s" % (f, 100 * v[0] / tot, 100 * v[1] / tot, tag))
        for f, v in by.items():
            agg[f] += v[0] / tot
        rows.append([n, P["final_prof_round"], "%.0f%%" % (100 * sum(o["us"] for o in ops if o["seed"].startswith(("hand", "fused-seed"))) / tot),
                     "; ".join(cells)])
    table(rows, ["problem", "prof", "hand+fused-seed", "top families (share of final time)"])
    print("Mean final-time share by family: " + ", ".join("%s %.0f%%" % (f, 100 * v / max(1, len(PS))) for f, v in sorted(agg.items(), key=lambda kv: -kv[1])))
    print("Origin: 'cache'/'cache:reuse' = kernel from the KB cache/reuse routing; 'hand' = unevolved hand kernel; 'fused-seed' = composed from unevolved seeds.")
    print()


def sec8(PS):
    print("## 8. What was missed")
    rows = []
    for n, P in PS.items():
        fk = collections.Counter(o["kind"] for o in P["first_ops"]); hk = collections.Counter(o["kind"] for o in P["final_ops"])
        miss = sorted(k for k in fk if k.startswith("fused__") and k not in hk)
        extra = sorted(k for k in hk if k.startswith("fused__") and k not in fk)
        hs = sum(len(r["steps"]) for r in P["runs"]); hu = sum(v[3] for r in P["runs"] for v in r["llm"].values())
        rows.append([n, "%.2fx" % P["vs_first"] if P["vs_first"] else "-", "%.2f" % (hs / P["first_steps"]) if P["first_steps"] else "-",
                     "%.2f" % (hu / P["first_usd"]) if P["first_usd"] else "-", len(P["first_ops"]), len(P["final_ops"]),
                     P["fused_first"], P["fused_head"], ",".join(k[7:40] for k in miss)[:80] or "-",
                     ",".join(k[7:40] for k in extra)[:60] or "-"])
    table(rows, ["problem", "first/head", "steps head/first", "$ head/first", "ops first", "ops head", "fused grp first", "fused grp head", "fused kinds only in first", "only in head"])
    print("Parity/regression flags (first/head < 1.05):")
    for n, P in PS.items():
        if P["vs_first"] and P["vs_first"] < 1.05:
            c, _ = step_stats(P)
            print("  %-28s %.2fx rounds=%s fixpoint=%s empty-rounds=%s all-sent-before=%s infra-fail-runs=%d err-steps=%d/%d first-rounds=%s" % (
                n, P["vs_first"], P["rounds_done"], P["fixpoint"], [r["r"] for r in P["rounds"] if r["items"] and r["selected"] == 0],
                [r["r"] for r in P["rounds"] if r["sent_before"] and r["sent_before"] == r["items"]],
                sum(1 for r in P["runs"] if r["infra"]), c["runtime_error"] + c["compile_error"], sum(c.values()), P["first_rounds"]))
    print()


def sec9(PS):
    print("## 9. Loop aggregates (rerun to track progress)")
    R = {k: run_effort(P) for k, P in PS.items()}
    allr = [r for v in R.values() for r in v]
    c = collections.Counter()
    for P in PS.values():
        c.update(step_stats(P)[0])
    tot = sum(c.values()); usd_ = sum(r["usd"] for r in allr)
    lo = [r for r in allr if r["gain_frac"] < 0.01]
    print("problems=%d  active wall=%.0f h  est LLM $=%.0f  KB steps=%d (improved %s, error %s)" % (
        len(PS), sum(P["wall_s"] for P in PS.values()) / 3600, usd_, tot, pct(c["improved"], tot),
        pct(c["runtime_error"] + c["compile_error"] + c["wrong_output"], tot)))
    print("slot runs <1%% whole-model gain: %s of runs, %s of $, %s of slot-wall" % (
        pct(len(lo), len(allr)), pct(sum(r["usd"] for r in lo), usd_), pct(sum(r["wall"] for r in lo), sum(r["wall"] for r in allr))))
    lowr = [(n, r) for n, P in PS.items() for i, r in enumerate(P["rounds"])
            if (i + 1 < len(P["rounds"]) and P["rounds"][i + 1]["part_us"] and r["part_us"] and 1 - P["rounds"][i + 1]["part_us"] / r["part_us"] < 0.01)
            or (i + 1 == len(P["rounds"]) and P["final_part_us"] and r["part_us"] and 1 - P["final_part_us"] / r["part_us"] < 0.01)]
    emp = [(n, r) for n, P in PS.items() for r in P["rounds"] if r["items"] and r["selected"] == 0]
    tr = sum((r["seconds"] or 0) for P in PS.values() for r in P["rounds"])
    print("rounds with <1%% partition drop: %d (%.1f h = %s of round time); rounds with an empty plan: %d (%.1f h)" % (
        len(lowr), sum(r["seconds"] or 0 for _, r in lowr) / 3600, pct(sum(r["seconds"] or 0 for _, r in lowr), tr),
        len(emp), sum(r["seconds"] or 0 for _, r in emp) / 3600))
    cut = collections.Counter()
    for P in PS.values():
        for r in P["runs"]:
            if r["phase"] == "decision":
                continue
            cut["planned"] += r["planned"] or 0; cut["committed"] += r["committed"]
            if r["stop"] == "never-started":
                cut["never-started"] += r["planned"] or 0
            elif r["stop"] == "external":
                cut["absorbed (not run)"] += max(0, (r["planned"] or 0) - r["committed"])
            elif r["stop"] == "patience":
                cut["patience (not run)"] += max(0, (r["planned"] or 0) - r["committed"])
        cut["survival (never planned)"] += sum(v or 0 for v in P["surv_saved"].values())
    print("steps: planned(after survival) %d, executed %d; cut: %s" % (cut["planned"], cut["committed"], {k: v for k, v in cut.items() if k not in ("planned", "committed")}))
    ab = [r for r in allr if r["stop"] == "external"]
    print("absorbed runs (stopped after an accepted fusion took their ops): %d runs, %d steps already spent, est $%.0f (%s of $), slot-wall %.1f h" % (
        len(ab), sum(r["committed"] for r in ab), sum(r["usd"] for r in ab), pct(sum(r["usd"] for r in ab), usd_), sum(r["wall"] for r in ab) / 3600))
    dr = [r for r in allr if r["phase"] == "decision"]
    print("decision (rung) runs: %d, steps %d, est $%.0f (%s of $); promising/undecided/rejected-after-spend hints cost: see section 5" % (
        len(dr), sum(r["committed"] for r in dr), sum(r["usd"] for r in dr), pct(sum(r["usd"] for r in dr), usd_)))
    print("problems with <3 rounds: %s" % [k for k, P in PS.items() if len(P["rounds"]) < 3])
    print("problems with an empty round: %s" % [k for k, P in PS.items() if any(r["items"] and r["selected"] == 0 for r in P["rounds"])])
    print("final-infra runs: %d; patience-stopped: %d; absorbed-stopped: %d" % (
        sum(1 for r in allr if r["infra"]), sum(1 for r in allr if r["stop"] == "patience"), sum(1 for r in allr if r["stop"] == "external")))
    print("first/head < 1.05: %s" % [(k, round(P["vs_first"], 2)) for k, P in PS.items() if P["vs_first"] and P["vs_first"] < 1.05])
    print()


def sec0(PS):
    print("## 0. One-line-per-problem summary (first/head > 1 = headline faster than first sweep)")
    rows = []
    for n, P in PS.items():
        R = run_effort(P)
        c, _ = step_stats(P)
        tot = sum(c.values()); tu = sum(r["usd"] for r in R)
        lo = sum(r["usd"] for r in R if r["phase"] != "decision" and r["cls"] in ("low-leverage(<1% share)", "stalled big slot(own gain<5%)", "never-started"))
        hs = collections.Counter(h["status"] for h in P["hints"])
        ops = P["final_ops"]; t_ = sum(o["us"] for o in ops) or 1
        rows.append([n, "%.2fx" % P["vs_first"] if P["vs_first"] else "-", "%.1f" % (P["wall_s"] / 3600), len(P["rounds"]),
                     sum(1 for r in P["rounds"] if r["items"] and r["selected"] == 0), tot, pct(c["improved"], tot),
                     c["runtime_error"] + c["compile_error"], sum(1 for r in P["runs"] if r["stop"] == "never-started" or r["infra"]),
                     "$%.0f" % tu, pct(lo, tu), "%d/%d/%d" % (hs["accepted"], hs["rejected"], hs["proposed"]),
                     pct(sum(o["us"] for o in ops if o["seed"].startswith(("hand", "fused-seed"))), t_)])
    table(rows, ["problem", "first/head", "wall h", "rounds", "empty rnds", "KB steps", "impr", "err steps", "infra/never-started runs",
                 "est $", "$ on low-yield slots", "hints acc/rej/undecided", "hand+fseed share"])


def sec10(PS):
    print("## 10. Where slot wall time goes (sum over runs with result.json telemetry; slots run in parallel, so these are slot-seconds)")
    rows = []
    T = collections.Counter()
    for n, P in PS.items():
        c = collections.Counter()
        for r in P["runs"]:
            for k, v in (r["tsec"] or {}).items():
                if k.startswith("llm.") and k != "llm.backoff":
                    c["llm"] += v
                elif k == "llm.backoff":
                    c["backoff"] += v
                elif k == "gpu.queue":
                    c["gpuq"] += v
                elif k.startswith(("build", "ncu", "compile")):
                    c["build+ncu"] += v
                elif k == "reflect":
                    c["reflect"] += v
            c["wall"] += r["wall"] if r["has_result"] else 0
        T.update(c)
        w = c["wall"] or 1
        rr = [x["seconds"] or 0 for x in P["rounds"]]
        dec = sum(x["decide_s"] or 0 for x in P["rounds"])
        rows.append([n, "%.1f" % (c["wall"] / 3600), pct(c["llm"], w), pct(c["backoff"], w), pct(c["gpuq"], w), pct(c["build+ncu"], w),
                     pct(c["reflect"], w), "%.1f" % (sum(rr) / 3600), "%.1f" % (rr[0] / 3600 if rr else 0), "%.2f" % (dec / 3600),
                     "%.1f" % (P["wall_s"] / 3600)])
    w = T["wall"] or 1
    rows.append(["ALL", "%.1f" % (T["wall"] / 3600), pct(T["llm"], w), pct(T["backoff"], w), pct(T["gpuq"], w), pct(T["build+ncu"], w),
                 pct(T["reflect"], w), "", "", "", ""])
    table(rows, ["problem", "slot-wall h", "llm gen wait", "llm backoff (relay 503)", "gpu queue", "build+ncu", "reflect", "sum round h", "round0 h",
                 "decide h", "active wall h"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", default=ROOT)
    ap.add_argument("-p", "--problems", default="")
    ap.add_argument("-s", "--sections", default="0,1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    dirs = find_dirs(a.root, [x for x in a.problems.split(",") if x])
    PS = {n: load_problem(a.root, n, d) for n, d in dirs.items()}
    print("# search_waste: %d finished problems: %s\n" % (len(PS), ", ".join(PS)))
    secs = {"1": sec1, "2": sec2, "3": sec3, "4": sec4, "5": sec5, "6": sec6, "7": sec7, "8": sec8, "9": sec9, "10": sec10, "0": sec0}
    for s in a.sections.split(","):
        if s in secs:
            secs[s](PS)
    if a.json:
        slim = {n: {k: v for k, v in P.items() if k not in ("runs", "final_ops", "first_ops", "decisions")} for n, P in PS.items()}
        with open(a.json, "w") as f:
            json.dump(slim, f, indent=1, default=str)


if __name__ == "__main__":
    main()
