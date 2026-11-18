#!/usr/bin/env python3
"""E1 (X-120) analysis: e1_data/runs/<P> (main session) + e1_data/runs_fixed/<P> (second session, write-back-fixed reuse)
-> e1_data/e1_results.json, e1_table.csv, e1_summary.json and the markdown tables printed on stdout.

Per session, per problem: every arm = 6 fresh-process CUDA-graph-replay runs, interleaved, on one node; the PyTorch graph
baseline = median of 3 fresh invocations on the same GPU right after (each a median of 50 events).
Arms:
  orig          shipped kernels, sequential walker (re-measured)
  orig+on       shipped + --graph-concurrency on (= orig when the graph has no parallel branch)
  orig+auto     decided on runs 1-3 of (orig, orig+on) back to back (concurrency kept only if not slower); value = median of
                runs 4-6 of the chosen arm (decision and reported number use different runs)
  sib / glob    --reuse-winners siblings / global (global = siblings' sources + 4 similar-shape sources of other problems),
                sequential, the kb_reuse pass AS SHIPPED (the pass reverts itself when its model guard fails)
  sibf / globf  the same with the write-back fix (e1_kb_reuse_writeback.patch), run only where the as-shipped pass reverted
  widef         audit set only: global + fix with max_kinds 8, min_share 0.3%
  X+on, X+auto  with concurrency on / auto
Times of arms from the second session are transported to the main session by their ratio to that session's own `orig`.
"""
import json, math, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]
DATA = L3 / "analysis" / "e1_data"
NOISE_MIN = 0.02            # a change below 2% is never called a win or a regression
STATES = ("orig", "sib", "glob", "sibf", "globf", "widef")


def med(x):
    return st.median(x) if x else None


def cv(x):
    return (st.pstdev(x) / st.mean(x)) if len(x) > 1 else None


def load_torch(d):
    out = []
    for i in (1, 2, 3):
        f = d / f"torch_{i}.txt"
        if f.exists():
            for ln in f.read_text().splitlines():
                if ln.startswith("@@POSTHOC@@"):
                    out.append(json.loads(ln[len("@@POSTHOC@@"):]))
    return out


def smi(d):
    f = d / "smi_before.csv"
    if not f.exists():
        return None
    rows = f.read_text().splitlines()
    try:
        hdr = [h.strip() for h in rows[0].split(",")]
        v = [x.strip() for x in rows[1].split(",")]
        row = dict(zip(hdr, v))
        return {"max_mem_mhz": row.get("clocks.max.memory [MHz]"), "vbios": row.get("vbios_version"),
                "gpu": row.get("uuid"), "others": [r for r in rows[2:] if r.strip()]}
    except Exception:  # noqa: BLE001
        return None


def session(d):
    """one e1.json -> {arm: {...}}, checks, notes"""
    f = d / "e1.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    if "times" not in j:
        return {"j": j, "arms": {}, "bad": {}, "error": "no timings"}
    alias, times, chk = j.get("alias", {}), j["times"], j.get("check", {})

    def res(a):
        while a in alias:
            a = alias[a]
        return a

    T = lambda a: times.get(res(a)) or []
    arms = {}
    for s in STATES:
        if f"{s}_seq" not in times and f"{s}_seq" not in alias:
            continue
        seq, conc = T(f"{s}_seq"), T(f"{s}_conc")
        conc = conc or seq
        arms[s] = {"runs": seq}
        arms[s + "+on"] = {"runs": conc}
        use = (med(conc[:3]) or 1e99) <= (med(seq[:3]) or 1e99)
        arms[s + "+auto"] = {"runs": (conc if use else seq)[3:], "conc": bool(use and res(f"{s}_conc") != res(f"{s}_seq"))}
    for v in arms.values():
        v["med"], v["cv"], v["min"], v["max"], v["n"] = med(v["runs"]), cv(v["runs"]), \
            (min(v["runs"]) if v["runs"] else None), (max(v["runs"]) if v["runs"] else None), len(v["runs"])
    bad = {}
    for a in set(list(times) + list(alias)):
        c = chk.get(res(a))
        why = []
        if c is None:
            why.append("no check run")
        else:
            if not c.get("passed"):
                why.append("not passed")
            if any(not o.get("ok") for o in (c.get("outputs") or {}).values()):
                why.append("output check")
            if c.get("bad_slots"):
                why.append(f"slot check {c['bad_slots'][:3]}")
            if c.get("deadlock"):
                why.append("deadlock")
        if not why and not times.get(res(a)):
            why.append("no timed run")
        if why:
            bad[a] = ", ".join(why)
    return {"j": j, "arms": arms, "bad": bad, "alias": alias, "chk": chk}


def arm_ok(k, bad):
    s = k.split("+")[0]
    need = [f"{s}_seq"] + ([f"{s}_conc"] if k.endswith(("+on", "+auto")) else [])
    return not any(n in bad for n in need)


def analyze(p):
    m = session(DATA / "runs" / p)
    if m is None:
        return None
    r = {"problem": p, "errors": m["j"].get("errors"), "identical": m["j"].get("as_shipped_identical"),
         "identity": m["j"].get("identity"), "reps": m["j"].get("reps"), "smi": smi(DATA / "runs" / p)}
    if not m["arms"]:
        r["status"] = m.get("error")
        return r
    fx = session(DATA / "runs_fixed" / p)
    arms = {k: dict(v, session="main") for k, v in m["arms"].items()}
    bad = {k: v for k, v in m["bad"].items()}
    o = arms["orig"]["med"]
    r["orig"] = o
    sess = {"main": {"orig": o}}
    if fx and fx["arms"]:
        fo = fx["arms"]["orig"]["med"]
        sess["fixed"] = {"orig": fo, "smi": smi(DATA / "runs_fixed" / p)}
        r["orig_repeat"] = {"main": o, "fixed": fo, "diff_pct": 100 * (fo / o - 1) if o and fo else None}
        for k, v in fx["arms"].items():
            if k.startswith("orig"):
                continue
            nm = k
            if nm in arms:                    # same state in both sessions (only-fixed mode never has sib/glob)
                nm = k + "@2"
            ratio = (v["med"] / fo) if v["med"] and fo else None
            arms[nm] = dict(v, session="fixed", ratio=ratio, med_session=v["med"], med=(o * ratio if ratio and o else None))
        for k, w in fx["bad"].items():
            if not k.startswith("orig"):
                bad[k] = w + " (fixed session)"
        r["fixed_session"] = {"errors": fx["j"].get("errors"), "reuse": {t: {"changed": v.get("changed"), "reverted": v.get("reverted")}
                                                                          for t, v in (fx["j"].get("reuse") or {}).items()}}
    # torch
    tor = load_torch(DATA / "runs" / p)
    gus, eus = [t["graph_us"] for t in tor if t.get("graph_us")], [t["eager_us"] for t in tor if t.get("eager_us")]
    r["torch"] = {"n": len(tor), "graph_runs": gus, "eager_runs": eus, "graph": med(gus), "eager": med(eus), "cv_graph": cv(gus),
                  "basis": "graph" if gus else ("eager" if eus else None)}
    if fx:
        t2 = load_torch(DATA / "runs_fixed" / p)
        g2 = [t["graph_us"] for t in t2 if t.get("graph_us")]
        r["torch"]["graph_runs_fixed_session"] = g2
    base = med(gus) or med(eus)
    r["torch_us"] = base
    r["vs_torch"] = {k: (base / v["med"] if base and v["med"] else None) for k, v in arms.items()}
    valid = {k: v for k, v in arms.items() if v["med"] and arm_ok(k.split("@")[0], bad)}
    best = min(valid, key=lambda k: valid[k]["med"]) if valid else "orig"
    r["best_arm"], r["best"] = best, (valid[best]["med"] if valid else o)
    r["gain"] = {k: ((o / v["med"] - 1) if v["med"] and o else None) for k, v in arms.items()}
    r["gain_best"] = (o / r["best"] - 1) if r["best"] and o else None
    r["vs_torch_best"] = base / r["best"] if base and r["best"] else None
    r["noise_thr"] = max(NOISE_MIN, 3 * (arms["orig"]["cv"] or 0))
    # which feature wins
    g = r["gain"]
    thr = r["noise_thr"]
    reuse_keys = [k for k in arms if k in ("sib", "glob", "sibf", "globf", "widef") or k.split("@")[0] in ("sib", "glob", "sibf", "globf", "widef")]
    r["reuse_gain"] = max([g[k] or -9 for k in reuse_keys] or [0])
    r["conc_gain"] = g.get("orig+auto") or 0
    r["win_reuse"] = r["reuse_gain"] > thr
    r["win_conc"] = r["conc_gain"] > thr
    r["bad"] = bad
    r["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "runs"} | {"runs": v.get("runs")} for k, v in arms.items()}
    r["concurrency"] = m["j"].get("concurrency")
    r["alias"] = m.get("alias")
    r["graph_skip"] = {a: c.get("graph_skip") for a, c in (m.get("chk") or {}).items() if c.get("graph_skip")}
    r["graph_nodes"] = {a: c.get("graph_nodes") for a, c in (m.get("chk") or {}).items()}
    rep = dict(m["j"].get("reuse") or {})
    if fx:
        rep.update({t: v for t, v in (fx["j"].get("reuse") or {}).items()})
    r["reuse"] = {t: {"changed": v.get("changed"), "reverted": v.get("reverted"), "wall_s": v.get("wall_s"), "evals": v.get("evals"),
                      "guard": [v.get("guard", {}).get("before_us"), v.get("guard", {}).get("after_us")],
                      "kinds": [{"kind": k["kind"], "share": k.get("share"), "ops": k.get("ops"), "pool": len(k.get("pool") or []),
                                 "result": k.get("result") or k.get("skipped") or k.get("error"), "before_us": k.get("before_us"),
                                 "after_us": k.get("after_us"), "incumbents_added": k.get("incumbents_added")}
                                for k in v.get("kinds", [])]} for t, v in rep.items()}
    r["wall_s"] = (m["j"].get("t1") or 0) - (m["j"].get("t0") or 0)
    # seed-left: share of the profile's op time that sits in slots still on a hand seed (in the arm's slots)
    try:
        j = m["j"]
        ops = {n: (k, us) for n, k, us in j["profile"]["ops"]}
        tot = sum(us for _, us in ops.values()) or 1
        ss = {}
        res = lambda a: (lambda f: f(f, a))(lambda f, a: f(f, m["alias"][a]) if a in m["alias"] else a)
        for a in ("orig_seq", "sib_seq", "glob_seq"):
            sl, om = j.get("arm_slots", {}).get(res(a)), j.get("arm_ops", {}).get(res(a))
            if not sl or not om:
                continue
            by = {}
            for n, sid in om.items():
                if n in ops and not str(sl[sid]["seed_source"]).startswith("cache:"):
                    by[ops[n][0]] = by.get(ops[n][0], 0) + ops[n][1]
            ss[a] = {"total": sum(by.values()) / tot, "by_kind": {k: round(v / tot, 4) for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:4]}}
        r["seed_share"] = ss
    except Exception as e:  # noqa: BLE001
        r["seed_share"] = {"error": repr(e)}
    return r


def geo(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    runs = DATA / "runs"
    rows = [analyze(p.name) for p in sorted(runs.iterdir(), key=lambda q: int(q.name.split("_")[0])) if p.is_dir()]
    rows = [r for r in rows if r]
    good = [r for r in rows if r.get("arms")]
    (DATA / "e1_results.json").write_text(json.dumps(rows, indent=1, default=str))
    f1 = lambda x, n=1: f"{x:.{n}f}" if x else ""
    hdr = ["problem", "torch_us", "orig", "conc_auto", "sib", "glob", "sibf", "globf", "widef", "sib+auto", "glob+auto", "best", "best_arm",
           "gain_best_pct", "vs_torch_orig", "vs_torch_best"]
    lines = [",".join(hdr)]
    for r in good:
        a = r["arms"]
        g = lambda k: f1(a[k]["med"]) if k in a else ""
        lines.append(",".join(map(str, [r["problem"], f1(r["torch_us"]), g("orig"), g("orig+auto"), g("sib"), g("glob"), g("sibf"),
                                         g("globf"), g("widef"), g("sib+auto"), g("glob+auto"), f1(r["best"]), r["best_arm"],
                                         f1(100 * r["gain_best"]), f1(r["vs_torch"].get("orig"), 3), f1(r["vs_torch_best"], 3)])))
    (DATA / "e1_table.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    def agg(fn):
        xs = [x for x in (fn(r) for r in good) if x]
        return {"mean": st.mean(xs), "geomean": geo(xs), "n": len(xs)} if xs else None

    def best_of(r, keys):
        c = [r["arms"][k]["med"] for k in keys if k in r["arms"] and r["arms"][k]["med"] and arm_ok(k.split("@")[0], r["bad"])]
        return min(c) if c else None

    def vs(r, t):
        return r["torch_us"] / t if t and r["torch_us"] else None

    summ = {
        "orig": agg(lambda r: r["vs_torch"].get("orig")),
        "conc_auto": agg(lambda r: r["vs_torch"].get("orig+auto")),
        "sib": agg(lambda r: vs(r, best_of(r, ["sib", "sibf"]))),
        "glob": agg(lambda r: vs(r, best_of(r, ["glob", "globf"]))),
        "sib+auto": agg(lambda r: vs(r, best_of(r, ["sib+auto", "sibf+auto"]))),
        "glob+auto": agg(lambda r: vs(r, best_of(r, ["glob+auto", "globf+auto"]))),
        "best": agg(lambda r: r["vs_torch_best"]),
        "n_problems": len(good),
    }
    (DATA / "e1_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
