#!/usr/bin/env python3
"""E0 noise floor analysis (X-103). Reads analysis/e0_runs/<problem>_s<seed>/ (new replicates, frozen tree 20261025-123252),
the sweep's results/<problem>/ (older code, seed 0: flagged extra sample) and, when present, the beam/hetero test's
control arm (same tree, seed 0). Writes e0_data/e0_runs.json, e0_runs.csv, e0_stats.json and e0_curves.png.

Cost: assumed Opus list prices ($/M): uncached input 5, output 25, cache write 6.25, cache read 0.50 (relay cost fields are 0).
Per round/role from state.json rounds[].telemetry.kb.llm.by_role_model; the relay ledger slice (ledger.jsonl) is a cross-check.
Run with the matplotlib venv python."""
import csv, datetime, json, math, sys
from pathlib import Path
import numpy as np

L3 = Path(__file__).resolve().parents[3]
RUNS = L3 / "analysis" / "e0_runs"
DATA = L3 / "analysis" / "e0_data"
PRICE = dict(uncached=5.0, cache_read=0.5, cache_write=6.25, output=25.0)
PROBLEMS = ["1_MLP", "41_GRUBidirectional", "43_MinGPTCausalAttention"]
EXTRA = json.loads((DATA / "extra_samples.json").read_text()) if (DATA / "extra_samples.json").exists() else {}


def usd(br):
    return (br.get("uncached_input_tokens", 0) * PRICE["uncached"] + br.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
            + br.get("cache_write_input_tokens", 0) * PRICE["cache_write"] + br.get("output_tokens", 0) * PRICE["output"]) / 1e6


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def ledger_tally(p):
    tot = dict(calls=0, input=0, output=0, cache_read=0, cache_write=0, usd=0.0, roles={})
    if not Path(p).exists():
        return None
    for ln in open(p):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        cr, cw = d.get("cache_read_input_tokens", 0) or 0, d.get("cache_write_input_tokens", 0) or 0
        pi, out = d.get("prompt_tokens", 0) or 0, d.get("completion_tokens", 0) or 0
        unc = max(0, pi - cr - cw)
        u = (unc * PRICE["uncached"] + cr * PRICE["cache_read"] + cw * PRICE["cache_write"] + out * PRICE["output"]) / 1e6
        n = d.get("calls", 1) or 1
        tot["calls"] += n; tot["input"] += pi; tot["output"] += out; tot["cache_read"] += cr; tot["cache_write"] += cw
        tot["usd"] += u
        role = next((w.split("=", 1)[1] for w in (d.get("what") or "").split() if w.startswith("role=")), "?")
        r = tot["roles"].setdefault(role, dict(calls=0, usd=0.0))
        r["calls"] += n; r["usd"] += u
    return tot


def one(name, wdir, rdir=None, kind="e0", meta=None):
    """wdir: the run's work dir (state.json, summary.json); rdir: the replicate dir (t0/t1, remeasure.json, posthoc.json)."""
    st, sm = jload(wdir / "state.json"), jload(wdir / "summary.json")
    if not st:
        return None
    rec = dict(name=name, kind=kind, **(meta or {}))
    ir = (sm or {}).get("irs", {}).get("kb_problem", {}) if sm else {}
    rec.update(final_us=ir.get("model_e2e_us"), seed_us=ir.get("seed_e2e_us"), speedup_vs_seed=ir.get("model_speedup"),
               model_passed=ir.get("model_passed"), torch_status=ir.get("torch_status"), torch_graph_us=ir.get("torch_graph_us"),
               torch_best_us=ir.get("torch_best_us"), torch_best_variant=ir.get("torch_best_variant"),
               flow_speedup_vs_torch=ir.get("speedup_vs_torch"),
               accepted=(sm or {}).get("accepted", []), rejected=(sm or {}).get("rejected", []),
               fused=ir.get("fused"), n_rounds=len(st["rounds"]), fixpoint=st.get("fixpoint"), complete=bool(sm))
    rounds, cum_calls, cum_usd = [], 0, 0.0
    roles = {}
    for r in st["rounds"]:
        llm = (((r.get("telemetry") or {}).get("kb") or {}).get("llm") or {})
        tot = llm.get("total") or {}
        ru = sum(usd(b) for b in llm.get("by_role_model", []))
        for b in llm.get("by_role_model", []):
            x = roles.setdefault(b["role"], dict(calls=0, usd=0.0, output_tokens=0, input_tokens=0))
            x["calls"] += b.get("calls", 0); x["usd"] += usd(b)
            x["output_tokens"] += b.get("output_tokens", 0); x["input_tokens"] += b.get("input_tokens", 0)
        rounds.append(dict(round=r["round"], partition_us=sum((r.get("partition_us") or {}).values()) or None,
                           seconds=r.get("seconds"), calls=tot.get("calls", 0), usd=ru,
                           input_tokens=tot.get("input_tokens", 0), output_tokens=tot.get("output_tokens", 0),
                           cache_read=tot.get("cache_read_input_tokens", 0), cache_write=tot.get("cache_write_input_tokens", 0),
                           calls_before=cum_calls, usd_before=cum_usd,
                           kb_problems=len(r.get("kb_runs") or []),
                           decisions=[(d.get("hint"), d.get("status"), d.get("gain")) for d in r.get("decisions", [])]))
        cum_calls += tot.get("calls", 0); cum_usd += ru
    rec.update(rounds=rounds, calls=cum_calls, usd=cum_usd, roles=roles,
               input_tokens=sum(r["input_tokens"] for r in rounds), output_tokens=sum(r["output_tokens"] for r in rounds),
               flow_seconds_sum_rounds=sum(r["seconds"] or 0 for r in rounds))
    if rdir is not None:
        def t(n):
            p = rdir / n
            return float(p.read_text().split()[0]) if p.exists() else None
        t0, te, t1 = t("t0"), t("t_extract"), t("t1")
        rec.update(wall_s=(t1 - te) if t1 and te else None, wall_total_s=(t1 - t0) if t1 and t0 else None,
                   flow_status=(rdir / "flow_status").read_text().strip() if (rdir / "flow_status").exists() else None)
        rm = jload(rdir / "remeasure.json")
        if rm:
            rec.update(remeasure_us=rm.get("model_e2e_us"), remeasure_basis=rm.get("model_basis"),
                       remeasure_seed_us=rm.get("seed_e2e_us"), remeasure_passed=rm.get("passed"),
                       remeasure_torch_status=rm.get("torch_status"), remeasure_torch_graph_us=rm.get("torch_graph_us"))
        ph = jload(rdir / "posthoc2.json") or jload(rdir / "posthoc.json")   # posthoc2: rerun with writable compile caches
        if ph:
            rec.update(posthoc_torch_eager_us=ph.get("eager_us"), posthoc_torch_graph_us=ph.get("graph_us"),
                       posthoc_torch_compile_us=ph.get("compile_us"), posthoc_others=ph.get("others_before"))
        rec["ledger"] = ledger_tally(rdir / "ledger.jsonl")
    tg = rec.get("posthoc_torch_graph_us") or rec.get("torch_graph_us")
    q = rec.get("remeasure_us") or rec.get("final_us")
    rec["torch_graph_ref_us"] = tg
    rec["speedup_vs_torch_graph"] = (tg / q) if tg and q else None
    return rec


def collect():
    recs = []
    for d in sorted(RUNS.glob("*_s*")):
        prob, seed = d.name.rsplit("_s", 1)
        r = one(d.name, d / "work", d, "e0", dict(problem=prob, seed=int(seed), code="frozen 20261025-123252"))
        if r:
            recs.append(r)
    for prob in PROBLEMS:   # the sweep's own sample: older code, seed 0
        w = L3 / "results" / prob / "work"
        r = one(f"{prob}_sweep", w, None, "sweep", dict(problem=prob, seed=0, code="sweep (older tree)"))
        if r:
            pj = jload(L3 / "results" / prob / "torch_posthoc.json")
            if pj:
                r["posthoc_torch_graph_us"] = pj.get("graph_us"); r["posthoc_torch_compile_us"] = pj.get("compile_us")
                r["torch_graph_ref_us"] = pj.get("graph_us") or r.get("torch_graph_us")
                r["speedup_vs_torch_graph"] = r["torch_graph_ref_us"] / r["final_us"] if r["torch_graph_ref_us"] and r["final_us"] else None
            recs.append(r)
    for name, x in EXTRA.items():   # e.g. the beam/hetero test's control arm (same tree, seed 0)
        d = Path(x["dir"])
        r = one(name, d / x.get("work", "work"), d if x.get("rdir") else None, x["kind"],
                dict(problem=x["problem"], seed=x.get("seed", 0), code=x.get("code", "")))
        if r:
            r.update({k: v for k, v in x.get("set", {}).items()})
            recs.append(r)
    return recs


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.array(xs, float)
    sd = float(a.std(ddof=1)) if len(a) > 1 else None
    return dict(n=len(a), mean=float(a.mean()), sd=sd, cv=(sd / a.mean()) if sd is not None else None,
                min=float(a.min()), max=float(a.max()), range_pct=float((a.max() - a.min()) / a.mean() * 100),
                sd_log=float(np.log(a).std(ddof=1)) if len(a) > 1 else None)


# Student t quantiles (two-sided alpha 0.05 -> 0.975; power 0.8 -> 0.80), df 1..60
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179,
        14: 2.145, 16: 2.120, 18: 2.101, 20: 2.086, 30: 2.042, 60: 2.000}


def tq(df):
    k = max(d for d in T975 if d <= df)
    return T975[k]


def power(delta_sd, n, sims=40000, rng=np.random.default_rng(0)):
    """Power of a two-sided two-sample Student t test (alpha 0.05, n per arm) for a true difference of delta_sd sd's."""
    df = 2 * n - 2
    z = rng.standard_normal(sims) + delta_sd / math.sqrt(2 / n)
    c = rng.chisquare(df, sims) / df
    return float(np.mean(np.abs(z / np.sqrt(c)) > tq(df)))


def mde_sd(n, target=0.8):
    lo, hi = 0.0, 40.0
    for _ in range(40):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if power(m, n) < target else (lo, m)
    return hi


def n_needed(effect_sd, target=0.8):
    for n in range(2, 101):
        if power(effect_sd, n) >= target:
            return n
    return None


def main():
    recs = collect()
    DATA.mkdir(exist_ok=True)
    (DATA / "e0_runs.json").write_text(json.dumps(recs, indent=1, default=str))
    cols = ["name", "kind", "problem", "seed", "code", "flow_status", "final_us", "remeasure_us", "seed_us", "speedup_vs_seed",
            "torch_graph_ref_us", "posthoc_torch_compile_us", "speedup_vs_torch_graph", "calls", "input_tokens", "output_tokens",
            "usd", "wall_s", "wall_total_s", "flow_seconds_sum_rounds", "n_rounds", "fixpoint", "accepted", "rejected"]
    with open(DATA / "e0_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["ledger_calls", "ledger_usd"] + [f"r{i}_partition_us" for i in range(3)])
        for r in recs:
            lg = r.get("ledger") or {}
            w.writerow([";".join(r[c]) if c in ("accepted", "rejected") else (round(r[c], 3) if isinstance(r.get(c), float) else r.get(c)) for c in cols]
                       + [lg.get("calls"), round(lg["usd"], 2) if lg else None]
                       + [(r["rounds"][i]["partition_us"] if i < len(r["rounds"]) else None) for i in range(3)])
    out = {"per_problem": {}, "pooled": {}}
    for prob in PROBLEMS:
        e0 = [r for r in recs if r["problem"] == prob and r["kind"] == "e0" and r.get("final_us")]
        same = [r for r in recs if r["problem"] == prob and r["kind"] == "same_tree" and r.get("final_us")]
        pp = {}
        for label, grp in (("e0", e0), ("e0+same_tree", e0 + same)):
            pp[label] = {m: stats([r.get(m) for r in grp]) for m in ("final_us", "remeasure_us", "usd", "calls", "wall_s",
                                                                       "output_tokens", "speedup_vs_torch_graph")}
            pp[label]["names"] = [r["name"] for r in grp]
        pp["accepted_sets"] = {r["name"]: r["accepted"] for r in recs if r["problem"] == prob}
        sw = next((r for r in recs if r["problem"] == prob and r["kind"] == "sweep"), None)
        if sw and e0:
            fs = [r["final_us"] for r in e0 + same]
            m, s = float(np.mean(fs)), float(np.std(fs, ddof=1)) if len(fs) > 1 else None
            pp["sweep_vs_new"] = dict(sweep_final_us=sw["final_us"], new_mean=m, new_sd=s,
                                      z=(sw["final_us"] - m) / s if s else None,
                                      within_range=min(fs) <= sw["final_us"] <= max(fs),
                                      sweep_usd=sw["usd"], sweep_calls=sw["calls"])
        out["per_problem"][prob] = pp
    # pooled within-problem sd on the log scale (= ~CV) with df = sum(n_i - 1)
    for metric in ("final_us", "remeasure_us", "usd", "wall_s", "calls"):
        for label in ("e0", "e0+same_tree"):
            ss, df = 0.0, 0
            for prob in PROBLEMS:
                grp = [r[metric] for r in recs if r["problem"] == prob and r.get(metric)
                       and r["kind"] in (("e0",) if label == "e0" else ("e0", "same_tree"))]
                if len(grp) > 1:
                    lg = np.log(grp)
                    ss += float(((lg - lg.mean()) ** 2).sum()); df += len(grp) - 1
            if df:
                out["pooled"].setdefault(metric, {})[label] = dict(sd_log=math.sqrt(ss / df), df=df)
    # minimum detectable effect (two-sided two-sample t test, alpha 0.05, power 0.8), in units of sd and in % (log scale)
    m2, m3, m5 = mde_sd(2), mde_sd(3), mde_sd(5)
    out["mde_sd_units"] = {"n2": m2, "n3": m3, "n5": m5, "test": "two-sided two-sample Student t, alpha 0.05, power 0.8, "
                           "equal n per arm, on log(metric); sd = pooled within-problem sd of log(metric)"}
    for metric, d in out["pooled"].items():
        for label, v in d.items():
            s = v["sd_log"]
            v["mde_pct"] = {f"n{n}": (math.exp(k * s) - 1) * 100 for n, k in ((2, m2), (3, m3), (5, m5))}
            v["n_for_effect"] = {f"{e}%": n_needed(math.log(1 + e / 100) / s) if s > 0 else 2 for e in (5, 10, 20, 30, 50)}
            # pooling K problems per arm (one replicate each problem per arm): sd of the mean log ratio shrinks by sqrt(K)
            v["mde_pct_k_problems_n1"] = {f"K{k}": (math.exp(1.96 * 1.0 * s * math.sqrt(2 / k) + 0.84 * s * math.sqrt(2 / k)) - 1) * 100
                                          for k in (3, 5, 10)}
    # per-problem MDE from the problem's own sd_log (df = n-1: the sd itself is very uncertain)
    for prob, pp in out["per_problem"].items():
        for label in ("e0", "e0+same_tree"):
            for metric in ("final_us", "usd", "calls", "wall_s"):
                st_ = pp[label].get(metric)
                if st_ and st_.get("sd_log"):
                    s_ = st_["sd_log"]
                    st_["mde_pct"] = {f"n{n}": (math.exp(k * s_) - 1) * 100 for n, k in ((2, m2), (3, m3), (5, m5))}
    (DATA / "e0_stats.json").write_text(json.dumps(out, indent=1, default=str))
    plot(recs)
    print(json.dumps(out, indent=1, default=str)[:6000])


def plot(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, SUB, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    E0C = ["#2a6fdb", "#e0731f", "#1a9e77"]

    def style(r, k):
        if r["kind"] == "e0":
            return E0C[k % 3], "-", f"seed {r['seed']} (frozen tree)"
        if r["kind"] == "sweep":
            return "#7a7a7a", "--", "sweep (older code, seed 0)"
        return "#b03a8c", ":", "beam-test control C (same tree, seed 0)"

    def curve(r, xkey):
        xs, ys, best = [], [], None
        for rd in r["rounds"]:
            if rd["partition_us"]:
                best = rd["partition_us"] if best is None else min(best, rd["partition_us"])
                xs.append(rd[xkey]); ys.append(best)
        if r.get("final_us"):
            xs.append(r["calls"] if xkey == "calls_before" else r["usd"]); ys.append(r["final_us"])
        return xs, ys

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor=BG)
    for j, prob in enumerate(PROBLEMS):
        grp = [r for r in recs if r["problem"] == prob]
        k = 0
        for r in grp:
            c, ls, lab = style(r, k)
            k += r["kind"] == "e0"
            for row, (xkey, zoom) in enumerate((("calls_before", False), ("calls_before", True), ("usd_before", True))):
                ax = axes[row][j]
                xs, ys = curve(r, xkey)
                if zoom:
                    xs, ys = xs[1:], ys[1:]
                ax.step(xs, ys, where="post", color=c, ls=ls, lw=1.8, marker="o", ms=4, label=lab)
        for row in range(3):
            a = axes[row][j]
            if row == 0:
                a.set_yscale("log")
            a.set_facecolor(BG); a.grid(True, color=GRID, lw=0.8)
            a.set_xlabel("cumulative LLM $ (assumed Opus prices)" if row == 2 else "cumulative LLM calls", color=SUB)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
            a.tick_params(colors=SUB)
        axes[0][j].set_title(prob, color=INK, fontsize=12, loc="left")
        axes[0][j].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("best-so-far model time (us, log)", color=SUB)
    axes[1][0].set_ylabel("best-so-far after round 0 (us)", color=SUB)
    axes[2][0].set_ylabel("best-so-far after round 0 (us)", color=SUB)
    fig.suptitle("E0 noise floor: best-so-far model time (start-of-round whole-model profile; last point = final model e2e, CUDA graph) "
                 "vs cumulative LLM calls (rows 1-2) and $ (row 3)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(DATA / "e0_curves.png", dpi=120, facecolor=BG)

    # spread: each run's final time / cost / wall relative to the problem's e0 mean
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=BG)
    for a, (m, lab) in zip(axes, (("final_us", "final model time"), ("usd", "LLM cost"), ("wall_s", "flow wall time"))):
        for j, prob in enumerate(PROBLEMS):
            e0 = [r[m] for r in recs if r["problem"] == prob and r["kind"] == "e0" and r.get(m)]
            if not e0:
                continue
            mu = float(np.mean(e0)); k = 0
            for r in recs:
                if r["problem"] != prob or not r.get(m):
                    continue
                c, ls, l2 = style(r, k); k += r["kind"] == "e0"
                mk = "o" if r["kind"] == "e0" else ("s" if r["kind"] == "sweep" else "D")
                a.scatter(r[m] / mu, j + (0 if r["kind"] == "e0" else 0.18), color=c, marker=mk, s=46, zorder=3)
        a.axvline(1, color=SUB, lw=0.8)
        a.set_yticks(range(len(PROBLEMS))); a.set_yticklabels(PROBLEMS if a is axes[0] else [""] * len(PROBLEMS))
        a.set_title(f"{lab} / e0 mean", loc="left", color=INK, fontsize=11)
        a.set_facecolor(BG); a.grid(True, axis="x", color=GRID, lw=0.8); a.tick_params(colors=SUB)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[2].text(1.0, -0.22, "circles: e0 replicates (frozen tree); squares: sweep (older code); diamond: beam-test control C",
                 transform=axes[2].transAxes, ha="right", color=SUB, fontsize=8)
    fig.tight_layout()
    fig.savefig(DATA / "e0_spread.png", dpi=120, facecolor=BG)


if __name__ == "__main__":
    main()
