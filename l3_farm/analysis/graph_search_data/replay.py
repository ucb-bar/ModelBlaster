#!/usr/bin/env python3
"""Offline replay of prior pre-application (A4 graph search, X-133) on the Level 3 sweep's logs (E7 dataset).

Question: how much round-0 slot-optimization spend (LLM calls, $) would pre-applying high-confidence fusions at
round 0 have saved, versus E7's "fused-away" spend?

Method (per problem P of the 46-problem replay sample; leave-one-problem-out priors):
- priors: graph_priors.table_from_decisions over every OTHER problem's decisions (main + degraded samples);
  a hint kind is pre-applied when its prior has >= MIN_N decisions and the Beta(acc+1, rej+1) 10% quantile >= CONF
  (the flow's defaults: 3, 0.55); exploration draws (graph_priors.explore, rate EPS) leave some to measurement.
- the hints considered are the ones the sweep decided for P (decisions.csv): the proposals we know of.
- a round-0 opt slot problem (kbruns.csv) of op kind K is "absorbed" by a pre-applied fusion F when K is a stage of
  F and F's fused ops end up in P's final model (fused_away_frac > 0). Its absorbed spend = usd x fused_away_frac.
  * upper: K is a stage of ANY pre-applied fusion of P that the sweep accepted;
  * lower: K's every accepted fusion in P is pre-applied (no non-eligible fusion could have absorbed it);
  * net: as lower, excluding head stages (a fused op keeps a slot problem of about the head's cost).
- also saved: the variant-decision runs of pre-applied hints (kbruns phase=variant), except explored ones.
- risk: pre-applied hints that the sweep REJECTED in P (false positives): count, and their measured loss.
- cost: the characterization pass (1 step x 1 rollout on the top-8 round-0 items) from steps.csv step-0 costs.
Outputs replay.json (all numbers) and prints the summary table.
"""
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
E7 = HERE.parent / "e7_data"
MB = Path.home() / "Documents/ModelBlaster-clone/ModelBlaster"
sys.path.insert(0, str(MB.parent))
from ModelBlaster.pipeline import graph_priors as gp  # noqa: E402

EXCL_ROUND = {"10_ResNet101", "22_EfficientNetB0"}       # E7: round 0 lost to infra


def base(k):
    return gp._base(k)


def run(conf=0.55, min_n=3, eps=0.0, seed=0):
    dec = list(csv.DictReader(open(E7 / "decisions.csv")))
    kb = list(csv.DictReader(open(E7 / "kbruns.csv")))
    steps = list(csv.DictReader(open(E7 / "steps.csv")))
    probs = sorted({r["problem"] for r in kb if r["sample"] == "main"} - EXCL_ROUND)
    spend = defaultdict(float)
    calls = defaultdict(int)
    for r in kb:
        if r["sample"] == "main":
            spend[r["problem"]] += float(r["usd"] or 0)
            calls[r["problem"]] += int(r["n_calls"] or 0)
    # characterization cost: mean $ / calls of a step-0 (rollout 0) round-0 opt step, x 8 items
    s0 = [r for r in steps if r["sample"] == "main" and r["round"] == "0" and r["phase"] == "opt"
          and r["step"] == "0" and r["rollout"] == "0"]
    s0_usd = statistics.mean(float(r["usd"] or 0) for r in s0)
    s0_calls = statistics.mean(int(r["n_calls"] or 0) for r in s0)
    out = {"conf": conf, "min_n": min_n, "eps": eps, "problems": {}, "char_step0_usd": s0_usd,
           "char_step0_calls": s0_calls}
    tot = defaultdict(float)
    for P in probs:
        t = gp.table_from_decisions(E7 / "decisions.csv", exclude_problem=P)
        # also drop the problem's @v1 first pass (same model)
        rows = [(r["hint"], r["status"]) for r in dec if r["sample"] in ("main", "degraded")
                and r["problem"].split("@")[0] != P]
        t["counts"] = gp._status_counts(rows)
        pri = gp.Priors("cuda:sm_89|fp16", tables=[t], use_a5=False, min_n=min_n)
        mine = [r for r in dec if r["problem"] == P and r["sample"] == "main" and r["hint"].startswith("fuse:")]
        pre, explored, fp = {}, [], []
        for r in mine:
            kind = r["hint"].split(":", 1)[1]
            p = pri.prior(kind)
            if p["n"] >= min_n and p["lb"] >= conf:
                if gp.explore(seed, r["hint"], eps):
                    explored.append(r["hint"])
                    continue
                pre[r["hint"]] = {"stages": [base(s) for s in gp.stages(kind)], "status": r["status"],
                                  "prior": [p["key"], p["accepted"], p["rejected"], p["lb"]],
                                  "gain": float(r["gain"] or 0), "round": int(r["round"])}
                if r["status"] == "rejected":
                    fp.append({"hint": r["hint"], "gain": float(r["gain"] or 0), "round": int(r["round"]),
                               "orig_us": float(r["original_us"] or 0), "var_us": float(r["variant_us"] or 0)})
        acc_all = [[base(s) for s in gp.stages(r["hint"].split(":", 1)[1])] for r in mine if r["status"] == "accepted"]
        acc_pre = [v["stages"] for v in pre.values() if v["status"] == "accepted"]
        r0 = [r for r in kb if r["problem"] == P and r["sample"] == "main" and r["phase"] == "opt"
              and r["round"] == "0"]
        up = lo = net = 0.0
        up_c = lo_c = net_c = 0.0
        fa_all = fa_c = 0.0
        for r in r0:
            frac = float(r["fused_away_frac"] or 0)
            if frac <= 0:
                continue
            k = base(r["op_kind"])
            usd, nc = float(r["usd"] or 0) * frac, int(r["n_calls"] or 0) * frac
            fa_all += usd
            fa_c += nc
            in_pre = [s for s in acc_pre if k in s]
            in_acc = [s for s in acc_all if k in s]
            if in_pre:
                up += usd
                up_c += nc
                if len(in_pre) == len(in_acc):
                    lo += usd
                    lo_c += nc
                    if all(s.index(k) > 0 for s in in_pre):       # never the head of a pre-applied fusion
                        net += usd
                        net_c += nc
        var_saved = sum(float(r["usd"] or 0) for r in kb if r["problem"] == P and r["phase"] == "variant"
                        and r["hint"] in pre)
        var_calls = sum(int(r["n_calls"] or 0) for r in kb if r["problem"] == P and r["phase"] == "variant"
                        and r["hint"] in pre)
        n_items0 = len(r0)
        char_usd = min(8, n_items0) * s0_usd
        rec = {"spend": spend[P], "calls": calls[P], "preapplied": sorted(pre), "explored": explored,
               "false_positive": fp, "r0_fused_away_usd": fa_all, "r0_fused_away_calls": fa_c,
               "saved_upper_usd": up, "saved_lower_usd": lo, "saved_net_usd": net,
               "saved_upper_calls": up_c, "saved_lower_calls": lo_c, "saved_net_calls": net_c,
               "variant_saved_usd": var_saved, "variant_saved_calls": var_calls, "char_cost_usd": char_usd}
        out["problems"][P] = rec
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                tot[k] += v
        tot["n_fp"] += len(fp)
        tot["n_pre"] += len(pre)
    out["total"] = dict(tot)
    return out


def main():
    res = {}
    for conf, eps in ((0.55, 0.0), (0.55, 0.1), (0.7, 0.0), (0.45, 0.0)):
        r = run(conf=conf, eps=eps)
        res[f"conf{conf}_eps{eps}"] = r
        t = r["total"]
        S = t["spend"]
        print(f"conf {conf} eps {eps}: {int(t['n_pre'])} hints pre-applied in {sum(1 for p in r['problems'].values() if p['preapplied'])}"
              f"/{len(r['problems'])} problems, {int(t['n_fp'])} false positives | slot spend ${S:,.0f} ({t['calls']:.0f} calls)")
        print(f"  round-0 fused-away (all accepted fusions): ${t['r0_fused_away_usd']:,.0f} = {100*t['r0_fused_away_usd']/S:.1f}% "
              f"({t['r0_fused_away_calls']:.0f} calls)")
        for k in ("upper", "lower", "net"):
            print(f"  saved {k:5s}: ${t[f'saved_{k}_usd']:,.0f} = {100*t[f'saved_{k}_usd']/S:.1f}% of spend, "
                  f"{t[f'saved_{k}_calls']:.0f} calls ({100*t[f'saved_{k}_calls']/t['calls']:.1f}%); "
                  f"{100*t[f'saved_{k}_usd']/max(t['r0_fused_away_usd'],1e-9):.0f}% of the round-0 fused-away spend")
        print(f"  + variant decisions skipped: ${t['variant_saved_usd']:,.0f} ({t['variant_saved_calls']:.0f} calls); "
              f"- characterization cost ~${t['char_cost_usd']:,.0f}")
    (HERE / "replay.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- survival-weighted round-0 budgets (component 3)

def survival_replay(head_weight=0.5, steps=4, conf=0.55, min_n=3, preapply=True):
    """Round-0 slot problems get max(1, round(steps x P(survive))) steps per rollout, P from the LOO priors of the
    fusions the sweep decided for the problem (absorbed stage: 1 - p; head: 1 - p x head_weight; pre-applied
    high-prior kinds: p = 1). Replayed on steps.csv: steps with index >= s are dropped. $ saved = their usd;
    loss = E7-style slot loss (best over kept steps - logged best, x scale, x (1 - fused_away_frac)) added to the
    logged final model time. With preapply, slots whose every op is absorbed by a pre-applied fusion are dropped
    entirely (0 steps) and counted in the pre-apply rows above, not here."""
    import math
    dec = list(csv.DictReader(open(E7 / "decisions.csv")))
    kb = {r["kb_run"] + "|" + r["problem"]: r for r in csv.DictReader(open(E7 / "kbruns.csv"))
          if r["sample"] == "main" and r["phase"] == "opt" and r["round"] == "0"}
    st = defaultdict(list)
    for r in csv.DictReader(open(E7 / "steps.csv")):
        if r["sample"] == "main" and r["phase"] == "opt" and r["round"] == "0":
            st[r["kb_run"] + "|" + r["problem"]].append(r)
    fin = {r["problem"]: float(r["final_us"]) for r in csv.DictReader(open(E7 / "problems.csv")) if r["final_us"]}
    probs = sorted({r["problem"] for r in kb.values()} - EXCL_ROUND)
    saved = spent = 0.0
    losses = []
    per = {}
    for P in probs:
        rows = [(r["hint"], r["status"]) for r in dec if r["sample"] in ("main", "degraded")
                and r["problem"].split("@")[0] != P]
        pri = gp.Priors("cuda:sm_89|fp16", tables=[{"part": "cuda:sm_89|fp16", "counts": gp._status_counts(rows)}],
                        use_a5=False, min_n=min_n)
        pend = []
        for r in dec:
            if r["problem"] != P or r["sample"] != "main" or not r["hint"].startswith("fuse:"):
                continue
            kind = r["hint"].split(":", 1)[1]
            p = pri.prior(kind)
            stg = [base(s) for s in gp.stages(kind)]
            hi = preapply and p["n"] >= min_n and p["lb"] >= conf
            pend.append({"p": 1.0 if hi else p["p_mean"], "heads": {stg[0]}, "absorb": set(stg[1:])})
        loss_us = s_usd = k_usd = 0.0
        for key, r in kb.items():
            if r["problem"] != P:
                continue
            k = base(r["op_kind"])
            surv = 1.0
            for h in pend:
                if k in h["absorb"]:
                    surv *= 1 - h["p"]
                elif k in h["heads"]:
                    surv *= 1 - h["p"] * head_weight
            s = max(1, min(steps, round(steps * surv)))
            rows_ = st.get(key, [])
            kept = [x for x in rows_ if int(x["step"]) < s]
            drop = [x for x in rows_ if int(x["step"]) >= s]
            du = sum(float(x["usd"] or 0) for x in drop)
            s_usd += du
            k_usd += sum(float(x["usd"] or 0) for x in rows_)
            if drop:
                b_all = min([float(x["best_after"]) for x in rows_ if x["best_after"]] or [math.inf])
                b_keep = min([float(x["best_after"]) for x in kept if x["best_after"]] or
                             [float(rows_[0]["best_before"] or math.inf)])
                if math.isfinite(b_all) and math.isfinite(b_keep):
                    sc = float(r["scale"] or 1.0)
                    loss_us += max(0.0, b_keep - b_all) * sc * (1 - float(r["fused_away_frac"] or 0))
        saved += s_usd
        spent += k_usd
        f = fin.get(P)
        lo = 1 - f / (f + loss_us) if f else 0.0
        losses.append(lo)
        per[P] = {"saved_usd": s_usd, "r0_usd": k_usd, "loss": lo}
    g = 1 - math.exp(sum(math.log(1 - x) for x in losses) / len(losses))
    return {"head_weight": head_weight, "saved_usd": saved, "r0_usd": spent, "geomean_loss": g,
            "worst": max(per.items(), key=lambda kv: kv[1]["loss"]), "problems": per}


if __name__ == "__main__" and "--survival" in sys.argv:
    out = {}
    tot = sum(float(r["usd"] or 0) for r in csv.DictReader(open(E7 / "kbruns.csv"))
              if r["sample"] == "main" and r["problem"] not in EXCL_ROUND)
    for hw in (0.0, 0.5, 1.0):
        for pre in (True, False):
            r = survival_replay(head_weight=hw, preapply=pre)
            out[f"hw{hw}_pre{pre}"] = r
            print(f"survival head_weight {hw} preapply {pre}: saved ${r['saved_usd']:,.0f} = "
                  f"{100 * r['saved_usd'] / tot:.1f}% of spend ({100 * r['saved_usd'] / r['r0_usd']:.1f}% of round-0 "
                  f"opt spend), geomean loss {100 * r['geomean_loss']:.2f}%, worst {r['worst'][0]} "
                  f"{100 * r['worst'][1]['loss']:.1f}%")
    (HERE / "survival_replay.json").write_text(json.dumps(out, indent=1, default=str))
