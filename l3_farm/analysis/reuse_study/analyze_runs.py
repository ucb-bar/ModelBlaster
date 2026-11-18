#!/usr/bin/env python3
"""Part 2 analysis: data/runs/<P>/ru.json (+ ru2.json for the early/late batch, optional) + torch_*.txt -> data/uplift.json, data/table_uplift.md"""
import json, glob, math, statistics as st
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
L3 = Path(__file__).resolve().parents[2]
PR = json.load(open(D / "problems.json"))
ARMS = ["orig", "sibw", "ex", "nk4", "gbest", "early", "late", "cold"]


def torch_us(d):
    g = []
    for i in (1, 2, 3):
        f = d / f"torch_{i}.txt"
        if f.exists():
            for ln in f.read_text().splitlines():
                if ln.startswith("@@POSTHOC@@"):
                    try:
                        g.append(json.loads(ln[len("@@POSTHOC@@"):]).get("graph_us"))
                    except ValueError:
                        pass
    g = [x for x in g if x]
    return (st.median(g) if g else None), g


def load(p):
    d = D / "runs" / p
    out = {"problem": p}
    meds, checks, alias, reuse, slots = {}, {}, {}, {}, {}
    for fn in ("ru.json", "ru2.json"):
        f = d / fn
        if not f.exists():
            continue
        r = json.load(open(f))
        if "t1" not in r:
            continue
        out.setdefault("identical", r.get("as_shipped_identical"))
        for k, v in (r.get("median_us") or {}).items():
            if fn == "ru2.json" and k == "orig":
                out["orig_batch2"] = v
                continue
            meds[k] = v
        for k, v in (r.get("check") or {}).items():
            if not (fn == "ru2.json" and k == "orig"):
                checks[k] = v
        alias.update(r.get("alias") or {})
        for k, v in (r.get("reuse") or {}).items():
            reuse[k] = v
        for k, v in (r.get("arm_slots") or {}).items():
            slots.setdefault(k, v)
        out["errors"] = (out.get("errors") or []) + [e["stage"] for e in r.get("errors", [])]
        out["profile_cold"] = r.get("profile_cold", {}).get("e2e_us", out.get("profile_cold"))
    if "orig" not in meds:
        return None
    for k, tgt in alias.items():
        if tgt in meds and k not in meds:
            meds[k] = meds[tgt]
            checks[k] = checks.get(tgt)
    out["med"], out["check"], out["alias"], out["reuse"] = meds, checks, alias, reuse
    out["torch_us"], out["torch_runs"] = torch_us(d)
    out["shipped_orig_sweep"] = PR[p]["e2e_us"]
    out["first"] = PR[p]["first_sweep_e2e_us"]
    return out


def gm(xs):
    xs = [x for x in xs if x]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def main():
    rows = []
    for p in sorted((x.name for x in (D / "runs").iterdir() if x.is_dir()), key=lambda s: int(s.split("_")[0])):
        r = load(p)
        if r:
            rows.append(r)
    json.dump(rows, open(D / "uplift.json", "w"), indent=1)
    lines = ["| problem | torch graph us | first sweep us | shipped (now) us | " + " | ".join(f"{a} us (x vs shipped)" for a in ARMS[1:]) + " | best sharing arm (not cold) | cold vs torch |",
             "|---|---|---|---|" + "---|" * (len(ARMS) - 1) + "---|---|"]
    acc = {a: [] for a in ARMS}
    acc_t = {a: [] for a in ARMS}
    acc_f = {a: [] for a in ARMS}
    best_rows = []
    for r in rows:
        m, o = r["med"], r["med"]["orig"]
        cells = []
        for a in ARMS[1:]:
            v = m.get(a)
            ok = (r["check"].get(a) or {}).get("passed") if v else None
            cells.append(f"{v:.0f} ({o / v:.3f}x){'' if ok in (True, None) else ' FAIL'}" if v else "-")
            if v and ok in (True, None):
                acc[a].append(o / v)
                if r["torch_us"]:
                    acc_t[a].append(r["torch_us"] / v)
                if r["first"]:
                    acc_f[a].append(r["first"] / v)
        cand = {a: m[a] for a in ("ex", "nk4", "gbest", "early") if m.get(a) and (r["check"].get(a) or {}).get("passed") in (True, None)}
        bestv = min([o] + list(cand.values()))
        best = min(cand, key=cand.get) if cand and min(cand.values()) < o else "orig"
        best_rows.append((r["problem"], o, bestv, best))
        acc["orig"].append(1.0)
        if r["torch_us"]:
            acc_t["orig"].append(r["torch_us"] / o)
        if r["first"]:
            acc_f["orig"].append(r["first"] / o)
        cv = (r["torch_us"] / m["cold"]) if m.get("cold") and r["torch_us"] else None
        lines.append(f"| {r['problem']} | {r['torch_us'] and round(r['torch_us']) or '-'} | {r['first'] and round(r['first']) or '-'} | {o:.0f} | " + " | ".join(cells) +
                     f" | {best} {bestv:.0f} ({o / bestv:.3f}x) | {cv and round(cv, 2) or '-'} |")
    # aggregates
    ag = ["", "| arm | n | mean x vs shipped | geomean x vs shipped | geomean x vs torch (n) | geomean x vs first sweep |", "|---|---|---|---|---|---|"]
    for a in ARMS:
        if acc[a]:
            ag.append(f"| {a} | {len(acc[a])} | {sum(acc[a]) / len(acc[a]):.3f} | {gm(acc[a]):.3f} | {gm(acc_t[a]) and round(gm(acc_t[a]), 3)} ({len(acc_t[a])}) | {gm(acc_f[a]) and round(gm(acc_f[a]), 3)} |")
    bb = [o / b for _, o, b, _ in best_rows]
    ag.append(f"| best sharing arm per problem (selection bias, upper bound) | {len(bb)} | {sum(bb) / len(bb):.3f} | {gm(bb):.3f} | | |")
    (D / "table_uplift.md").write_text("\n".join(lines + ag))
    print("\n".join(lines + ag))



def donors():
    """Which donor kernels were chosen (per arm, kind, shape): origin problem, slot-level gain estimate (matrix measurement)."""
    rows = json.load(open(D / "uplift.json"))
    out = []
    for r in rows:
        for arm, rep in (r.get("reuse") or {}).items():
            if rep.get("reverted") or not rep.get("changed"):
                continue
            for k in rep.get("kinds", []):
                ch = k.get("choice") or {}
                pool = {c["sha"]: c["from"] for c in k.get("pool", [])}
                for si, sha in ch.items():
                    if not sha:
                        continue
                    row = (k.get("table") or {}).get(si, {})
                    inc = (row.get("incumbent") or {}).get("us")
                    cu = ((row.get("cands") or {}).get(sha) or {}).get("us")
                    fr = pool.get(sha) or next((v for kk, v in pool.items() if kk.startswith(sha)), "?")
                    org = fr.split(":")[2] if fr.startswith("impl:") and len(fr.split(":")) > 2 else fr.split(":")[0]
                    out.append({"consumer": r["problem"], "arm": arm, "kind": k["kind"], "share": k.get("share"), "shape_idx": si,
                                "n": row.get("n"), "inc_us": inc, "new_us": cu, "gain_us": (row.get("n") or 1) * (inc - cu) if inc and cu else None,
                                "donor": org, "from": fr})
    json.dump(out, open(D / "donors.json", "w"), indent=1)
    return out



def donor_timing():
    """Attach availability of each chosen donor kernel to its consumer: created before the consumer's seed snapshot (early),
    delivered in its seed / a refresh, or only created later (late)."""
    import collections
    pool = json.load(open(D / "pool.json"))
    by_id = {}
    for r in pool:
        for i in r["ids"]:
            by_id.setdefault(i, r)
    code_rows = collections.defaultdict(list)
    for r in pool:
        code_rows[r["code_sha"]].append(r)
    d = json.load(open(D / "donors.json"))
    agg = collections.defaultdict(float)
    cnt = collections.Counter()
    for x in d:
        iid = x["from"].split(":")[1] if x["from"].startswith("impl:") else None
        r = by_id.get(iid)
        if x["from"].startswith("cache:") or x["from"] == "incumbent":
            x["avail"] = "own cache (sibling reuse, not foreign)"
            if x["arm"] != "cold" and x.get("gain_us"):
                agg[x["avail"]] += x["gain_us"]; cnt[x["avail"]] += 1
            continue
        if not r:
            x["avail"] = "?"
            continue
        p = PR[x["consumer"]]
        t_seed = p["t_seed"] or p["t_start"]
        v = [y["create_ts"] for y in code_rows[r["code_sha"]] if y["origin"] != x["consumer"] and y["create_ts"]]
        ct = min(v) if v else None
        refc = set().union(*[set(q["codes"]) for q in p["refresh"]]) if p["refresh"] else set()
        if r["code_sha"] in set(p["seed_codes"]):
            a = "seed"
        elif r["code_sha"] in refc:
            a = "refresh"
        elif ct and ct < t_seed:
            a = "pre-start (not delivered)"
        elif ct and ct < (p["t_finish"] or 1e18):
            a = "during run (not delivered)"
        else:
            a = "after finish"
        x["avail"] = a
        x["create_min_vs_seed"] = round((ct - t_seed) / 60, 1) if ct else None
        x["in_store"] = r["in_store"]
        if x["arm"] != "cold" and x.get("gain_us"):
            agg[a] += x["gain_us"]
            cnt[a] += 1
    json.dump(d, open(D / "donors.json", "w"), indent=1)
    return dict(agg), dict(cnt)


if __name__ == "__main__":
    main()
    donors()
    print(donor_timing())
