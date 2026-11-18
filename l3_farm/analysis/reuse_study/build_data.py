#!/usr/bin/env python3
"""Reuse-uplift study, step 1 (no GPU, no LLM, read-only on all results): extract per-problem final-model op tables and the
donor-kernel pool from the headline sweep into analysis/reuse_study/data/.

  problems.json : per finished problem: e2e, first-sweep e2e, per-op rows (name, kind, shape, us, share, slot, seed_source,
                  class own/foreign-in/seed/weak/region, origin problem, seed_speedup), timing (start, seed snapshot time,
                  refresh times, finish), delivered foreign codes (seed / refresh).
  pool.json     : donor kernels: one row per (origin problem, kind, code_sha): shapes verified, speedup_vs_seed, slot_us,
                  create_ts, store merge ts, tiers (store / win / all), path of the .cu.
"""
import glob, json, os, sqlite3, sys, hashlib, datetime as dt, collections
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]
ROOT = L3.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)
STORE = ROOT / "knowledge_global_headline"
BASES = [L3 / "results_headline", L3 / "results_headline_heavy"]


def ts(s):
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    if len(s) > 5 and s[-5] in "+-" and s[-3] != ":":
        s = s[:-2] + ":" + s[-2:]
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def jl(p):
    out = []
    for ln in open(p):
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def dirs():
    """problem -> list of dirs (heavy first); finished = has work/model_runtime.json."""
    d = collections.defaultdict(list)
    for b in BASES:
        for p in sorted(b.iterdir()):
            if p.is_dir() and p.name[0].isdigit():
                d[p.name].append(p)
    return d


def finished_dir(ds):
    for x in ds:
        if (x / "work" / "model_runtime.json").exists() and (x / "work" / "final").exists():
            return x
    return None


# ------------------------------------------------------------------ store: gk_impl rows with merge time
def store_rows():
    db = sqlite3.connect(STORE / "snapshots" / "v0258.db")
    rows = {}
    for id_, part, kkey, slot_us, data, prov in db.execute("select id,part,kkey,slot_us,data,prov from gk_impl"):
        r = json.loads(data)
        pv = json.loads(prov) if prov else {}
        r["_merge_ts"] = ts(pv.get("date"))
        r["_merge_problem"] = pv.get("problem")
        rows[id_] = r
    return rows


def store_versions():
    return [(h["version"], ts(h["ts"]), h["run"]["problem"]) for h in jl(STORE / "history.jsonl")]


def origin_of(rec, home):
    pid = rec.get("problem_id") or ""
    return pid.split("::")[0] if "::" in pid else home


def build_pool(dmap):
    """all verified records (>=1.02x vs seed, cuda) of every results dir, grouped by (origin, op, code_sha)."""
    srows = store_rows()
    scodes = {r["code_sha"]: r for r in srows.values()}
    recs = {}
    for p, ds in dmap.items():
        for d in ds:
            f = d / "work" / "knowledge" / "impl" / "index.jsonl"
            if not f.exists():
                continue
            for r in jl(f):
                if r.get("target", "cuda") != "cuda":
                    continue
                if r["id"] not in recs:
                    r["_home"] = p
                    r["_code"] = str(d / "work" / "knowledge" / "impl" / "code" / f"{r['code_sha']}.cu")
                    recs[r["id"]] = r
    # winners: every cache entry of any problem -> its code sha
    win_codes = {}
    for p, ds in dmap.items():
        for d in ds:
            f = d / "work" / "cache" / "index.json"
            if f.exists():
                for e in json.load(open(f))["entries"]:
                    win_codes.setdefault(Path(e["file"]).stem, set()).add(p)
    pool = {}
    for r in recs.values():
        if (r.get("speedup_vs_seed") or 0) < 1.02:
            continue
        o = origin_of(r, r["_home"])
        key = (o, r["op"], r["code_sha"])
        sh = [s for s in (r.get("shapes") or []) if isinstance(s, dict)]
        row = pool.setdefault(key, {"origin": o, "op": r["op"], "code_sha": r["code_sha"], "sig": r.get("signature_sha"),
                                    "dtype": r.get("dtype"), "arch": r.get("arch"), "shapes": [], "best_speedup": 0.0,
                                    "best_slot_us": None, "create_ts": ts(r.get("ts")), "path": r["_code"], "ids": []})
        for s in sh:
            if s not in row["shapes"]:
                row["shapes"].append(s)
        sp = r.get("speedup_vs_seed") or 0
        if sp > row["best_speedup"]:
            row["best_speedup"] = sp
            row["best_slot_us"] = r.get("slot_us")
            row["best_seed_us"] = r.get("seed_us")
        t = ts(r.get("ts"))
        if t and (row["create_ts"] is None or t < row["create_ts"]):
            row["create_ts"] = t
        row["ids"].append(r["id"])
    for row in pool.values():
        sr = scodes.get(row["code_sha"])
        row["in_store"] = bool(sr)
        row["store_ts"] = sr["_merge_ts"] if sr else None
        row["in_win"] = row["in_store"] or row["code_sha"] in win_codes
        row["ids"] = row["ids"][:3]
    return list(pool.values()), srows


# ------------------------------------------------------------------ per-problem final-model op table
def seed_left_like(ops):
    pass


def problem_info(p, d, pool_by_sha):
    W = d / "work"
    man = json.load(open(W / "final" / p / "problem" / "manifest.json"))
    g = json.load(open(W / "final" / p / "ir" / "graph.json"))
    gshape = {o["name"]: o.get("shape", {}) for o in g["ops"]}
    mr = json.load(open(W / "model_runtime.json"))[p]
    res = json.load(open(W / "final_model" / p / "kb_time" / "model.result.json"))
    runs = res["jobs"]["model"]["runs"] if isinstance(res["jobs"], dict) else res["jobs"][0]["model"]["runs"]
    ok = [x for x in runs if x.get("ok") and x.get("op_us")]
    import statistics as st
    opus = {k: st.median([x["op_us"][k] for x in ok if k in x["op_us"]]) for k in ok[0]["op_us"]} if ok else {}
    # first profile: seed times per op (hand seed)
    seeds = {}
    f0 = sorted(glob.glob(str(W / "r00" / "profile_pre" / p / "profile.json")))
    if f0:
        for r in json.load(open(f0[0]))["ops"]:
            if r.get("us") is not None and r.get("seed") == "hand":
                seeds[r["name"]] = r["us"]
    # cache index (for origin chain)
    ce = json.load(open(W / "cache" / "index.json"))["entries"] if (W / "cache" / "index.json").exists() else []
    byid = {e["impl_id"]: e for e in ce}

    def resolve(e, depth=0):
        s = e.get("source") or {}
        if s.get("by") and s.get("from") and depth < 8:
            n = byid.get(str(s["from"]).removeprefix("cache:"))
            if n:
                return resolve(n, depth + 1)
        pid = s.get("problem_id") or ""
        return pid.split("::")[0] if "::" in pid else p, e

    slots = {s["id"]: s for s in man["slots"]}
    tot = sum(opus.values()) or 1.0
    rows = []
    for o in man["ops"]:
        s = slots[o["slot"]]
        us = opus.get(o["name"])
        ss = str(s.get("seed_source", ""))
        parts = o["name"].split("+")
        s_us = sum(seeds[n] for n in parts) if all(n in seeds for n in parts) else None
        row = {"name": o["name"], "kind": o["op"], "slot": o["slot"], "shape": gshape.get(o["name"], {}), "us": us,
               "share": (us or 0) / tot, "seed_source": ss, "seed_us": s_us,
               "speedup_vs_seed": (s_us / us) if s_us and us else None, "origin": None, "code_sha": None,
               "ctype": None}
        if o["op"].startswith("region") or str(o["slot"]).startswith("region"):
            row["ctype"] = "region"
        elif ss == "hand" or ss.startswith("fused-seed") or not ss.startswith("cache:"):
            row["ctype"] = "seed"
        else:
            iid = ss.split(":", 1)[1]
            e = byid.get(iid)
            if e is None:
                row["ctype"] = "own?"
            else:
                org, e0 = resolve(e)
                row["origin"] = org
                row["code_sha"] = Path(e0["file"]).stem
                row["ctype"] = "own" if org == p else "foreign-in"
                if row["speedup_vs_seed"] is not None and row["speedup_vs_seed"] < 1.25:
                    row["weak"] = True
        rows.append(row)
    meta = json.load(open(d / "run_meta.json"))
    start = next((e for e in meta["events"] if e["event"] == "start" and not e.get("resumed")), meta["events"][0])
    fin = [e for e in meta["events"] if e["event"] == "finish"]
    seedj = d / "gk" / "seed" / "seed.json"
    seed = json.load(open(seedj)) if seedj.exists() else {}
    refresh = []
    for rd in sorted((d / "gk" / "refresh").glob("*-v*")) if (d / "gk" / "refresh").exists() else []:
        if rd.is_dir() and (rd / "refresh.json").exists():
            rj = json.load(open(rd / "refresh.json"))
            ids = {x.stem for x in (rd / "impl" / "code").glob("*.cu")} if (rd / "impl" / "code").exists() else set()
            refresh.append({"snap": rj.get("to", {}).get("snapshot"), "dir": rd.name, "n_impl": len(ids),
                            "codes": sorted(x for x in ids if x)})
    seedcodes = set()
    if (d / "gk" / "seed" / "impl" / "index.jsonl").exists():
        seedcodes = {r.get("code_sha") for r in jl(d / "gk" / "seed" / "impl" / "index.jsonl")}
    if (d / "gk" / "seed" / "impl" / "code").exists():
        seedcodes |= {x.stem for x in (d / "gk" / "seed" / "impl" / "code").glob("*.cu")}
    first = None
    fm = L3 / "results" / p / "work" / "model_runtime.json"
    if fm.exists():
        first = json.load(open(fm))[p].get("model_e2e_us")
    return {"problem": p, "dir": str(d), "e2e_us": mr["model_e2e_us"], "seed_e2e_us": mr.get("seed_e2e_us"),
            "first_sweep_e2e_us": first, "ops": rows, "sum_op_us": tot,
            "t_start": start["t"], "t_finish": fin[-1]["t"] if fin else None,
            "t_seed": ts(seed.get("created")), "seed_snapshot": seed.get("snapshot"),
            "seed_codes": sorted(x for x in seedcodes if x), "refresh": refresh,
            "n_ops": len(rows), "cache_entries": len(ce)}


def main():
    dmap = dirs()
    pool, srows = build_pool(dmap)
    json.dump(pool, open(OUT / "pool.json", "w"))
    print("pool rows", len(pool), "in_store", sum(r["in_store"] for r in pool), "in_win", sum(r["in_win"] for r in pool))
    pbs = collections.defaultdict(list)
    for r in pool:
        pbs[r["code_sha"]].append(r)
    probs = {}
    for p, ds in sorted(dmap.items(), key=lambda kv: int(kv[0].split("_")[0])):
        d = finished_dir(ds)
        if not d:
            print("skip (unfinished)", p)
            continue
        probs[p] = problem_info(p, d, pbs)
        print(p, probs[p]["e2e_us"], probs[p]["n_ops"])
    json.dump(probs, open(OUT / "problems.json", "w"))
    json.dump({"versions": store_versions()}, open(OUT / "store_versions.json", "w"))


if __name__ == "__main__":
    main()
