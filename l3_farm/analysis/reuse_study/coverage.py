#!/usr/bin/env python3
"""Reuse-uplift study, part 1: COVERAGE of each finished headline problem's final-model ops by compatible kernels of OTHER problems.
Reads data/problems.json + data/pool.json (build_data.py). Writes data/coverage.json and data/cand.json (per op nearest donors).

Per op (non-region): same kind (= same entry + signature + dtype) donors of other origin problems; exact = the op's shape is in the donor's verified shapes
(json-equal); near = min shape_distance (search_fixes definition) over donor shapes. Pools: store (gk_impl v0258) / win (store + any cache winner) / all (any verified >=1.02x record).
"""
import json, math, collections, sys
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
K = 4


def shape_distance(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0 if a == b else 8.0
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    tot = 0.0
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) or isinstance(y, dict):
            tot += shape_distance(x or {}, y or {})
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            tot += abs(math.log2((abs(x) + 1.0) / (abs(y) + 1.0)))
        elif x != y:
            tot += 2.0
    return tot / len(keys)


def skey(s):
    return json.dumps(s, sort_keys=True)


def main():
    P = json.load(open(D / "problems.json"))
    pool = json.load(open(D / "pool.json"))
    by_op = collections.defaultdict(list)
    for r in pool:
        by_op[r["op"]].append(r)
    for rs in by_op.values():
        for r in rs:
            r["_keys"] = {skey(s) for s in r["shapes"]}
    code_rows = collections.defaultdict(list)
    for r in pool:
        code_rows[r["code_sha"]].append(r)

    def first_ts(code, p):
        v = [x["create_ts"] for x in code_rows[code] if x["origin"] != p and x["create_ts"]]
        return min(v) if v else None
    out, cand = {}, {}
    tiers = {"store": lambda r: r["in_store"], "win": lambda r: r["in_win"], "all": lambda r: True}
    for p, d in P.items():
        t_seed = d["t_seed"] or d["t_start"]
        seedc, refc = set(d["seed_codes"]), [(x["snap"], set(x["codes"])) for x in d["refresh"]]
        allref = set().union(*[c for _, c in refc]) if refc else set()
        rows = []
        for o in d["ops"]:
            row = {k: o[k] for k in ("name", "kind", "us", "share", "ctype", "origin", "seed_source", "speedup_vs_seed")}
            row["weak"] = bool(o.get("weak"))
            if o["ctype"] == "region":
                rows.append(row)
                continue
            sk = skey(o["shape"])
            for tn, f in tiers.items():
                ex, near = [], []
                for r in by_op.get(o["kind"], []):
                    if r["origin"] == p or not f(r):
                        continue
                    if sk in r["_keys"]:
                        ex.append(r)
                    if r["shapes"]:
                        dist = min(shape_distance(o["shape"], s) for s in r["shapes"])
                        near.append((dist, r))
                near.sort(key=lambda x: (x[0], -x[1]["best_speedup"]))
                # distinct codes, nearest K
                seen, nk = set(), []
                for dist, r in near:
                    if r["code_sha"] in seen:
                        continue
                    seen.add(r["code_sha"])
                    nk.append((dist, r))
                    if len(nk) >= K:
                        break
                row[tn] = {"n_exact": len({r["code_sha"] for r in ex}), "n_same_kind": len(near),
                           "dmin": round(nk[0][0], 3) if nk else None,
                           "exact_best_speedup": max([r["best_speedup"] for r in ex], default=None),
                           "near_best_speedup_k": max([r["best_speedup"] for _, r in nk], default=None)}
                # availability of the exact / near-K donors relative to this consumer
                def avail(rs):
                    a = {"seed": False, "refresh": False, "created_before_start": False, "created_before_finish": False,
                         "in_store_before_seed": False, "earliest_create_min_rel_start": None}
                    for r in rs:
                        c = r["code_sha"]
                        a["seed"] |= c in seedc
                        a["refresh"] |= (c in allref and c not in seedc)
                        ct = first_ts(r["code_sha"], p)
                        if ct:
                            a["created_before_start"] |= ct < t_seed
                            a["created_before_finish"] |= ct < (d["t_finish"] or 1e18)
                            m = (ct - t_seed) / 60
                            if a["earliest_create_min_rel_start"] is None or m < a["earliest_create_min_rel_start"]:
                                a["earliest_create_min_rel_start"] = round(m, 1)
                        a["in_store_before_seed"] |= bool(r["store_ts"] and r["store_ts"] < t_seed)
                    a["cls"] = ("seed" if a["seed"] else "refresh" if a["refresh"] else "pre_start_undelivered" if a["created_before_start"]
                                else "during_run" if a["created_before_finish"] else "after_finish")
                    return a
                row[tn]["avail_exact"] = avail(ex) if ex else None
                row[tn]["avail_near"] = avail([r for _, r in nk]) if nk else None
                if tn == "win":
                    own = [r["best_speedup"] for r in by_op.get(o["kind"], []) if r["origin"] == p and sk in r["_keys"]]
                    row["own_best_sp_exact"] = max(own, default=None)
                    if nk:
                        r0 = nk[0][1]
                        row["top1"] = {"dist": round(nk[0][0], 3), "origin": r0["origin"], "sha": r0["code_sha"], "sp": r0["best_speedup"],
                                       "create_min_rel_start": round((first_ts(r0["code_sha"], p) - t_seed) / 60, 1) if first_ts(r0["code_sha"], p) else None,
                                       "delivered": r0["code_sha"] in seedc and "seed" or (r0["code_sha"] in allref and "refresh") or "no"}
                    cand[f"{p}|{o['name']}"] = {"exact": [(r["origin"], r["code_sha"]) for r in ex][:8],
                                                "near": [(round(dist, 3), r["origin"], r["code_sha"], r["best_speedup"]) for dist, r in nk]}
            rows.append(row)
        out[p] = {"ops": rows, "e2e_us": d["e2e_us"], "first": d["first_sweep_e2e_us"]}
    json.dump(out, open(D / "coverage.json", "w"))
    json.dump(cand, open(D / "cand.json", "w"))
    print("ok", len(out))


if __name__ == "__main__":
    main()
