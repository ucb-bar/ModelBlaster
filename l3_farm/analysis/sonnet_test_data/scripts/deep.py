#!/usr/bin/env python3
"""Deep tables for the Sonnet routing test: deep.py  -> ../deep.json and markdown on stdout.
Arms S3 S5 S0 H5 from ../res/<arm>; controls C C2 C3 and X-037's H from the X-037 scratch copies (res dirs saved there)."""
import glob, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent
CTRL = Path("/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad/bt/res")
ARMS = {a: DATA / "res" / a for a in ("S3", "S5", "S0", "H5")}
OLD = {a: CTRL / a for a in ("C", "C2", "C3", "H")}


def fam(m):
    m = (m or "").lower()
    return "sonnet" if "sonnet" in m else "haiku" if "haiku" in m else "opus"


def load(d):
    """-> list of (round, problem, steps in commit order, slot summary)"""
    out = []
    for f in sorted(glob.glob(f"{d}/work/r0*/run/*/result.json")):
        r = json.load(open(f))
        parts = f.split("/")
        out.append((parts[-4], parts[-2].replace("kb_problem__", ""), r["steps"], next(iter(r["slots"].values()))))
    return out


def steps_table(d):
    T = defaultdict(Counter)
    esc = Counter()
    fixes = []
    wins = Counter()
    final_by = {}
    for rnd, prob, steps, slot in load(d):
        if "fused" in prob:
            pass
        base = next((s["slot_us_before"] for s in steps if s["step"] == 0 and s.get("slot_us_before") is not None), None)
        if base is None and steps:
            base = steps[0].get("slot_us_before")
        best, who = base, None
        for s in steps:
            c = s.get("context") or {}
            m = fam(c.get("model"))
            why = c.get("model_why") or ""
            kind = "hard-technique" if why.startswith("hard") else re.sub(r"\d+", "N", why) if why.startswith("slot stalled") else "default"
            key = (m, kind)
            T[key]["n"] += 1
            T[key][s["status"]] += 1
            esc[kind] += 1
            after = s.get("slot_us_after")
            if s["status"] == "improved" and after is not None and best is not None and after < best * 0.995:
                T[key]["new_best"] += 1
                best, who = after, (m, kind)
            if s.get("fix_attempts"):
                fm = c.get("fix_models") or []
                fixes.append({"slot": f"{rnd}/{prob}", "writer": m, "kind": kind, "fix_models": [fam(x) for x in fm],
                              "attempts": s["fix_attempts"], "status": s["status"]})
        final_by[f"{rnd}/{prob}"] = {"best_found_by": who, "base_us": base, "best_us": best}
    return {f"{k[0]}|{k[1]}": dict(v) for k, v in T.items()}, dict(esc), fixes, final_by


def slot_speedups(d):
    out = {}
    for rnd, prob, steps, slot in load(d):
        key = f"{rnd}/{re.sub(r'__r0[0-9]$', '', prob)}"
        out[key] = {"base": slot["baseline_us"], "final": slot["final_us"], "sp": slot["baseline_us"] / slot["final_us"]}
    return out


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main():
    res = {"arms": {}}
    sp = {}
    for a, d in {**ARMS, **OLD}.items():
        t, esc, fx, fb = steps_table(d)
        sp[a] = slot_speedups(d)
        res["arms"][a] = {"steps": t, "esc": esc, "fixes": fx, "final_by": fb, "slots": sp[a]}
        print(f"## {a}: steps by (model, why-kind)")
        for k, v in sorted(t.items()):
            n = v["n"]
            print(f"  {k:28s} n={n:3d} impr={v.get('improved',0):3d} ({100*v.get('improved',0)/n:3.0f}%) neutral={v.get('neutral',0):3d} "
                  f"regr={v.get('regressed',0):3d} new_best={v.get('new_best',0):3d}")
        print("  fixes:", [(f["writer"], f["fix_models"], f["status"]) for f in fx])
    # slot comparisons vs controls
    print("\n## per-slot final speedup (baseline/final), arms vs controls")
    keys = [k for k in sp["C2"] if all(k in sp[a] for a in sp)]
    print("slots not common to all arms (different graph hints):", {a: [k for k in sp[a] if k not in keys] for a in sp})
    hdr = ["slot"] + list(sp.keys())
    print(" | ".join(hdr))
    for k in keys:
        print(k, *[f"{sp[a][k]['sp']:.2f}" for a in sp], sep=" | ")
    print("geomean", *[f"{gm([sp[a][k]['sp'] for k in keys]):.2f}" for a in sp], sep=" | ")
    cmp = {}
    for a in ARMS:
        for c in ("C2", "C3", "H"):
            w = l = t = 0
            for k in keys:
                r = sp[a][k]["sp"] / sp[c][k]["sp"]
                if r > 1.01: w += 1
                elif r < 0.99: l += 1
                else: t += 1
            cmp[f"{a}_vs_{c}"] = [w, l, t]
    print("\nwins/losses/ties (+-1%):", json.dumps(cmp))
    res["slot_cmp"] = cmp
    # final slot time relative to the mean of the same-GPU controls C2, C3 (lower = better) for the slots that matter
    print("\n## final slot us vs mean(C2,C3), key slots")
    for k in keys:
        ref = (sp["C2"][k]["final"] + sp["C3"][k]["final"]) / 2
        print(k, f"ref {ref:.1f}", *[f"{a}:{sp[a][k]['final']/ref:.3f}" for a in ARMS], sep=" | ")
    res["slot_cmp"] = cmp
    (DATA / "deep.json").write_text(json.dumps(res, indent=1, default=str))


main()
