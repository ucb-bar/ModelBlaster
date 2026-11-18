#!/usr/bin/env python3
"""L40S summary for KernelBench L1, L2 (X-162) and L3 (X-152): speedup over PyTorch eager, torch.compile,
the better of the two, and CUDA graph.

Our final time:
  L1/L2: <farm>/results_l12/6NNNN_*/work/summary.json (model_e2e_us, model_passed)
  L3:    <farm>/plots/data/flow_results.csv rows with dataset == l3_l40s_sweep (model_us)
Baselines (X-168, same harness for all 250 problems): <farm>/analysis/l12_sweep/torch_compile/summary.csv
  columns eager_us, graph_us, compile_us. "best" = min(eager, compile) per problem (compile missing -> eager).
Prints the stats table, ready-to-paste LaTeX rows (vs best), the problems still missing, and writes l40s_summary.json.
Usage: python3 l40s_summary.py [FARM_DIR]   (default /Users/krisd/Documents/ModelBlaster-clone/l3_farm)
"""
import csv, glob, json, math, statistics as st, sys
from pathlib import Path

FARM = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/krisd/Documents/ModelBlaster-clone/l3_farm")
POOL = {1: 100, 2: 100, 3: 46}
BASES = ("eager", "compile", "best", "graph")
gm = lambda a: math.exp(sum(map(math.log, a)) / len(a))


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def baselines():
    out = {}
    with open(FARM / "analysis/l12_sweep/torch_compile/summary.csv") as fh:
        for r in csv.DictReader(fh):
            out[r["name"]] = {"eager": num(r["eager_us"]), "graph": num(r["graph_us"]), "compile": num(r["compile_us"])}
    return out


def ours():
    res = {1: {}, 2: {}, 3: {}}
    for f in sorted(glob.glob(str(FARM / "results_l12/6*/work/summary.json"))):
        n = f.split("/")[-3]
        try:
            r = list(json.load(open(f))["irs"].values())[0]
        except Exception:
            continue
        if r.get("model_passed") and num(r.get("model_e2e_us")):
            res[1 if n.startswith("61") else 2][n] = r["model_e2e_us"]
    with open(FARM / "plots/data/flow_results.csv") as fh:
        for r in csv.DictReader(fh):
            if r["dataset"] == "l3_l40s_sweep" and num(r["model_us"]):
                res[3][r["problem"]] = float(r["model_us"])
    return res


def speedups(level_ours, base):
    out = {b: [] for b in BASES}
    for n, m in level_ours.items():
        b = base.get(n)
        if not b or not b["eager"]:
            continue
        e, c, g = b["eager"], b["compile"], b["graph"]
        out["eager"].append(e / m)
        if c:
            out["compile"].append(c / m)
        out["best"].append(min(e, c) / m if c else e / m)
        if g:
            out["graph"].append(g / m)
    return out


def stats(sp, pool):
    return dict(n=len(sp), valid_pct=100 * len(sp) / pool, mean=st.mean(sp), geomean=gm(sp), median=st.median(sp),
                min=min(sp), max=max(sp), pct_gt1=100 * sum(x > 1 for x in sp) / len(sp),
                pct_lt1=100 * sum(x < 1 for x in sp) / len(sp), n_ge2=sum(x >= 2 for x in sp))


def main():
    base, mine = baselines(), ours()
    res, allbest = {}, []
    for lvl in (1, 2, 3):
        sp = speedups(mine[lvl], base)
        for b in BASES:
            if sp[b]:
                res[f"L{lvl}_{b}"] = stats(sp[b], POOL[lvl])
        allbest += sp["best"]
    for k, s in res.items():
        print(f"{k:11s} n={s['n']:3d} valid={s['valid_pct']:5.1f}% mean={s['mean']:7.3f} geo={s['geomean']:6.3f} med={s['median']:6.3f} "
              f"min={s['min']:.3f} max={s['max']:.2f} >1x={s['pct_gt1']:5.1f}% <1x={s['pct_lt1']:5.1f}% >=2x:{s['n_ge2']}")
    res["ALL_best"] = stats(allbest, sum(POOL.values()))
    print(f"ALL (L1+L2+L3) vs best of eager/compile: n={len(allbest)} geo={gm(allbest):.3f}")
    print("\nLaTeX rows (vs best of eager and compile; Valid, Mean, GeoM, Med, Min, Max, %>1x, %<1x):")
    for lvl in (1, 2, 3):
        s = res[f"L{lvl}_best"]
        print(f"{lvl} & \\textbf{{Ours}} & {s['valid_pct']:.0f}\\% & {s['mean']:.3f} & {s['geomean']:.3f} & {s['median']:.3f} & "
              f"{s['min']:.4f} & {s['max']:.2f} & {s['pct_gt1']:.2f}\\% & {s['pct_lt1']:.2f}\\%")
    print("\nMissing vs pool:", {f"L{l}": POOL[l] - res[f"L{l}_best"]["n"] for l in (1, 2, 3)})
    json.dump(res, open(Path(__file__).with_name("l40s_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
