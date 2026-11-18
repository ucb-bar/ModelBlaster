#!/usr/bin/env python3
"""E8 analysis: reads analysis/e8_data/{bench,flowtime,fixed,rebuild}/ and the recorded sweep / baseline numbers, writes
analysis/e8_data/e8_results.json (all derived numbers) and prints the tables used in analysis/e8_measurement.md.

Statistics:
  block      one implementation timed once (warm-up, then n graph replays; value = upper median of the n replays)
  session    the main node's blocks: K processes x 4 rounds = 16 alternations per (problem, impl); reported mean /
             median / stdev / min / max of the block values; within-process vs between-process split (one-way ANOVA)
  ratio      speedup A over B = t_B / t_A, computed per alternation (same process, same round -> paired) and summarised
             by its mean and a 95% interval over alternations (t quantile); node-to-node interval = min..max of the
             per-node median ratios over the 5 nodes
  verdict    a reported claim "survives" when the corrected interval (union of the same-node 95% CI and the
             node-to-node range) excludes 1.0 on the claimed side, and "magnitude holds" when the reported value
             lies inside that interval
"""
import csv, io, json, math, re, statistics as st, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
L3 = HERE.parents[1]
DATA = L3 / "analysis" / "e8_data"
sys.path.insert(0, str(HERE))
import recompose  # noqa: E402

PS = ["1_MLP", "11_VGG16", "19_MobileNetV1", "41_GRUBidirectional", "43_MinGPTCausalAttention"]
IMPLS = ["torch_eager", "torch_graph", "flow", "freeform", "cudaonly"]
MAIN = 5298204
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
        12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 30: 2.042}


def t975(df):
    return T975.get(df) or (T975[30] if df >= 30 else T975[min(T975, key=lambda k: abs(k - df))])


def desc(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = st.fmean(xs)
    sd = st.stdev(xs) if len(xs) > 1 else 0.0
    return {"n": len(xs), "mean": m, "median": st.median(xs), "stdev": sd, "cv_pct": 100 * sd / m if m else None,
            "min": min(xs), "max": max(xs), "range_pct": 100 * (max(xs) - min(xs)) / m if m else None,
            "ci95": [m - t975(len(xs) - 1) * sd / math.sqrt(len(xs)), m + t975(len(xs) - 1) * sd / math.sqrt(len(xs))] if len(xs) > 1 else None}


def load_bench(tag):
    """{(job, problem): [bench json of each process]}"""
    out = defaultdict(list)
    for f in sorted((DATA / "bench" / tag).glob("*_*.json")):
        m = re.match(r"(\d+)_(.+)_(\d+)\.json$", f.name)
        if not m:
            continue
        d = json.loads(f.read_text())
        if d.get("failed"):
            continue
        out[(int(m.group(1)), m.group(2))].append(d)
    return out


def blocks(runs):
    """[(proc, round, impl, med)]"""
    res = []
    for d in runs:
        k = d["cfg"].get("proc", 0)
        for b in d["blocks"]:
            res.append((k, b["round"], b["impl"], b["med"]))
    return res


def anova(blk, impl):
    """within-process vs between-process stdev (% of mean) for one impl."""
    by = defaultdict(list)
    for k, r, i, v in blk:
        if i == impl:
            by[k].append(v)
    groups = [g for g in by.values() if len(g) > 1]
    if len(groups) < 2:
        return None
    allv = [v for g in groups for v in g]
    m = st.fmean(allv)
    within = math.sqrt(st.fmean([st.pvariance(g) * len(g) / (len(g) - 1) for g in groups]))
    means = [st.fmean(g) for g in groups]
    n0 = st.fmean([len(g) for g in groups])
    var_b = max(0.0, st.variance(means) - within ** 2 / n0)
    return {"within_pct": 100 * within / m, "between_pct": 100 * math.sqrt(var_b) / m, "proc_means": means}


def paired_ratios(blk, num, den):
    """t_den / t_num per (proc, round) alternation: >1 means `num` is faster."""
    v = {(k, r, i): x for k, r, i, x in blk}
    return [v[(k, r, den)] / v[(k, r, num)] for (k, r, i) in v if i == num and (k, r, den) in v]


def smi_stats(txt):
    rows = list(csv.reader(io.StringIO(txt.strip()))) if txt else []
    cl, mem, pw, tp, thr = [], [], [], [], defaultdict(int)
    for r in rows:
        if len(r) < 6:
            continue
        try:
            cl.append(float(r[1].split()[0])); mem.append(float(r[2].split()[0])); pw.append(float(r[3].split()[0]))
            tp.append(float(r[4]))
            thr[r[5].strip()] += 1
        except Exception:
            pass
    if not cl:
        return None
    busy = [i for i, c in enumerate(cl) if pw[i] > 120]   # samples while kernels run
    pick = lambda xs: [xs[i] for i in busy] if busy else xs
    return {"samples": len(cl), "busy_samples": len(busy), "sm_clock_busy": desc(pick(cl)), "mem_clock_busy": desc(pick(mem)),
            "power_busy": desc(pick(pw)), "temp": desc(tp), "temp_max": max(tp), "throttle_reasons": dict(thr)}


def recorded():
    """the numbers the earlier reports used (flow sweep, baseline finals, validated.md torch)."""
    vt = {"1_MLP": 2206, "11_VGG16": 3230, "19_MobileNetV1": 577, "41_GRUBidirectional": 12153, "43_MinGPTCausalAttention": 11919}
    rec = {}
    for p in PS:
        a = recompose.artifacts(p)
        r = {"flow": a.get("model_e2e_us"), "flow_runs": a.get("model_e2e_runs"), "flow_gpu_uuid": a.get("model_gpu_uuid"),
             "torch_graph_validated_md": vt[p]}
        tp = L3 / "results" / p / "torch_posthoc.json"
        if tp.exists():
            t = json.loads(tp.read_text())
            r["torch_posthoc"] = {k: t.get(k) for k in ("eager_us", "graph_us", "host", "gpu_uuid", "when")}
        sj = json.loads((L3 / "results" / p / "work" / "summary.json").read_text())
        v = next(iter(sj["irs"].values()))
        if v.get("torch_status") == "ok":
            r["torch_builtin"] = {k: v.get(k) for k in ("torch_graph_us", "torch_eager_us", "torch_us", "torch_mode")}
        for tag, sfx in (("freeform", "__T2"), ("cudaonly", "__CUDA")):
            run = sorted((L3 / "baseline_cc" / "runs").glob(f"{p}__*{sfx}"))[-1]
            fe = json.loads((run / "final_eval.json").read_text())["result"]
            r[tag] = {"time_us": fe["time_us"], "torch_graph_us": fe["torch_graph_us"], "torch_graph_after_us": fe.get("torch_graph_after_us"),
                      "host": fe["host"], "when": json.loads((run / "final_eval.json").read_text())["when"]}
        rec[p] = r
    return rec


# reported claims: (label, problem, faster impl, slower impl, reported ratio, source)
def claims(rec):
    C = []
    for p in PS:
        r = rec[p]
        C += [("free-form vs flow", p, "freeform", "flow", r["flow"] / r["freeform"]["time_us"], "X-033 / FEATURE_TRACKER (1.06x-3.2x)"),
              ("CUDA-only vs flow", p, "cudaonly", "flow", r["flow"] / r["cudaonly"]["time_us"], "X-034 / baseline_cc README"),
              ("flow vs torch graph", p, "flow", "torch_graph", r["torch_graph_validated_md"] / r["flow"], "results/validated.md"),
              ("free-form vs torch graph", p, "freeform", "torch_graph", r["freeform"]["torch_graph_us"] / r["freeform"]["time_us"], "final_eval (same node)"),
              ("CUDA-only vs torch graph", p, "cudaonly", "torch_graph", r["cudaonly"]["torch_graph_us"] / r["cudaonly"]["time_us"], "final_eval (same node)"),
              ("free-form vs CUDA-only", p, "freeform", "cudaonly", r["cudaonly"]["time_us"] / r["freeform"]["time_us"], "implied by X-033/X-034 finals")]
    return C


def main():
    rec = recompose_rec = recorded()
    res = {"recorded": rec, "main": {}, "nodes": {}, "fixed": {}, "flowtime": {}, "claims": [], "validation": {}}
    mainb = load_bench("main")
    spread = load_bench("spread")
    # ---- main session
    for p in PS:
        runs = mainb.get((MAIN, p), [])
        blk = blocks(runs)
        d = {"processes": len(runs), "impl": {}, "checks": {}, "smi": None}
        for i in IMPLS:
            vals = [v for k, r, ii, v in blk if ii == i]
            if vals:
                d["impl"][i] = {**desc(vals), "anova": anova(blk, i),
                                "replay_cv_pct_median": st.median([100 * b["std"] / b["mean"] for x in runs for b in x["blocks"] if b["impl"] == i])}
        for x in runs:
            for i, v in x["impls"].items():
                d["checks"].setdefault(i, []).append(v.get("check", {}).get("ok"))
        d["errors"] = [x.get("errors") for x in runs if x.get("errors")]
        d["others"] = sorted({str(x.get("others_before_timing")) for x in runs} | {str(x.get("others_end")) for x in runs})
        d["ratios"] = {}
        for a in IMPLS:
            for b in IMPLS:
                if a != b:
                    rr = paired_ratios(blk, a, b)
                    if rr:
                        d["ratios"][f"{a}/{b}"] = desc(rr)
        smis = [smi_stats((DATA / "bench" / "main" / f"{MAIN}_{p}_{x['cfg'].get('proc', 0)}_smi.csv").read_text()) for x in runs]
        d["smi"] = [s for s in smis if s]
        res["main"][p] = d
    # ---- node-to-node
    allnodes = sorted({j for (j, p) in spread} | {MAIN})
    for p in PS:
        per = {}
        for j in allnodes:
            runs = (mainb if j == MAIN else spread).get((j, p), [])
            blk = blocks(runs)
            if not blk:
                continue
            per[j] = {i: st.median([v for k, r, ii, v in blk if ii == i]) for i in IMPLS if any(ii == i for k, r, ii, v in blk)}
            per[j]["ratios"] = {f"{a}/{b}": st.median(paired_ratios(blk, a, b)) for a in IMPLS for b in IMPLS
                                if a != b and paired_ratios(blk, a, b)}
            per[j]["host"] = runs[0]["host"]
            per[j]["uuid"] = (runs[0].get("smi_start") or {}).get("uuid")
        sp = {}
        for i in IMPLS:
            xs = [per[j][i] for j in per if i in per[j]]
            if len(xs) > 1:
                sp[i] = desc(xs)
        rs = {}
        for key in {k for j in per for k in per[j]["ratios"]}:
            xs = [per[j]["ratios"][key] for j in per if key in per[j]["ratios"]]
            rs[key] = desc(xs)
        res["nodes"][p] = {"per_node": per, "spread": sp, "ratio_spread": rs}
    # ---- fixed kernels
    for f in sorted((DATA / "fixed").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("failed"):
            continue
        res["fixed"][f.stem] = {"host": d.get("host"), "job": d.get("job"), "gpu_info": d.get("gpu_info"),
                                "kernels": {k: desc([b["med"] for b in v]) for k, v in d.get("kernels", {}).items()},
                                "smi": smi_stats("\n".join(l for l in d.get("smi", "").splitlines() if l.strip()))}
    # ---- flowtime (the flow's own driver)
    for f in sorted((DATA / "flowtime").glob("*.json")):
        d = json.loads(f.read_text())
        res["flowtime"][f.stem] = {k: d.get(k) for k in ("passed", "e2e_us", "e2e_runs", "graph_nodes", "recorded_model_e2e_us",
                                                         "recorded_runs", "host", "when", "job", "eager_us")}
    # ---- claims
    for lab, p, fa, sl, rep, src in claims(rec):
        m = res["main"][p]["ratios"].get(f"{fa}/{sl}")
        ns = res["nodes"][p]["ratio_spread"].get(f"{fa}/{sl}")
        if not m:
            continue
        lo = min(m["ci95"][0] if m["ci95"] else m["mean"], ns["min"] if ns else m["mean"])
        hi = max(m["ci95"][1] if m["ci95"] else m["mean"], ns["max"] if ns else m["mean"])
        res["claims"].append({"claim": lab, "problem": p, "reported": rep, "source": src, "same_node_mean": m["mean"],
                              "same_node_ci95": m["ci95"], "node_range": [ns["min"], ns["max"]] if ns else None,
                              "interval": [lo, hi], "direction_holds": (lo > 1.0) if rep > 1 else (hi < 1.0),
                              "magnitude_holds": lo <= rep <= hi, "dev_pct": 100 * (m["mean"] / rep - 1)})
    # ---- tool validation: rebuilt program, flow's own driver, vs recorded model_e2e_us
    inv = {r["problem"]: r for r in json.loads((DATA / "inventory.json").read_text())}
    for p in recompose.problems():
        fts = [v for k, v in res["flowtime"].items() if k.split("_", 1)[1].startswith(p + "_") or k.split("_", 1)[1] == p]
        fts = [v for v in fts if v.get("job") != 5298298]          # the 8001 MHz-memory node is reported separately
        rb = sorted((DATA / "rebuild").glob(f"*_{p}.json"))
        rbd = json.loads(rb[0].read_text()) if rb else {}
        e = [v["e2e_us"] for v in fts if v.get("e2e_us")]
        recd = inv.get(p, {}).get("model_e2e_us")
        res["validation"][p] = {"level": inv.get(p, {}).get("level"), "recorded": recd, "rebuilt_med": st.median(e) if e else None,
                                "n_timings": len(e), "dev_pct": 100 * (st.median(e) / recd - 1) if e and recd else None,
                                "passed": all(v.get("passed") for v in fts) if fts else None,
                                "data_sha1_same": (rbd.get("checks") or {}).get("data_sha1_same"),
                                "ops_buffers_same": bool((rbd.get("checks") or {}).get("ops_same") and (rbd.get("checks") or {}).get("buffers_same")) if rbd else None,
                                "recorded_runs": inv.get(p, {}).get("model_e2e_runs")}
    (DATA / "e8_results.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n## validation")
    for p, v in res["validation"].items():
        print(f"{p:34s} {str(v['level']):11s} rec={v['recorded']} rebuilt={v['rebuilt_med']} n={v['n_timings']} "
              f"dev={'%+.2f%%' % v['dev_pct'] if v['dev_pct'] is not None else '-'} passed={v['passed']} data_same={v['data_sha1_same']}")
    # ---- print
    print("\n## main session (us, block medians over alternations)")
    for p in PS:
        for i, d in res["main"][p]["impl"].items():
            a = d["anova"] or {}
            print(f"{p:26s} {i:12s} n={d['n']:2d} mean={d['mean']:10.1f} med={d['median']:10.1f} sd={d['stdev']:8.1f} "
                  f"cv={d['cv_pct']:.2f}% min={d['min']:10.1f} max={d['max']:10.1f} within={a.get('within_pct', 0):.2f}% "
                  f"between={a.get('between_pct', 0):.2f}% replaycv={d['replay_cv_pct_median']:.2f}%")
        print("   checks", {k: all(v) for k, v in res["main"][p]["checks"].items()}, "others", res["main"][p]["others"])
    print("\n## node-to-node (per-node median us; spread over nodes)")
    for p in PS:
        for i, d in res["nodes"][p]["spread"].items():
            print(f"{p:26s} {i:12s} nodes={d['n']} mean={d['mean']:10.1f} cv={d['cv_pct']:.2f}% range={d['range_pct']:.2f}% "
                  f"min={d['min']:.1f} max={d['max']:.1f}")
    print("\n## fixed kernels")
    for k, d in res["fixed"].items():
        print(k, d["host"], {n: round(v["median"], 1) for n, v in d["kernels"].items()},
              (d["smi"] or {}).get("sm_clock_busy", {}) and round(d["smi"]["sm_clock_busy"]["median"]),
              (d["smi"] or {}).get("power_busy", {}) and round(d["smi"]["power_busy"]["median"]),
              (d["smi"] or {}).get("temp_max"), (d["smi"] or {}).get("throttle_reasons"))
    print("\n## claims")
    for c in res["claims"]:
        print(f"{c['claim']:26s} {c['problem']:26s} reported={c['reported']:.2f}x now={c['same_node_mean']:.3f}x "
              f"ci={['%.3f' % x for x in c['same_node_ci95']]} nodes={['%.3f' % x for x in (c['node_range'] or [])]} "
              f"dir={c['direction_holds']} mag={c['magnitude_holds']} dev={c['dev_pct']:+.1f}%")


if __name__ == "__main__":
    main()
