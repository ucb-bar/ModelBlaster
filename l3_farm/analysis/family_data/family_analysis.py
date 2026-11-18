"""Shape-family analysis of the Level 3 sweep (no GPU): which slot problems are the same op kind at
different shapes inside one model, what they cost, how similar their winners are, and what the family
holds of the model time. Reads results/*/work (state.json, rNN/plan.json, rNN/run/*/result.json,
best sources). Writes analysis/family_data/{runs.json,families.json,similarity.json} and prints a summary.

    cd l3_farm && python3 analysis/family_data/family_analysis.py
"""
import collections, difflib, glob, json, math, re, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRICE = dict(uncached=5.0, cache_read=0.5, cache_write=6.25, output=25.0)   # $/M, as analysis/cost_breakdown.py


def usd(t):
    return (t.get("uncached_input_tokens", 0) * PRICE["uncached"] + t.get("cache_read_input_tokens", 0) * PRICE["cache_read"]
            + t.get("cache_write_input_tokens", 0) * PRICE["cache_write"] + t.get("output_tokens", 0) * PRICE["output"]) / 1e6


def op_class(op):
    o = op.replace("fused__", "").split("__")[0]
    for k, c in (("conv", "conv"), ("linear", "gemm"), ("matmul", "gemm"), ("bmm", "gemm"), ("addmm", "gemm"),
                 ("batchnorm", "norm"), ("layernorm", "norm"), ("softmax", "reduction"), ("cat", "data_movement"),
                 ("copy", "data_movement"), ("pool", "pool")):
        if k in o:
            return c
    return "other"


runs = []
for sp in sorted(ROOT.glob("results/[0-9]*/work/state.json")):
    model = sp.parts[-3]
    for rdir in sorted(sp.parent.glob("r[0-9][0-9]")):
        pf = rdir / "plan.json"
        if not pf.exists():
            continue
        plan = json.loads(pf.read_text())
        items = {it["key"]: it for it in plan["items"]}
        for q in plan["problems"]:
            pid = q.get("id", q["ir"])
            rd = rdir / "run" / f"{pid}__{rdir.name}"
            rf = rd / "result.json"
            if not rf.exists():
                continue
            r = json.loads(rf.read_text())
            tel = (r.get("telemetry") or {}).get("llm", {}).get("total", {})
            key = q["keys"][0]
            it = items.get(key, {})
            op = it.get("op") or key.split("|")[0].replace("kernel_", "")
            slots = r.get("slots") or {}
            sid, sl = next(iter(slots.items())) if slots else (None, {})
            src = None
            if sid and (rd / "best" / "slots" / f"{sid}.cu").exists():
                src = (rd / "best" / "slots" / f"{sid}.cu").read_text()
            runs.append(dict(model=model, round=rdir.name, pid=pid, op=op, op_class=op_class(op), shape=it.get("shape"),
                             share_at_plan=it.get("share"), plan_us=it.get("total_us"), n_inst=it.get("count"),
                             base_us=sl.get("baseline_attr_us") or sl.get("baseline_us"),
                             best_us=sl.get("final_attr_us") or sl.get("best_us"),
                             techniques=sl.get("techniques") or [], calls=r.get("llm_calls") or tel.get("calls", 0),
                             out_tok=tel.get("output_tokens", 0), usd=usd(tel),
                             wall_s=(r.get("telemetry") or {}).get("wall_s"), src=src))

with_llm = [x for x in runs if x["calls"]]
fam = collections.defaultdict(list)
for x in with_llm:
    fam[(x["model"], x["op"])].append(x)

# family time share: the latest profile of each model (before its last round), op kinds grouped
share = {}
for sp in sorted(ROOT.glob("results/[0-9]*/work/state.json")):
    model = sp.parts[-3]
    profs = sorted(sp.parent.glob("r[0-9][0-9]/profile/*/profile.json"))
    if not profs:
        continue
    p = json.loads(profs[-1].read_text())
    tot = sum(o["us"] or 0 for o in p["ops"]) or 1
    by = collections.Counter()
    nshape = collections.defaultdict(set)
    for o in p["ops"]:
        by[o["op"]] += (o["us"] or 0) / tot
        nshape[o["op"]].add(json.dumps(o["shape"], sort_keys=True))
    share[model] = {"profile": str(profs[-1].relative_to(ROOT)), "kernel_us": tot, "e2e_us": p.get("e2e_us"),
                    "ops": {k: {"share": v, "shapes": len(nshape[k])} for k, v in by.most_common()}}

fams = []
for (model, op), xs in fam.items():
    sp = [x["base_us"] / x["best_us"] for x in xs if x["base_us"] and x["best_us"]]
    fams.append(dict(model=model, op=op, op_class=xs[0]["op_class"], n=len(xs), calls=sum(x["calls"] for x in xs),
                     usd=sum(x["usd"] for x in xs), wall_s=sum(x["wall_s"] or 0 for x in xs),
                     speedup_min=min(sp) if sp else None, speedup_max=max(sp) if sp else None,
                     speedup_med=st.median(sp) if sp else None,
                     share_latest=share.get(model, {}).get("ops", {}).get(op, {}).get("share"),
                     shapes_latest=share.get(model, {}).get("ops", {}).get(op, {}).get("shapes")))
fams.sort(key=lambda f: -f["n"])

# ---------------------------------------------------------------- winner similarity
NUM = re.compile(r"\b\d+(\.\d+)?[fFuU]?\b")
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def norm(src):
    s = COMMENT.sub("", src)
    s = NUM.sub("N", s)
    return [t for t in re.findall(r"[A-Za-z_]\w*|\S", s)]


def feats(src):
    s = COMMENT.sub("", src)
    return {
        "tensor_core": bool(re.search(r"wmma::|mma\.sync|mma_sync|ldmatrix", s)),
        "cp_async": "cp.async" in s or "__pipeline_memcpy_async" in s or "cp_async" in s,
        "smem": "__shared__" in s,
        "half2": "half2" in s or "__hfma2" in s,
        "templated": bool(re.search(r"template\s*<", s)),
        # the kernel branches on a shape scalar (a dispatch over configurations / specialisations)
        "shape_dispatch": bool(re.search(r"\b(if|switch)\s*\(\s*\(?\s*(IC|OC|KH|KW|IH|IW|OH|OW|N|K|M|C|H|W)\b\s*[=<>!%]", s)),
        "n_kernels": len(re.findall(r"__global__", s)),
        "lines": s.count("\n"),
    }


def tile_params(src):
    """Integers named like tile / block / thread constants (BM, BN, BK, TILE_*, THREADS, WARPS...)."""
    s = COMMENT.sub("", src)
    out = {}
    for m in re.finditer(r"(?:#define|constexpr\s+(?:int|unsigned|size_t)|const\s+int|static\s+constexpr\s+int)\s+"
                         r"([A-Z_][A-Z0-9_]*)\s*=?\s*\(?\s*(\d+)\s*\)?", s):
        n, v = m.group(1), int(m.group(2))
        if re.search(r"(TILE|BLOCK|BM|BN|BK|^TM|^TN|THREAD|WARP|STAGE|PIX|OCB|ICB|CO|CI|K_|M_|N_)", n):
            out[n] = v
    return out


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


rng = __import__("random").Random(0)
simrows = []
by_op = collections.defaultdict(list)
for x in with_llm:
    if x["src"]:
        by_op[x["op"]].append(x)
within, across = [], []
fam_sim = {}
for (model, op), xs in fam.items():
    xs = [x for x in xs if x["src"]]
    if len(xs) < 3:
        continue
    toks = [norm(x["src"]) for x in xs]
    pairs = [(i, j) for i in range(len(xs)) for j in range(i + 1, len(xs))]
    rng.shuffle(pairs)
    vals = [sim(toks[i], toks[j]) for i, j in pairs[:60]]
    other = [y for y in by_op[op] if y["model"] != model]
    oth = []
    for _ in range(min(30, len(other) * len(xs))):
        a, b = rng.choice(xs), rng.choice(other) if other else None
        if b is not None:
            oth.append(sim(norm(a["src"]), norm(b["src"])))
    F = [feats(x["src"]) for x in xs]
    tp = [tile_params(x["src"]) for x in xs]
    # technique lineage overlap: Jaccard of the technique sets of two shapes' winners
    T = [set(x["techniques"]) for x in xs]
    jac = [len(T[i] & T[j]) / max(1, len(T[i] | T[j])) for i, j in pairs[:60]]
    top_tech = collections.Counter(t for s in T for t in s).most_common(4)
    fam_sim[f"{model}|{op}"] = dict(
        n=len(xs), code_sim_within_med=st.median(vals), code_sim_across_models_med=st.median(oth) if oth else None,
        technique_jaccard_med=st.median(jac) if jac else None, top_techniques=top_tech,
        frac_tensor_core=sum(f["tensor_core"] for f in F) / len(F), frac_smem=sum(f["smem"] for f in F) / len(F),
        frac_cp_async=sum(f["cp_async"] for f in F) / len(F), frac_shape_dispatch=sum(f["shape_dispatch"] for f in F) / len(F),
        frac_templated=sum(f["templated"] for f in F) / len(F),
        median_lines=st.median(f["lines"] for f in F),
        distinct_tile_param_sets=len({json.dumps(t, sort_keys=True) for t in tp}),
        tile_param_examples=[t for t in tp[:6]])
    within += vals
    across += oth

# ---------------------------------------------------------------- what-if: families -> 1..3 problems
cls = collections.Counter()
for f in fams:
    cls["1" if f["n"] == 1 else "2" if f["n"] == 2 else "3-9" if f["n"] < 10 else "10+"] += f["n"]
tot_runs, tot_calls, tot_usd = len(with_llm), sum(x["calls"] for x in with_llm), sum(x["usd"] for x in with_llm)
med_calls = st.median(x["calls"] for x in with_llm)
med_usd = st.median(x["usd"] for x in with_llm)


def whatif(k_of_n, budget_mult):
    """Every family of n>=3 shapes becomes k(n) problems, each with budget_mult x the median per-problem calls."""
    calls = usd = 0.0
    nprob = 0
    for f in fams:
        if f["n"] < 3:
            calls += f["calls"]
            usd += f["usd"]
            nprob += f["n"]
        else:
            k = k_of_n(f["n"])
            nprob += k
            per_call = f["usd"] / max(f["calls"], 1)
            c = k * budget_mult * (f["calls"] / f["n"])
            calls += c
            usd += c * per_call * 1.15         # family prompts carry a per-shape table: ~15% more tokens per call
    return dict(problems=nprob, calls=round(calls), usd=round(usd), calls_frac=calls / tot_calls, usd_frac=usd / tot_usd)


scen = {
    "1 problem/family, 4x a slot problem's budget": whatif(lambda n: 1, 4),
    "1 problem/family, 8x budget": whatif(lambda n: 1, 8),
    "2 problems/family (n>=10), 6x each": whatif(lambda n: 2 if n >= 10 else 1, 6),
    "3 problems/family (n>=10), 6x each": whatif(lambda n: 3 if n >= 10 else 1, 6),
    "budget ~ 25-30% of the family's per-shape calls": None,
}
f3 = [f for f in fams if f["n"] >= 3]
c3 = sum(f["calls"] for f in f3)
u3 = sum(f["usd"] for f in f3)
scen["budget ~ 25-30% of the family's per-shape calls"] = dict(
    calls_frac=(tot_calls - c3 + 0.275 * c3) / tot_calls, usd_frac=(tot_usd - u3 + 0.275 * u3 * 1.15) / tot_usd,
    calls=round(tot_calls - c3 + 0.275 * c3), usd=round(tot_usd - u3 + 0.275 * u3 * 1.15))

summary = dict(
    slot_problems_with_llm=tot_runs, llm_calls=tot_calls, usd=round(tot_usd), median_calls=med_calls,
    median_usd=round(med_usd, 2), families=len(fams), families_ge3=len(f3),
    frac_problems_in_families_ge3=sum(f["n"] for f in f3) / tot_runs, problems_by_family_size=dict(cls),
    calls_in_families_ge3=c3, usd_in_families_ge3=round(u3),
    by_op_top=collections.Counter({op: sum(f["n"] for f in fams if f["op"] == op) for op in {f["op"] for f in fams}}).most_common(12),
    code_sim_within_family_med=st.median(within) if within else None,
    code_sim_same_op_other_model_med=st.median(across) if across else None,
    whatif=scen)
for x in runs:
    x.pop("src", None)
(OUT / "runs.json").write_text(json.dumps(runs, indent=0))
(OUT / "families.json").write_text(json.dumps(dict(summary=summary, families=fams, share=share), indent=1))
(OUT / "similarity.json").write_text(json.dumps(fam_sim, indent=1))
print(json.dumps(summary, indent=1))
print("\nlargest families:")
for f in fams[:25]:
    print(f"  {f['model']:<28} {f['op']:<44} n={f['n']:<3} calls={f['calls']:<5} ${f['usd']:<7.0f} share={f['share_latest'] or 0:.2f} "
          f"speedup {f['speedup_min'] or 0:.2f}-{f['speedup_max'] or 0:.2f}")
print("\nsimilarity (families >= 8 shapes):")
for k, v in sorted(fam_sim.items(), key=lambda kv: -kv[1]["n"]):
    if v["n"] >= 8:
        print(f"  {k:<60} n={v['n']:<3} sim_in={v['code_sim_within_med']:.2f} sim_x={v['code_sim_across_models_med'] or 0:.2f} "
              f"jac={v['technique_jaccard_med'] or 0:.2f} tc={v['frac_tensor_core']:.2f} smem={v['frac_smem']:.2f} "
              f"disp={v['frac_shape_dispatch']:.2f} tiles={v['distinct_tile_param_sets']}")
