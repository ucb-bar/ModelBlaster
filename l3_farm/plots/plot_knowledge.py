#!/usr/bin/env python3
"""Knowledge-base / implementation-database / optimisation-step plots of the flow, in the conventions of plot_flow.py (14x8, bold labels, grid alpha 0.3, dpi 300, steelblue palette).
Sources (read-only): the global knowledge stores' newest snapshot (sqlite, immutable mode): knowledge_global_headline (KernelBench L3 on L40S, 278 merges) and l3_farm/pilot_sol/*_instance/gk_store
(SOL-ExecBench on B200, one fresh store per arm); tables gk_ledger (one row per merged run), gk_outcomes (one row per applied optimisation step), gk_claims/hypotheses/states/techniques;
and the per-run implementation indexes work/knowledge/impl/index.jsonl of every results dir (verified candidate kernels: op, speedup vs seed, parent, techniques, rel. error).
  plot_knowledge.py all      -> out/knowledge/*.png
Clustering = numpy only (TF-IDF over technique-name tokens + PCA + k-means; average-linkage ordering for the heatmap)."""
from __future__ import annotations
import collections, glob, json, math, re, sqlite3
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
H = Path(__file__).resolve().parent; REPO = H.parents[1]; F = H.parent; OUT = H / "out/knowledge"
STORES = {"KernelBench L3 / L40S sweep": REPO / "knowledge_global_headline",
          "SOL pilot (B200)": F / "pilot_sol/instance/gk_store", "SOL A6 pilot36": F / "pilot_sol/a6sol3_instance/gk_store", "SOL small base": F / "pilot_sol/basesmall_instance/gk_store",
          "SOL small A6": F / "pilot_sol/a6small_instance/gk_store", "SOL large base": F / "pilot_sol/baselarge_instance/gk_store", "SOL large A6": F / "pilot_sol/a6large_instance/gk_store",
          "SOL over-cap A6": F / "pilot_sol/a6big_instance/gk_store"}
PAL = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#7f7f7f", "#bcbd22"]
def db(d):
    fs = sorted(glob.glob(str(d / "snapshots/*.db")))
    return sqlite3.connect(f"file:{fs[-1]}?mode=ro&immutable=1", uri=True) if fs else None
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
def ledger(c):
    rows = []
    for seq, t, prob, rep in c.execute("select seq, ts, problem, report from gk_ledger order by seq"):
        r = json.loads(rep); g = lambda *k: (lambda d: d if isinstance(d, (int, float)) else 0)(__import__("functools").reduce(lambda a, b: a.get(b, {}) if isinstance(a, dict) else {}, k, r))
        rows.append(dict(seq=seq, t=ts(t), problem=prob, claims=g("claims", "added"), hyp=g("hypotheses", "added"), tech=g("techniques", "added"), states=g("states", "added"),
                         impl=g("impl", "kept"), outcomes=r.get("outcomes", 0) if isinstance(r.get("outcomes"), (int, float)) else 0, stats=g("stats", "rows")))
    return rows
def outcomes(c):
    out = []
    for (rec, t, prob) in c.execute("select rec, ts, problem from gk_outcomes"):
        o = json.loads(rec); o["_t"] = ts(t) if isinstance(t, str) else t; o["_problem"] = prob; out.append(o)
    out.sort(key=lambda o: o["_t"] or 0); return out
def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / name, dpi=300, bbox_inches="tight"); plt.close(fig); print("saved", OUT / name)
# ---------------------------------------------------------------- growth of the knowledge base
def growth(L):
    series = [("claims", "claims"), ("hyp", "hypotheses"), ("outcomes", "applied steps (outcomes)"), ("stats", "stat rows touched"), ("tech", "techniques promoted"), ("states", "states discovered"), ("impl", "implementations kept")]
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]; name = "KernelBench L3 / L40S sweep"; rows = L[name]; x = np.arange(1, len(rows) + 1)
    for i, (k, lab) in enumerate(series):
        y = np.cumsum([r[k] for r in rows]); 
        if y[-1] > 0: ax.plot(x, np.maximum(y, 0.8), color=PAL[i], linewidth=2.2, label=f"{lab} ({int(y[-1]):,})")
    probs = [r["problem"] for r in rows]; seen = set(); firsts = []
    for i, p in enumerate(probs):
        if p not in seen: seen.add(p); firsts.append(i + 1)
    ax.set_yscale("log"); ax.set_xlabel("merged runs (in merge order)", fontsize=12, fontweight="bold"); ax.set_ylabel("cumulative additions [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title(f"{name}: {len(rows)} merges, {len(seen)} problems", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(firsts[::6]); ax2.set_xticklabels(range(1, len(firsts) + 1)[::6]); ax2.set_xlabel("distinct problems seen", fontsize=10)
    ax = axs[1]
    for i, (sname, rows) in enumerate((k, v) for k, v in L.items() if k.startswith("SOL")):
        y = np.cumsum([r["claims"] for r in rows]); ax.plot(np.arange(1, len(rows) + 1), y, color=PAL[i % len(PAL)], linewidth=2.2, label=f"{sname} ({len(rows)} merges, {int(y[-1]):,})")
    ax.set_xlabel("merged runs (in merge order)", fontsize=12, fontweight="bold"); ax.set_ylabel("cumulative claims", fontsize=12, fontweight="bold")
    ax.set_title("SOL-ExecBench / B200: claims per store (each arm starts from an empty store)", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Growth of the knowledge base across the sweeps", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, "knowledge_growth.png")
    fig, axs = plt.subplots(2, 3, figsize=(18, 9)); 
    for ax, (k, lab) in zip(axs.ravel(), [("hyp", "hypotheses"), ("outcomes", "applied steps"), ("states", "states"), ("tech", "techniques promoted"), ("stats", "stat rows touched"), ("impl", "implementations kept")]):
        for i, (sname, rows) in enumerate(L.items()):
            y = np.cumsum([r[k] for r in rows])
            if y[-1] > 0: ax.plot(np.arange(1, len(rows) + 1) / len(rows), y, color=PAL[i % len(PAL)], linewidth=2, label=sname if k == "hyp" else None)
        ax.set_title(lab, fontsize=12, fontweight="bold"); ax.grid(alpha=0.3); ax.set_xlabel("fraction of the sweep's merges", fontsize=10, fontweight="bold")
    axs[0, 0].legend(fontsize=7, loc="upper left"); fig.suptitle("Knowledge-base growth by kind (sweep progress normalised to 1)", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, "knowledge_growth_by_kind.png")
def step_quality(O):
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (title, names) in zip(axs, (("KernelBench L3 / L40S sweep", ["KernelBench L3 / L40S sweep"]), ("SOL-ExecBench / B200 (all arms pooled by time)", [k for k in O if k.startswith("SOL")]))):
        steps = sorted([o for n in names for o in O[n]], key=lambda o: o["_t"] or 0); nb = 12; edges = np.linspace(0, len(steps), nb + 1).astype(int)
        cats = [("improved", "steelblue"), ("neutral", "lightgray"), ("regressed", "lightcoral")]; bottoms = np.zeros(nb)
        for cat, col in cats + [("error", "red")]:
            fr = []
            for a, b in zip(edges[:-1], edges[1:]):
                chunk = steps[a:b]; n = max(len(chunk), 1)
                fr.append(sum(1 for o in chunk if (o.get("status") == cat if cat != "error" else o.get("status") not in ("improved", "neutral", "regressed"))) / n)
            ax.bar(np.arange(nb), fr, bottom=bottoms, color=col, alpha=0.85, edgecolor="black", linewidth=0.4, label=cat); bottoms += np.array(fr)
        ax.set_xticks(range(nb)); ax.set_xticklabels([f"{i+1}" for i in range(nb)]); ax.set_xlabel(f"time bin of applied steps ({len(steps):,} steps, {nb} bins)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("share of applied optimisation steps", fontsize=12, fontweight="bold"); axs[1].legend(loc="upper right"); fig.suptitle("Outcome of applied optimisation steps over the course of the sweeps", fontsize=15, fontweight="bold")
    plt.tight_layout(); save(fig, "step_outcomes_over_time.png")
# ---------------------------------------------------------------- implementation database
def impls(pat):
    seen = {}
    for f in glob.glob(pat):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            i = r.get("id")
            if i and (i not in seen or (r.get("ts") or "") < (seen[i].get("ts") or "")): seen[i] = r
    return list(seen.values())
def impl_sets():
    return {"KernelBench L3 / L40S": impls(str(F / "results_headline/*/work/knowledge/impl/index.jsonl")),
            "SOL idx0 (B200: pilot + A6 + small/large + over-cap)": impls(str(F / "pilot_sol/*/results/9*/work/knowledge/impl/index.jsonl")),
            "SOL adaptation (other workloads)": impls(str(F / "sol_adapt/instance/results/*/work/knowledge/impl/index.jsonl"))}
def impl_plots(I):
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, (n, rs) in enumerate(I.items()):
        t = np.sort([ts(r["ts"]) for r in rs if r.get("ts")]); h = (t - t[0]) / 3600
        ax.plot(h, np.arange(1, len(h) + 1), color=PAL[i], linewidth=2.4, label=f"{n}  ({len(h):,} verified implementations)")
    ax.set_yscale("log"); ax.set_xlabel("hours since the first implementation of the sweep", fontsize=12, fontweight="bold"); ax.set_ylabel("cumulative distinct verified implementations [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title("Growth of the implementation database", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="lower right"); plt.tight_layout(); save(fig, "impl_growth.png")
    classes = collections.Counter(r.get("op_class") for rs in I.values() for r in rs); top = [c for c, _ in classes.most_common(8)]
    fig, axs = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]; w = 0.8 / len(I); pos0 = np.arange(len(top))
    for i, (n, rs) in enumerate(I.items()):
        data = [[math.log10(r["speedup_vs_seed"]) for r in rs if r.get("op_class") == c and r.get("speedup_vs_seed") and r["speedup_vs_seed"] > 0] for c in top]
        bp = ax.boxplot([d if d else [0] for d in data], positions=pos0 + (i - (len(I) - 1) / 2) * w, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]: b.set_facecolor(PAL[i]); b.set_alpha(0.75)
        for m in bp["medians"]: m.set_color("black")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5); ax.set_xticks(pos0); ax.set_xticklabels([f"{c}\n(n={classes[c]:,})" for c in top], fontsize=9)
    ax.set_ylabel("log10 speedup of the implementation over the seed kernel", fontsize=12, fontweight="bold"); ax.set_title("Quality by op class (box = IQR, line = median)", fontsize=13, fontweight="bold"); ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=PAL[i], alpha=0.75, label=n) for i, n in enumerate(I)], loc="upper right", fontsize=8)
    ax = axs[1]
    for i, (n, rs) in enumerate(I.items()):
        sp = np.sort([r["speedup_vs_seed"] for r in rs if r.get("speedup_vs_seed") and r["speedup_vs_seed"] > 0]); ax.plot(sp, 1 - np.arange(len(sp)) / len(sp), color=PAL[i], linewidth=2.2, label=f"{n}: {100*np.mean(sp>1):.0f}% faster than seed")
    ax.set_xscale("log"); ax.axvline(1, color="gray", linestyle="--", alpha=0.5); ax.set_xlim(0.1, 300); ax.set_xlabel("speedup over seed kernel [Log Scale]", fontsize=12, fontweight="bold"); ax.set_ylabel("fraction of implementations above p", fontsize=12, fontweight="bold")
    ax.set_title("Speedup distribution (survival)", fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Quality of the implementation database", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, "impl_quality.png")
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    for i, (n, rs) in enumerate(I.items()):
        byid = {r["id"]: r for r in rs}
        def depth(r, memo={}):
            d = 0; cur = r
            while cur.get("parent_id") in byid and d < 60: cur = byid[cur["parent_id"]]; d += 1
            return d
        ds = np.array([depth(r) for r in rs]); sp = np.array([r.get("speedup_vs_seed") or np.nan for r in rs])
        cnt = np.bincount(ds)[:15]; axs[0].plot(np.arange(len(cnt)), cnt / cnt.sum(), marker="o", color=PAL[i], linewidth=2, label=n)
        med = [np.nanmedian(sp[ds == d]) if (ds == d).sum() >= 5 else np.nan for d in range(15)]; axs[1].plot(range(15), med, marker="o", color=PAL[i], linewidth=2, label=n)
        err = np.array([r.get("max_rel_err") if r.get("max_rel_err") is not None else np.nan for r in rs]); e = np.sort(err[~np.isnan(err)] + 1e-12)
        axs[2].plot(e, np.arange(len(e)) / max(len(e), 1), color=PAL[i], linewidth=2, label=n)
    axs[0].set_xlabel("lineage depth (parent -> child chain)", fontsize=11, fontweight="bold"); axs[0].set_ylabel("share of implementations", fontsize=11, fontweight="bold"); axs[0].set_title("How deep the optimisation chains go", fontsize=12, fontweight="bold")
    axs[1].set_xlabel("lineage depth", fontsize=11, fontweight="bold"); axs[1].set_ylabel("median speedup over seed", fontsize=11, fontweight="bold"); axs[1].set_yscale("log"); axs[1].set_title("Deeper = better?", fontsize=12, fontweight="bold")
    axs[2].set_xscale("log"); axs[2].set_xlabel("max relative error vs reference", fontsize=11, fontweight="bold"); axs[2].set_ylabel("cumulative share", fontsize=11, fontweight="bold"); axs[2].set_title("Numerical accuracy of verified implementations", fontsize=12, fontweight="bold")
    for a in axs: a.grid(alpha=0.3)
    axs[0].legend(fontsize=8); fig.suptitle("Implementation database: lineage and correctness", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, "impl_lineage_accuracy.png")
# ---------------------------------------------------------------- clustering of applied optimisation steps
def tokens(t): return [w for w in re.split(r"[^a-z0-9]+", t.lower()) if len(w) > 1 and not w.isdigit()]
def kmeans(X, k, seed=0, iters=60):
    rng = np.random.default_rng(seed); c = [X[rng.integers(len(X))]]
    for _ in range(k - 1): d = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0); c.append(X[rng.choice(len(X), p=d / d.sum())] if d.sum() > 0 else X[rng.integers(len(X))])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - c[None]) ** 2).sum(2), 1); nc = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c): break
        c = nc
    return lab, c
def cluster_techniques(O, tag, k=9, min_n=4):
    steps = [o for rs in O.values() for o in rs]; byt = collections.defaultdict(list)
    for o in steps:
        if o.get("technique"): byt[o["technique"]].append(o)
    T = [t for t, v in byt.items() if len(v) >= min_n]; docs = [tokens(t) for t in T]; vocab = sorted({w for d in docs for w in d}); vi = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros((len(T), len(vocab)))
    for i, d in enumerate(docs):
        for w in d: tf[i, vi[w]] += 1
    idf = np.log((1 + len(T)) / (1 + (tf > 0).sum(0))) + 1; X = tf * idf; X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False); P = U[:, :2] * S[:2]; Z = U[:, :12] * S[:12]
    lab, _ = kmeans(Z, k, seed=1); n = np.array([len(byt[t]) for t in T])
    impr = np.array([np.mean([o.get("status") == "improved" for o in byt[t]]) for t in T]); gain = np.array([np.median([math.log10(max(o.get("speedup") or 1, 1e-3)) for o in byt[t]]) for t in T])
    fig, axs = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax = axs[0]; names = {}
    for j in range(k):
        m = lab == j; w = collections.Counter()
        for i in np.where(m)[0]:
            for tok in docs[i]: w[tok] += n[i]
        names[j] = ", ".join(t for t, _ in w.most_common(4))
        ax.scatter(P[m, 0], P[m, 1], s=20 + 380 * np.sqrt(n[m] / n.max()), color=PAL[j % len(PAL)], alpha=0.6, edgecolor="black", linewidth=0.4, label=f"C{j+1}: {names[j]}  ({m.sum()} techniques, {n[m].sum():,} steps)")
    for i in np.argsort(-n)[:6]: ax.annotate(T[i][:26], (P[i, 0], P[i, 1]), fontsize=7, alpha=0.8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PC1 of technique-name TF-IDF", fontsize=12, fontweight="bold"); ax.set_ylabel("PC2", fontsize=12, fontweight="bold"); ax.grid(alpha=0.3)
    ax.set_title(f"{tag}: {len(T)} techniques (>= {min_n} applications) clustered by name; marker size = number of applications", fontsize=12, fontweight="bold"); ax.legend(loc="best", fontsize=8)
    ax = axs[1]; order = np.argsort([-n[lab == j].sum() for j in range(k)]); y = np.arange(k)
    imp = [np.average(impr[lab == j], weights=n[lab == j]) for j in order]; ax.barh(y, imp, color=[PAL[j % len(PAL)] for j in order], alpha=0.8, edgecolor="black", linewidth=0.4)
    for yy, j, v in zip(y, order, imp): ax.text(v + 0.01, yy, f"C{j+1}  n={n[lab==j].sum():,}  median log10 speedup {np.median(gain[lab==j]):+.2f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels([f"C{j+1}" for j in order], fontsize=10); ax.invert_yaxis(); ax.set_xlim(0, 0.9); ax.set_xlabel("share of applications that improved the kernel", fontsize=12, fontweight="bold")
    ax.set_title("Success rate per cluster", fontsize=12, fontweight="bold"); ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Clustering of applied optimisation steps ({tag}, {len(steps):,} steps)", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, f"technique_clusters_{tag.split()[0].lower()}.png")

def cluster_behavior(O, tag, k=7, min_n=25):
    """Cluster the techniques by how they are USED: op-class mix, the bottleneck state they were applied in, success / regression rate, size of the gain when they help."""
    steps = [o for rs in O.values() for o in rs if o.get("technique") and o.get("op_class")]; byt = collections.defaultdict(list)
    for o in steps: byt[o["technique"]].append(o)
    T = [t for t, v in byt.items() if len(v) >= min_n]
    ocs = [c for c, _ in collections.Counter(o["op_class"] for o in steps).most_common(8)]; sts = [c for c, _ in collections.Counter(o.get("state_before") for o in steps if o.get("state_before")).most_common(8)]
    cols = [f"op:{c}" for c in ocs] + [f"state:{c}" for c in sts] + ["improved", "regressed", "log10 gain when improved"]; X = np.zeros((len(T), len(cols)))
    for i, t in enumerate(T):
        v = byt[t]; n = len(v)
        for j, c in enumerate(ocs): X[i, j] = sum(1 for o in v if o["op_class"] == c) / n
        for j, c in enumerate(sts): X[i, len(ocs) + j] = sum(1 for o in v if o.get("state_before") == c) / n
        X[i, -3] = sum(1 for o in v if o.get("status") == "improved") / n; X[i, -2] = sum(1 for o in v if o.get("status") == "regressed") / n
        g = [math.log10(max(o.get("speedup") or 1, 1e-3)) for o in v if o.get("status") == "improved"]; X[i, -1] = float(np.mean(g)) if g else 0.0
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9); U, S, Vt = np.linalg.svd(Z, full_matrices=False); P = U[:, :2] * S[:2]; lab, _ = kmeans(U[:, :8] * S[:8], k, seed=2)
    n = np.array([len(byt[t]) for t in T]); fig, axs = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axs[0]; order = sorted(range(k), key=lambda j: -n[lab == j].sum()); rank = {j: r + 1 for r, j in enumerate(order)}
    for j in order:
        m = lab == j; ax.scatter(P[m, 0], P[m, 1], s=30 + 420 * np.sqrt(n[m] / n.max()), color=PAL[rank[j] - 1], alpha=0.65, edgecolor="black", linewidth=0.4, label=f"B{rank[j]}: " + ", ".join(T[i][:22] for i in np.where(m)[0][np.argsort(-n[m])][:3]) + f"  ({m.sum()} techniques, {n[m].sum():,} steps)")
        for i in np.where(m)[0][np.argsort(-n[m])][:2]: ax.annotate(T[i][:24], (P[i, 0], P[i, 1]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("PC1 of technique usage profile", fontsize=12, fontweight="bold"); ax.set_ylabel("PC2", fontsize=12, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{tag}: {len(T)} techniques (>= {min_n} applications) clustered by usage behaviour; marker size = applications", fontsize=12, fontweight="bold")
    ax = axs[1]; M = np.array([X[lab == j].mean(0) for j in order]); Mz = (M - X.mean(0)) / (X.std(0) + 1e-9)
    im = ax.imshow(Mz, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto"); ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8); ax.set_yticks(range(k)); ax.set_yticklabels([f"B{r+1}" for r in range(k)])
    for r in range(k):
        for c in range(len(cols)): ax.text(c, r, f"{M[r, c]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("Cluster profiles (colour = standardised, text = mean value)", fontsize=12, fontweight="bold"); fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(f"Behavioural clusters of applied optimisation steps ({tag}, {len(steps):,} steps)", fontsize=15, fontweight="bold"); plt.tight_layout(); save(fig, f"technique_behaviour_clusters_{tag.split()[0].lower()}.png")
def hier_order(M):
    n = len(M); D = np.sqrt(((M[:, None, :] - M[None]) ** 2).sum(2)); cl = {i: [i] for i in range(n)}; nxt = n
    while len(cl) > 1:
        ks = list(cl); best = (1e18, None)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                d = D[np.ix_(cl[ks[a]], cl[ks[b]])].mean()
                if d < best[0]: best = (d, (ks[a], ks[b]))
        a, b = best[1]; cl[nxt] = cl.pop(a) + cl.pop(b); nxt += 1
    return list(cl.values())[0]
def heatmap(O, tag, top_t=34):
    steps = [o for rs in O.values() for o in rs if o.get("technique") and o.get("op_class") and o.get("speedup")]
    tc = collections.Counter(o["technique"] for o in steps); oc = collections.Counter(o["op_class"] for o in steps); T = [t for t, _ in tc.most_common(top_t)]; C = [c for c, _ in oc.most_common(10)]
    M = np.full((len(T), len(C)), np.nan); N = np.zeros_like(M)
    for i, t in enumerate(T):
        for j, c in enumerate(C):
            v = [math.log10(max(o["speedup"], 1e-3)) for o in steps if o["technique"] == t and o["op_class"] == c]; N[i, j] = len(v)
            if len(v) >= 3: M[i, j] = float(np.mean(v))
    Mf = np.where(np.isnan(M), 0, M); r = hier_order(Mf); c = hier_order(Mf.T)
    fig, ax = plt.subplots(figsize=(15, 13)); im = ax.imshow(M[np.ix_(r, c)], cmap="RdBu", vmin=-0.4, vmax=0.4, aspect="auto")
    ax.set_xticks(range(len(C))); ax.set_xticklabels([C[j] for j in c], rotation=45, ha="right", fontsize=10); ax.set_yticks(range(len(T))); ax.set_yticklabels([f"{T[i][:44]}  (n={tc[T[i]]:,})" for i in r], fontsize=8)
    for a, i in enumerate(r):
        for b, j in enumerate(c):
            if N[i, j] >= 3: ax.text(b, a, int(N[i, j]), ha="center", va="center", fontsize=6, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("mean log10 speedup of the step (blue = faster, red = slower)", fontsize=11, fontweight="bold")
    ax.set_title(f"{tag}: effect of the {top_t} most-applied techniques by op class (hierarchically ordered; numbers = applications, blank < 3)", fontsize=12, fontweight="bold")
    plt.tight_layout(); save(fig, f"technique_opclass_heatmap_{tag.split()[0].lower()}.png")
def main():
    L, O = {}, {}
    for n, d in STORES.items():
        c = db(d)
        if c: L[n] = ledger(c); O[n] = outcomes(c)
    growth(L); step_quality(O)
    cluster_techniques({"L3": O["KernelBench L3 / L40S sweep"]}, "L3 on L40S"); cluster_techniques({k: v for k, v in O.items() if k.startswith("SOL")}, "SOL on B200")
    cluster_behavior({"L3": O["KernelBench L3 / L40S sweep"]}, "L3 on L40S"); cluster_behavior({k: v for k, v in O.items() if k.startswith("SOL")}, "SOL on B200"); heatmap({"L3": O["KernelBench L3 / L40S sweep"]}, "L3 on L40S"); heatmap({k: v for k, v in O.items() if k.startswith("SOL")}, "SOL on B200")
    impl_plots(impl_sets())
if __name__ == "__main__": main()
