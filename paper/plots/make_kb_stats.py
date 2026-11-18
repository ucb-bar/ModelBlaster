#!/usr/bin/env python3
"""Section 4.5 statistics and Figure 4: what the Knowledge Base holds after the L40S KernelBench Level 1+2 sweep.

Source: the global store knowledge_global_l12, current snapshot (gk_outcomes rows whose problem starts with 61 or 62).
Prints the numbers quoted in the paper and writes opt_pie_l12_l40s.{png,pdf}.
Usage: python3 make_kb_stats.py [SNAPSHOT_DB] [OUT_DIR]
"""
import collections, json, sqlite3, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB = sys.argv[1] if len(sys.argv) > 1 else "/Users/krisd/Documents/ModelBlaster-clone/knowledge_global_l12/snapshots/v0586.db"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/Users/krisd/Documents/ModelBlaster-clone/Blaster/plot")
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rows = []
for prob, ts, rec in c.execute("select problem, ts, rec from gk_outcomes"):
    d = json.loads(rec); d["_p"] = prob or ""; d["_ts"] = ts; rows.append(d)
l12 = [d for d in rows if d["_p"][:2] in ("61", "62")]
N = len(l12)
state = collections.Counter((d.get("state_before") or "no state recorded") for d in l12)
tech = collections.Counter((d.get("technique") or "unknown") for d in l12)
print(f"L1+L2 applications {N}, benchmark problems {len({d['_p'] for d in l12})}, states {len(state)}, techniques {len(tech)}")
print("top states:", [(s, round(100 * n / N, 1)) for s, n in state.most_common(4)])
print("top techniques:", [(s, round(100 * n / N, 1)) for s, n in tech.most_common(4)])
per = collections.defaultdict(set)
for d in l12:
    for k in ("state_before", "state_after"):
        if d.get(k):
            per[d["_p"]].add(d[k])
print("avg distinct states per benchmark problem:", round(st.mean(len(v) for v in per.values()), 2))

def pie(ax, counter, title, top):
    items = counter.most_common(top)
    rest = sum(counter.values()) - sum(n for _, n in items)
    labels = [k.replace("_", " ") for k, _ in items] + (["other"] if rest else [])
    vals = [n for _, n in items] + ([rest] if rest else [])
    cmap = plt.get_cmap("tab20")
    cols = [cmap(i % 20) for i in range(len(vals))]
    if rest:
        cols[-1] = (0.8, 0.8, 0.8, 1)
    w, _ = ax.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(edgecolor="white", linewidth=0.6))
    ax.set_title(title, fontsize=12)
    tot = sum(vals)
    ax.legend(w, [f"{l} ({100 * v / tot:.1f}%)" for l, v in zip(labels, vals)], loc="upper center", bbox_to_anchor=(0.5, -0.02),
              fontsize=8.5, ncol=1, frameon=False)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2))
pie(axes[0], state, f"Performance state before the application (n={N})", 9)
pie(axes[1], tech, "Technique applied", 12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"opt_pie_l12_l40s.{ext}", dpi=250, bbox_inches="tight")
print("wrote", OUT / "opt_pie_l12_l40s.png")
