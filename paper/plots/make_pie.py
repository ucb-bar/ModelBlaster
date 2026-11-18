#!/usr/bin/env python3
"""Pie in the style of plot/pie.png: slices are optimization families, colored by performance-state group
(one hue per state; shade = optimization family). Left legend: State Groups; right legend: Optimizations.
Reads counts_state_family.csv (extract_counts.py + families.py). Output: opt_pie_l40s_l3.{png,pdf}"""
import csv, collections, colorsys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
N_GROUPS, MIN_FRAC = 10, 0.08    # top state groups; families below 8% of a group merge into "Other small opts"
rows = [(r["state"], r["family"], int(r["applications"])) for r in csv.DictReader(open(HERE / "counts_state_family.csv"))]
total = sum(n for *_, n in rows)
by_state = collections.defaultdict(collections.Counter)
for st, fm, n in rows: by_state[st][fm] += n
order = sorted(by_state, key=lambda s: -sum(by_state[s].values()))
groups = order[:N_GROUPS]
rest = collections.Counter()
for st in order[N_GROUPS:]: rest.update(by_state[st])
fam_rank = [f for f, _ in collections.Counter({f: sum(by_state[s][f] for s in by_state) for s in [0] for f in {f for s2 in by_state for f in by_state[s2]}}).most_common()
            if f not in ("Other", "Agent-proposed (unresolved)")]
fam_rank += ["Other small opts"]
def light(c, f):
    k = fam_rank.index(f) if f in fam_rank else len(fam_rank) - 1
    h, l, s = colorsys.rgb_to_hls(*c[:3]); t = k / max(1, len(fam_rank) - 1)
    return colorsys.hls_to_rgb(h, 0.30 + 0.52 * t, s * (1 - 0.25 * t))
base = [plt.get_cmap("tab20")(i) for i in (0, 2, 4, 6, 8, 10, 12, 14, 16, 18)]
slices, colors, group_info = [], [], []
def add_group(name, counter, color):
    g = sum(counter.values()); small = 0; items = []
    for fm, n in counter.most_common():
        if fm in ("Other", "Agent-proposed (unresolved)") or n / g < MIN_FRAC: small += n
        else: items.append((fm, n))
    items.sort(key=lambda x: fam_rank.index(x[0]))
    if small: items.append(("Other small opts", small))
    for fm, n in items:
        slices.append((name, fm, n)); colors.append((0.82, 0.82, 0.82, 1) if fm == "Other small opts" else light(color, fm))
    group_info.append((name, len(items), color))
for g, c in zip(groups, base): add_group(g, by_state[g], c)
add_group("other states", rest, (0.45, 0.45, 0.45, 1))

fig = plt.figure(figsize=(24, 8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.0], wspace=0.02)
axl, axp, axr = (fig.add_subplot(gs[0, i]) for i in range(3))
axp.pie([n for *_, n in slices], colors=colors, startangle=90, counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=0.7), autopct=lambda p: f"{p:.1f}%" if p >= 2.5 else "",
        pctdistance=0.78, textprops={"fontsize": 11, "color": "white", "fontweight": "bold"})
axp.set_title(f"{total:,} optimization applications (Level 3, L40S headline sweep)", fontsize=14)
axl.axis("off"); axr.axis("off")
axl.legend(handles=[Patch(color=c, label=f"{g.replace('_', ' ').title()} ({k} opts)") for g, k, c in group_info],
           title="State Groups", loc="center", fontsize=12, title_fontsize=14, frameon=True)
axr.legend(handles=[Patch(color=light((0.35, 0.35, 0.35, 1), f), label=f) for f in fam_rank[:-1]] + [Patch(color=(0.82, 0.82, 0.82), label="Other small opts")],
           title="Optimizations (shade within each state hue)", loc="center", fontsize=11.5, title_fontsize=13, frameon=True)
for ext in ("png", "pdf"):
    fig.savefig(HERE / f"opt_pie_l40s_l3.{ext}", dpi=250, bbox_inches="tight")
print(total, "applications;", len(slices), "slices;", len(fam_rank) - 1, "families")
