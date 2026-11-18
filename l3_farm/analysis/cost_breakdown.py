"""Token / cost breakdown of the Level 3 sweep (from per-round KernelBlaster LLM telemetry in results/*/work/state.json).
Lower bound: restarted/killed attempts whose telemetry was lost are not counted."""
import json, glob, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRICE = dict(uncached=5.0, cache_read=0.5, cache_write=6.25, output=25.0)   # $/M tokens, ASSUMED Opus-class list price
TYPES = [("output", "output"), ("cache_write", "cache write"), ("cache_read", "cache read"), ("uncached", "uncached input")]
FAM = {
    "CNN": {4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27},
    "Attention / transformer": {28,29,30,31,32,43,44,45,50},
    "Recurrent": {33,34,35,36,37,38,39,40,41,42},
    "MLP": {1,2,3},
    "Other (NetVlad, Mamba)": {46,47,48,49},
}
def fam(name):
    k = int(name.split("_")[0])
    return next(f for f, s in FAM.items() if k in s)

role_tok = collections.defaultdict(collections.Counter)
role_calls = collections.Counter()
prob_cost = collections.Counter(); prob_tok = collections.defaultdict(collections.Counter)
round_cost = collections.Counter()
def cost_of(c): return sum(c[t] * PRICE[t] / 1e6 for t, _ in TYPES)
for f in glob.glob("results/*/work/state.json") + glob.glob("results/_v1_pre_plan_fix/*/work/state.json"):
    parts = f.split("/"); name = parts[2] if parts[1] == "_v1_pre_plan_fix" else parts[1]
    if not name[0].isdigit(): continue
    st = json.load(open(f))
    for r in st["rounds"]:
        llm = (((r.get("telemetry") or {}).get("kb") or {}).get("llm") or {})
        for br in llm.get("by_role_model", []):
            c = collections.Counter(output=br.get("output_tokens", 0), cache_write=br.get("cache_write_input_tokens", 0),
                                    cache_read=br.get("cache_read_input_tokens", 0), uncached=br.get("uncached_input_tokens", 0))
            role_tok[br["role"]].update(c); role_calls[br["role"]] += br.get("calls", 0)
            prob_tok[name].update(c); prob_cost[name] += cost_of(c); round_cost[r["round"]] += cost_of(c)

total = sum(cost_of(c) for c in role_tok.values())
print("total est. cost $%.0f; calls %d" % (total, sum(role_calls.values())))
json.dump(dict(total_usd=total, role={k: dict(v, calls=role_calls[k], usd=cost_of(v)) for k, v in role_tok.items()},
               problem_usd=dict(prob_cost), round_usd=dict(round_cost), prices_per_M=PRICE), open("analysis/cost_breakdown.json", "w"), indent=1)

# ---------- plot ----------
COL = {"output": "#2a78d6", "cache_write": "#eb6834", "cache_read": "#1baf7a", "uncached": "#eda100"}
INK, SUB, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB,
                     "axes.facecolor": BG, "figure.facecolor": BG, "text.color": INK})
fig = plt.figure(figsize=(15, 9.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.42, wspace=0.34, left=0.145, right=0.985, top=0.82, bottom=0.07)
fig.text(0.012, 0.965, "Where the tokens and dollars went: Level 3 sweep (50 problems, Opus 5.5, ~34.8k LLM calls)", fontsize=15, weight="bold", ha="left")
fig.text(0.012, 0.925, f"Estimated ${total:,.0f} at assumed Opus-class list prices ($5 in / $25 out / $6.25 cache write / $0.50 cache read per M tokens); lower bound "
         "(lost restarts and hand runs not counted).\nAll calls used one model (azure/anthropic/claude-opus-5-5); \"agent\" below = the flow's LLM role.", fontsize=9.5, color=SUB, ha="left", va="top")

def stacked_h(ax, labels, series, usd=True, title="", total_fmt=None, label_cols=("output",)):
    left = [0]*len(labels)
    for key, nice in TYPES:
        vals = [series[l].get(key, 0) * (PRICE[key]/1e6 if usd else 1e-6) for l in labels]
        ax.barh(labels, vals, left=left, color=COL[key], edgecolor=BG, linewidth=1.5, label=nice, height=0.62)
        left = [a+b for a, b in zip(left, vals)]
    for i, l in enumerate(labels):
        ax.text(left[i] + max(left)*0.01, i, (f"${left[i]:,.0f}" if usd else f"{left[i]:,.1f}M"), va="center", fontsize=9, color=INK)
    ax.invert_yaxis(); ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.set_xlim(0, max(left)*1.14)

roles = sorted(role_tok, key=lambda r: -cost_of(role_tok[r]))
nice_role = {"codegen_simple": "codegen (simple)", "codegen_hard": "codegen (hard)", "fix": "fix (build/check errors)", "fix_escalated": "fix (escalated)",
             "reflect": "reflect (knowledge)", "hypothesis": "hypothesis (knowledge)", "discovery": "discovery (knowledge)"}
ax1 = fig.add_subplot(gs[0, 0])
stacked_h(ax1, [nice_role.get(r, r) for r in roles], {nice_role.get(r, r): role_tok[r] for r in roles}, title="Cost by agent role ($, by token type)")
ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

ax2 = fig.add_subplot(gs[0, 1])
stacked_h(ax2, [nice_role.get(r, r) for r in roles], {nice_role.get(r, r): role_tok[r] for r in roles}, usd=False, title="Tokens by agent role (millions)")
ax2.set_yticklabels([])

# calls per role
ax3 = fig.add_subplot(gs[0, 2])
calls = [role_calls[r] for r in roles]; avg_out = [role_tok[r]["output"]/max(role_calls[r], 1) for r in roles]
ax3.barh([nice_role.get(r, r) for r in roles], calls, color="#6250d6", edgecolor=BG, linewidth=1.5, height=0.62)
for i, (c, a) in enumerate(zip(calls, avg_out)): ax3.text(c + max(calls)*0.01, i, f"{c:,} calls; {a/1000:.1f}k output/call", va="center", fontsize=8.5)
ax3.invert_yaxis(); ax3.set_title("Calls, and output length per call", loc="left", fontsize=11, weight="bold", pad=8)
ax3.set_yticklabels([]); ax3.set_xticks([0,10000,20000]); ax3.set_xticklabels(["0","10k","20k"]); ax3.grid(axis="x", color=GRID, linewidth=0.8); ax3.set_axisbelow(True); ax3.set_xlim(0, max(calls)*2.0)
for s in ("top", "right", "left"): ax3.spines[s].set_visible(False)
ax3.tick_params(axis="y", length=0)

# by family
fams = list(FAM)
fam_tok = {f: collections.Counter() for f in fams}
for n, c in prob_tok.items(): fam_tok[fam(n)].update(c)
fam_n = {f: len([n for n in prob_tok if fam(n) == f]) for f in fams}
ax4 = fig.add_subplot(gs[1, 0])
order = sorted(fams, key=lambda f: -cost_of(fam_tok[f]))
stacked_h(ax4, [f"{f} ({fam_n[f]})" for f in order], {f"{f} ({fam_n[f]})": fam_tok[f] for f in order}, title="Cost by model family ($, problems in parentheses)")

# top problems
ax5 = fig.add_subplot(gs[1, 1])
top = [n for n, _ in prob_cost.most_common(12)]
stacked_h(ax5, top, {n: prob_tok[n] for n in top}, title="Twelve most expensive problems ($)")
ax5.tick_params(axis="y", labelsize=8.5)

# by round
ax6 = fig.add_subplot(gs[1, 2])
rounds = sorted(round_cost); vals = [round_cost[r] for r in rounds]
ax6.bar([f"round {r}" for r in rounds], vals, color="#2a78d6", edgecolor=BG, linewidth=1.5, width=0.6)
for i, v in enumerate(vals): ax6.text(i, v + max(vals)*0.015, f"${v:,.0f}\n({v/total:.0%})", ha="center", fontsize=9)
ax6.set_title("Cost by optimization round ($)", loc="left", fontsize=11, weight="bold", pad=8)
ax6.grid(axis="y", color=GRID, linewidth=0.8); ax6.set_axisbelow(True); ax6.set_ylim(0, max(vals)*1.22)
for s in ("top", "right"): ax6.spines[s].set_visible(False)
ax6.tick_params(axis="x", length=0)
fig.savefig("analysis/cost_breakdown.png", dpi=130)
print("saved")
