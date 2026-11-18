#!/usr/bin/env python3
"""sonnet_test.png: final model time (quiet, one common node) and LLM $ by model per arm."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = Path(__file__).resolve().parent.parent
cross = json.load(open(D / "res/cross_one_node.json"))["results"]
CT = Path("/private/tmp/claude-501/-Users-krisd-Documents/bf39cca1-7e22-482f-8c91-bf7b1d5240f4/scratchpad/bt/res")
def led(a):
    p = (CT / a) if a in ("C", "C2", "C3", "H") else (D / "res" / a)
    return json.load(open(p / "analysis.json"))["ledger"]
order = ["C", "C2", "C3", "H", "H5", "S3", "S5", "S0"]
label = {"C": "C ctrl\n(slow node)", "C2": "C2 ctrl", "C3": "C3 ctrl", "H": "H Haiku\nstall:3", "H5": "H5 Haiku\nstall:5+ff", "S3": "S3 Sonnet\nstall:3", "S5": "S5 Sonnet\nstall:5+ff", "S0": "S0 Sonnet\nno stall"}
col = {"opus": "#4c78a8", "sonnet": "#f58518", "haiku": "#54a24b"}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
ys = [cross[a]["model_e2e_us"] for a in order]
cs = ["#9aa0a6" if a in ("C", "C2", "C3") else "#4c78a8" for a in order]
a1.bar(range(len(order)), ys, color=cs)
for i, y in enumerate(ys):
    a1.text(i, y + 15, f"{y:.0f}", ha="center", fontsize=8)
m = sum(cross[a]["model_e2e_us"] for a in ("C2", "C3")) / 2
a1.axhline(m, color="k", lw=.8, ls="--"); a1.axhspan(m * 0.985, m * 1.015, color="k", alpha=.07)
a1.set_ylim(7500, 8150); a1.set_xticks(range(len(order))); a1.set_xticklabels([label[a] for a in order], fontsize=7)
a1.set_ylabel("final model us (CUDA graph), quiet, one common node"); a1.set_title("Final quality (n=1 each; band = +-1.5% around mean of C2, C3)", fontsize=9)
bot = [0] * len(order)
for fam in ("opus", "sonnet", "haiku"):
    v = []
    for a in order:
        v.append(sum(x["usd"] for k, x in led(a).items() if fam in k))
    a2.bar(range(len(order)), v, bottom=bot, color=col[fam], label=fam)
    bot = [b + x for b, x in zip(bot, v)]
for i, b in enumerate(bot):
    a2.text(i, b + 0.6, f"${b:.1f}", ha="center", fontsize=8)
a2.set_xticks(range(len(order))); a2.set_xticklabels([label[a] for a in order], fontsize=7)
a2.set_ylabel("LLM $ per run (assumed prices)"); a2.legend(fontsize=8); a2.set_title("Cost per run", fontsize=9)
fig.suptitle("43_MinGPTCausalAttention: Sonnet / Haiku / Opus routing (ff = fix-follow)", fontsize=11)
fig.tight_layout(); fig.savefig(D.parent / "sonnet_test.png", dpi=130)
