#!/usr/bin/env python3
"""Detailed flowchart of step 7 (Graph move: decide hints) of the ModelBlaster flow. Drawn from ModelBlaster/pipeline/kb_graph_opt.py (_cands, evaluate_hints, finish_hint),
kb_proposals.py (build_variant, remeasure, decide; ACCEPT_GAIN = 3%) and docs/graph_optimizer.md (parallel hint decisions). -> out/modelblaster_graph_move.{png,svg,pdf}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(26, 15.5)); ax.set_xlim(0, 130); ax.set_ylim(-3.5, 80); ax.axis("off")
MB, KB, STORE, GRAY, A6C, GPU, LLM = "#cfe2f3", "#fce5cd", "#d9ead3", "#eeeeee", "#f4cccc", "#d9d2e9", "#ffe599"
OKC, BADC, UNDC = "#b6d7a8", "#ea9999", "#fff2cc"
def box(x, y, w, h, text, fc, fs=9.4, a6=False, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#c00000" if a6 else "#444", lw=2.4 if a6 else 1.2, ls="--" if a6 else "-", zorder=3))
    lines = text.split("\n"); ty = y + h - 0.9
    ax.text(x + 0.8, ty, lines[0], fontsize=fs + 1, fontweight="bold", va="top", ha="left", zorder=4)
    if len(lines) > 1: ax.text(x + 0.8, ty - 1.7, "\n".join(lines[1:]), fontsize=fs, va="top", ha="left", zorder=4, linespacing=1.35)
def gate(cx, cy, w, h, text, fc="#ffffff", fs=9):
    ax.add_patch(Polygon([(cx - w / 2, cy), (cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2)], closed=True, fc=fc, ec="#444", lw=1.3, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, ha="center", va="center", zorder=4, linespacing=1.3)
def poly(pts, text=None, tp=None, color="#333", ls="-", lw=1.8, rot=0, fs=8.8):
    for p, q in zip(pts[:-2], pts[1:-1]): ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
    if text: ax.text(tp[0], tp[1], text, fontsize=fs, ha="center", va="center", rotation=rot, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
CHIP = {"R": ("#9fc5e8", "#1f4e79"), "W": ("#ea9999", "#990000"), "RW": ("#d5a6bd", "#741b47")}
def chips(xr, y, items, h=1.5):
    for L, m in reversed(items):
        w = 2.5 + 0.75 * len(m); fc, ec = CHIP[m]; x0 = xr - w
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=1.3, zorder=7))
        ax.text(x0 + w / 2, y + h / 2, f"{L}:{m}", fontsize=6.9, fontweight="bold", ha="center", va="center", color=ec, zorder=8); xr = x0 - 0.2

ax.text(65, 78.2, "Step 7: Graph move, deciding graph hints (fuse / split / merge-siblings / region ...) by paired kernel runs", fontsize=19, fontweight="bold", ha="center")
ax.text(65, 76.3, "kb_graph_opt, once per round after the kernel move and ingest; a hint is accepted only if its VARIANT graph beats the ORIGINAL by >= 3% after an independent re-measurement", fontsize=11.5, ha="center", color="#555")

# inputs
box(1, 63, 27, 10, "Input: graph_hints.json\nhints with status proposed, from\nstep 2 (profile / sibling convs / region windows),\nstep 6 (KernelBlaster proposals: fuse, split),\nand the kernel cache's shape guards", STORE)
chips(27.6, 71.6, [("H", "R")])
# 7.1a
box(33, 62, 31, 11, "7.1a Select candidates  [CPU]\nkind in fuse, split_slot, (A6) fuse_siblings /\nfuse_attention / fold_layout / region; target cuda\nest. gain >= min_gain_pct (2%); sorted by est. gain\nkeep the top max_variants (3) that have an IR", MB)
chips(63.6, 71.6, [("H", "R")])
box(68, 62, 29, 11, "A6 round controller\nthe move runs only if the controller picked 'graph'\n(graph-only round 0 when graph gain >= 8% or\nsmall-op share >= 25%); a move is dropped after\n< 3% for 2 rounds", A6C, a6=True)
poly([(28, 68), (33, 68)]); poly([(68, 68), (64, 68)])
# 7.1b
box(1, 46, 31, 12.5, "7.1b  Footprints + conflict waves  [CPU]\nfootprint = ops it touches, cache entries it reads,\nentries an acceptance writes.  Conflict = shares an\nop, or one writes what the other reads.\nrank order -> hint i goes in the first wave after\nevery earlier conflicting hint  (--parallel-hints)", MB)
poly([(48.5, 62), (48.5, 60.3), (16.5, 60.3), (16.5, 58.5)])
# 7.2
box(36, 46, 29, 12.5, "7.2  Build ORIGINAL + VARIANT  [CPU]\nsame ops, same budget; the variant applies the\nhint (fuse chain / split by shape bucket / ...),\nseeds from best stage kernels or cached winners\napply fails or graph walk differs -> rejected\nearly (no LLM spent)", MB)
poly([(32, 52), (36, 52)], "per wave", (34, 53.3))
# 7.3
box(69, 46, 29, 12.5, "7.3  KernelBlaster run / wave [LLM+GPU]\none run_multislot, all problems of the wave,\n--parallel-problems; both sides read the same\nstore state, one writer for knowledge.db;\nsame rollouts / steps as step 5 (5a, 5.1-5.8)", KB)
chips(97.6, 56.6, [("K", "RW"), ("I", "RW")])
poly([(65, 52), (69, 52)])
# gate: final_ok
gate(83.5, 38.5, 25, 8, "both sides final_ok?\n(driver-checked composite)")
poly([(83.5, 46), (83.5, 42.5)])
# undecided branch
box(100, 36, 29, 9, "undecided  (infrastructure failure)\nfinal run lost to GPU / link / staging, not to\nthe composite: no vote, hint stays a candidate\nfor a later round", UNDC)
poly([(96, 38.5), (100, 38.5)], "infra only", (98, 39.8), fs=8.2)
# reject branch (not ok)
box(100, 24, 29, 9, "rejected  (side failed / no passing run)\nevidence recorded with the reason; the hint's\nstatus is the majority over every problem\nthat tried it (a tie rejects)", BADC)
poly([(83.5, 34.5), (83.5, 28.5), (100, 28.5)], "a side failed (not infra)", (92, 29.7), fs=8.2)
# 7.4 re-measure
box(1, 25, 33, 16, "7.4  Re-measure both bests  [GPU]\nbuild both composite bests, run them INTERLEAVED on\none leased GPU (GpuLeases + per-GPU flock file), so\nthe two sides share the device and time; reps = 5,\nmedian per side; <= 2 workers per GPU (one builds,\nthe other times); retries; infra failure -> undecided", GPU)
poly([(71, 38.5), (34, 38.5)], "yes", (52, 39.8))
chips(33.6, 39.2, [("C", "R")])
ax.text(52, 36.6, "(hint is re-timed as soon as both result.json exist, while\nKernelBlaster still works on the other hints of the wave)", fontsize=8.3, ha="center", va="center", color="#555", style="italic")
# 7.5 decide
gate(17.5, 15.5, 31, 9.5, "7.5  gain = (orig - variant) / orig\n>= accept_gain (3%) ?", fs=9.4)
poly([(17.5, 25), (17.5, 20.25)])
# outcomes
box(38, 9.5, 27, 11, "accepted\nhint status accepted (by majority of its decisions);\nevidence with both re-measured series is stored;\nnew slots (fused / split / region) are ingested into\nthe kernel cache", OKC)
chips(64.6, 18.7, [("H", "W"), ("C", "W"), ("I", "W")])
box(38, -1.5, 27, 9, "rejected\nstatus rejected; kept as evidence so it is not\nproposed again for this IR; its ingest is skipped", BADC)
chips(64.6, 5.5, [("H", "W")])
poly([(33, 15.5), (38, 15.5)], "yes", (35.5, 16.8))
poly([(17.5, 10.75), (17.5, 3), (38, 3)], "no", (23, 4.3))
# 7.6 next
box(70, 7, 29, 14, "7.6  Next round\naccepted hints are applied at the next profile (step 1):\nthe fused / split graph is what the next round's kernel\nmove optimises; cached variant kernels seed it.\nunder A6, step 8 (zero-LLM reuse) then re-routes\nthem across shapes before the fixpoint check (9)", MB)
poly([(65, 15), (70, 15)])
poly([(65, 3), (68, 3), (68, 9), (70, 9)])
# footer / legend
box(102, 8, 27, 12, "Knowledge touched\nH graph_hints.json: read (7.1a), evidence + status (7.5)\nC kernel cache: read seeds (7.2), write accepted slots\nK / I knowledge.db, impl store: outcomes and kernels\nof both sides' runs (7.3) are learned either way", STORE, fs=8.6)
lx = 1
for i, (c, t) in enumerate([(MB, "CPU step"), (KB, "LLM-driven step"), (GPU, "GPU measurement"), (A6C, "A6 add-on"), (OKC, "accepted"), (BADC, "rejected"), (UNDC, "undecided (retried later)"), (STORE, "store")]):
    xx = lx + 1 + i * 15.5; ax.add_patch(Rectangle((xx, -3.0), 1.6, 1.3, fc=c, ec="#444", lw=1, zorder=3)); ax.text(xx + 2.1, -2.35, t, fontsize=9, va="center")
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"modelblaster_graph_move.{ext}", dpi=170, bbox_inches="tight")
print("saved", OUT / "modelblaster_graph_move.png")
