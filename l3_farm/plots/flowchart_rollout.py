#!/usr/bin/env python3
"""Decision flow of the agentic rollout phase (KernelBlaster run_multislot, box 5 of modelblaster_flow.png): initial setup, then R rollouts x S steps with every decision point, then the final composite.
Drawn from KernelBlaster docs/multislot_optimization.md, docs/semantic_update_pass.md and agents/opt_multislot.py (initialize, _seed_candidates, _rollout, _step_body, _decide_e2e, patience / resync).
-> out/modelblaster_rollout_decisions.{png,svg,pdf}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(26, 17)); ax.set_xlim(0, 130); ax.set_ylim(-5.4, 91); ax.axis("off")
MB, KB, STORE, GRAY, A6C, DEC, LLMC, GPUC = "#cfe2f3", "#fce5cd", "#d9ead3", "#eeeeee", "#f4cccc", "#fff2cc", "#ffe599", "#d9d2e9"
CHIP = {"R": ("#9fc5e8", "#1f4e79"), "W": ("#ea9999", "#990000"), "RW": ("#d5a6bd", "#741b47")}
def box(x, y, w, h, title, body, fc, a6=False, fs=8.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#c00000" if a6 else "#444", lw=2.6 if a6 else 1.2, ls="--" if a6 else "-", zorder=3))
    ax.text(x + 0.7, y + h - 0.8, title, fontsize=fs + 1.2, fontweight="bold", va="top", ha="left", zorder=4)
    ax.text(x + 0.7, y + h - 2.5, body, fontsize=fs, va="top", ha="left", zorder=4, linespacing=1.32)
def diamond(cx, cy, w, h, text, fc=DEC, fs=9):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc=fc, ec="#7f6000", lw=1.6, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, fontweight="bold", ha="center", va="center", zorder=4, linespacing=1.2)
def frame(x, y, w, h, title, color="#1f4e79"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6", fc="none", ec=color, lw=2.3, ls="--", zorder=1))
    ax.text(x + 1, y + h - 1.0, title, fontsize=12.5, fontweight="bold", color=color, va="center", zorder=4)
def poly(pts, text=None, tp=None, color="#333", ls="-", lw=1.8, rot=0, fs=8.5):
    for p, q in zip(pts[:-2], pts[1:-1]): ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
    if text: ax.text(tp[0], tp[1], text, fontsize=fs, ha="center", va="center", rotation=rot, zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
def chips(xr, y, items, h=1.5):
    for L, m in reversed(items):
        w = 2.5 + 0.75 * len(m); fc, ec = CHIP[m]; x0 = xr - w
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=1.3, zorder=7))
        ax.text(x0 + w / 2, y + h / 2, f"{L}:{m}", fontsize=6.9, fontweight="bold", ha="center", va="center", color=ec, zorder=8); xr = x0 - 0.2
ax.text(65, 89.3, "Decision making in the agentic rollout phase (KernelBlaster run_multislot)", fontsize=19, fontweight="bold", ha="center")
ax.text(65, 87.6, "one problem = one partition with N kernel slots; R rollouts x S steps run in parallel and share a per-slot composite best; every step changes exactly one slot", fontsize=11.5, ha="center", color="#555")
# =========================== A. setup
frame(1, 64.2, 128, 21.3, "A. Initial setup (once per problem)")
box(2.5, 66.4, 22, 14.4, "A1  Baseline run", "hand seeds of every slot, one\npartition run, CUDA-graph replay\nrecords per-slot times, shares,\nend-to-end (deployed) time", GPUC)
diamond(29.6, 73.3, 7.6, 8.6, "ok?")
ax.text(29.6, 68.9, "no: retry\nsequentially, else\nthe problem fails", fontsize=7.4, ha="center", va="top", zorder=5)
box(35, 66.4, 22, 14.4, "A2  Bookkeeping", "slot share of partition time,\nroofline state per slot,\nbest[slot] = seed, seed impl\nrecords, baseline profile (ncu opt.)", MB)
box(58.5, 66.4, 28, 14.4, "A3  Seed from the impl store", "DECISION: stored verified kernel for this\nop + signature + arch + shape bucket?\nyes -> trial composite; a slot keeps it only\nif it passes its isolated check and is faster;\nthe composite must not be slower end to end", DEC)
box(88, 66.4, 23.5, 14.4, "A4  Reuse candidates", "DECISION per candidate (sibling / global\nwinners of other shapes, tried by rank):\nkeep iff it passes on this shape and\nis < 0.98x the slot's time; composite\nguard as in A3", DEC)
box(113, 66.4, 14.5, 14.4, "A5  Region phase", "A6, when on: LLM tool\nloop on region slots\n(own budget) before the\ngeneric steps", KB, a6=True)
for x0, x1 in ((24.5, 25.8), (33.4, 35), (57, 58.5), (86.5, 88), (111.5, 113)): poly([(x0, 73.3), (x1, 73.3)], lw=1.6)
chips(56.8, 78.3, [("I", "W")]); chips(85.8, 78.3, [("I", "R")]); chips(111.0, 78.3, [("I", "R")])
poly([(119.8, 66.4), (119.8, 65.3), (9.5, 65.3), (9.5, 60.6)], "composite best C0; start R rollouts in parallel", (62, 65.3))
# =========================== B. rollout steps
frame(1, 24, 128, 39.5, "B. Rollout step loop (repeated S times per rollout; R rollouts in parallel; successful slots compose)")
# row 1: what to try
diamond(9.5, 56.2, 13, 9.2, "resync?")
box(18, 51.8, 19, 8.8, "1  Pick slot", "weight = time share x 0.6^stalls\nx exploration bonus; slots\nunder min_share skipped", MB)
box(39, 51.8, 19, 8.8, "2  Classify state", "roofline from bytes / FLOPs vs\nmeasured time: latency-bound,\nunderutilised, memory, compute", MB)
box(60, 51.8, 22, 8.8, "3  Pick technique", "UCB1 per (op_class, state, dtype);\nunseen arm starts from its mean\nelsewhere; v2: test a hypothesis", DEC)
box(84, 51.8, 13.5, 8.8, "3b  Pick model", "router policy:\nstructural / stalled / big\nshare -> Opus; else Haiku", DEC, fs=8.2)
for x0, x1 in ((16, 18), (37, 39), (58, 60), (82, 84)): poly([(x0, 56.2), (x1, 56.2)], lw=1.6)
chips(36.8, 58.9, []); chips(57.8, 52.2, [("K", "R")]); chips(81.8, 52.2, [("K", "R")])
# row 2: generate and test (right to left)
box(75, 37.8, 22.5, 10.4, "4  Build the context", "slot source (this rollout's lineage), neighbours'\nsignatures read-only, C reference, shapes,\nmeasured state, impl-store seed / adapt /\nexamples, claims, 'already tried', pitfalls", MB)
chips(97.3, 38.2, [("K", "R"), ("I", "R")])
box(59, 37.8, 14, 10.4, "5  LLM rewrites", "only this slot;\nfull source back;\nresponse journaled\n(resume-safe)", LLMC)
box(40, 37.8, 17.5, 10.4, "6  Build + evaluate", "persistent driver host:\npartition + isolated-op\nchecks; CUDA-graph replay\npaired with the incumbent", GPUC)
diamond(33.4, 43, 11.5, 9.6, "passes\nall checks?")
diamond(14, 43, 17, 10, "paired gain?")
poly([(97.5 - 22.5 + 0, 52.5), (97.5 - 22.5 + 0, 52.5)], lw=0)
poly([(91, 51.8), (91, 48.2)], "next", (92.8, 50.0), lw=1.8)
for x0, x1 in ((75, 73), (59, 57.5), (40, 39.2), (27.8, 22.5)): poly([(x0, 43), (x1, 43)], lw=1.6)
ax.text(25.2, 44.3, "yes", fontsize=8.5, fontweight="bold", color="#38761d", zorder=6)
# fix loop (row 3 under D2/Build)
box(22.5, 25.4, 33.5, 9.6, "Fix loop (no, or capture defect)", "<= max_fix attempts with the error text, failing slot, known pitfalls;\nfix model = Haiku, last attempt escalates to Opus; a verified kernel that\nbreaks CUDA-graph capture also returns here; still failing: compile_error\n(-0.05) / runtime or wrong_output (-0.1) -> discarded, recorded", LLMC)
poly([(33.4, 38.2), (33.4, 35.3)], "no", (35.2, 36.8)); poly([(50.5, 35.1), (50.5, 37.6)], "re-test", (53.4, 36.4))
# outcome classes and commit
poly([(14, 37.9), (14, 35.3)], None)
ax.text(14.5, 36.6, "improved / neutral / regressed", fontsize=7.6, ha="left", va="center", zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))
box(2.5, 25.4, 19, 9.6, "Commit and learn", "outcome record, impl store, bandit +\nclaim tallies, composite merge, stalls,\njournal checkpoint; then the stop test:\npatience 8 misses, S steps, min_share", MB)
chips(21.3, 33.2, [("K", "W"), ("I", "W")])
ax.text(2.6, 44.3, "", fontsize=1)
box(58, 25.4, 39.5, 9.6, "Reflect / upkeep  (LLM, every N new records)", "reflect: claim edits (validated against the records); v2 upkeep: hypothesis generation, state\ndiscovery, promotions; model = the cheap reflect / upkeep role; edits feed the next steps\nthrough the claims in the prompt (4) and the technique statistics (3)", LLMC)
chips(97.3, 33.2, [("K", "W")])
poly([(2.5, 30.2), (1.8, 30.2), (1.8, 56.2), (3.0, 56.2)], "otherwise: next step", (1.8, 41), rot=90)
poly([(97.5, 27.3), (98.2, 27.3), (98.2, 24.6), (97.5, 24.6)], lw=0) if False else None
poly([(11, 25.4), (11, 20.4)], "stop test met: finish", (11, 22.9))
# =========================== decision inputs panel
ax.add_patch(FancyBboxPatch((100, 24.6), 28.6, 38.6, boxstyle="round,pad=0.25,rounding_size=0.8", fc="#f3f3f3", ec="#444", lw=1.2, zorder=3))
ax.text(101, 62.0, "What each decision uses", fontsize=11.5, fontweight="bold", va="top", zorder=4)
rows = [("Stop / continue", "patience: N=8 consecutive committed steps without a new slot best; steps S; min_share"),
        ("Resync", "a rollout that stalls while lagging the composite restarts from it"),
        ("Which slot", "time share x 0.6^stalls x (1 + 0.5 sqrt(ln T / (1 + tries)))"),
        ("Which state", "roofline numbers vs arch peaks (+ learned v2 states)"),
        ("Which technique", "UCB1 stats keyed (op_class, state, dtype), cross-key prior, v2 hypotheses (explore)"),
        ("Which model", "category | stall:N | share:X | explore -> hard; fix-escalate on the last fix"),
        ("What context", "impl-store seed / adapt / examples, semantic claims, tried-notes, pitfalls"),
        ("Accept?", "paired end-to-end gain vs the incumbent; >= 1% of the partition: re-measure vs the composite, reward = the smaller gain (improved); < 1%: neutral, kept unless > 0.5% slower; slower: regressed, dropped"),
        ("What is learned", "outcome log, bandit stats, impl store, claims, hypotheses, states")]
import textwrap
y = 59.6
for k, v in rows:
    lines = textwrap.wrap(v, 60)
    ax.text(101, y, k, fontsize=8.6, fontweight="bold", va="top", zorder=4, color="#1f4e79")
    ax.text(101, y - 1.3, "\n".join(lines), fontsize=7.3, va="top", zorder=4, linespacing=1.2); y -= 1.2 + 1.0 * len(lines) + 0.3
# =========================== C. finish
frame(1, 2, 128, 18, "C. Finish (after all rollouts or a patience stop)")
box(2.5, 3.6, 29, 11.2, "C1  Rebuild the composite", "best verified source of every slot, assembled\nand run as one partition (every check, CUDA-graph\nreplay); a slot that breaks falls back to its seed", GPUC)
box(35, 3.6, 29, 11.2, "C2  Write results", "result.json (per-step records, efficiency, usage by\nmodel), best/slots/*.cu, winners.json (>= 1.05x per\nop / signature / arch / shape set), journal", STORE)
box(67.5, 3.6, 29, 11.2, "C3  Propose graph changes", "propose_fusions / propose_splits -> proposals.json\n(kb-proposal/1) for ModelBlaster's hint layer", MB)
box(100, 3.6, 28, 11.2, "C4  Back in kb_graph_opt", "ingest: winners -> shape-guarded cache (step 6);\nhints decided by paired runs (step 7); next round\nstarts from the cache and the accepted hints", GRAY)
for x0, x1 in ((31.5, 35), (64, 67.5), (96.5, 100)): poly([(x0, 9.2), (x1, 9.2)], lw=1.6)
from matplotlib.patches import Rectangle
lx, ly = 2, -1.6
ax.text(lx, ly + 1.3, "Legend:", fontsize=10.5, fontweight="bold", va="center")
for i, (c, t, a6) in enumerate([(MB, "ModelBlaster-side step", False), (LLMC, "LLM call", False), (GPUC, "GPU evaluation", False), (DEC, "decision", False), (STORE, "output / store", False), (GRAY, "hand-back to the graph optimiser", False), (KB, "A6 add-on (dashed red border)", True)]):
    xx = lx + 9 + i * 17.5; ax.add_patch(Rectangle((xx, ly + 0.6), 1.6, 1.4, fc=c, ec="#c00000" if a6 else "#444", lw=2 if a6 else 1, ls="--" if a6 else "-", zorder=3)); ax.text(xx + 2.1, ly + 1.3, t, fontsize=9, va="center")
ax.text(lx, -3.1, "Knowledge chips:", fontsize=10, fontweight="bold", va="center")
chips(lx + 28, -3.85, [("K", "R")]); ax.text(lx + 28.5, -3.1, "read", fontsize=9, va="center"); chips(lx + 42, -3.85, [("K", "W")]); ax.text(lx + 42.5, -3.1, "write", fontsize=9, va="center")
chips(lx + 56, -3.85, [("K", "RW")]); ax.text(lx + 57, -3.1, "read + write", fontsize=9, va="center")
ax.text(lx + 74, -3.1, "K knowledge.db   I impl store", fontsize=9, va="center")
ax.text(lx, -4.6, "Beam search (--search beam) replaces the independent rollouts by a beam of states, each expanded beam_expand ways per step. Resumable at step granularity (per-problem journal).", fontsize=8.8, color="#333", va="center")
for ext in ("png", "svg", "pdf"): fig.savefig(OUT / f"modelblaster_rollout_decisions.{ext}", dpi=170, bbox_inches="tight")
print("saved", OUT / "modelblaster_rollout_decisions.png")
