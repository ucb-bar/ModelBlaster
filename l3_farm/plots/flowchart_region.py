#!/usr/bin/env python3
"""Flowchart of A5, the region phase (--region-loop on): a bounded tool-using LLM loop on the region slots before the generic rollouts. Drawn from KernelBlaster docs/region_loop.md and
agents/opt_multislot.py (_region_phase, _region_one, _region_event, _region_knowledge). -> out/modelblaster_region_loop.{png,svg,pdf}"""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(26, 17.5)); ax.set_xlim(0, 130); ax.set_ylim(-6, 90); ax.axis("off")
MB, KB, STORE, GRAY, A6C, DEC, LLMC, GPUC = "#cfe2f3", "#fce5cd", "#d9ead3", "#eeeeee", "#f4cccc", "#fff2cc", "#ffe599", "#d9d2e9"
CHIP = {"R": ("#9fc5e8", "#1f4e79"), "W": ("#ea9999", "#990000"), "RW": ("#d5a6bd", "#741b47")}
def box(x, y, w, h, title, body, fc, a6=False, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#c00000" if a6 else "#444", lw=2.6 if a6 else 1.2, ls="--" if a6 else "-", zorder=3))
    ax.text(x + 0.7, y + h - 0.8, title, fontsize=fs + 1.2, fontweight="bold", va="top", ha="left", zorder=4)
    body = body.replace("$", r"\$")
    ax.text(x + 0.7, y + h - 2.4, body, fontsize=fs, va="top", ha="left", zorder=4, linespacing=1.3)
def diamond(cx, cy, w, h, text, fc=DEC, fs=8.8):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True, fc=fc, ec="#7f6000", lw=1.6, zorder=3))
    ax.text(cx, cy, text, fontsize=fs, fontweight="bold", ha="center", va="center", zorder=4, linespacing=1.2)
def frame(x, y, w, h, title, color="#1f4e79"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.6", fc="none", ec=color, lw=2.3, ls="--", zorder=1))
    ax.text(x + 1, y + h - 1.0, title, fontsize=12.5, fontweight="bold", color=color, va="center", zorder=4)
def poly(pts, text=None, tp=None, color="#333", ls="-", lw=1.8, rot=0, fs=8.3):
    for p, q in zip(pts[:-2], pts[1:-1]): ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, color=color, lw=lw, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
    if text: ax.text(tp[0], tp[1], text, fontsize=fs, ha="center", va="center", rotation=rot, zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))
def chips(xr, y, items, h=1.5):
    for L, m in reversed(items):
        w = 2.5 + 0.75 * len(m); fc, ec = CHIP[m]; x0 = xr - w
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=1.3, zorder=7))
        ax.text(x0 + w / 2, y + h / 2, f"{L}:{m}", fontsize=6.9, fontweight="bold", ha="center", va="center", color=ec, zorder=8); xr = x0 - 0.2
ax.text(65, 88.2, "A5: the region phase (--region-loop on), a tool-using LLM loop on the region slots", fontsize=19, fontweight="bold", ha="center")
ax.text(65, 86.4, "runs once, after the baseline, store seeds and reuse candidates (A1-A4) and before the generic rollouts; slots it works on are then skipped by the rollouts", fontsize=11.5, ha="center", color="#555")
# ================= entry and per-slot setup
frame(1, 66, 128, 19, "A5.0  Entry and per-slot setup")
box(2.5, 67.6, 22, 13.6, "MB pre-loop gate (CPU)", "a hint variant with region ops is\nchecked by graph_verify against the\noriginal IR before export; failing =\nrejected hint, never timed, no LLM", MB)
diamond(30.2, 74.4, 9, 9, "loop\non?")
ax.text(30.2, 68.4, "no: A5 skipped", fontsize=7.6, ha="center", va="top", zorder=5)
box(35.5, 67.6, 22, 13.6, "Find the region slots", "slots with a region (fuse_region)\nnot yet done (resume-safe), sorted by\nruntime share, largest first", MB)
box(59.5, 67.6, 24, 13.6, "Per-slot budget and model", "problem budget split by share (>= 20%,\n>= 6 turns); default 40 turns / 120 tool\ncalls / $8 / 3600 s (A6: $4); model =\nthe hard role (Opus)", DEC)
box(85.5, 67.6, 42, 13.6, "Build the loop's prompt", "region contract (inputs, outputs, scratch, weights, dtypes incl. native),\ndtype rules, hardware + B200 facts (ncu disabled: measure with\nprofile / scratch / paired time), kernel rules, what a region may do\n(persistent / megakernel, side streams joined before return),\nknowledge: claims in scope, already-tried, technique menu incl.\nlaunch-bound, impl-store examples", MB)
chips(126.8, 78.4, [("K", "R"), ("I", "R")])
for x0, x1 in ((24.5, 25.7), (34.7, 35.5), (57.5, 59.5), (83.5, 85.5)): poly([(x0, 74.4), (x1, 74.4)], lw=1.6)
poly([(106, 67.6), (106, 66.2), (13.5, 66.2), (13.5, 64.4)], "slot loop starts (one slot at a time)", (60, 66.2))
# ================= turn loop
frame(1, 24, 99, 40, "A5.1  The turn loop (one region slot)")
diamond(9, 57.6, 13, 8.8, "budget\\nleft?".replace("\\n", "\n"))
box(17.5, 53.6, 24, 8.4, "LLM turn", "reasoning + ONE action block; code in a\ncuda fence or SEARCH / REPLACE edit\nblocks; old turns compacted (sha)", LLMC)
diamond(52, 57.6, 13, 8.8, "reply\nparses?")
diamond(70, 57.6, 13, 8.8, "which\ntool?")
ax.text(81, 57.6, "one action (tool call) per turn", fontsize=8.4, ha="left", va="center", color="#555", zorder=5)
poly([(15.5, 57.6), (17.5, 57.6)], "yes", (16.5, 59.0), lw=1.6); poly([(41.5, 57.6), (45.5, 57.6)], lw=1.6); poly([(58.5, 57.6), (63.5, 57.6)], "yes", (61, 59.0), lw=1.6)
poly([(52, 53.2), (52, 52.2), (29, 52.2), (29, 53.6)], "no: error text back (costs a turn; 4 in a row end the loop; a cut-off reply is never executed)", (40.5, 51.6), fs=7.6)
# tool bus
poly([(70, 53.2), (70, 50.4)], lw=1.6)
ax.plot([27, 94], [50.4, 50.4], color="#333", lw=1.6, zorder=2)
tools = [(14, 22, "show / profile / scratch / edit", "show: region, source / outline, history\nprofile: e2e, per-slot share, launches\nscratch: standalone program (<= 12 per loop)\nedit: SEARCH / REPLACE on the pending source", MB),
         (38, 16, "verify", "static lint, then evaluator with\nevery check: compile, region slot's\nisolated check vs the reference walk,\npartition outputs, repeat_run", GPUC),
         (56, 16, "time", "paired candidate vs committed:\n5 alternating reps, median of\nratios; refused on a clock-\nsuspect GPU", GPUC),
         (74, 12, "commit", "needs the gain rule\nbelow; final-verified\nat the end", DEC),
         (88, 11, "revert / status / finish", "drop pending or last\nN commits; budgets;\nstop the loop", GRAY)]
for x, w, t, b, c in tools:
    box(x, 39.4, w, 9.6, t, b, c, fs=7.6); poly([(x + w / 2, 50.4), (x + w / 2, 49.1)], lw=1.4)
chips(35.9, 40.0, []); 
# results row
box(38, 27.2, 16, 9.6, "verify result", "ok: the pending source is\nverified. Failure: OutcomeRecord\n(compile / run / wrong_output,\nreward -0.05 / -0.1) + error text", DEC, fs=7.6)
chips(53.8, 27.6, [("K", "W")])
box(56, 27.2, 16, 9.6, "time result", "gain = 1 - median(cand /\ncommitted); every timing is an\nOutcomeRecord (reward = gain),\nresult returned to the model", DEC, fs=7.6)
chips(71.8, 27.6, [("K", "W")])
box(74, 27.2, 12, 9.6, "commit rule", "gain >= 15% in one\ntiming, or two\nconsecutive >= 3%;\nelse refused", DEC, fs=7.6)
poly([(46, 39.4), (46, 36.8)], lw=1.4); poly([(64, 39.4), (64, 36.8)], lw=1.4); poly([(80, 39.4), (80, 36.8)], lw=1.4)
ax.text(25, 33.0, "results of show / profile / scratch / edit return directly\nwith the budgets left (turns, tools, $, wall)", fontsize=7.8, ha="center", va="center", color="#444", zorder=5)
poly([(25, 39.4), (25, 37.2)], lw=1.2)
# bus back to the budget check
for x in (46, 64, 80): poly([(x, 27.2), (x, 25.6)], lw=1.2) if False else ax.plot([x, x], [27.2, 25.6], color="#333", lw=1.4, zorder=2)
ax.plot([25, 25], [32.0, 25.6], color="#333", lw=1.4, zorder=2); ax.plot([93.5, 93.5], [39.4, 25.6], color="#333", lw=1.4, zorder=2)
poly([(93.5, 25.6), (9, 25.6), (9, 53.1)], "next turn", (50, 25.6))
poly([(2.5, 57.6), (1.8, 57.6), (1.8, 14.6), (2.6, 14.6)], "no: budget spent, finish, or interrupt", (1.8, 42), rot=90)
# ================= end of slot
frame(1, 1.0, 128, 21, "A5.2  End of the slot's loop")
box(2.5, 3.6, 24, 12.8, "Stop reason", "finish / budget (turns, tool calls, $,\nwall) / interrupted (SIGTERM) / parse\nerrors; committed work stays, a pending\ncandidate is lost", GRAY)
diamond(33, 10, 9.5, 9, "anything\ncommitted?")
diamond(48.5, 10, 12, 9.4, "final re-verify\nof the committed\ncomposite ok?")
box(58, 3.6, 28, 12.8, "yes: adopt", "source -> impl store (techniques + region_loop),\nbest[slot], composite and end-to-end time\nupdated; recorded in region_done / result.json", STORE)
chips(85.8, 4.2, [("I", "W")])
box(88, 3.6, 20, 12.8, "no: revert", "the slot goes back to its start;\nthe loop's trace stays in\nregion_loop/<slot>/events.jsonl", GRAY)
box(110, 3.6, 18.5, 12.8, "Next", "more region slots: next slot\n(budget split). Else phase done:\nmaybe_reflect / upkeep, then\ngeneric rollouts (B) on the\nremaining slots", MB)
chips(128.3, 4.2, [("K", "W")])
for x0, x1 in ((26.5, 28.3), (37.8, 42.4), (54.5, 58)): poly([(x0, 10), (x1, 10)], lw=1.6)
ax.text(40, 11.8, "yes", fontsize=8, fontweight="bold", color="#38761d"); ax.text(56.2, 11.8, "yes", fontsize=8, fontweight="bold", color="#38761d")
poly([(33, 5.5), (33, 3.0), (98, 3.0), (98, 3.7)], None) if False else None
poly([(33, 5.5), (33, 2.9), (96, 2.9), (96, 3.6)], "no", (35, 4.2), lw=1.4) if False else None
poly([(48.5, 5.3), (48.5, 3.0), (98, 3.0), (98, 3.5)], "no", (50.5, 4.1), lw=1.4)
poly([(33, 5.5), (33, 2.0), (119, 2.0), (119, 3.5)], "no: nothing to adopt or revert", (60, 2.0), lw=1.4)
poly([(86, 10), (88, 10)], lw=0) if False else None
poly([(108, 10), (110, 10)], lw=1.6)
# ================= right panel
ax.add_patch(FancyBboxPatch((101, 24.6), 28.6, 38.6, boxstyle="round,pad=0.25,rounding_size=0.8", fc="#f3f3f3", ec="#444", lw=1.2, zorder=3))
ax.text(102, 62.0, "Numbers and rules", fontsize=11.5, fontweight="bold", va="top", zorder=4)
rows = [("Budgets", "turns (LLM calls), tool calls, $ (relay usage), wall; A6: 40 / 120 / $4 / 3600 s per problem, split by slot share"),
        ("Protocol", "plain chat over the relay: reasoning + one action block per turn; harness answers with the result and budgets"),
        ("Cost", "about $0.05-0.4 per turn on an Opus-class model; the pilot's sessions cost $2-7 (25-56 turns)"),
        ("Gain rule", ">= 15% in one paired timing or two consecutive timings >= 3%; commits are final-verified"),
        ("Who is skipped", "slots the loop ran on get no generic rollout steps; others do"),
        ("Knowledge", "every failed verify and every timing becomes an OutcomeRecord (technique region_loop); the committed source is stored as an implementation")]
y = 59.6
for k, v in rows:
    lines = textwrap.wrap(v, 60)
    ax.text(102, y, k, fontsize=8.6, fontweight="bold", va="top", zorder=4, color="#1f4e79")
    ax.text(102, y - 1.3, "\n".join(lines).replace("$", r"\$"), fontsize=7.4, va="top", zorder=4, linespacing=1.2); y -= 1.7 + 1.05 * len(lines) + 0.9
# ================= legend
lx, ly = 2, -3.6
ax.text(lx, ly + 1.3, "Legend:", fontsize=10.5, fontweight="bold", va="center")
for i, (c, t) in enumerate([(MB, "ModelBlaster-side step"), (LLMC, "LLM call"), (GPUC, "GPU evaluation"), (DEC, "decision / rule"), (STORE, "output / store"), (GRAY, "control / stop")]):
    xx = lx + 9 + i * 17.5; ax.add_patch(Rectangle((xx, ly + 0.6), 1.6, 1.4, fc=c, ec="#444", lw=1, zorder=3)); ax.text(xx + 2.1, ly + 1.3, t, fontsize=9, va="center")
ax.text(lx, -5.3, "Knowledge chips:", fontsize=10, fontweight="bold", va="center")
chips(lx + 28, -6.05, [("K", "R")]); ax.text(lx + 28.5, -5.3, "read", fontsize=9, va="center"); chips(lx + 42, -6.05, [("K", "W")]); ax.text(lx + 42.5, -5.3, "write", fontsize=9, va="center")
ax.text(lx + 56, -5.3, "K knowledge.db   I impl store", fontsize=9, va="center")
for ext in ("png", "svg", "pdf"): fig.savefig(OUT / f"modelblaster_region_loop.{ext}", dpi=170, bbox_inches="tight")
print("saved", OUT / "modelblaster_region_loop.png")
