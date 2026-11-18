#!/usr/bin/env python3
"""Flowchart of the current ModelBlaster agentic flow (kb_graph_opt rounds + KernelBlaster multi-slot agent + A6 add-ons + knowledge stores). Drawn from the docs of the tree
(graph_optimizer.md, kernelblaster_multislot.md, round_controller.md, region_loop.md, region_windows.md, merge_sibling_convs.md). -> out/modelblaster_flow.{png,svg,pdf}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path
OUT = Path(__file__).resolve().parent / "out"; OUT.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(26, 17)); ax.set_xlim(0, 130); ax.set_ylim(-2.2, 86); ax.axis("off")
MB, KB, STORE, GRAY, A6C, INFRA = "#cfe2f3", "#fce5cd", "#d9ead3", "#eeeeee", "#f4cccc", "#e6e0f3"
def box(x, y, w, h, text, fc, a6=False, bold=None, fs=9.5, ec="#444444", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#c00000" if a6 else ec, lw=2.6 if a6 else 1.2, ls="--" if a6 else ls, zorder=3))
    lines = text.split("\n"); n = len(lines); ty = y + h - 0.9
    ax.text(x + 0.8, ty, lines[0], fontsize=fs + 1, fontweight="bold", va="top", ha="left", zorder=4)
    ax.text(x + 0.8, ty - 1.6, "\n".join(lines[1:]), fontsize=fs, va="top", ha="left", zorder=4, linespacing=1.35)
def arrow(p, q, text=None, rad=0.0, color="#333333", ls="-", lw=1.8, tpos=0.5, dx=0, dy=0, fs=8.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if text: ax.text(p[0] + (q[0] - p[0]) * tpos + dx, p[1] + (q[1] - p[1]) * tpos + dy, text, fontsize=fs, ha="center", va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
ax.text(65, 84.4, "ModelBlaster agentic flow: graph-level optimiser (kb_graph_opt) driving the KernelBlaster multi-slot kernel agent", fontsize=19, fontweight="bold", ha="center")
ax.text(65, 82.6, "one problem = one work dir; rounds repeat until a fixpoint or --rounds; every stage resumes after a kill (rNN/progress.json, per-problem journals)", fontsize=11.5, ha="center", color="#555")
# ---- top pipeline
box(1, 72, 20, 8, "PyTorch problem\nKernelBench L1-L3 / SOL-ExecBench bench file\nfixed shapes (workload), fp16 or native dtypes\nQUANT mode, SOL reference outputs", GRAY)
box(25, 72, 27, 8, "extract_graph  [CPU]\nconcrete trace of the model -> IR: graph.json (ops, shapes,\ndtypes) + weights.npz + io.npz; unique op names;\nvalue-specialised inputs for data-dependent ops (FlashInfer)", MB)
box(56, 72, 33, 8, "Seed the work dir\nglobal knowledge store: seed for the (arch | dtype) partition (strict)\nadaptation runs: best index-0 solution's hints + cache + impl store\nsiblings / earlier rounds: graph_hints.json, cache/, knowledge.db", STORE)
box(93, 72, 36, 8, "Farm / infrastructure\nssh dispatcher: GPU claims (B200 slots, per-GPU flock), detached workers\nLLM relay (budget, ledger, model per role: Opus main/hard/reflect,\nHaiku simple/fix), resume packs, board lock + Adreno evaluator (OpenCL)", INFRA)
arrow((21, 76), (25, 76)); arrow((52, 76), (56, 76)); arrow((72, 72), (72, 68.6), "work dir ready")
# ---- round frame
ax.add_patch(Rectangle((1, 15.5), 98, 53, fc="none", ec="#1f4e79", lw=2.4, ls="--", zorder=1))
ax.text(2, 67.5, "kb_graph_opt: ROUND r  (graph_hints.json, cache/, knowledge/ and state.json persist in the work dir)", fontsize=12.5, fontweight="bold", color="#1f4e79", va="center")
# column A
box(3, 55, 29, 9, "1. Profile  [GPU, CPU for lowering]\naccepted hints applied, cached winners deployed\nwhole-model per-op times (MB_OP_US), CUDA-graph basis\nkb_plan.profile_ir; seed / final guard", MB)
box(3, 44.5, 29, 9, "2. Proposals  [CPU, no LLM]\nfuse elementwise consumers, grow chains, split\nA6: merge sibling convs (derived weights),\nregion windows (whole / chunk / blocks / motif)", MB, a6=True)
box(3, 33, 29, 9.5, "3. Round controller  [A6, LLM-free rule]\nper round pick: kernel | graph | region-loop | stop\ngraph-only round 0 if graph gain >= 8% or small-op\nshare >= 25%; stop a move after <3% for 2 rounds", A6C, a6=True)
box(3, 22.5, 29, 8.5, "4. Plan  [CPU]\nrank by (op, signature, shape) x runtime share,\ndedup across IRs, select top-k, export problem dirs\n(slots, walker, driver, data.bin)", MB)
# column B: KernelBlaster
box(36, 53.4, 31, 10.6, "5a. Region loop  [A6; runs first in step 5]\nLLM tool loop on region (fused-block) slots, own budget:\nprofile, scratch experiments, paired timing, verify,\ncommit (>=15% or two >=3% timings) / revert;\nJSON actions over the relay; budget: turns, tools, $, wall\nthen the generic 5.1-5.8 steps run on what is left", KB, a6=True)
box(36, 18, 31, 5, "6. Ingest  [CPU + GPU verify]\nwinners (>=1.05x over seed) -> shape-guarded kernel cache;\nKB proposals (fuse, split) -> hints", MB)
# column C

def container(x, y, w, h, title, fc, subtitle=None, a6=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=fc, ec="#c00000" if a6 else "#444444", lw=2.6 if a6 else 1.2, ls="--" if a6 else "-", zorder=3))
    ax.text(x + 0.8, y + h - 0.9, title, fontsize=10.5, fontweight="bold", va="top", ha="left", zorder=4)
    if subtitle: ax.text(x + 0.8, y + h - 2.7, subtitle, fontsize=8.6, va="top", ha="left", zorder=4, color="#333")
def substeps(x, ytop, w, items, fc="white", hh=2.3, gap=0.45, fs=8.2):
    """stack of numbered sub-step boxes with small arrows; returns the list of (ybottom, ytop) of each box"""
    pos = []
    for k, (tag, txt, col) in enumerate(items):
        yb = ytop - (k + 1) * hh - k * gap
        ax.add_patch(FancyBboxPatch((x, yb), w, hh, boxstyle="round,pad=0.12,rounding_size=0.4", fc=col or fc, ec="#666666", lw=0.9, zorder=4))
        ax.text(x + 0.5, yb + hh / 2, tag, fontsize=fs, fontweight="bold", va="center", ha="left", zorder=5)
        ax.text(x + 3.3, yb + hh / 2, txt, fontsize=fs, va="center", ha="left", zorder=5)
        if k: ax.add_patch(FancyArrowPatch((x + w / 2, pos[-1][0] - 0.02), (x + w / 2, yb + hh + 0.02), arrowstyle="-|>", mutation_scale=9, color="#555", lw=1.1, zorder=5, shrinkA=0, shrinkB=0))
        pos.append((yb, yb + hh))
    return pos
LLMC, GPUC = "#ffe599", "#d9d2e9"
container(36, 24, 31, 28, "5. KernelBlaster run_multislot  [kernel move]", KB, "per problem: R rollouts x S steps; shared composite best, re-verified")
p5 = substeps(37.1, 48.0, 28.2, [("5.1", "Sample slot: runtime share x stall decay x UCB bonus", None),
    ("5.2", "State: roofline class (latency / memory / compute-bound)", None),
    ("5.3", "Technique: UCB1 per (op_class, state)", None),
    ("5.4", "Prompt: source, shapes, stats, claims, seed", None),
    ("5.5", "[LLM] rewrite slot: Haiku (simple, fix), Opus (hard)", LLMC),
    ("5.6", "[GPU] build + <= max_fix repairs; paired graph replay", GPUC),
    ("5.7", "Reward = time saved / partition; log outcome", None),
    ("5.8", "[LLM] reflect / upkeep: claims, hypotheses, states", LLMC)], hh=2.2, gap=0.4)
ax.plot([66.0, 66.0], [p5[0][0] + 1.1, p5[-1][0] + 1.1], color="#555", lw=1.1, zorder=5)
ax.plot([65.4, 66.0], [p5[-1][0] + 1.1, p5[-1][0] + 1.1], color="#555", lw=1.1, zorder=5)
ax.add_patch(FancyArrowPatch((66.0, p5[0][0] + 1.1), (65.4, p5[0][0] + 1.1), arrowstyle="-|>", mutation_scale=9, color="#555", lw=1.1, zorder=5, shrinkA=0, shrinkB=0))
ax.text(51, 25.3, "loop: next step (x S), next rollout (x R)", fontsize=8, ha="center", va="center", color="#555", zorder=5)
container(71, 43, 27, 21, "7. Graph move: decide hints  [LLM + GPU]", KB)
p7 = substeps(72.1, 61.2, 24.8, [("7.1", "Select top undecided hints, in waves", None),
    ("7.2", "Build ORIGINAL + VARIANT problems, same budget", None),
    ("7.3", "[LLM+GPU] KB run per wave (as in 5)", LLMC),
    ("7.4", "[GPU] re-time both bests back to back", GPUC),
    ("7.5", "Accept / reject by gain; infra failure = undecided", None),
    ("7.6", "Accepted: cache new slots; apply next round", None)], hh=2.45, gap=0.5)
box(71, 32.3, 27, 9.8, "8. Zero-LLM reuse  (cross-apply / seed_nearest)\npool of verified kernels per op kind tried on every\nshape slot with a per-slot correctness check; fastest\npassing source else incumbent; whole-model guard reverts", MB)
box(71, 21.3, 27, 9.8, "9. Fixpoint check\nstop when a round accepts nothing, sends nothing and has no\nmore undecided hints than it can evaluate, or --rounds\nreached (A6 controller can stop earlier)", GRAY)

# arrows (orthogonal routing)
def poly(pts, text=None, tp=None, color="#333333", ls="-", lw=1.8, rot=0, fs=8.5):
    for p, q in zip(pts[:-2], pts[1:-1]): ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls, zorder=2, shrinkA=0, shrinkB=0))
    if text: ax.text(tp[0], tp[1], text, fontsize=fs, ha="center", va="center", rotation=rot, zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95))
arrow((17.5, 55), (17.5, 53.7)); arrow((17.5, 44.5), (17.5, 42.7)); arrow((17.5, 33), (17.5, 31.4))
poly([(32, 27), (34, 27), (34, 58), (36, 58)], "kernel move", (34, 42), rot=90)
arrow((51.5, 53.4), (51.5, 52.4), "then 5.1-5.8", dx=9, dy=0.3, fs=8); arrow((51.5, 24), (51.5, 23.2))
poly([(67, 20.5), (69.2, 20.5), (69.2, 58), (71, 58)], "then: graph move (decide hints)", (69.2, 40), rot=90)
arrow((84.5, 43), (84.5, 42.4)); arrow((84.5, 32.3), (84.5, 31.4))
poly([(72, 72), (72, 65.6), (17.5, 65.6), (17.5, 64.2)], "enter round 1: profile", (45, 65.6))
poly([(84.5, 21.3), (84.5, 16.4), (1.9, 16.4), (1.9, 59.5), (3, 59.5)], "no fixpoint: next round (accepted hints applied)", (60, 16.4), color="#1f4e79", ls="--", lw=2.2)
poly([(24, 16.4), (24, 14.6)], "fixpoint or --rounds reached: finalise", (24, 15.6)) if False else ax.add_patch(FancyArrowPatch((24, 16.4), (24, 14.7), arrowstyle="-|>", mutation_scale=16, color="#333", lw=1.8, zorder=2))
ax.text(25, 15.55, "fixpoint / --rounds reached", fontsize=8.5, va="center", ha="left", zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.95))
# ---- stores column (with explicit read / write lists)
CHIP = {"R": ("#9fc5e8", "#1f4e79"), "W": ("#ea9999", "#990000"), "RW": ("#d5a6bd", "#741b47")}
def chips(xr, y, items, h=1.55):
    for L, m in reversed(items):
        w = 2.5 + 0.75 * len(m); fc, ec = CHIP[m]; x0 = xr - w
        ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=1.3, zorder=7))
        ax.text(x0 + w / 2, y + h / 2, f"{L}:{m}", fontsize=6.9, fontweight="bold", ha="center", va="center", color=ec, zorder=8); xr = x0 - 0.2
def store(y, L, title, desc, rtxt, wtxt, h=9.3):
    ax.add_patch(FancyBboxPatch((102, y), 27, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=STORE, ec="#444", lw=1.2, zorder=3))
    ax.add_patch(FancyBboxPatch((102.6, y + h - 2.5), 2.3, 1.8, boxstyle="round,pad=0.05,rounding_size=0.3", fc="white", ec="#38761d", lw=1.4, zorder=5)); ax.text(103.75, y + h - 1.6, L, fontsize=9.5, fontweight="bold", ha="center", va="center", color="#38761d", zorder=6)
    ax.text(105.6, y + h - 1.6, title, fontsize=10, fontweight="bold", va="center", ha="left", zorder=4)
    ax.text(102.8, y + h - 3.1, desc, fontsize=8, va="top", ha="left", zorder=4, color="#333")
    ax.text(102.8, y + h - 4.7, "R", fontsize=8.2, fontweight="bold", va="top", ha="left", zorder=4, color="#1f4e79"); ax.text(104.2, y + h - 4.7, rtxt, fontsize=7.9, va="top", ha="left", zorder=4, linespacing=1.25)
    ax.text(102.8, y + h - 7.0, "W", fontsize=8.2, fontweight="bold", va="top", ha="left", zorder=4, color="#990000"); ax.text(104.2, y + h - 7.0, wtxt, fontsize=7.9, va="top", ha="left", zorder=4, linespacing=1.25)
ax.text(115.5, 66.6, "Persistent knowledge: who reads (R) / writes (W)", fontsize=10.5, fontweight="bold", ha="center")
store(55.2, "K", "knowledge.db (v2)", "outcome of every applied step, UCB stats, claims, hypotheses, states", "5.2-5.3 slot state and technique stats, 5.4 claims in the prompt", "5.7 outcome + stats, 5.8 claims / hypotheses / states (also 7.3)")
store(45.2, "I", "impl store", "content-addressed kernel code + index (op, shapes, speedup, lineage)", "5.4 seed / adapt / examples, 6 export winners, 8 reuse pool", "5.7 every verified kernel (also 7.3), 8 reuse records")
store(35.2, "C", "kernel cache (shape-guarded)", "winner usable only where every shape is in its guard", "1 deploy winners, 4 skip solved items, 8 pool, 10 final verify", "6 winners (>= 1.05x), 7.6 accepted variants, 8 reuse routing")
store(25.2, "H", "graph_hints.json", "fuse / split / merge / region hints + evidence", "1 apply accepted, 2-3 candidates, 7.1 select, 9, 10", "2 proposed, 6 KB proposals, 7.5 accept / reject + evidence")
store(15.0, "G", "global knowledge store", "snapshots per (arch | dtype) partition, strict", "seed at start; refresh at round starts (other runs)", "merge of this run's delta after each round and at the end")
# chips on steps
chips(31.6, 62.0, [("C", "R"), ("H", "R")]); chips(31.6, 51.9, [("H", "RW")]); chips(31.6, 40.9, [("H", "R")]); chips(31.6, 29.4, [("C", "R")])
chips(66.2, 24.5, [("K", "RW"), ("I", "RW")]); chips(66.4, 21.6, [("I", "R"), ("C", "W"), ("H", "W")])
chips(88.4, 77.9, [("G", "R"), ("K", "W"), ("I", "W")]); chips(97.0, 32.8, [("I", "R"), ("C", "RW")]); chips(97.0, 29.5, [("H", "R")])
for (yb, yt), items in zip(p5, [[], [("K", "R")], [("K", "R")], [("K", "R"), ("I", "R")], [], [], [("K", "W"), ("I", "W")], [("K", "W")]]):
    if items: chips(65.0, yb + (yt - yb - 1.55) / 2, items)
for (yb, yt), items in zip(p7, [[("H", "R")], [], [("K", "W"), ("I", "W")], [], [("H", "W")], [("C", "W")]]):
    if items: chips(96.6, yb + (yt - yb - 1.55) / 2, items)
# ---- bottom: finalisation
box(1, 5.2, 40, 9.3, "10. Final verify + model run  [GPU]\nwhole model with every hint and cached kernel applied, in ONE runtime:\ncorrectness (golden bound; SOL tolerance gate accept / final / off),\nCUDA-graph replay time vs the hand-seeded model (kb_model_run);\nslots that break fall back to the seed", MB)
box(45, 5.2, 40, 9.3, "Torch baselines  [same GPU, same run]\neager / CUDA graph / compile (best-of incl. TF32), graph-replay basis\nspeedup vs torch eager, graph and the flow's own seed;\nOpenCL (Adreno): speedup vs the flow's seed only (no torch on the board)\nTVM-OpenCL baseline for comparison", GRAY)
box(89, 5.2, 40, 8.5, "Outputs\nsummary.json (final_us, model_e2e_us, hints, sol gate, torch),\nfinal/ problem dir, cache, knowledge; merged into the global store\n(by round and final); telemetry and relay ledger", STORE)
arrow((41, 9.8), (45, 9.8)); arrow((85, 9.8), (89, 9.8))
chips(40.2, 12.9, [("C", "R"), ("H", "R")]); chips(128.2, 12.0, [("G", "W")])
# ---- legend
lx, ly = 2, 2.0
items = [(MB, "ModelBlaster step", False), (KB, "KernelBlaster / LLM-driven step", False), (A6C, "A6 add-on (round controller)", True), (STORE, "persistent store / output", False), (GRAY, "baseline / control step", False), (INFRA, "infrastructure", False)]
ax.text(lx, ly + 1.5, "Legend:", fontsize=10.5, fontweight="bold")
for i, (c, t, a6) in enumerate(items):
    xx = lx + 9 + i * 15.5; ax.add_patch(Rectangle((xx, ly + 0.55), 1.6, 1.4, fc=c, ec="#c00000" if a6 else "#444", lw=2 if a6 else 1, ls="--" if a6 else "-", zorder=3)); ax.text(xx + 2.1, ly + 1.25, t, fontsize=9, va="center")
ax.text(1.5, -1.3, "A6 = --round-controller auto + --merge-sibling-convs auto + --region-windows auto + --region-loop on (+ --rounds 3). The plain flow (A0) runs steps 1, 2 (fusion / split only), 4-10 with every move on every round.   Targets: CUDA (L40S / B200) and OpenCL (Adreno 650, kb_graph_opt --target opencl).", fontsize=9.5, color="#333")
ax.text(lx, 0.55, "Knowledge chips:", fontsize=10, fontweight="bold", va="center")
chips(lx + 28, -0.2, [("K", "R")]); ax.text(lx + 28.5, 0.55, "read", fontsize=9, va="center"); chips(lx + 42, -0.2, [("K", "W")]); ax.text(lx + 42.5, 0.55, "write", fontsize=9, va="center")
chips(lx + 56, -0.2, [("K", "RW")]); ax.text(lx + 57, 0.55, "read + write", fontsize=9, va="center")
ax.text(lx + 74, 0.55, "K knowledge.db   I impl store   C kernel cache   H graph hints   G global store   (numbers in the store boxes = step numbers)", fontsize=9, va="center")
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"modelblaster_flow.{ext}", dpi=170, bbox_inches="tight")
print("saved", OUT / "modelblaster_flow.png")
