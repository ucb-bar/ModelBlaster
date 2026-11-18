#!/usr/bin/env python3
"""Best-so-far curves vs cumulative LLM calls. For each arm: within each round, an ESTIMATED model time
= round-start profiled model time - sum over that round's slot problems of (slot baseline e2e - best slot e2e so far),
updated at each step's completion (journal timestamp of its last LLM call); cumulative LLM calls = ledger rows up to
that time (all roles), or for the sweep (no ledger) the journal calls. Measured points: round starts + final.
Writes curves.json and curves.png."""
import json, sys
from bisect import bisect_right
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import tsf

BT = Path(__file__).resolve().parent
ARMS = [("S", "sweep ref (x = codegen steps, no ledger)"), ("C", "C control"), ("B", "B beam 2x2"), ("H", "H hetero Haiku/Opus")]
ARMS += [(t, f"{t} control seed") for t in sys.argv[1:]]
out = {}
for tag, label in ARMS:
    d = BT / "res" / ("sweep" if tag == "S" else tag)
    A = json.loads((d / "analysis.json").read_text())
    led = []
    if (d / "ledger.jsonl").exists():
        mk = (d / "markers.txt").read_text().split()
        t0, t1 = float(mk[1]), float(mk[5])
        led = sorted(tsf(json.loads(l)["ts"]) for l in (d / "ledger.jsonl").read_text().splitlines() if l.strip()
                     and t0 - 5 <= tsf(json.loads(l)["ts"]) <= t1 + 5)
    jts = sorted(s["ts"] for s in A["steps"])
    def calls_at(ts):
        return bisect_right(led, ts) if led else bisect_right(jts, ts)
    pts = []
    for r in A["rounds"]:
        rnd = f"r{r['round']:02d}"
        start = r["start_partition_us"]
        steps = sorted((s for s in A["steps"] if s["round"] == rnd and "fused" not in s["problem"]), key=lambda s: s["ts"])
        best = {}
        base = {}
        allr = sorted(s["ts"] for s in A["steps"] if s["round"] == rnd)
        if allr:
            pts.append((calls_at(allr[0] - 1), start, "round start"))
        for s in steps:
            p = s["problem"]
            base.setdefault(p, s["base_e2e"])
            v = s["e2e_after"] if s["status"] in ("improved", "neutral") and s["e2e_after"] else None
            if v is not None:
                best[p] = min(best.get(p, base[p]), v)
            est = start - sum(base[q] - best[q] for q in best)
            pts.append((calls_at(s["ts"]), est, "est"))
    total = len(led) if led else len(jts)
    pts.append((total, A["final"]["model_e2e_us"], "final"))
    out[tag] = {"label": label, "points": pts, "rounds": A["rounds"], "final": A["final"]["model_e2e_us"], "calls": total}
(BT / "curves.json").write_text(json.dumps(out, indent=1))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = {"S": "#9aa0a6", "C": "#1f77b4", "B": "#d62728", "H": "#2ca02c"}
    for tag, v in out.items():
        xs = [p[0] for p in v["points"]]
        ys = [p[1] for p in v["points"]]
        ax.step(xs, ys, where="post", label=f"{v['label']} -> {v['final']:.0f} us", color=cols.get(tag), lw=1.4,
                ls="--" if tag not in cols else "-")
        m = [p for p in v["points"] if p[2] != "est"]
        ax.scatter([p[0] for p in m], [p[1] for p in m], color=cols.get(tag), s=14, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(6000, 30000)
    ax.set_xlabel("cumulative LLM calls (all roles; relay ledger)")
    ax.set_ylabel("model e2e us (est. within rounds; dots measured)")
    ax.set_title("43_MinGPTCausalAttention: best-so-far vs LLM calls")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(BT / "curves.png", dpi=130)
    print("wrote curves.png")
except ImportError:
    print("no matplotlib")
for tag, v in out.items():
    pts = v["points"]
    row = []
    for c in (50, 100, 150, 200, 250, 300, 350):
        prev = [p[1] for p in pts if p[0] <= c]
        row.append(f"{prev[-1]:.0f}" if prev else "-")
    print(tag, v["label"], "| current est. at 50..350 calls:", " ".join(row), "| final", v["final"], "calls", v["calls"])
