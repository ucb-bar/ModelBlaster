#!/usr/bin/env python3
"""ModelBlaster / KernelBlaster flow plots, in the style of KernelBlaster's scripts/plot_speedup_opencl_adreno_v2.py (per-problem vertical bars sorted by speedup descending,
log-scale Y, steelblue = faster / lightsteelblue = slower than the baseline, red = no verified result, dotted geomean/median lines with boxed labels, a vertical line where the bars drop
below 1x annotated with the share of problems that are faster, 14x8 figure, bold axis labels, grid alpha 0.3, dpi 300) plus fast_p curves (fraction of problems whose verified result is
more than p times faster than the baseline, log-x) drawn with the same conventions.

  plot_flow.py bars DATASET [--baseline eager|graph|seed] [--title T] [--xlabel X] [--out PNG]
  plot_flow.py fastp DATASET [DATASET ...] [--labels L ...] [--baseline eager|graph] [--title T] [--out PNG]
  plot_flow.py all                       the standard set into out/ (reads data/flow_results.csv from collect_flow_results.py)
DATASET = a dataset name of the CSV, or A+B (union of rows, e.g. sol_small_a6+sol_large_a6). Baseline = PyTorch eager / PyTorch CUDA graph measured on the same device in the same run
(L40S L3: E1's torch_bench); 'seed' = the flow's own unoptimised seed (the only baseline the OpenCL runs have). Speedup = baseline_us / flow final whole-model us.
Rows with status failed (no verified result) count as not faster in fast_p and are red bars; 'unsupported' (extract refused) rows are not part of N and are reported in the title."""
from __future__ import annotations
import argparse, csv, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
H = Path(__file__).resolve().parent
CSV = H / "data/flow_results.csv"
def load():
    rows = list(csv.DictReader(open(CSV)))
    return rows
def sel(rows, ds):
    names = ds.split("+"); return [r for r in rows if r["dataset"] in names]
def fnum(x):
    try: return float(x) if x not in ("", None) else None
    except ValueError: return None
def speedups(rows, baseline):
    """-> (list of (problem, speedup or None when failed), n_unsupported, n_no_baseline)"""
    out, unsup, nob = [], 0, 0
    for r in rows:
        if r["status"] == "unsupported": unsup += 1; continue
        if r["status"] == "failed": out.append((r["problem"], None)); continue
        m = fnum(r["model_us"])
        if baseline == "seed":
            note = re.search(r"speedup_vs_seed=([0-9.eE+-]+)", r["note"] or "")
            b = None; sp = float(note.group(1)) if note else None
            if sp is None:
                s = fnum(r["seed_us"]); sp = s / m if (s and m) else None
            if sp is None: nob += 1; continue
            out.append((r["problem"], sp)); continue
        b = fnum(r["torch_eager_us" if baseline == "eager" else "torch_graph_us"])
        if not m or not b: nob += 1; continue
        out.append((r["problem"], b / m))
    return out, unsup, nob
BASE_LABEL = {"eager": "PyTorch eager", "graph": "PyTorch CUDA graph", "seed": "flow seed (initial kernels)"}
def gm(v): return float(np.exp(np.mean(np.log(v)))) if v else float("nan")
def bars(rows, ds, baseline, title, xlabel, out):
    data, unsup, nob = speedups(sel(rows, ds), baseline)
    if not data: print("no data for", ds, baseline); return
    valid = sorted([(p, s) for p, s in data if s is not None], key=lambda t: -t[1]); failed = [(p, None) for p, s in data if s is None]
    ordered = valid + failed; heights = [s if s is not None else 1.0 for _, s in ordered]
    colors = ["steelblue" if (s or 0) >= 1 else "lightsteelblue" if s is not None else "red" for _, s in ordered]
    vs = [s for _, s in valid]; geomean, median = gm(vs), float(np.median(vs)) if vs else float("nan"); n = len(ordered)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(np.arange(n), heights, color=colors, alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.set_yscale("log"); ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    for val, lab in ((geomean, "geomean"), (median, "median")):
        ax.axhline(y=val, color="#1f77b4", linestyle=":", linewidth=1.5, alpha=0.85)
        ax.text(n - 0.5, val, f" {lab} {val:.2f}×", va="center", ha="left", fontsize=9, fontweight="bold", color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1f77b4", alpha=0.9))
    faster = sum(1 for s in vs if s >= 1.0); pct = faster / len(ordered) * 100
    transition = next((i for i, (_, s) in enumerate(ordered) if s is not None and s < 1.0), None)
    if transition:
        ax.axvline(x=transition - 0.5, color="darkblue", linestyle=":", alpha=0.8, linewidth=2)
        ax.text(transition - 0.5, max(heights) ** 0.5 if max(heights) > 4 else 3.0, f"{pct:.1f}% of problems are faster", ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7), rotation=90)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Speedup ({BASE_LABEL[baseline]} / flow final) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_title(title + f"  |  N={len(vs)}, geomean={geomean:.2f}×, median={median:.2f}×" + (f", {len(failed)} no verified result" if failed else "") + (f", {unsup} unsupported (not in N)" if unsup else ""),
                 fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_xlim(-0.6, n - 0.4); ax.grid(axis="y", alpha=0.3)
    leg = [Patch(facecolor="steelblue", alpha=0.8, label="Flow faster (>1×)"), Patch(facecolor="lightsteelblue", alpha=0.8, label="Flow slower (<1×)")]
    if failed: leg.append(Patch(facecolor="red", alpha=0.8, label="No verified result"))
    leg += [plt.Line2D([0], [0], color="gray", linestyle="--", alpha=0.5, label="No Speedup (1×)"),
            plt.Line2D([0], [0], color="#1f77b4", linestyle=":", linewidth=1.5, label=f"Geomean ({geomean:.2f}×)"),
            plt.Line2D([0], [0], color="#1f77b4", linestyle=":", linewidth=1.5, label=f"Median ({median:.2f}×)")]
    ax.legend(handles=leg, loc="upper right"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
    print(f"saved {out}  N={len(vs)} geomean={geomean:.2f}x median={median:.2f}x faster={pct:.1f}% failed={len(failed)} unsupported={unsup} no-baseline={nob}")
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
def fastp(rows, dss, labels, baseline, title, out):
    fig, ax = plt.subplots(figsize=(12, 8)); grid = np.logspace(np.log10(0.1), np.log10(300), 800); pmax = 0
    for i, ds in enumerate(dss):
        data, unsup, nob = speedups(sel(rows, ds), baseline)
        if not data: print("no data for", ds); continue
        s = np.array([x if x is not None else 0.0 for _, x in data]); n = len(s)
        y = np.array([(s > p).mean() for p in grid]); f1 = float((s > 1.0).mean()); g = gm([x for _, x in data if x])
        lab = labels[i] if labels else ds
        ax.step(grid, y, where="post", color=PALETTE[i % len(PALETTE)], linewidth=2.2, alpha=0.9, label=f"{lab}  (N={n}, fast₁={100*f1:.0f}%, geomean {g:.2f}×)")
        pmax = max(pmax, s.max())
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5); ax.set_xscale("log"); ax.set_xlim(0.1, min(300, max(10, pmax * 1.5))); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"p  (speedup over {BASE_LABEL[baseline]}) [Log Scale]", fontsize=12, fontweight="bold")
    ax.set_ylabel("fast_p  (fraction of problems verified and faster than p×)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True); fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(); print("saved", out)
def standard(rows):
    O = H / "out"; A6 = "sol_a6_pilot36+sol_small_a6+sol_large_a6+sol_overcap_a6"
    for base in ("eager", "graph"):
        T = f"B200, flow vs {BASE_LABEL[base]}"
        fastp(rows, ["sol_a6_pilot36", "sol_small_a6", "sol_large_a6", "sol_overcap_a6", A6], ["SOL pilot (36, A6)", "SOL small (A6)", "SOL large (A6)", "SOL over-cap (A6)", "SOL all A6 (idx0)"], base,
              f"fast_p vs {BASE_LABEL[base]} — SOL-ExecBench on B200 (A6 flow)", O / f"fastp_sol_a6_vs_{base}.png")
        fastp(rows, ["sol_small_base", "sol_small_a6", "sol_large_base", "sol_large_a6"], ["small baseline", "small A6", "large baseline", "large A6"], base,
              f"fast_p vs {BASE_LABEL[base]} — A6 vs baseline flow on SOL small / large problems (B200)", O / f"fastp_sol_a6_vs_baseline_{base}.png")
        fastp(rows, ["l3_l40s_sweep", "l3_l40s_A0", "l3_l40s_A6"], ["L3 sweep (46, L40S)", "L3 ablation A0 (8)", "L3 ablation A6 (8)"], base,
              f"fast_p vs {BASE_LABEL[base]} — KernelBench Level 3 on L40S (fp16)", O / f"fastp_l3_l40s_vs_{base}.png")
        fastp(rows, [A6, "sol_adapt"], ["SOL idx0 solutions (full flow)", "SOL other workloads (haiku adaptation)"], base,
              f"fast_p vs {BASE_LABEL[base]} — SOL variant adaptation vs the index-0 solutions (B200)", O / f"fastp_sol_adapt_vs_{base}.png")
        for ds, ttl, xl in (("sol_a6_pilot36", "SOL-ExecBench pilot, B200, A6 flow", "SOL-ExecBench problem (workload 0)"), ("sol_small_a6", "SOL small problems, B200, A6 flow", "SOL-ExecBench problem (workload 0)"),
                            ("sol_large_a6", "SOL large problems, B200, A6 flow", "SOL-ExecBench problem (workload 0)"), ("sol_overcap_a6", "SOL over-cap problems (smallest workload), B200, A6 flow", "SOL-ExecBench problem"),
                            ("sol_adapt", "SOL other workloads, B200, haiku adaptation", "SOL-ExecBench workload"), ("l3_l40s_sweep", "KernelBench L3, L40S, flow sweep", "Kernel Bench L3 Problem")):
            bars(rows, ds, base, ttl, xl, O / f"bars_{ds}_vs_{base}.png")
    bars(rows, "qrb_l2", "seed", "QRB5165 OpenCL/Adreno, KernelBench L2 fp16, A6 flow (Sonnet)", "Kernel Bench L2 Problem", O / "bars_qrb_l2_vs_seed.png")
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bars"); b.add_argument("dataset"); b.add_argument("--baseline", default="eager", choices=["eager", "graph", "seed"]); b.add_argument("--title", default=None)
    b.add_argument("--xlabel", default="Problem"); b.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("fastp"); f.add_argument("datasets", nargs="+"); f.add_argument("--labels", nargs="*"); f.add_argument("--baseline", default="eager", choices=["eager", "graph"])
    f.add_argument("--title", default=None); f.add_argument("--out", type=Path, default=None)
    sub.add_parser("all"); a = ap.parse_args(); rows = load()
    if a.cmd == "all": standard(rows)
    elif a.cmd == "bars": bars(rows, a.dataset, a.baseline, a.title or a.dataset, a.xlabel, a.out or H / f"out/bars_{a.dataset}_vs_{a.baseline}.png")
    else: fastp(rows, a.datasets, a.labels, a.baseline, a.title or f"fast_p vs {BASE_LABEL[a.baseline]}", a.out or H / f"out/fastp_{'_'.join(a.datasets)[:60]}_vs_{a.baseline}.png")
if __name__ == "__main__": main()
