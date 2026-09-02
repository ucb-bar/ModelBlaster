#!/usr/bin/env python3
"""Fusion-benefit decision aid — "what is worth fusing, before we spend a codegen+measure cycle".

For each network's MEASURED per-op K1 profile (build/k1_xpurt/<net>/int8/profile_k1.csv),
split cycles into three buckets:

  * heavy      — compute-bound matmul-class ops (conv2d, linear, lstm, matmul). Fusing an
                 epilogue INTO these does not remove their MACs, so their cycles are a floor.
  * epilogue   — memory-bound elementwise/reduction ops that CAN be folded into the heavy op's
                 epilogue (batchnorm, silu/elu/relu, add, cast, small cats, pooling).
  * fused      — ops already emitted as a genuine fused kernel (conv2d_batchnorm2d[_silu], etc.)

The `epilogue` share is the CEILING on what conv+BN+act fusion can save on wall cycles: even if
the folded op became free, you save at most its cycles, and the real fused kernel still computes
it (it only skips the intermediate memory round-trip). So a net that is 95% heavy has a ~few-%
fusion ceiling and should be fused for the DISPATCH-COUNT collapse (scheduling), not for speed.
A net with a large epilogue share is where fusion actually buys cycles.

Cross-check: the measured yolov8_nano baseline->fused run collapsed 204->147 dispatches for
+0.85% cycles (slightly worse) — this tool PREDICTS that (yolov8_nano epilogue ceiling ~2.6%).

Usage:  python3 scripts/fusion_benefit.py [--out-dir artifacts/fusion_benefit]
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

HEAVY = ("conv2d_s8", "linear_s8", "linear_f16", "lstm_f16", "matmul", "sdpa")
# ops already emitted as one genuine fused kernel (heavy + folded epilogue in a single dispatch)
ALREADY_FUSED = ("conv2d_batchnorm2d_s8", "conv2d_batchnorm2d_silu_s8",
                 "batchnorm2d_silu_s8", "linear_elu_s8", "linear_relu_s8")
# memory-bound elementwise / reduction ops that CAN fold into a heavy op's epilogue
EPILOGUE = ("batchnorm2d", "silu", "elu", "relu", "add_", "cast_", "cat", "upsample",
            "maxpool", "avgpool")


def classify(op: str) -> str:
    if op in ALREADY_FUSED:
        return "fused"
    if op in HEAVY:
        return "heavy"
    if any(k in op for k in EPILOGUE):
        return "epilogue"
    return "other"


def analyze(csv_path: str):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows or "cycles" not in rows[0]:
        return None
    buckets = collections.Counter()
    byop = collections.Counter()
    for r in rows:
        cy = int(float(r["cycles"]))
        buckets[classify(r["op"])] += cy
        byop[r["op"]] += cy
    tot = sum(buckets.values()) or 1
    return dict(
        n_dispatch=len(rows),
        total=tot,
        heavy=buckets["heavy"], fused=buckets["fused"],
        epilogue=buckets["epilogue"], other=buckets["other"],
        epilogue_ceiling_pct=100.0 * buckets["epilogue"] / tot,
        already_fused_pct=100.0 * buckets["fused"] / tot,
        byop=byop,
    )


def verdict(a) -> str:
    ceil = a["epilogue_ceiling_pct"]
    if a["already_fused_pct"] > 50:
        return "ALREADY FUSED (deployed) — no headroom left"
    if ceil < 5:
        return f"COMPUTE-BOUND — fuse for dispatch-collapse only ({ceil:.1f}% cycle ceiling)"
    if ceil < 20:
        return f"MODEST — some epilogue headroom ({ceil:.1f}%)"
    return f"MEMORY-BOUND — real fusion win available ({ceil:.1f}%)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="build/k1_xpurt/*/int8/profile_k1.csv")
    ap.add_argument("--out-dir", default="artifacts/fusion_benefit")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for p in sorted(glob.glob(args.profiles)):
        net = p.split("/")[-3]
        a = analyze(p)
        if a:
            results[net] = a

    # table
    hdr = f"{'network':22s} {'disp':>5s} {'Mcyc':>8s} {'heavy%':>7s} {'fused%':>7s} {'epi-ceil%':>9s}  verdict"
    lines = [hdr, "-" * len(hdr)]
    for net, a in sorted(results.items(), key=lambda kv: -kv[1]["epilogue_ceiling_pct"]):
        lines.append(f"{net:22s} {a['n_dispatch']:5d} {a['total']/1e6:8.2f} "
                     f"{100*a['heavy']/a['total']:7.1f} {a['already_fused_pct']:7.1f} "
                     f"{a['epilogue_ceiling_pct']:9.1f}  {verdict(a)}")
    table = "\n".join(lines)
    print(table)
    open(os.path.join(args.out_dir, "fusion_benefit.txt"), "w").write(table + "\n")
    with open(os.path.join(args.out_dir, "fusion_benefit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["network", "n_dispatch", "total_cycles", "heavy_pct", "already_fused_pct",
                    "epilogue_ceiling_pct", "verdict"])
        for net, a in results.items():
            w.writerow([net, a["n_dispatch"], a["total"], round(100*a["heavy"]/a["total"], 1),
                        round(a["already_fused_pct"], 1), round(a["epilogue_ceiling_pct"], 1),
                        verdict(a)])

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        nets = sorted(results, key=lambda n: -results[n]["epilogue_ceiling_pct"])
        heavy = [100*results[n]["heavy"]/results[n]["total"] for n in nets]
        fused = [results[n]["already_fused_pct"] for n in nets]
        epi = [results[n]["epilogue_ceiling_pct"] for n in nets]
        other = [100 - h - f - e for h, f, e in zip(heavy, fused, epi)]
        y = np.arange(len(nets))
        fig, ax = plt.subplots(figsize=(9.2, 0.7*len(nets)+1.6))
        l = np.zeros(len(nets))
        for vals, col, lab in [(heavy, "#4477aa", "heavy / compute-bound (MAC floor)"),
                               (fused, "#228833", "already fused (deployed kernel)"),
                               (epi, "# ee6677".replace(" ", ""), "fusible epilogue = fusion CEILING"),
                               (other, "#bbbbbb", "other")]:
            ax.barh(y, vals, left=l, color=col, label=lab, height=0.62)
            l = l + np.array(vals)
        for i, n in enumerate(nets):
            ax.text(101, i, verdict(results[n]).split(" —")[0], va="center", fontsize=8, color="#333")
        ax.set_yticks(y); ax.set_yticklabels(nets, fontsize=9)
        ax.set_xlim(0, 100); ax.set_xlabel("share of measured K1 cycles (%)")
        ax.invert_yaxis()
        ax.set_title("Fusion-benefit decision aid — the epilogue slice is the cycle ceiling for fusion",
                     fontsize=10, weight="bold")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fusion_benefit.png"), dpi=140)
        fig.savefig(os.path.join(args.out_dir, "fusion_benefit.pdf"))
        print("\nwrote", os.path.join(args.out_dir, "fusion_benefit.{png,pdf,txt,csv}"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
