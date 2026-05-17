"""Per-op + aggregate speedup plotter.

Reads multiple results.csv files (each one per-dispatch cycles for a
(network, target, backend) combination), aligns by dispatch_id, computes
speedup = baseline_cycles / variant_cycles, and renders:

  * one per-op bar chart per (network, target) — sorted by descending
    baseline cost so the layers that matter visually dominate the chart;
  * one aggregate bar chart per (network, target) — geometric mean of
    per-op speedups + arithmetic mean weighted by baseline cycles
    (= speedup of the whole network's serial execution).

Usage:
    python -m modelblaster.scripts.plot_speedup \\
        --baseline scalar=path/to/scalar.csv \\
        --variant  rvv=path/to/rvv.csv \\
        --variant  gemmini_q31=path/to/gemmini_q31.csv \\
        --net dronet \\
        --target spike \\
        --out-prefix plots/speedup/dronet_spike
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


def load_per_op(path: str) -> Dict[int, Tuple[str, str, int]]:
    """Map dispatch_id -> (op, shape, cycles) from a results.csv."""
    out: Dict[int, Tuple[str, str, int]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                did = int(r["dispatch_id"])
            except (ValueError, KeyError):
                continue
            try:
                cyc = int(r["cycles"])
            except (ValueError, KeyError):
                # Some old rows have missing cycles; skip
                continue
            if cyc <= 0:
                continue
            op = r.get("op", "")
            shape = r.get("shape", "")
            # Some dispatches appear multiple times across topo subdirs;
            # keep the first occurrence (deterministic).
            if did not in out:
                out[did] = (op, shape, cyc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="LABEL=path/to/results.csv — the reference "
                         "(e.g. scalar=...)")
    ap.add_argument("--variant", action="append", default=[],
                    help="LABEL=path/to/results.csv — repeat once per "
                         "backend (rvv, gemmini_q31, ...). Speedup is "
                         "baseline_cycles / variant_cycles per dispatch.")
    ap.add_argument("--net", required=True, help="network name (used in titles).")
    ap.add_argument("--target", default="firesim",
                    help="profile target label for titles.")
    ap.add_argument("--out-prefix", required=True,
                    help="output prefix; the tool writes "
                         "{prefix}_perop.png + {prefix}_aggregate.png "
                         "+ {prefix}.csv")
    args = ap.parse_args()

    if "=" not in args.baseline:
        raise SystemExit("--baseline expects LABEL=path")
    base_label, base_path = args.baseline.split("=", 1)
    base = load_per_op(base_path)
    if not base:
        raise SystemExit(f"no per-op rows in baseline {base_path}")

    variants: "OrderedDict[str, Dict[int, Tuple[str, str, int]]]" = OrderedDict()
    for spec in args.variant:
        if "=" not in spec:
            raise SystemExit(f"--variant expects LABEL=path, got {spec!r}")
        lbl, p = spec.split("=", 1)
        v = load_per_op(p)
        if not v:
            raise SystemExit(f"no per-op rows in variant {p}")
        variants[lbl] = v

    if not variants:
        raise SystemExit("at least one --variant required")

    # Build the dispatch order from baseline: dispatch_id -> (op, shape, base_cyc)
    dispatch_ids = sorted(base.keys())

    # Per-op rows + speedups
    rows: List[Dict[str, object]] = []
    speedups_by_variant: Dict[str, List[float]] = {lbl: [] for lbl in variants}
    base_cyc_for_aggregate: Dict[str, List[int]] = {lbl: [] for lbl in variants}
    op_label_for_aggregate: Dict[str, List[str]] = {lbl: [] for lbl in variants}
    for did in dispatch_ids:
        op, shape, b_cyc = base[did]
        row = {
            "dispatch_id": did,
            "op": op,
            "shape": shape,
            f"{base_label}_cycles": b_cyc,
        }
        for lbl, v in variants.items():
            if did not in v:
                row[f"{lbl}_cycles"] = ""
                row[f"{lbl}_speedup"] = ""
                continue
            v_cyc = v[did][2]
            sp = b_cyc / v_cyc if v_cyc > 0 else float("nan")
            row[f"{lbl}_cycles"] = v_cyc
            row[f"{lbl}_speedup"] = f"{sp:.3f}"
            speedups_by_variant[lbl].append(sp)
            base_cyc_for_aggregate[lbl].append(b_cyc)
            op_label_for_aggregate[lbl].append(op)
        rows.append(row)

    out_csv = f"{args.out_prefix}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    cols = ["dispatch_id", "op", "shape", f"{base_label}_cycles"]
    for lbl in variants:
        cols += [f"{lbl}_cycles", f"{lbl}_speedup"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    # ---------- plots ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise SystemExit("matplotlib + numpy required for plot rendering.")

    palette = plt.get_cmap("tab10")
    color_for = {lbl: palette(i % 10) for i, lbl in enumerate(variants)}

    # ---------- per-op chart ----------
    # Sort by descending baseline cycles so the layers that matter most
    # are visually dominant. Show top-N in detail; cap labels for layout.
    n_ops = len(dispatch_ids)
    sorted_ids = sorted(dispatch_ids, key=lambda d: -base[d][2])
    fig_w = max(12, 0.18 * n_ops)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    bar_w = 0.8 / max(len(variants), 1)
    x = np.arange(n_ops)
    for i, lbl in enumerate(variants):
        v = variants[lbl]
        ys = []
        for did in sorted_ids:
            if did in v and v[did][2] > 0:
                ys.append(base[did][2] / v[did][2])
            else:
                ys.append(float("nan"))
        ax.bar(x + (i - (len(variants) - 1) / 2) * bar_w, ys,
               width=bar_w, color=color_for[lbl], edgecolor="black",
               linewidth=0.3, label=lbl)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7,
               label=f"{base_label} (1.0×)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    # Compose tick labels: dispatch_id + op (truncated)
    def _short_op(op: str) -> str:
        if len(op) <= 14:
            return op
        return op[:12] + ".."
    ax.set_xticklabels(
        [f"{did}\n{_short_op(base[did][0])}" for did in sorted_ids],
        rotation=70, fontsize=7, ha="right")
    ax.set_ylabel(f"speedup over {base_label} (log)")
    ax.set_title(
        f"{args.net} on {args.target}: per-dispatch speedup over {base_label} "
        f"(sorted by descending {base_label} cost)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_perop = f"{args.out_prefix}_perop.png"
    fig.savefig(out_perop, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_perop}")

    # ---------- aggregate chart ----------
    # Two summary stats per variant:
    #   geomean = geometric mean of per-op speedups (gives equal weight
    #             to every op regardless of cost) — closer to "average
    #             kernel speedup".
    #   weighted = total_baseline_cycles / total_variant_cycles (= speedup
    #             of the whole network's serial execution if the schedule
    #             is unchanged).
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(variants) + 2), 4.5))
    geomeans, weighteds, labels = [], [], []
    for lbl in variants:
        sp = speedups_by_variant[lbl]
        if not sp:
            geomeans.append(float("nan"))
            weighteds.append(float("nan"))
            labels.append(lbl)
            continue
        ln = sum(math.log(s) for s in sp) / len(sp)
        geomeans.append(math.exp(ln))
        v = variants[lbl]
        bcyc_sum = sum(base[did][2] for did in dispatch_ids if did in v)
        vcyc_sum = sum(v[did][2] for did in dispatch_ids if did in v)
        weighteds.append(bcyc_sum / vcyc_sum if vcyc_sum > 0 else float("nan"))
        labels.append(lbl)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, geomeans, width=0.36, color="#4472c4",
           edgecolor="black", linewidth=0.3, label="geomean (per-op)")
    ax.bar(x + 0.18, weighteds, width=0.36, color="#ed7d31",
           edgecolor="black", linewidth=0.3, label="weighted (whole-net)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7,
               label=f"{base_label} (1.0×)")
    for i, (g, w) in enumerate(zip(geomeans, weighteds)):
        if not math.isnan(g):
            ax.text(i - 0.18, g, f"{g:.2f}×", ha="center", va="bottom",
                    fontsize=9)
        if not math.isnan(w):
            ax.text(i + 0.18, w, f"{w:.2f}×", ha="center", va="bottom",
                    fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"speedup over {base_label}")
    ax.set_title(
        f"{args.net} on {args.target}: aggregate speedup over {base_label}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_agg = f"{args.out_prefix}_aggregate.png"
    fig.savefig(out_agg, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_agg}")

    # ---------- print tabular summary ----------
    print("--- summary ---")
    for lbl in variants:
        sp = speedups_by_variant[lbl]
        if not sp:
            print(f"  {lbl}: no overlapping dispatches with baseline")
            continue
        ln = sum(math.log(s) for s in sp) / len(sp)
        gm = math.exp(ln)
        v = variants[lbl]
        bsum = sum(base[did][2] for did in dispatch_ids if did in v)
        vsum = sum(v[did][2] for did in dispatch_ids if did in v)
        wt = bsum / vsum if vsum > 0 else float("nan")
        print(f"  {lbl}: geomean={gm:.3f}× weighted={wt:.3f}× "
              f"(over {len(sp)}/{len(dispatch_ids)} dispatches; "
              f"total baseline {bsum} → variant {vsum})")


if __name__ == "__main__":
    main()
