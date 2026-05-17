"""Per-op-instance latency distributions on FireSim, aggregated across
all profiled networks (dronet + yolov8_nano), bucketed by operator
type, dodged by backend (scalar / rvv / gemmini_q31). Rendered as
box-and-whisker (primary) plus a strip plot for context.

Modeled after qnn_models/plot_op_distributions.py: one sample = one
operator instance from one (network, backend) profile run. Multiple
networks contribute to the same bucket, so e.g. the "Conv" bucket
mixes every conv2d_s8 dispatch from dronet + yolov8_nano.

Inputs: per-(network, backend) profile.csv files. Each row has
`dispatch_id, name, op, shape, cycles` (the MODELBLASTER_PROFILE_BEGIN/END
schema emitted by harness/src/main.c) OR
`dispatch_id, ..., op, shape, cycles` (the sweep_v8 results.csv
schema, post-processed).

Default cycle counter: k_cycle_get_64 mtime ticks at 1 MHz, so ms =
cycles / 1000.

Usage:
    python -m modelblaster.scripts.plot_firesim_op_distributions \\
        --backend scalar=dronet=plots/speedup/dronet_scalar_firesim.csv \\
        --backend scalar=yolov8_nano=plots/speedup/yolov8_scalar_firesim.csv \\
        --backend rvv=dronet=$DRONET_RVV_CSV \\
        ... \\
        --out-prefix plots/op_distributions/firesim
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

# Op-type bucket regex (simplified vs qnn_models — our op_name space is
# small and consistent: every dispatch's op field is something like
# "conv2d_s8", "relu_s8", etc.).  Order matters: first match wins.
OP_BUCKETS = [
    ("Conv",            re.compile(r"^conv2d")),
    ("DepthwiseConv",   re.compile(r"depthwise|dwconv")),
    ("FC/MatMul",       re.compile(r"^linear|^matmul|^gemm")),
    ("Norm",            re.compile(r"batchnorm|layernorm|rmsnorm")),
    ("Softmax",         re.compile(r"^softmax")),
    ("Activation",      re.compile(r"^(relu|silu|sigmoid|tanh|gelu|swish|"
                                   r"hardswish|prelu)")),
    ("Pool",            re.compile(r"pool|maxpool|avgpool")),
    ("Resize/Upsample", re.compile(r"upsample|resize|interpolate")),
    ("ElementWise",     re.compile(r"^(add|mul|sub|div|sqrt|pow|exp|"
                                   r"reciprocal)")),
    ("Concat",          re.compile(r"^cat[0-9]?|concat")),
    ("Split/Slice",     re.compile(r"^split|^slice|^chunk|^gather")),
    ("Reshape/Layout",  re.compile(r"reshape|flatten|transpose|squeeze|"
                                   r"unsqueeze|cast|permute|view")),
    ("Reduce",          re.compile(r"^reduce")),
]

BUCKET_ORDER = [
    "Conv", "DepthwiseConv", "FC/MatMul",
    "Norm", "Softmax", "Activation",
    "Pool", "ElementWise",
    "Resize/Upsample", "Reshape/Layout", "Reduce",
    "Concat", "Split/Slice", "Other",
]

# Display order + color map for the box/swarm plots. Includes the
# legacy "rvv" label and the per-config tags (V256D128_rvv = the saturn
# bitstream we run today; future configs like V512D128_rvv get added
# below). Unknown labels fall through to BACKEND_ORDER.index() returning
# 99 in the sort key, so they end up rightmost in arbitrary order.
BACKEND_ORDER = [
    "scalar",
    "rvv", "V256D128_rvv", "V512D256_rvv", "V512D128_rvv",
    "gemmini_q31", "gemmini",
]
BACKEND_COLORS = {
    "scalar":         "#3498db",   # blue
    "rvv":            "#2ecc71",   # green   (legacy alias)
    "V256D128_rvv":   "#2ecc71",   # green   (saturn V256/D128 RVV)
    "V512D256_rvv":   "#27ae60",   # darker green (saturn V512/D256 RVV — big-saturn bitstream)
    "V512D128_rvv":   "#16a085",   # teal (hypothetical V512/D128 config)
    "gemmini_q31":    "#9b59b6",   # purple
    "gemmini":        "#8e44ad",   # darker purple (legacy float-acc-scale)
}


def classify(op: str) -> str:
    op_l = op.lower()
    for label, rx in OP_BUCKETS:
        if rx.search(op_l):
            return label
    return "Other"


def collect(specs: list[str]) -> dict[tuple[str, str], list[float]]:
    """specs: list of "<backend>=<network>=<csv-path>" strings.

    Returns {(bucket, backend): [cycles, ...]} aggregated across all
    (network) inputs for that backend.
    """
    points: dict[tuple[str, str], list[float]] = defaultdict(list)
    for spec in specs:
        parts = spec.split("=", 2)
        if len(parts) != 3:
            raise SystemExit(
                f"--backend expects backend=network=csv-path, got {spec!r}")
        backend, network, path = parts
        if not os.path.exists(path):
            raise SystemExit(f"--backend {spec}: {path} not found")
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                op = row.get("op", "")
                if not op:
                    continue
                try:
                    cyc = int(row["cycles"])
                except (ValueError, KeyError):
                    continue
                if cyc <= 0:
                    continue
                bucket = classify(op)
                points[(bucket, backend)].append(float(cyc))
    return points


def setup_axes(buckets, kind: str, target_label: str = "",
               ylabel: str = "cycles (log)"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(14, 1.5 * len(buckets) + 4), 8))
    ax.set_yscale("log")
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(ylabel)
    title = (f"FireSim per-op latency distribution "
             f"by op type and backend — {kind}")
    if target_label:
        title += f"  [{target_label}]"
    title += ("\n(one sample = one op instance from one "
              "(network, backend) profile; aggregated over dronet + yolov8_nano)")
    ax.set_title(title, fontsize=11)
    for x in range(len(buckets) - 1):
        ax.axvline(x + 0.5, color="#dddddd", linewidth=0.6, zorder=0)
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.5, len(buckets) - 0.5)
    return fig, ax


def add_legend(ax, points, buckets, backends):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=BACKEND_COLORS[b], label=b)
        for b in backends
        if any((bk, b) in points for bk in buckets)
    ]
    ax.legend(handles=handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5),
              title="Backend", fontsize=10, title_fontsize=11,
              frameon=True)


def sub_centers(x_idx: int, n_be: int, bucket_width: float = 0.9):
    sub_width = bucket_width / n_be
    return ([x_idx - bucket_width / 2 + sub_width / 2 + i * sub_width
             for i in range(n_be)],
            sub_width)


def plot_box(points, buckets, backends, out_path,
             target_label: str = "", ylabel: str = "cycles (log)"):
    import matplotlib.pyplot as plt
    fig, ax = setup_axes(buckets, "box-and-whisker", target_label, ylabel)
    n_be = len(backends)
    for x_idx, bucket in enumerate(buckets):
        centers, sub_w = sub_centers(x_idx, n_be)
        for be_idx, backend in enumerate(backends):
            vals = points.get((bucket, backend), [])
            if not vals:
                continue
            color = BACKEND_COLORS[backend]
            ax.boxplot(
                vals,
                positions=[centers[be_idx]],
                widths=sub_w * 0.85,
                patch_artist=True,
                showfliers=True,
                manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.55,
                              edgecolor=color, linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color=color, linewidth=1.0),
                capprops=dict(color=color, linewidth=1.0),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markeredgecolor="none",
                                markersize=3, alpha=0.5),
            )
    add_legend(ax, points, buckets, backends)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_swarm(points, buckets, backends, out_path,
               target_label: str = "", ylabel: str = "cycles (log)"):
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    fig, ax = setup_axes(buckets, "swarm / strip", target_label, ylabel)
    n_be = len(backends)
    rng = random.Random(0xBEEF)
    jitter = 0.65
    for x_idx, bucket in enumerate(buckets):
        centers, sub_w = sub_centers(x_idx, n_be)
        for be_idx, backend in enumerate(backends):
            vals = points.get((bucket, backend), [])
            if not vals:
                continue
            sc = centers[be_idx]
            xs = [sc + rng.uniform(-1, 1) * sub_w * jitter / 2
                  for _ in vals]
            ax.scatter(xs, vals, s=14, alpha=0.5,
                       color=BACKEND_COLORS[backend], edgecolors="none")
            med = float(np.median(vals))
            ax.plot([sc - sub_w * 0.4, sc + sub_w * 0.4], [med, med],
                    color=BACKEND_COLORS[backend], linewidth=1.6, alpha=0.95)
    add_legend(ax, points, buckets, backends)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", action="append", required=True,
                    help="<backend>=<network>=<csv-path>; repeat once "
                         "per (backend, network) profile.")
    ap.add_argument("--out-prefix", required=True,
                    help="output prefix; writes "
                         "{prefix}_box.png + {prefix}_swarm.png + "
                         "{prefix}_summary.txt")
    ap.add_argument("--unit", choices=["cycles", "ms"], default="cycles",
                    help="report y-axis in raw cycles (default) or ms "
                         "assuming 1 MHz mtime ticks.")
    ap.add_argument("--target-label", default="firesim_rocket_saturn",
                    help="label appended to plot titles.")
    args = ap.parse_args()

    points = collect(args.backend)
    if not points:
        raise SystemExit("no (bucket, backend) data points collected.")

    # The "cycles" column in the sweep_v8 CSVs and the harness MODELBLASTER_PROFILE
    # output is actually nanoseconds (matches mean_time_ns; 1 unit = 1 ns
    # at the FireSim Saturn target frequency, ~60 MHz). To get ms we
    # divide by 1e6.
    if args.unit == "ms":
        scaled = defaultdict(list)
        for k, vs in points.items():
            scaled[k] = [v / 1e6 for v in vs]
        points = scaled
        ylabel = "per-op latency (ms, log)"
    else:
        ylabel = "per-op latency (ns, log)"

    backends_present = sorted({b for (_, b) in points},
                              key=lambda b: BACKEND_ORDER.index(b)
                                            if b in BACKEND_ORDER else 99)
    buckets_present = {bk for (bk, _) in points}
    buckets = [b for b in BUCKET_ORDER if b in buckets_present]

    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)

    out_box = f"{args.out_prefix}_box.png"
    out_swarm = f"{args.out_prefix}_swarm.png"
    plot_box(points, buckets, backends_present, out_box,
             args.target_label, ylabel)
    plot_swarm(points, buckets, backends_present, out_swarm,
               args.target_label, ylabel)
    print(f"wrote {out_box}")
    print(f"wrote {out_swarm}")

    # Summary table
    import numpy as np
    out_summary = f"{args.out_prefix}_summary.txt"
    lines = []
    hdr = f"{'bucket':<18s} " + " ".join(
        f"{b:>14s}" for b in backends_present) + "  n_total"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    unit = "cyc" if args.unit == "cycles" else "ms"
    for bucket in buckets:
        cells = []
        n_pts = 0
        for be in backends_present:
            vals = points.get((bucket, be), [])
            n_pts += len(vals)
            if not vals:
                cells.append(f"{'-':>14s}")
            else:
                med = float(np.median(vals))
                cells.append(f"{med:>11.0f} {unit}" if args.unit == "cycles"
                             else f"{med:>11.3f} {unit}")
        lines.append(f"  {bucket:<16s} " + " ".join(cells)
                     + f"  {n_pts:>4d}")
    summary = "\n".join(lines)
    print()
    print(summary)
    with open(out_summary, "w") as f:
        f.write(summary + "\n")
    print(f"\nwrote {out_summary}")


if __name__ == "__main__":
    main()
