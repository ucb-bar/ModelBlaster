"""Side-by-side Gantt comparison of two runtime traces.

Used for the coarse-vs-fine slicing experiment: each network can be
dispatched as one full-network sub-DLC per instance (coarse) or as
multiple per-segment sub-DLCs honoring the schedule's per-op routing
(fine). The comparison plot stacks both runs sharing one x-axis so
it's immediately visible where the per-call overhead of the finer
slicing eats into the per-segment compute savings.

Usage:
    python3 plot_runtime_compare.py \\
        --coarse-log /tmp/qnn_coarse.log --coarse-label coarse \\
        --fine-log   /tmp/qnn_fine.log   --fine-label   fine \\
        --out plots/qrb5165_qnn_coarse_vs_fine.png
"""

from __future__ import annotations

import argparse
import os
import sys

# Reuse the parser from plot_runtime_trace.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from plot_runtime_trace import extract_trace_csv, _color_for


def render(coarse_rows, fine_rows, coarse_label, fine_label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    def lane_name(r):
        if r.get("actual_backend") and r["actual_backend"] != "?":
            return r["actual_backend"]
        return r.get("backend_label") or r.get("kind", "?")

    # Union of lanes across both runs so the y-axis lines up.
    lane_keys = []
    for r in coarse_rows + fine_rows:
        n = lane_name(r)
        if n not in lane_keys:
            lane_keys.append(n)
    kind_y = {k: i for i, k in enumerate(lane_keys)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 4 + 0.5 * len(lane_keys)),
                              sharex=True,
                              gridspec_kw={"hspace": 0.18})
    ax_top, ax_bot = axes

    palette: dict[tuple[str, int], str] = {}
    label_seen: set[tuple[str, int]] = set()

    def draw_pane(ax, rows, title):
        nonlocal label_seen
        max_t = 0.0
        for r in rows:
            net, inst = r["network"], r["instance"]
            color = _color_for(net, inst, palette)
            py = kind_y[lane_name(r)]
            if r["actual_end_ms"] >= 0 and r["actual_start_ms"] >= 0:
                ax.barh(y=py, width=r["actual_end_ms"] - r["actual_start_ms"],
                         left=r["actual_start_ms"], height=0.7,
                         color=color, edgecolor="black", linewidth=0.5,
                         label=f"{net}#{inst}" if (net, inst) not in label_seen else None)
                ax.text((r["actual_start_ms"] + r["actual_end_ms"]) / 2, py,
                         str(r["seg_id"]), ha="center", va="center",
                         fontsize=6, color="white")
                if r["dep_wait_done_ms"] >= 0 and r["dep_wait_done_ms"] < r["actual_start_ms"]:
                    ax.barh(y=py, width=r["actual_start_ms"] - r["dep_wait_done_ms"],
                             left=r["dep_wait_done_ms"], height=0.7,
                             color=color, alpha=0.18, edgecolor="none")
                max_t = max(max_t, r["actual_end_ms"])
            label_seen.add((net, inst))
        ax.set_yticks(list(kind_y.values()))
        ax.set_yticklabels(lane_keys)
        ax.set_ylabel("backend")
        ax.set_title(title, loc="left", fontsize=10)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        return max_t

    coarse_max = draw_pane(ax_top, coarse_rows, f"{coarse_label}  ({len(coarse_rows)} segments)")
    fine_max   = draw_pane(ax_bot, fine_rows,   f"{fine_label}  ({len(fine_rows)} segments)")
    ax_bot.set_xlabel("time (ms)")
    xmax = max(coarse_max, fine_max) * 1.02
    ax_top.set_xlim(0, xmax)
    ax_bot.set_xlim(0, xmax)

    handles = [mpatches.Patch(color=c, label=f"{n}#{i}") for (n, i), c in palette.items()]
    fig.legend(handles=handles, loc="upper center",
                ncol=min(len(handles), 8),
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)

    fig.suptitle(
        f"QNN runtime — {coarse_label}={coarse_max:.1f} ms, "
        f"{fine_label}={fine_max:.1f} ms  "
        f"({fine_max/coarse_max:.2f}× ratio)",
        y=1.06, fontsize=11,
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    print(f"wrote {out_path}  (coarse={coarse_max:.1f} ms, fine={fine_max:.1f} ms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coarse-log",   required=True)
    ap.add_argument("--coarse-label", default="coarse")
    ap.add_argument("--fine-log",     required=True)
    ap.add_argument("--fine-label",   default="fine")
    ap.add_argument("--out",          required=True)
    args = ap.parse_args()
    coarse = extract_trace_csv(args.coarse_log)
    fine   = extract_trace_csv(args.fine_log)
    print(f"coarse: {len(coarse)} segments    fine: {len(fine)} segments")
    render(coarse, fine, args.coarse_label, args.fine_label, args.out)


if __name__ == "__main__":
    main()
