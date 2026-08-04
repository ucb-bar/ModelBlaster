"""Plot full SmolVLA unrolled10 schedules: original (coarse vision) vs
new (v3 bundle vision). Highlight where each component falls and what
backends are utilized.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO = Path(__file__).parent.parent.parent

# Color by network type (group-by-job, color per category).
JOB_COLOR = {
    "state_projector_coarse":         "#7f7f7f",
    "smolvlm_text_coarse":            "#e377c2",
    "smolvlm_vision_coarse":          "#9467bd",
    "smolvlm_vision_v3_bundles":      "#9467bd",
    "smolvlm_expert_prefill_coarse":  "#ff7f0e",
    "action_in_projector_coarse":     "#17becf",
    "time_in_projector_coarse":       "#17becf",
    "time_out_projector_coarse":      "#17becf",
    "smolvlm_expert_decode_coarse":   "#d62728",
    "action_out_projector_coarse":    "#17becf",
}
HW_PATTERN = {"HTA": None, "DSP": "///", "CPU": ""}


def load_sched(path: Path):
    with open(path) as f:
        sched = json.load(f)
    hw_map = sched["metadata"]["profile_hw"]
    rows = []
    for d in sched["dispatches"].values():
        rows.append({
            "id": d["id"],
            "job": d["job_name"],
            "name": d["module_name"],
            "hw": hw_map[d["hardware_target"].split("#")[0]],
            "start": d["start_time"],
            "dur": d["duration"],
        })
    rows.sort(key=lambda r: r["start"])
    return rows, sched["metadata"]["makespan"]


def draw_strip(ax, rows, max_t, title, y_label):
    # Each row plotted as one bar; color by job; hatch by HW.
    for r in rows:
        c = JOB_COLOR.get(r["job"], "#cccccc")
        hatch = HW_PATTERN.get(r["hw"], "")
        ax.broken_barh([(r["start"], r["dur"])], (0.15, 0.7),
                        facecolors=c, edgecolor="black", linewidth=0.15,
                        hatch=hatch)
    ax.set_xlim(0, max_t)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)

    # Annotate the component spans inline so the reader can see where each
    # network falls in the timeline. We compute per-job min_start/max_end
    # and put a small label centered above each span.
    by_job = {}
    for r in rows:
        j = r["job"]
        by_job.setdefault(j, [r["start"], r["start"] + r["dur"]])
        by_job[j][0] = min(by_job[j][0], r["start"])
        by_job[j][1] = max(by_job[j][1], r["start"] + r["dur"])
    SHORT = {
        "smolvlm_vision_coarse": "vision",
        "smolvlm_vision_v3_bundles": "vision (sliced)",
        "smolvlm_expert_prefill_coarse": "prefill",
        "smolvlm_expert_decode_coarse": "decode×10",
        "smolvlm_text_coarse": "text",
        "state_projector_coarse": "state",
    }
    for j, (s, e) in by_job.items():
        span = e - s
        # Only label jobs occupying >5% of the timeline to avoid clutter
        if span < 0.04 * max_t: continue
        label = SHORT.get(j, j.split("_")[0])
        ax.text((s + e) / 2, 0.92, label, ha="center", va="top", fontsize=8.5,
                weight="bold")


def main():
    schedules = {
        "original\n(coarse v, all CPU)":
            _REPO / "schedules/scheduled_networks_smolvla_unrolled10_qrb5165_greedy_profiled.json",
        "v3-monolithic\n(over-optimistic, unrealizable)":
            _REPO / "schedules/scheduled_networks_smolvla_v3_unrolled10_qrb5165_greedy_profiled.json",
        "v3-BUNDLE budget=9\n(HONEST, ready to run)":
            _REPO / "schedules/scheduled_networks_smolvla_v3_bundles_unrolled10_qrb5165_greedy_profiled.json",
    }
    loaded = []
    for label, p in schedules.items():
        rows, ms = load_sched(p)
        loaded.append((label, rows, ms))
        print(f"  {label}: {len(rows)} dispatches, {ms:.1f} ms makespan")

    max_t = max(ms for _, _, ms in loaded) * 1.02

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.4, 1.4, 1.4, 1.6])

    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    for ax, (label, rows, ms) in zip(axes, loaded):
        draw_strip(ax, rows, max_t,
                    title=f"{label.replace(chr(10), ' — ')}: makespan {ms:.1f} ms ({len(rows)} dispatches)",
                    y_label=label)
    axes[-1].set_xlabel("Time (ms)", fontsize=11)

    # Bar chart at bottom
    ax_bar = fig.add_subplot(gs[3])
    labels = [l for l, _, _ in loaded]
    vals = [ms for _, _, ms in loaded]
    colors = ["#1f77b4", "#cccccc", "#2ca02c"]
    ax_bar.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.5)
    base = vals[0]
    for i, v in enumerate(vals):
        ax_bar.text(i, v + 100, f"{v:.0f} ms", ha="center", fontsize=10)
        if i > 0:
            speedup = base / v
            ax_bar.text(i, v / 2, f"{speedup:.2f}×", ha="center", fontsize=11,
                         color="white", weight="bold")
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels([l.replace("\n", "\n") for l in labels], fontsize=9)
    ax_bar.set_ylabel("Makespan (ms)", fontsize=10)
    ax_bar.set_title("SmolVLA unrolled10 makespan summary", fontsize=11, loc="left")
    ax_bar.grid(axis="y", alpha=0.3)
    for sp in ("top", "right"): ax_bar.spines[sp].set_visible(False)
    ax_bar.set_ylim(0, max(vals) * 1.18)

    # Legend
    legend_patches = [
        mpatches.Patch(color=JOB_COLOR["smolvlm_vision_v3_bundles"], label="vision"),
        mpatches.Patch(color=JOB_COLOR["smolvlm_expert_prefill_coarse"], label="prefill"),
        mpatches.Patch(color=JOB_COLOR["smolvlm_expert_decode_coarse"], label="decode"),
        mpatches.Patch(color=JOB_COLOR["action_in_projector_coarse"], label="projectors"),
        mpatches.Patch(color=JOB_COLOR["smolvlm_text_coarse"], label="text"),
        mpatches.Patch(color=JOB_COLOR["state_projector_coarse"], label="state"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="///", label="DSP work"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=7, frameon=False,
                fontsize=9, bbox_to_anchor=(0.5, 0.995))

    fig.suptitle("SmolVLA inference (1 prefix + 10 denoise iters) — heterogeneous scheduling progress",
                  fontsize=12, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = _REPO / "plots/unrolled10_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    main()
