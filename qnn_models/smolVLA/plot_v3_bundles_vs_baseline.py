"""Plot the bundle-aware v3 runtime traces against:
   - the scheduler's predicted timeline (multiple variants),
   - a CPU-monolithic baseline.

6-panel layout:
   1) over-optimistic prediction (conv-only HTA times — unrealizable)
   2) CPU-monolithic baseline (all on CPU, no HTA, no bundles)
   3) MEASURED: v3 + CPU-trampoline bundles
   4) MEASURED: v3 + DSP-trampoline bundles (budget-limited to 9 segments)
   5) PREDICTED: v3 + DSP-trampoline bundles (all 23 inner segs — blocked
                 by DSP context-count limit, needs lazy ctx load to realize)
   6) Side-by-side makespan bars
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_REPO = Path(__file__).parent.parent.parent

COLOR = {"HTA": "#2ca02c", "DSP": "#ff7f0e", "CPU": "#1f77b4"}


def load_trace(path: Path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "name": r["backend_label"],
                "hw": r["actual_backend"] or "?",
                "actual_start": float(r["actual_start_ms"]),
                "actual_end": float(r["actual_end_ms"]),
            })
    return rows


def load_predicted(sched_path: Path):
    with open(sched_path) as f:
        sched = json.load(f)
    hw_map = sched["metadata"]["profile_hw"]
    rows = []
    for d in sched["dispatches"].values():
        rows.append({
            "name": d["module_name"],
            "hw": hw_map[d["hardware_target"].split("#")[0]],
            "start": d["start_time"],
            "dur": d["duration"],
        })
    rows.sort(key=lambda r: r["start"])
    return rows, sched["metadata"]["makespan"]


def cpu_mono_baseline_rows(perf_path: Path):
    with open(perf_path) as f:
        perf = json.load(f)
    rows = []
    t = 0.0
    for i in range(25):
        seg = f"dsp_seg_{i:02d}"
        dur = perf[seg]["Cpu"]["mean_us"] / 1000.0
        rows.append({"name": seg, "hw": "CPU", "start": t, "dur": dur})
        t += dur
        if i < 24:
            cseg = f"cpu_seg_{i:02d}"
            cdur = perf[cseg]["Cpu"]["mean_us"] / 1000.0
            rows.append({"name": cseg, "hw": "CPU", "start": t, "dur": cdur})
            t += cdur
    return rows, t


def draw_predicted(ax, rows, *, title, y_label, max_t):
    for r in rows:
        ax.broken_barh([(r["start"], r["dur"])], (0.2, 0.6),
                        facecolors=COLOR.get(r["hw"], "#999999"),
                        edgecolor="black", linewidth=0.2)
    ax.set_xlim(0, max_t); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)


def draw_measured(ax, trace, *, title, y_label, max_t):
    for r in trace:
        ax.broken_barh([(r["actual_start"], r["actual_end"] - r["actual_start"])],
                        (0.2, 0.6),
                        facecolors=COLOR.get(r["hw"], "#999999"),
                        edgecolor="black", linewidth=0.2)
    ax.set_xlim(0, max_t); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.3)
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)


def main():
    trace_cpu      = load_trace(_REPO / "runs/v3_bundles/trace.csv")
    trace_dsp9     = load_trace(_REPO / "runs/v3_bundles_dsp9/trace.csv")
    trace_dsp14_lz = load_trace(_REPO / "runs/v3_bundles_dsp14_lazy/trace.csv")
    trace_dsp_reset = load_trace(_REPO / "runs/v3_bundles_dsp_all_reset/trace.csv")
    dsp9_pred,    dsp9_makespan    = load_predicted(
        _REPO / "schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled.json")
    dsp14_pred,   dsp14_makespan   = load_predicted(
        _REPO / "schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled_dsp14.json")
    dsp_all_pred, dsp_all_makespan = load_predicted(
        _REPO / "schedules/scheduled_networks_smolvla_vision_v3_bundles_qrb5165_greedy_profiled_dsp_all.json")
    optimistic_pred, optimistic_makespan = load_predicted(
        _REPO / "schedules/scheduled_networks_smolvla_vision_v3_qrb5165_greedy_profiled.json")
    cpu_mono, cpu_mono_total = cpu_mono_baseline_rows(
        _REPO / "qnn_models/boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json")
    actual_cpu_tramp = max(r["actual_end"] for r in trace_cpu)
    actual_dsp9      = max(r["actual_end"] for r in trace_dsp9)
    actual_dsp14_lz  = max(r["actual_end"] for r in trace_dsp14_lz)
    actual_reset    = max(r["actual_end"] for r in trace_dsp_reset)
    max_t = max(cpu_mono_total, actual_cpu_tramp, actual_dsp14_lz,
                 actual_reset, dsp_all_makespan, optimistic_makespan) * 1.02

    fig = plt.figure(figsize=(15, 16))
    gs = fig.add_gridspec(8, 1, height_ratios=[1, 1, 1, 1, 1, 1, 1, 1.8])

    ax0 = fig.add_subplot(gs[0])
    draw_predicted(ax0, optimistic_pred, max_t=max_t,
        title=f"v3 over-optimistic (conv-only HTA, no trampolines) — predicted {optimistic_makespan:.1f} ms (UNREALIZABLE)",
        y_label="Optimistic\nprediction")

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    draw_predicted(ax1, cpu_mono, max_t=max_t,
        title=f"CPU-monolithic baseline — {cpu_mono_total:.1f} ms (all 49 segments on CPU)",
        y_label="Baseline\n(CPU only)")

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    draw_measured(ax2, trace_cpu, max_t=max_t,
        title=f"v3 HTA-bundle with CPU trampolines — MEASURED {actual_cpu_tramp:.1f} ms "
              f"(all 23 inner segs HTA-bundle-CPU, B-types only)",
        y_label="Measured\n(CPU tramp)")

    ax3 = fig.add_subplot(gs[3], sharex=ax0)
    draw_measured(ax3, trace_dsp9, max_t=max_t,
        title=f"v3 HTA-bundle with DSP trampolines — MEASURED {actual_dsp9:.1f} ms "
              f"(eager, budget=9 inner segs, fits in ~27 DSP ctxs; error {(actual_dsp9 - dsp9_makespan)/dsp9_makespan*100:+.2f}%)",
        y_label="Measured\n(DSP tramp\neager bud=9)")

    ax4 = fig.add_subplot(gs[4], sharex=ax0)
    draw_measured(ax4, trace_dsp14_lz, max_t=max_t,
        title=f"v3 HTA-bundle with DSP trampolines, LAZY load + LRU evict — MEASURED {actual_dsp14_lz:.1f} ms "
              f"(budget=14 segs, 42 DSP ctxs > 30 simul. limit; reload cost slows it)",
        y_label="Measured\n(DSP tramp\nlazy bud=14)")

    ax5 = fig.add_subplot(gs[5], sharex=ax0)
    draw_measured(ax5, trace_dsp_reset, max_t=max_t,
        title=f"v3 ALL 23 segs DSP-tramp w/ backend RESET (cum threshold=28) — MEASURED {actual_reset:.1f} ms "
              f"(2 resets × ~5s each: reset dwarfs the segment-routing win)",
        y_label="Measured\n(reset)")

    ax6 = fig.add_subplot(gs[6], sharex=ax0)
    draw_predicted(ax6, dsp_all_pred, max_t=max_t,
        title=f"v3 HTA-bundle with DSP trampolines, ALL 23 inner segs — PREDICTED {dsp_all_makespan:.1f} ms "
              f"(unblocked makespan target — needs DLC sharing to fit context budget cheaply)",
        y_label="Predicted\n(unblocked)")
    ax6.set_xlabel("Time (ms)", fontsize=10)

    ax7 = fig.add_subplot(gs[7])
    labels = ["CPU-mono\nbaseline",
              "v3 + CPU\ntramps\n(measured)",
              "v3 + DSP\ntramps eager\nbudget=9\n(measured)",
              "v3 + DSP\ntramps lazy\nbudget=14\n(measured)",
              "v3 + DSP\ntramps all\nw/ reset\n(measured)",
              "v3 + DSP\ntramps all\n(predicted)"]
    vals = [cpu_mono_total, actual_cpu_tramp, actual_dsp9, actual_dsp14_lz, actual_reset, dsp_all_makespan]
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#ff7f0e", "#d62728", "#cccccc"]
    ax7 = ax7
    ax7.bar(range(len(labels)), vals, color=colors, edgecolor="black", linewidth=0.5)
    for i, v in enumerate(vals):
        ax7.text(i, v + 100, f"{v:.0f} ms", ha="center", fontsize=10)
        if i > 0:
            speedup = cpu_mono_total / v
            ax7.text(i, v / 2, f"{speedup:.2f}×", ha="center", fontsize=11,
                      color="white" if speedup >= 0.85 else "black", weight="bold")
    ax7.set_xticks(range(len(labels)))
    ax7.set_xticklabels(labels, fontsize=8)
    ax7.set_ylabel("Makespan (ms)", fontsize=10)
    ax7.set_title("Makespan summary (lower is better)", fontsize=11, loc="left")
    ax7.grid(axis="y", alpha=0.3)
    for sp in ("top","right"): ax7.spines[sp].set_visible(False)
    ax7.set_ylim(0, max(vals) * 1.18)

    legend_patches = [
        mpatches.Patch(color=COLOR["CPU"], label="CPU (ARM Kryo)"),
        mpatches.Patch(color=COLOR["DSP"], label="DSP (Hexagon v66)"),
        mpatches.Patch(color=COLOR["HTA"], label="HTA"),
    ]
    fig.legend(handles=legend_patches, loc="upper center", ncol=3, frameon=False,
                fontsize=10, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(
        "SmolVLA Vision v3: Heterogeneous Scheduling — Predicted vs Measured",
        fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = _REPO / "plots/v3_bundles_vs_baseline.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    print()
    print("=== Makespan summary ===")
    print(f'  Over-optimistic prediction (v3, conv-only HTA): {optimistic_makespan:>8.1f} ms (unrealizable)')
    print(f'  CPU-monolithic baseline (no HTA):               {cpu_mono_total:>8.1f} ms')
    print(f'  v3 + CPU trampolines (MEASURED):                {actual_cpu_tramp:>8.1f} ms  ({cpu_mono_total/actual_cpu_tramp:.2f}× speedup)')
    print(f'  v3 + DSP trampolines eager bud=9 (MEASURED):    {actual_dsp9:>8.1f} ms  ({cpu_mono_total/actual_dsp9:.2f}× speedup)')
    print(f'  v3 + DSP trampolines lazy bud=14 (MEASURED):    {actual_dsp14_lz:>8.1f} ms  ({cpu_mono_total/actual_dsp14_lz:.2f}× speedup)')
    print(f'  v3 ALL DSP-tramp w/ reset (MEASURED):           {actual_reset:>8.1f} ms  ({cpu_mono_total/actual_reset:.2f}× speedup)')
    print(f'  v3 + DSP trampolines all (PREDICTED):           {dsp_all_makespan:>8.1f} ms  ({cpu_mono_total/dsp_all_makespan:.2f}× speedup)')


if __name__ == "__main__":
    main()
