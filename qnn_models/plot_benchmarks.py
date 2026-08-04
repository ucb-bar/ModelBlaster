#!/usr/bin/env python3
"""Plot QNN benchmark results across models and backends.

Usage:
    python plot_benchmarks.py [--results PATH] [--output PATH]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MODEL_META = {
    "dronet":                  {"label": "DroNet",              "params": "300K params\n13M MACs"},
    "mobilenet_v2":            {"label": "MobileNetV2",         "params": "3.5M params\n300M MACs"},
    "yolov8s":                 {"label": "YOLOv8s",             "params": "11.2M params\n14.4G MACs"},
    "action_in_projector":     {"label": "ActionIn\nProjector"},
    "action_out_projector":    {"label": "ActionOut\nProjector"},
    "state_projector":         {"label": "State\nProjector"},
    "time_in_projector":       {"label": "TimeIn\nProjector"},
    "time_out_projector":      {"label": "TimeOut\nProjector"},
    "smolvlm_text":            {"label": "SmolVLM\nText"},
    "smolvlm_vision":          {"label": "SmolVLM\nVision"},
    "smolvlm_expert_decode":   {"label": "SmolVLM\nExpert Dec"},
    "smolvlm_expert_prefill":  {"label": "SmolVLM\nExpert Pre"},
}

BACKENDS = ["CPU", "GPU", "DSP"]
COLORS = {"CPU": "#3498db", "GPU": "#e74c3c", "DSP": "#2ecc71"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=None,
                        help="Path to benchmark_results.json")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, "plots", "qnn_benchmark.png")
    results_path = args.results or os.path.join(script_dir, "benchmark_results.json")

    with open(results_path) as f:
        raw = json.load(f)

    # Build ordered lists: known models first (MODEL_META order), then any extras
    models = [k for k in MODEL_META if k in raw]
    models += [k for k in sorted(raw) if k not in MODEL_META]
    labels = [MODEL_META.get(m, {}).get("label", m) for m in models]
    param_info = [MODEL_META.get(m, {}).get("params", "") for m in models]

    latency = {}
    for m in models:
        latency[m] = {}
        for b in BACKENDS:
            val = raw[m].get(b)
            latency[m][b] = float(val) if val is not None else None

    fig_width = max(14, len(models) * 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 6))

    x = np.arange(len(models))
    width = 0.25

    # --- Left: Latency ---
    ax = axes[0]
    for i, backend in enumerate(BACKENDS):
        vals = [latency[m][backend] or 0 for m in models]
        bars = ax.bar(x + i * width, vals, width, label=backend,
                      color=COLORS[backend], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val <= 0:
                continue
            offset = 8 if val < 50 else 15
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{val:.1f}" if val < 50 else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_title("Inference Latency by Backend", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{l}\n{p}" if p else l for l, p in zip(labels, param_info)],
                       fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    all_vals = [v for m in models for v in latency[m].values() if v]
    ax.set_ylim(min(all_vals) * 0.5, max(all_vals) * 2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Right: FPS ---
    ax = axes[1]
    for i, backend in enumerate(BACKENDS):
        vals = [1000.0 / latency[m][backend] if latency[m][backend] else 0 for m in models]
        bars = ax.bar(x + i * width, vals, width, label=backend,
                      color=COLORS[backend], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val <= 0:
                continue
            offset = 5 if val > 10 else 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{val:.0f}" if val > 10 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Throughput (FPS)", fontsize=11)
    ax.set_title("Inference Throughput by Backend", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{l}\n{p}" if p else l for l, p in zip(labels, param_info)],
                       fontsize=8, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.set_yscale("log")
    all_fps = [1000.0 / v for v in all_vals]
    ax.set_ylim(min(all_fps) * 0.5, max(all_fps) * 2)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("QNN Benchmark — QRB5165 (Kryo 585 / Adreno 650 / Hexagon v66)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
