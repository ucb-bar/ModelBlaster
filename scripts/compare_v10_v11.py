#!/usr/bin/env python3
"""Side-by-side compare of v10 and v11 breakdown.json plus produce
the FINAL_COMPARISON.md row for v11. Run after parse_runtime_breakdown
has emitted v11's breakdown.json.

Usage:
    python3 scripts/compare_v10_v11.py [V11_RUN_DIR]

V11_RUN_DIR defaults to artifacts/runtime_optimization/v11_conv_opu_fixed/.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization")
V10 = ROOT / "v10_silu_kernel" / "breakdown.json"


def main(argv: list[str]) -> int:
    v11_dir = Path(argv[1]) if len(argv) > 1 else ROOT / "v11_conv_opu_fixed"
    v11_bd = v11_dir / "breakdown.json"
    if not v11_bd.exists():
        print(f"ERROR: {v11_bd} not found. Run parse_runtime_breakdown first.",
              file=sys.stderr)
        return 1
    v10 = json.loads(V10.read_text())
    v11 = json.loads(v11_bd.read_text())

    print("=== v10 → v11 per-hart attribution (µs) ===")
    print(f"{'kind':<10} {'metric':<22} {'v10':>10} {'v11':>10} {'Δ':>10} {'pct':>8}")
    for h10, h11 in zip(v10["harts"], v11["harts"]):
        kind = h10["kind"]
        for k in ("wall_total_us", "kernel_us", "dep_wait_us",
                  "sync_overhead_us", "target_gate_spin_us", "hart_idle_us"):
            v10v = h10[k]
            v11v = h11[k]
            d = v11v - v10v
            pct = (100.0 * d / v10v) if v10v else 0.0
            print(f"{kind:<10} {k:<22} {v10v:>10} {v11v:>10} {d:>+10} {pct:>+7.1f}%")
        print()

    print("=== v10 → v11 per-network wall (µs) ===")
    print(f"{'network':<15} {'v10':>10} {'v11':>10} {'Δ':>10} "
          f"{'v10_err':>8} {'v11_err':>8}")
    for net in sorted(set(v10["per_network"]) | set(v11["per_network"])):
        v10n = v10["per_network"].get(net, {})
        v11n = v11["per_network"].get(net, {})
        v10w = v10n.get("wall_clock_cycles_us", 0)
        v11w = v11n.get("wall_clock_cycles_us", 0)
        d = v11w - v10w
        v10e = v10n.get("max_abs_err", "?")
        v11e = v11n.get("max_abs_err", "?")
        print(f"{net:<15} {v10w:>10} {v11w:>10} {d:>+10} {v10e:>8} {v11e:>8}")

    v10_makespan = max(h["wall_total_us"] for h in v10["harts"])
    v11_makespan = max(h["wall_total_us"] for h in v11["harts"])
    print(f"\nMakespan: v10={v10_makespan:,} µs  v11={v11_makespan:,} µs  "
          f"Δ={v11_makespan - v10_makespan:+,} µs  "
          f"({100.0 * (v11_makespan - v10_makespan) / v10_makespan:+.1f}%)")

    print(f"\n=== Suggested FINAL_COMPARISON.md row for v11 ===")
    gh11 = v11["harts"][0]
    rh11 = v11["harts"][1]
    mlp = v11["per_network"].get("mlp_control", {})
    dr = v11["per_network"].get("dronet", {})
    yl = v11["per_network"].get("yolov8_nano", {})
    print(f"| v11 | + RVV im2col conv2d_s8 kernel (G6) | "
          f"{gh11['wall_total_us'] // 1000} ms | "
          f"**{rh11['wall_total_us'] // 1000} ms** | "
          f"**{max(gh11['wall_total_us'], rh11['wall_total_us']) // 1000} ms** | "
          f"{mlp.get('max_abs_err', '?')} | "
          f"{dr.get('max_abs_err', '?')} | "
          f"{yl.get('max_abs_err', '?')} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
