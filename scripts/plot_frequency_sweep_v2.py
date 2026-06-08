"""Real per-network frequency sweep — vary one network's instance count
at a time while holding the other two fixed.

The original frequency_sweep_makespan.png used pre-MOSEK naive layouts
with yolov8_nano@160 (~420 ms compute alone), which dominated the
makespan regardless of dronet/mlp count. This script generates MOSEK
schedules for two sweeps:

  Sweep A: vary dronet ∈ {0, 1, 2, 3, 4}, fix mlp_control=4, yolov8_nano_64=1
  Sweep B: vary mlp_control ∈ {0, 2, 4, 6, 8, 12}, fix dronet=2, yolov8_nano_64=1

Each point is a fresh HEFT schedule (MOSEK doesn't converge on the
yolov8_nano_64 + 2d + 4m case; HEFT is the practical solver here). We
record predicted makespan + per-tile utilization.

Output: notes/figures/frequency_sweep_v2.png — two subplots side by side.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
# XPU-RT + merlin are sibling checkouts next to this repo (../XPU-RT, ../merlin);
# override with the XPURT_ROOT / MERLIN_DIR env vars if they live elsewhere.
_XPURT_PKG = os.environ.get("XPURT_ROOT", str(REPO.parent / "XPU-RT")) + "/xpu-rt"
_MERLIN_DIR = os.environ.get("MERLIN_DIR", str(REPO.parent / "merlin"))


CONFIG_TEMPLATE = """bitstream: GemminiAndOPUShuttleConfig
registry:  cores/chipyard_gemmini_opu_hetero.json
cores:
  - {{ id: gemmini0, kind: CPU_P, hart: 0 }}
  - {{ id: opu0,     kind: CPU_E, hart: 1 }}
requantize_ops: [conv2d_s8, linear_s8]
networks:
  - {{ name: yolov8_nano_64, quant: int8, instances: {n_yolo} }}
  - {{ name: dronet,         quant: int8, instances: {n_dronet} }}
  - {{ name: mlp_control,    quant: int8, instances: {n_mlp} }}
horizon_ms: 200.0
solver: HEFT
time_limit_s: 0
cycles_source: db
cycles_agg: median
cycles_per_ms: 1000000
"""


def _solve(n_yolo: int, n_dronet: int, n_mlp: int) -> dict | None:
    if n_yolo == 0 and n_dronet == 0 and n_mlp == 0:
        return {"makespan_ms": 0.0, "n_ops": 0}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # If a count is 0 we omit the network from the list.
        nets = []
        if n_yolo > 0:
            nets.append((n_yolo, "yolov8_nano_64"))
        if n_dronet > 0:
            nets.append((n_dronet, "dronet"))
        if n_mlp > 0:
            nets.append((n_mlp, "mlp_control"))
        body = (
            "bitstream: GemminiAndOPUShuttleConfig\n"
            "registry:  cores/chipyard_gemmini_opu_hetero.json\n"
            "cores:\n"
            "  - { id: gemmini0, kind: CPU_P, hart: 0 }\n"
            "  - { id: opu0,     kind: CPU_E, hart: 1 }\n"
            "requantize_ops: [conv2d_s8, linear_s8]\n"
            "networks:\n"
        )
        for cnt, name in nets:
            body += f"  - {{ name: {name}, quant: int8, instances: {cnt} }}\n"
        body += (
            "horizon_ms: 200.0\n"
            "solver: HEFT\n"
            "time_limit_s: 0\n"
            "cycles_source: db\n"
            "cycles_agg: median\n"
            "cycles_per_ms: 1000000\n"
        )
        f.write(body)
        cfg_path = f.name

    out = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False).name
    proc = subprocess.run(
        ["uv", "run", "python", "scripts/run_xpurt_scheduler_multi.py",
         "--config", cfg_path, "--output", out],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ,
             "PYTHONPATH": f"{REPO}:{_XPURT_PKG}:{_MERLIN_DIR}"},
    )
    if proc.returncode != 0:
        print(f"  solver failed for ({n_yolo},{n_dronet},{n_mlp}): {proc.stderr[-300:]}")
        return None
    fx = json.loads(pathlib.Path(out).read_text())
    prov = fx.get("_provenance", {})
    util = (prov.get("scheduler_report", {}) or {}).get("utilization", {})
    return {
        "makespan_ms": float(prov["makespan_ms"]),
        "n_ops": len(fx["dispatches"]),
        "util_cpu_p": util.get("CPU_P#0", util.get("frac_busy_per_kind", {}).get("CPU_P", 0.0)),
        "util_cpu_e": util.get("CPU_E#0", util.get("frac_busy_per_kind", {}).get("CPU_E", 0.0)),
    }


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("== Sweep A: dronet count (yolov8_nano_64=1, mlp_control=4) ==")
    sweep_a = []
    for n_d in [0, 1, 2, 3, 4]:
        r = _solve(1, n_d, 4)
        if r is None:
            continue
        print(f"  n_dronet={n_d}: makespan={r['makespan_ms']:.2f} ms, n_ops={r['n_ops']}, "
              f"util P={r.get('util_cpu_p',0):.2f} E={r.get('util_cpu_e',0):.2f}")
        sweep_a.append((n_d, r))

    print("\n== Sweep B: mlp_control count (yolov8_nano_64=1, dronet=2) ==")
    sweep_b = []
    for n_m in [0, 2, 4, 6, 8, 12]:
        r = _solve(1, 2, n_m)
        if r is None:
            continue
        print(f"  n_mlp={n_m}: makespan={r['makespan_ms']:.2f} ms, n_ops={r['n_ops']}, "
              f"util P={r.get('util_cpu_p',0):.2f} E={r.get('util_cpu_e',0):.2f}")
        sweep_b.append((n_m, r))

    # Plot.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    if sweep_a:
        xs = [p[0] for p in sweep_a]
        ys = [p[1]["makespan_ms"] for p in sweep_a]
        ax_a.plot(xs, ys, "o-", color="#3b82f6", linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            ax_a.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=8)
        ax_a.set_xlabel("# dronet instances")
        ax_a.set_ylabel("Predicted makespan (ms)")
        ax_a.set_title(f"Sweep A — vary dronet  (yolov8_nano_64=1, mlp_control=4)")
        ax_a.grid(linestyle=":", alpha=0.4)
        ax_a.axhline(75.71, color="#16a34a", linestyle="--", linewidth=1, alpha=0.7,
                     label="qrb 75.71 ms")
        ax_a.legend(fontsize=8)

    if sweep_b:
        xs = [p[0] for p in sweep_b]
        ys = [p[1]["makespan_ms"] for p in sweep_b]
        ax_b.plot(xs, ys, "o-", color="#10b981", linewidth=2, markersize=8)
        for x, y in zip(xs, ys):
            ax_b.annotate(f"{y:.1f}", xy=(x, y), xytext=(0, 10), textcoords="offset points",
                          ha="center", fontsize=8)
        ax_b.set_xlabel("# mlp_control instances")
        ax_b.set_ylabel("Predicted makespan (ms)")
        ax_b.set_title(f"Sweep B — vary mlp_control  (yolov8_nano_64=1, dronet=2)")
        ax_b.grid(linestyle=":", alpha=0.4)
        ax_b.axhline(75.71, color="#16a34a", linestyle="--", linewidth=1, alpha=0.7,
                     label="qrb 75.71 ms")
        ax_b.legend(fontsize=8)

    fig.suptitle(
        "3-way frequency sweep — HEFT on yolov8_nano_64 (smaller variant) hetero bitstream\n"
        "Predicted makespan as a function of per-network instance count",
        fontsize=11,
    )
    out = REPO / "notes" / "figures" / "frequency_sweep_v2.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
