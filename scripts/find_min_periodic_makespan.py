"""Binary-search the minimum makespan M such that the periodic
partition scheduler can place every instance within its proportional
slot [i * M/N, (i+1) * M/N] with zero deadline misses.

Different from the qrb-rate periodic schedule (75 ms horizon): here
the per-network frequencies are RATIOS — for N instances of a network,
each owns 1/N of the makespan. Minimizing makespan ⇒ all networks run
at their maximum sustainable rate while still respecting per-instance
periodicity.

The binding constraint will usually be the heaviest single inference
(yolov8 critical path), since its slot is the whole M. Lighter
networks with multiple instances (mlp_control × 4) get tighter slots
proportionally but their compute is much smaller.

Run: python scripts/find_min_periodic_makespan.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent

NETWORKS = [
    ("yolov8_nano_64", "int8", 1),
    ("dronet",         "int8", 2),
    ("mlp_control",    "int8", 4),
]

# Bounds: lower from theoretical critical-path (55.77 ms for yolov8),
# upper from "definitely fits" (200 ms).
M_LO = 40.0
M_HI = 200.0
TOL = 0.5  # ms — stop when interval is smaller than this


def _try_makespan(M: float) -> tuple[bool, int, float]:
    """Return (feasible, deadline_misses, actual_makespan) for horizon=M."""
    yaml_text = (
        "bitstream: GemminiAndOPUShuttleConfig\n"
        "registry:  cores/chipyard_gemmini_opu_hetero.json\n"
        "cores:\n"
        "  - { id: gemmini0, kind: CPU_P, hart: 0 }\n"
        "  - { id: opu0,     kind: CPU_E, hart: 1 }\n"
        # NO requantize_ops: drift constraint dropped so conv2d_s8 +
        # linear_s8 can split across both tiles, matching qrb's
        # both-tiles-balanced approach.
        "networks:\n"
    )
    for net, q, n in NETWORKS:
        yaml_text += f"  - {{ name: {net}, quant: {q}, instances: {n} }}\n"
    yaml_text += f"horizon_ms: {M}\n"
    yaml_text += "enforce_periodic: true\n"
    yaml_text += "solver: HEFT\n"
    yaml_text += "cycles_source: db\n"
    yaml_text += "cycles_agg: median\n"
    yaml_text += "cycles_per_ms: 1000000\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        cfg = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        out = f.name

    import os
    env = {**os.environ,
           "PYTHONPATH": f"{REPO}:/scratch2/agustin/XPU-RT/xpu-rt:/scratch2/agustin/merlin"}
    r = subprocess.run(
        ["python3", "scripts/periodic_partition_schedule.py",
         "--config", cfg, "--output", out],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return (False, -1, 0.0)
    fx = json.loads(pathlib.Path(out).read_text())
    prov = fx.get("_provenance", {})
    return (
        prov.get("deadline_misses", 0) == 0,
        prov.get("deadline_misses", 0),
        prov.get("makespan_ms", 0.0),
    )


def main():
    print(f"Binary-searching min feasible makespan for 1y + 2d + 4m with periodic slots...")
    print(f"  bounds: [{M_LO}, {M_HI}] ms, tolerance {TOL} ms")
    print()

    # First, probe both ends
    print("Probe ends:")
    feas_hi, miss_hi, mks_hi = _try_makespan(M_HI)
    print(f"  M={M_HI:.1f}: feasible={feas_hi}, misses={miss_hi}, achieved={mks_hi:.2f}")
    feas_lo, miss_lo, mks_lo = _try_makespan(M_LO)
    print(f"  M={M_LO:.1f}: feasible={feas_lo}, misses={miss_lo}, achieved={mks_lo:.2f}")

    if not feas_hi:
        print(f"\nNo feasible makespan up to {M_HI} ms (workload genuinely too big).")
        return 1
    if feas_lo:
        print(f"\nM_LO={M_LO} already feasible (achieved {mks_lo:.2f}). "
              f"Lower bound was too generous.")

    lo, hi = M_LO, M_HI
    best_M = M_HI
    best_makespan = mks_hi
    print()
    print(f"{'iter':>4s}  {'M (ms)':>10s}  {'feasible?':>10s}  {'misses':>7s}  {'achieved (ms)':>15s}")
    print("-" * 60)
    it = 0
    while hi - lo > TOL:
        mid = (lo + hi) / 2.0
        feas, miss, achieved = _try_makespan(mid)
        it += 1
        print(f"{it:>4d}  {mid:>10.2f}  {str(feas):>10s}  {miss:>7d}  {achieved:>15.2f}")
        if feas:
            best_M = mid
            best_makespan = achieved
            hi = mid
        else:
            lo = mid

    print()
    print(f"Minimum feasible periodic makespan (1y + 2d + 4m, all instances in slot):")
    print(f"  horizon target: ~{best_M:.2f} ms")
    print(f"  achieved makespan: {best_makespan:.2f} ms")
    print(f"  implied per-network frequencies:")
    for net, _, n in NETWORKS:
        slot = best_M / n
        hz = 1000.0 / slot
        print(f"    {net}: {n} instance(s), each in {slot:.2f} ms slot → {hz:.1f} Hz")
    # Compare to qrb
    print()
    print(f"vs qrb image (1y + 2d + 4m in 75.71 ms = 13.2 Hz yolo / 26.4 Hz dronet / 52.8 Hz mlp):")
    qrb = 75.71
    if best_M < qrb:
        print(f"  BEAT qrb by {qrb - best_M:.2f} ms ({(1 - best_M/qrb)*100:.1f}%)")
    else:
        print(f"  Over qrb by {best_M - qrb:.2f} ms ({(best_M/qrb - 1)*100:.1f}%)")

    # Emit the best fixture as a real output
    out = REPO / "schedule_fixtures" / "3way_partitioned_qrb_y64_minM.json"
    _, _, _ = _try_makespan(best_M)  # regenerate at exact best M
    # Actually we want to emit the fixture from the BEST run
    yaml_text = (
        "bitstream: GemminiAndOPUShuttleConfig\n"
        "registry:  cores/chipyard_gemmini_opu_hetero.json\n"
        "cores:\n"
        "  - { id: gemmini0, kind: CPU_P, hart: 0 }\n"
        "  - { id: opu0,     kind: CPU_E, hart: 1 }\n"
        "requantize_ops: [conv2d_s8, linear_s8]\n"
        "networks:\n"
    )
    for net, q, n in NETWORKS:
        yaml_text += f"  - {{ name: {net}, quant: {q}, instances: {n} }}\n"
    yaml_text += f"horizon_ms: {best_M}\nenforce_periodic: true\nsolver: HEFT\n"
    yaml_text += "cycles_source: db\ncycles_agg: median\ncycles_per_ms: 1000000\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        cfg = f.name
    import os
    env = {**os.environ,
           "PYTHONPATH": f"{REPO}:/scratch2/agustin/XPU-RT/xpu-rt:/scratch2/agustin/merlin"}
    subprocess.run(
        ["python3", "scripts/periodic_partition_schedule.py",
         "--config", cfg, "--output", str(out)],
        cwd=str(REPO), env=env,
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
