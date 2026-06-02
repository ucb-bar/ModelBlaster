"""Same binary-search shape as find_min_periodic_makespan.py but uses
MOSEK MILP instead of the partition scheduler — lets the solver
interleave ops from different instances within each tile to reclaim
the idle time the greedy partition leaves on the table.

For each candidate horizon M:
  - Each instance i of network with N instances gets
    min_start_t = i * M/N and max_end_t = (i+1) * M/N (set by the
    bridge's enforce_periodic path).
  - MOSEK minimizes makespan subject to: per-op time-windows,
    cross-backend drift constraint, machine non-overlap, precedence.
  - If status=OPTIMAL/OPTIMAL_INACCURATE with all ops scheduled
    in their windows, the horizon is feasible at that frequency.

Binary-search the smallest feasible M.

Caveat: 300-op MILP is heavy. Previous experiments showed MOSEK
needs 5+ min per solve on this scale. The binary search may use
~10 solves. Plan for ~1 hour total.
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

# Start at lower bound = HEFT's 84 ms packed (we KNOW we need at least
# critical-path-ish), upper at partition's 160 ms.
M_LO = 70.0
M_HI = 160.0
TOL = 2.0  # ms — coarser for MOSEK (each solve is expensive)

TIME_LIMIT_S = 300  # per-MOSEK-solve budget; balance vs total binary-search wall


def _try_makespan(M: float, label: str = "") -> tuple[bool, str, float]:
    """Return (feasible, status, achieved_makespan_ms)."""
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
    yaml_text += f"horizon_ms: {M}\n"
    yaml_text += "enforce_periodic: true\n"
    yaml_text += "solver: MOSEK\n"
    yaml_text += f"time_limit_s: {TIME_LIMIT_S}\n"
    yaml_text += "cycles_source: db\ncycles_agg: median\ncycles_per_ms: 1000000\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        cfg = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        out = f.name

    import os
    env = {**os.environ,
           "PYTHONPATH": f"{REPO}:/scratch2/agustin/XPU-RT/xpu-rt:/scratch2/agustin/merlin"}
    r = subprocess.run(
        ["/scratch2/agustin/xpu-rt-integration/.venv/bin/python",
         "scripts/run_xpurt_scheduler_multi.py",
         "--config", cfg, "--output", out],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )
    if r.returncode != 0:
        return (False, "FAILED", 0.0)
    try:
        fx = json.loads(pathlib.Path(out).read_text())
    except Exception:
        return (False, "PARSE_ERR", 0.0)
    prov = fx.get("_provenance", {})
    rep = prov.get("scheduler_report") or {}
    status = rep.get("solver_status", "?")
    mks = prov.get("makespan_ms", 0.0)
    feasible = status in {"optimal", "optimal_inaccurate"} and mks > 0 and mks <= M + 0.5
    return (feasible, status, mks)


def main() -> int:
    print(f"MOSEK periodic min-makespan search for 1y + 2d + 4m")
    print(f"  bounds: [{M_LO}, {M_HI}] ms, tol {TOL} ms, per-solve {TIME_LIMIT_S}s budget")
    print()

    print(f"{'step':>5s}  {'M (ms)':>8s}  {'status':>22s}  {'achieved':>10s}  {'feasible?':>10s}")
    print("-" * 70)

    # Probe ends
    f_hi, s_hi, m_hi = _try_makespan(M_HI, "hi")
    print(f"{'hi':>5s}  {M_HI:>8.2f}  {s_hi:>22s}  {m_hi:>10.2f}  {str(f_hi):>10s}")
    if not f_hi:
        print(f"\nMOSEK could not find feasible solution even at M={M_HI}. Aborting.")
        return 1

    f_lo, s_lo, m_lo = _try_makespan(M_LO, "lo")
    print(f"{'lo':>5s}  {M_LO:>8.2f}  {s_lo:>22s}  {m_lo:>10.2f}  {str(f_lo):>10s}")

    lo, hi = M_LO, M_HI
    best_M = M_HI
    best_mks = m_hi
    if f_lo:
        best_M, best_mks = M_LO, m_lo
        print(f"\nLow end already feasible at {m_lo:.2f} ms — done")
    else:
        it = 0
        while hi - lo > TOL:
            mid = (lo + hi) / 2.0
            f, s, m = _try_makespan(mid)
            it += 1
            print(f"{it:>5d}  {mid:>8.2f}  {s:>22s}  {m:>10.2f}  {str(f):>10s}")
            if f:
                hi = mid
                best_M = mid
                best_mks = m
            else:
                lo = mid

    print()
    print(f"Minimum feasible periodic MOSEK makespan (1y + 2d + 4m):")
    print(f"  horizon target: ~{best_M:.2f} ms")
    print(f"  MOSEK achieved makespan: {best_mks:.2f} ms")
    print(f"  per-network frequencies (slot = M/N):")
    for net, _, n in NETWORKS:
        slot = best_M / n
        hz = 1000.0 / slot
        print(f"    {net}: {n} inst, slot={slot:.2f} ms → {hz:.1f} Hz")
    qrb = 75.71
    delta = (best_M / qrb - 1) * 100
    print()
    print(f"  vs qrb {qrb} ms: {'BEAT' if best_M < qrb else 'over'} by {abs(best_M - qrb):.2f} ms ({delta:+.1f}%)")

    # Emit best fixture
    out_path = REPO / "schedule_fixtures" / "3way_mosek_qrb_y64_periodic_minM.json"
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
    yaml_text += f"horizon_ms: {best_M}\nenforce_periodic: true\nsolver: MOSEK\n"
    yaml_text += f"time_limit_s: {TIME_LIMIT_S}\ncycles_source: db\ncycles_agg: median\ncycles_per_ms: 1000000\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        cfg = f.name
    import os
    env = {**os.environ,
           "PYTHONPATH": f"{REPO}:/scratch2/agustin/XPU-RT/xpu-rt:/scratch2/agustin/merlin"}
    subprocess.run(
        ["/scratch2/agustin/xpu-rt-integration/.venv/bin/python",
         "scripts/run_xpurt_scheduler_multi.py",
         "--config", cfg, "--output", str(out_path)],
        cwd=str(REPO), env=env,
    )
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
