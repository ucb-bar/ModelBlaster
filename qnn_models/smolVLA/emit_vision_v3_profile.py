"""Emit XPURT-compatible results.csv from v3 vision slice profiling data.

Two modes:
  --from-perf-json <path>  (preferred) Read segment_perf.json produced by
      profile_vision_v3_correct.sh (wallclock around QnnGraph_execute on
      context binaries — the same measurement the generated runtime uses).

  --profile-dir <path>  (legacy fallback) Read QNN profiling CSVs from
      qnn-net-run --profiling_level. Less accurate (only backend-reported
      compute, not full dispatch overhead).

Emits gen/profile/<hw>/qrb5165_v66/smolvlm_vision_v3/smolvlm_vision_v3.int8/topo_0/results.csv
for each backend (DSP, CPU).

The dispatch_graph has 49 dispatches in a linear chain:
  dispatch_0 = dsp_seg_00, dispatch_1 = cpu_seg_00, ...
  dispatch_2k = dsp_seg_k, dispatch_2k+1 = cpu_seg_k

Usage:
  python emit_vision_v3_profile.py --from-perf-json boards/qrb5165_v66/profiles/smolvlm_vision_v3/segment_perf.json
  python emit_vision_v3_profile.py --profile-dir boards/qrb5165_v66/profiles/smolvlm_vision_v3
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_HERE = Path(__file__).parent
_REPO_ROOT = _HERE.parent.parent


def parse_execute_time_us(csv_path: Path) -> float | None:
    """Extract median EXECUTE time from a QNN profiling CSV.

    Looks for BACKEND,ROOT EXECUTE entries (one per iteration).
    Falls back to NETRUN,ROOT if no BACKEND data.
    Returns time in microseconds, or None if no data.
    """
    backend_times = []
    netrun_times = []

    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7:
                continue
            msg = row[1].strip()
            if msg != "EXECUTE":
                continue
            try:
                t = float(row[2].strip())
            except (ValueError, IndexError):
                continue
            unit = row[3].strip()
            if unit == "MS":
                t *= 1000
            elif unit == "S":
                t *= 1_000_000
            source = row[4].strip()
            level = row[5].strip()

            if source == "BACKEND" and level == "ROOT":
                backend_times.append(t)
            elif source == "NETRUN" and level == "ROOT":
                netrun_times.append(t)

    times = backend_times if backend_times else netrun_times
    if not times:
        return None

    # Drop first 2 warmup iterations, take median
    warmup = min(2, len(times) // 4)
    steady = sorted(times[warmup:])
    if not steady:
        return times[-1]
    return steady[len(steady) // 2]


# Wall-clock fallback estimates for DSP segments that don't capture profiling.
# The DSP profiling infrastructure fails to capture BACKEND ROOT events for
# some segments (likely large graphs where the DSP firmware doesn't report).
# Pattern from segments that DO profile:
#   A-type (even DSP idx >=2: QKV + reshape + scale + Q×K^T): 134-175ms on DSP
#   B-type (odd DSP idx: proj + fc1 + fc2 + LN + Add): 76-84ms on DSP
#   seg_00 (patch embed conv + first A block): ~150ms
#   seg_01 (first B-type): ~80ms (profiled data was corrupted by concurrency)
DSP_FALLBACK_US = {
    "dsp_seg_00": 150_000,   # patch embed + first QKV block
    "dsp_seg_01":  80_000,   # B-type (first post-attention)
    "dsp_seg_04": 145_000,   # A-type
    "dsp_seg_05":  80_000,   # B-type
    "dsp_seg_06": 145_000,   # A-type
    "dsp_seg_07":  80_000,   # B-type
    "dsp_seg_08": 145_000,   # A-type
    "dsp_seg_09":  80_000,   # B-type
    "dsp_seg_22": 145_000,   # A-type
    "dsp_seg_23":  80_000,   # B-type
}

# Wall-clock fallback for missing CPU segment profiles
CPU_SEG_FALLBACK_US = {
    "cpu_seg_02": 40_000,   # softmax block, matches neighbors
    "cpu_seg_06": 39_000,   # tanh
    "cpu_seg_11": 39_000,
    "cpu_seg_20": 39_000,
}


def emit_from_perf_json(perf_json_path: Path, target: str):
    """Emit results.csv from segment_perf.json (profile_segments.cpp output)."""
    with open(perf_json_path) as f:
        perf = json.load(f)

    n_dsp = 25
    n_cpu = 24
    n_total = n_dsp + n_cpu

    dispatch_to_seg = {}
    for i in range(n_total):
        if i % 2 == 0:
            dispatch_to_seg[i] = f"dsp_seg_{i // 2:02d}"
        else:
            dispatch_to_seg[i] = f"cpu_seg_{i // 2:02d}"

    # hw_key maps our output "HW" directory to the backend key in perf JSON
    hw_to_backend_key = {"DSP": "Dsp", "CPU": "Cpu", "HTA": "Hta"}

    for hw in ("CPU", "DSP", "HTA"):
        out_dir = _REPO_ROOT / "gen" / "profile" / hw / target / "smolvlm_vision_v3" / "smolvlm_vision_v3.int8" / "topo_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"

        total_us = 0
        n_profiled = 0
        n_fallback = 0

        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dispatch_id", "module_name", "mean_time", "mean_unit"
            ])
            w.writeheader()

            for dispatch_id in range(n_total):
                seg_name = dispatch_to_seg[dispatch_id]
                time_us = None

                seg_data = perf.get(seg_name, {})
                if seg_name.startswith("cpu_seg"):
                    be_key = "Cpu"
                else:
                    be_key = hw_to_backend_key[hw]

                be_data = seg_data.get(be_key, {})
                if be_data.get("status") in ("ok", "partial"):
                    time_us = be_data.get("mean_us")
                    if time_us is not None:
                        n_profiled += 1

                if time_us is None:
                    # Fallback: for HTA, use DSP time as proxy since HTA
                    # only covers the conv ops (not the full segment)
                    if hw == "HTA":
                        dsp_data = seg_data.get("Dsp", {})
                        if dsp_data.get("status") == "ok":
                            time_us = dsp_data.get("mean_us")
                        if time_us is None and seg_name in DSP_FALLBACK_US:
                            time_us = DSP_FALLBACK_US[seg_name]
                    elif hw == "DSP" and seg_name in DSP_FALLBACK_US:
                        time_us = DSP_FALLBACK_US[seg_name]
                    elif seg_name in CPU_SEG_FALLBACK_US:
                        time_us = CPU_SEG_FALLBACK_US[seg_name]

                    if time_us is None:
                        cpu_data = seg_data.get("Cpu", {})
                        if cpu_data.get("status") == "ok":
                            time_us = cpu_data.get("mean_us")
                        if time_us is None:
                            time_us = 100_000
                    n_fallback += 1

                total_us += time_us
                w.writerow({
                    "dispatch_id": dispatch_id,
                    "module_name": seg_name,
                    "mean_time": f"{time_us:.2f}",
                    "mean_unit": "us",
                })

        total_ms = total_us / 1000
        print(f"  {hw}: {out_path}")
        print(f"       {n_profiled} profiled, {n_fallback} fallback")
        print(f"       Total serial time: {total_ms:.1f} ms")

    print(f"\nDone. Results emitted to gen/profile/<hw>/{target}/smolvlm_vision_v3/")


def emit_from_profile_csvs(profile_dir: Path, target: str):
    """Legacy: emit results.csv from qnn-net-run profiling CSVs."""
    n_dsp = 25
    n_cpu = 24
    n_total = n_dsp + n_cpu

    dispatch_to_seg = {}
    for i in range(n_total):
        if i % 2 == 0:
            dispatch_to_seg[i] = f"dsp_seg_{i // 2:02d}"
        else:
            dispatch_to_seg[i] = f"cpu_seg_{i // 2:02d}"

    for hw in ("CPU", "DSP"):
        out_dir = _REPO_ROOT / "gen" / "profile" / hw / target / "smolvlm_vision_v3" / "smolvlm_vision_v3.int8" / "topo_0"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.csv"

        total_us = 0
        n_profiled = 0
        n_fallback = 0

        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "dispatch_id", "module_name", "mean_time", "mean_unit"
            ])
            w.writeheader()

            for dispatch_id in range(n_total):
                seg_name = dispatch_to_seg[dispatch_id]

                if seg_name.startswith("cpu_seg"):
                    csv_path = profile_dir / f"{seg_name}__CPU.csv"
                elif hw == "DSP":
                    csv_path = profile_dir / f"{seg_name}__DSP.csv"
                else:
                    csv_path = profile_dir / f"{seg_name}__CPU.csv"

                time_us = None
                if csv_path.exists():
                    time_us = parse_execute_time_us(csv_path)

                if time_us is None:
                    if hw == "DSP" and seg_name in DSP_FALLBACK_US:
                        time_us = DSP_FALLBACK_US[seg_name]
                    elif seg_name in CPU_SEG_FALLBACK_US:
                        time_us = CPU_SEG_FALLBACK_US[seg_name]
                    else:
                        cpu_csv = profile_dir / f"{seg_name}__CPU.csv"
                        if cpu_csv.exists():
                            time_us = parse_execute_time_us(cpu_csv)
                        if time_us is None:
                            time_us = 100_000
                    n_fallback += 1
                else:
                    n_profiled += 1

                total_us += time_us
                w.writerow({
                    "dispatch_id": dispatch_id,
                    "module_name": seg_name,
                    "mean_time": f"{time_us:.2f}",
                    "mean_unit": "us",
                })

        total_ms = total_us / 1000
        print(f"  {hw}: {out_path}")
        print(f"       {n_profiled} profiled, {n_fallback} fallback")
        print(f"       Total serial time: {total_ms:.1f} ms")

    print(f"\nDone. Results emitted to gen/profile/<hw>/{target}/smolvlm_vision_v3/")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="qrb5165_v66")
    ap.add_argument("--from-perf-json", default=None,
                    help="Path to segment_perf.json from profile_vision_v3_correct.sh")
    ap.add_argument("--profile-dir", default=None,
                    help="Legacy: path to qnn-net-run profiling CSVs")
    args = ap.parse_args()

    if args.from_perf_json:
        emit_from_perf_json(Path(args.from_perf_json), args.target)
    else:
        profile_dir = Path(args.profile_dir) if args.profile_dir else (
            _HERE.parent / "boards" / args.target / "profiles" / "smolvlm_vision_v3"
        )
        emit_from_profile_csvs(profile_dir, args.target)


if __name__ == "__main__":
    main()
