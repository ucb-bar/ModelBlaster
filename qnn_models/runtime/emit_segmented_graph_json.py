"""Emit per-network dispatch_graph.json + per-(hw, network) profile CSVs
from the QNN per-segment per-backend profiling sweep.

Reads:
  qnn_models/runtime/gen/<gen_dir>/segment_perf.json
    {<segment_name>: {<backend>: {mean_us, median_us, p99_us, ...}}}
  where <segment_name> looks like:
    "dronet_HTA_split_seg0", "dronet_CPU_seg1", "yolov8n_HTA_split_seg100"
  and <backend> in {"Hta", "Dsp", "Cpu"}.

Writes (under repo root):
  gen/qnn_vmfb/<network>_segmented/<target>/<hw>/<basename>/<basename>_dispatch_graph.json
  gen/profile/<hw>/<target>/<network>_segmented/<basename>/topo_0/results.csv

The dispatches in the new graph.json are the *segments* (one dispatch per
sub-DLC), not the original per-op dispatches. Each segment becomes
dispatch_id N where N is a contiguous index within the network. Deps form
a linear chain (the segments came from a linear chain of the original
ONNX). This matches what the QNN runtime actually executes.

The CSV `mean_time` column is the wallclock-around-graphExecute mean
from profile_segments.cpp (launch + RPC + sync + compute), unit "us".
That is the cost the scheduler should plan against — it is the same
number the gantt-chart trace dump uses.

For (network, backend) cells with no measurement (e.g. yolov8n on HTA —
HTA can't run yolov8's head op set), we emit a sentinel cost of
1_000_000 us so the scheduler avoids those routes without us needing to
modify profile_loader.

Network → segments mapping is read directly from segment_perf.json: any
key starting with "<net>_" belongs to that network. Segment ids are
recovered by parsing the trailing "_seg<N>" off the segment name and
sorting numerically — that gives us the dependency chain.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict


BACKENDS = ("Hta", "Dsp", "Cpu")  # case as stored in segment_perf.json
SENTINEL_MISSING_US = 100_000.0  # 100 ms — cost-prohibitive vs the real per-segment alternatives (yolov8 worst-real ~39 ms on CPU) but small enough that the workload-factory horizon heuristic doesn't blow up the periodic-instance count (it sums max-per-op across non-periodic ops to estimate the makespan upper bound; a sentinel of 1e6 us = 1000 ms inflated horizon to 4000 ms → 800 dronet instances at 5 ms period)


def _parse_seg_index(seg_name: str) -> int:
    m = re.search(r"_seg(\d+)$", seg_name)
    if not m:
        raise ValueError(f"can't parse segment index from {seg_name!r}")
    return int(m.group(1))


def _network_for(seg_name: str) -> str:
    # "dronet_HTA_split_seg0" -> "dronet"
    # "yolov8n_HTA_split_seg100" -> "yolov8n"
    # "dronet_CPU_seg1" -> "dronet"
    return seg_name.split("_", 1)[0]


def _group_by_network(perf: dict) -> dict[str, list[tuple[int, str]]]:
    """{network: [(seg_idx, seg_name), ...]} sorted by seg_idx."""
    out: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for seg_name in perf.keys():
        net = _network_for(seg_name)
        try:
            idx = _parse_seg_index(seg_name)
        except ValueError as e:
            print(f"  (skip) {e}", file=sys.stderr); continue
        out[net].append((idx, seg_name))
    for net in out:
        out[net].sort(key=lambda t: t[0])
    return out


def _emit_dispatch_graph(net: str,
                         segs: list[tuple[int, str]],
                         out_path: str) -> None:
    """One dispatch per segment, linear-chain dependencies. Dispatch ids
    are 0..N-1 in segment order; module_name carries the original
    segment label so the gantt-plot/runtime can correlate."""
    dispatches: dict[str, dict] = {}
    for new_id, (orig_idx, seg_name) in enumerate(segs):
        deps = [f"dispatch_{new_id - 1}"] if new_id > 0 else []
        dispatches[f"dispatch_{new_id}"] = {
            "id": new_id,
            "ordinal": 1,
            "total": 1,
            "dependencies": deps,
            "vmfb_path": f"sub_dlc/{seg_name}_quantized.dlc",
            "segment_name": seg_name,
            "orig_seg_idx": orig_idx,
        }
    payload = {
        "dot_file": "",
        "dispatch_vmfb_dir": "sub_dlc",
        "_comment": (f"Segmented {net}: {len(segs)} dispatches; each "
                     "dispatch is one sub-DLC profiled across HTA/DSP/CPU "
                     "in segment_perf.json."),
        "dispatches": dispatches,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_path}  ({len(segs)} dispatches)")


def _emit_profile_csv(net: str,
                      segs: list[tuple[int, str]],
                      perf: dict,
                      backend: str,
                      out_path: str) -> int:
    """One row per dispatch. mean_time = mean_us from profile_segments
    (wallclock-around-graphExecute, includes launch/sync overhead).
    Returns count of "real" (non-sentinel) rows."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    real = 0
    with open(out_path, "w") as f:
        f.write("dispatch_id,module_name,mean_time,mean_unit\n")
        for new_id, (_, seg_name) in enumerate(segs):
            stats = perf.get(seg_name, {}).get(backend)
            if stats and stats.get("status") == "ok" and "mean_us" in stats:
                t_us = float(stats["mean_us"])
                real += 1
            else:
                t_us = SENTINEL_MISSING_US
            f.write(f"{new_id},{seg_name},{t_us:.4f},us\n")
    return real


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--perf-json", required=True,
                    help="path to segment_perf.json from profile_sweep.sh")
    ap.add_argument("--repo-root", required=True,
                    help="FreshScheduler repo root (gen/ written under here)")
    ap.add_argument("--target",   default="qrb5165_v66",
                    help="profile target tag (matches hardware.profile.target)")
    ap.add_argument("--basename-suffix", default="int8",
                    help="basename suffix; full basename = <net>_segmented.<suffix>")
    args = ap.parse_args()

    with open(args.perf_json, "r") as f:
        perf = json.load(f)
    by_net = _group_by_network(perf)
    print(f"==> networks: {sorted(by_net.keys())}")
    for net, segs in by_net.items():
        print(f"    {net}: {len(segs)} segments  ids={[i for i,_ in segs]}")

    # Backend label used in paths/profile_hw (Hta -> HTA, Dsp -> DSP, Cpu -> CPU).
    # The scheduler's profile_hw values can be arbitrary; we pick the
    # canonical uppercase form for clarity.
    HW_LABEL = {"Hta": "HTA", "Dsp": "DSP", "Cpu": "CPU"}

    for net, segs in by_net.items():
        net_segmented = f"{net}_segmented"
        basename = f"{net_segmented}.{args.basename_suffix}"

        # Emit one dispatch graph per (hw) so the toplevel can point each
        # net's lookup at a hw-specific subdir. The contents are identical
        # across hw — the scheduler reads dispatch deps from one CSV path,
        # but for symmetry with the existing layout we keep per-hw copies.
        # The per-hw copies don't hurt (small) and make it easier for the
        # plotter to introspect.
        for be in BACKENDS:
            hw = HW_LABEL[be]
            graph_path = os.path.join(
                args.repo_root, "gen", "qnn_vmfb",
                net_segmented, args.target, hw, basename,
                f"{basename}_dispatch_graph.json",
            )
            _emit_dispatch_graph(net, segs, graph_path)

            csv_path = os.path.join(
                args.repo_root, "gen", "profile",
                hw, args.target,
                net_segmented, basename, "topo_0",
                "results.csv",
            )
            n_real = _emit_profile_csv(net, segs, perf, be, csv_path)
            print(f"  wrote {csv_path}  ({n_real}/{len(segs)} measured)")


if __name__ == "__main__":
    main()
