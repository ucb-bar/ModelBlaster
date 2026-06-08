"""Compare per-network predicted (solver) makespan vs v20b measured walls.

For each network in the canonical 4 MLP + 2 Dronet + 1 Yolo workload,
sum the per-dispatch predicted duration in the new hybrid fixture and
compare with the measured per-instance wall clock from v20b's run.log.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

FIXTURE = Path("/scratch2/agustin/XPU-RT/schedules/scheduled_networks_1yolo_4mlp_2dronet_firesim_hybrid.json")
RUNLOG = Path("/scratch2/agustin/ModelBlaster/artifacts/runtime_optimization/v20b_transpose_elim_retry/run.log")


def predicted_per_net() -> dict:
    fx = json.loads(FIXTURE.read_text())
    sums = defaultdict(float)
    counts = defaultdict(int)
    for k, v in fx.get("dispatches", {}).items():
        job = v.get("job_name", "")
        # job_name = "<net><inst>", e.g. "dronet0", "mlp_control3"
        m = re.match(r"(.+?)(\d+)$", job)
        if not m:
            continue
        net = m.group(1)
        sums[net] += float(v.get("duration", 0.0))
        counts[net] += 1
    return {n: {"predicted_sum_us": sums[n], "n_dispatch": counts[n]}
            for n in sums}


def measured_per_net() -> dict:
    """Parse `WALL_CYCLES_INST [<net>#k]` lines from run.log."""
    text = RUNLOG.read_text()
    pattern = re.compile(r"WALL_CYCLES_INST \[(\w+)#(\d+)\] === (\d+)")
    per_net = defaultdict(list)
    for net, inst, val in pattern.findall(text):
        per_net[net].append((int(inst), int(val)))
    return per_net


def main() -> int:
    pred = predicted_per_net()
    meas = measured_per_net()
    print(f"{'net':<14} {'pred_sum_ms':>12} {'meas_max_ms':>12} {'meas_mean_ms':>13} {'ratio':>8}")
    for net, p in pred.items():
        m = meas.get(net, [])
        if not m:
            print(f"{net:<14} {p['predicted_sum_us']/1000:>12.2f}  (no measurement)")
            continue
        max_m = max(v for _, v in m)
        mean_m = sum(v for _, v in m) / len(m)
        # measurements in mtime ticks (≈ µs at 1 MHz)
        ratio = p["predicted_sum_us"] / max_m if max_m else 0.0
        print(f"{net:<14} {p['predicted_sum_us']/1000:>12.2f} "
              f"{max_m/1000:>12.2f} {mean_m/1000:>13.2f} {ratio:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
