"""Emit per-network 1-dispatch graph.json + per-(hw, network) profile CSVs
for the COARSE-GRAINED scheduling experiment: each network is one
QnnGraph_execute on a single backend (no per-segment routing).

Why coarse: the per-segment cost model lets MILP exploit per-op
flexibility, but the resulting MILP has n_ops × n_combinations × n_ops^2
boolean variables which blew up to 31M with 800 dronet instances. With
1 op per network × 24 instances + 1 yolov8 = 25 ops, MILP shrinks to
2k variables — solves in seconds, and the cost model is grounded in
actual whole-network measurements (no per-segment-launch overhead
double-counting).

Reads measurements from a small embedded table (the user's bench run
on QRB5165 / Hexagon v66, captured via profile_segments.cpp on
ctx_<net>_full__<Be>.bin). HTA's whole-network compose fails for both
networks (BN rewrite + yolov8 head ops) so we use a 100ms sentinel.

Outputs the same layout as emit_segmented_graph_json.py:
  gen/qnn_vmfb/<network>_coarse/<target>/<hw>/<basename>/<basename>_dispatch_graph.json
  gen/profile/<hw>/<target>/<network>_coarse/<basename>/topo_0/results.csv
"""

from __future__ import annotations

import argparse
import json
import os


# Per-(network, backend) wallclock-around-graphExecute mean, in microseconds,
# from profile_segments.cpp on ctx_<net>_full__<Be>.bin (50 iters each).
COARSE_PROFILE = {
    "dronet": {
        # Uses dronet_full_hta_quantized.dlc — bnfree + #8 conv-head + #11
        # drop trailing Reshape per qnn_models/optimization_flow.md so HTA
        # composes (the unmodified bnfree DLC fails: HTA can't run the
        # boundary Transpose ops).
        "HTA": 2654.64,
        "DSP": 920.69,
        "CPU": 7499.78,
    },
    "yolov8n": {
        "DSP": 62163.39,
        "CPU": 85572.84,
        "HTA": 100_000.0,    # yolov8 head ops (Resize/Slice/Softmax) reject on HTA — sentinel
    },
    "mlp_control": {
        # Trained drone-control actor MLP (16 → 256 → 128 → 64 → 4, ELU)
        # exported from zephyr-chipyard-sw/agents/models/mlp_control.py.
        # Compute is trivial (~70k MACs); CPU wins because DSP dispatch
        # overhead (~500 µs FastRPC RTT) dominates anything this small.
        # HTA rejects: v66 HTA's ElementWiseNeuron op set has no ELU
        # entry (only ReLU/ReLU6/Sigmoid/Tanh/HardSwish).
        "CPU": 113.74,
        "DSP": 543.34,
        "HTA": 100_000.0,    # ELU unsupported — sentinel
    },
}

BACKENDS = ("HTA", "DSP", "CPU")


def _emit_dispatch_graph(net: str, out_path: str) -> None:
    """One dispatch per network — the whole graphExecute."""
    payload = {
        "dot_file": "",
        "dispatch_vmfb_dir": "ctx",
        "_comment": (f"Coarse-grained {net}: 1 dispatch = whole-network DLC."
                     " Used for the small-MILP experiment."),
        "dispatches": {
            "dispatch_0": {
                "id": 0,
                "ordinal": 1,
                "total": 1,
                "dependencies": [],
                "vmfb_path": f"ctx/ctx_{net}_full.bin",
                "segment_name": f"{net}_full",
            },
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out_path}")


def _emit_profile_csv(net: str, backend: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t_us = COARSE_PROFILE[net][backend]
    with open(out_path, "w") as f:
        f.write("dispatch_id,module_name,mean_time,mean_unit\n")
        f.write(f"0,{net}_full,{t_us:.4f},us\n")
    print(f"  wrote {out_path}  ({t_us:.1f} us)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--target",   default="qrb5165_v66")
    ap.add_argument("--basename-suffix", default="int8")
    args = ap.parse_args()

    for net in COARSE_PROFILE.keys():
        net_coarse = f"{net}_coarse"
        basename = f"{net_coarse}.{args.basename_suffix}"
        for hw in BACKENDS:
            graph_path = os.path.join(
                args.repo_root, "gen", "qnn_vmfb",
                net_coarse, args.target, hw, basename,
                f"{basename}_dispatch_graph.json",
            )
            _emit_dispatch_graph(net, graph_path)

            csv_path = os.path.join(
                args.repo_root, "gen", "profile",
                hw, args.target,
                net_coarse, basename, "topo_0",
                "results.csv",
            )
            _emit_profile_csv(net, hw, csv_path)


if __name__ == "__main__":
    main()
