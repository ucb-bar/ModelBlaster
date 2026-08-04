"""Side-by-side gantt: ROS baseline (B1 / B2) vs the XPURT-scheduled
QNN runtime.

ROS rows come from /tmp/ros_baseline/<scenario>_<node>.csv (one CSV per
node, columns seq, callback_start_us, exec_start_us, exec_end_us,
callback_end_us; yolov8n adds backbone_end_us). XPURT rows come from
the runtime's AGENTS_QNN_TRACE_BEGIN..END block in the run log.

The plot lays out two axes vertically with a shared x-range so a
reviewer can read across: "for the same time window, XPURT fits all 26
dispatches in 32 ms while ROS-B2 keeps the lane busy ~140% longer with
99.8% deadline misses on dronet."

For yolov8n, both panels split the inference into its two segments
(backbone + head) and draw a faint underlay span across both segments
so the visual reads as "one yolo, two graphs" rather than "two unrelated
DSP calls."

Color scheme matches plot_runtime_trace.py (darkened xpurt palette) so
this looks consistent with the other gantts in the writeup.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Reuse extract_trace_csv + KIND_TO_CMAP from plot_runtime_trace so this
# comparison plot uses the same per-network colormap families as the
# runtime-walk gantts (Blues for dronet, Greens for mlp_control, etc).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from plot_runtime_trace import KIND_TO_CMAP, extract_trace_csv

# ROS baseline lanes — one row per backend the network ran on. Hardcoded
# matchup matches launch/all_nodes.py.
B1_LANES = {"dronet": "DSP", "mlp_control": "DSP",  "yolov8n": "DSP"}
B2_LANES = {"dronet": "HTA", "mlp_control": "CPU",  "yolov8n": "DSP"}


def _family_color(network: str, shade: float = 0.7):
    """Mid-shade from the network's family colormap. shade=0.7 gives a
    saturated tone that reads cleanly against white grid + dark text."""
    import matplotlib.pyplot as plt
    cmap_name = KIND_TO_CMAP.get(network, "Greys")
    return tuple(plt.get_cmap(cmap_name)(shade)[:3])


# Lazy-built dict of (network, shade) → RGB so callers can use it like
# the previous NETWORK_COLOR map. yolov8n's "yolov8" key in KIND_TO_CMAP
# is aliased.
class _NetworkColorMap:
    def __getitem__(self, k):
        return _family_color(k)
    def get(self, k, default=None):
        return _family_color(k) if k in KIND_TO_CMAP else default
    def __iter__(self):
        # Order the legend reads in: dronet, mlp_control, yolov8n
        return iter(["dronet", "mlp_control", "yolov8n"])

NETWORK_COLOR = _NetworkColorMap()


def _read_ros_csv(path: str) -> list[dict]:
    with open(path) as f:
        rows = [
            {k: float(v) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]
    return rows


def _ros_window_ms(rows: list[dict]) -> float:
    if not rows: return 0.0
    return max(r["callback_end_us"] for r in rows) / 1000.0


def _ros_deadline_misses(rows: list[dict], period_ms: float) -> tuple[int, int]:
    misses = 0
    for r in rows:
        seq = int(r["seq"])
        # ROS wall_timer fires the first callback at t0 + period — so seq
        # N runs during [(N+1)*T, (N+2)*T] in t0-relative time.
        deadline_us = (seq + 2) * period_ms * 1000
        if r["callback_end_us"] > deadline_us:
            misses += 1
    return misses, len(rows)


def render(ros_dir: str, scenario: str, milp_log: str, out_path: str,
            window_ms: float = 2200.0,
            snapshot_yolo_seq: int | None = None,
            snapshot_pad_ms: float = 60.0):
    """`snapshot_yolo_seq`: when set, zoom the ROS panel around the
    callback_start of yolov8n's seq=N call, with `snapshot_pad_ms` of
    context on each side. The MILP panel shifts its origin so both
    panels span the same window width, MILP starting at 0. This is the
    apples-to-apples figure for the paper: same x-width, same workload,
    one yolo frame visible in each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    lanes_map = B1_LANES if scenario == "B1" else B2_LANES
    nodes = [
        ("dronet",      f"{ros_dir}/{scenario}_dronet.csv",  5.0),
        ("mlp_control", f"{ros_dir}/{scenario}_mlp.csv",     2.0),
        ("yolov8n",     f"{ros_dir}/{scenario}_yolov8.csv",  33.33),
    ]
    ros_rows = {n: _read_ros_csv(p) for n, p, _ in nodes}

    # Collect ALL distinct lanes seen on either side so the y-axis matches.
    lane_set = ["HTA", "DSP", "CPU"]
    lane_y = {l: i for i, l in enumerate(lane_set)}

    # In snapshot mode both panels render with t=0 = "start of the
    # snapshot window". For ROS that means we subtract ros_x0 from every
    # bar's coordinates as we draw it; for XPURT t=0 is already the
    # natural origin. The two panels share x so a reviewer can read
    # vertically across "for the same ms of time, what's running on
    # each lane on each runtime".
    is_snap = snapshot_yolo_seq is not None
    fig, axes = plt.subplots(2, 1, figsize=(13 if is_snap else 15, 5),
                              sharex=True,
                              gridspec_kw={"hspace": 0.25})
    ax_ros, ax_milp = axes

    # Compute the ROS x-range. In snapshot mode the window is the chosen
    # yolo frame's exact callback span (so each panel shows one yolo
    # makespan side-by-side), with `snapshot_pad_ms` of margin on each
    # side. The MILP panel shifts to start at t=0 and shows the same
    # x-width — its full natural run is ~32ms which lines up with one
    # yolo makespan, making the comparison "one yolo's worth of work,
    # ROS vs MILP" direct.
    if is_snap:
        yolo_rows = ros_rows.get("yolov8n", [])
        match = [r for r in yolo_rows if int(r["seq"]) == snapshot_yolo_seq]
        if not match:
            raise SystemExit(f"snapshot: yolov8n seq={snapshot_yolo_seq} not found "
                              f"in {ros_dir}/{scenario}_yolov8.csv "
                              f"(have {[int(r['seq']) for r in yolo_rows[:5]]}...)")
        yolo_start = match[0]["callback_start_us"] / 1000.0
        yolo_end   = match[0]["callback_end_us"]   / 1000.0
        center_ms  = (yolo_start + yolo_end) / 2.0
        ros_x0 = yolo_start - snapshot_pad_ms
        ros_x1 = yolo_end   + snapshot_pad_ms
        view_width = ros_x1 - ros_x0
    else:
        ros_x0, ros_x1 = 0.0, window_ms
        view_width = window_ms

    # Plot ROS callbacks. cb_start/cb_end are in microseconds since each
    # node's t0 — roughly aligned across nodes (each node calls now() at
    # construction, all spawn within ~ms of each other under launch).
    # In snapshot mode we shift bar coords by ros_x0 so the snapshot's
    # left edge sits at displayed t=0, lining up with XPURT's natural
    # t=0 origin in the bottom panel.
    # For yolov8 we split each callback into its two graphExecute spans
    # (backbone + head) so the segmentation looks the same as the XPURT
    # panel. A faint underlay span connects them: same yolo, two graphs.
    ros_shift = ros_x0 if is_snap else 0.0
    for net, _, _ in nodes:
        rows = ros_rows[net]
        if not rows: continue
        ly = lane_y[lanes_map[net]]
        color = NETWORK_COLOR[net]
        for r in rows:
            x0 = r["callback_start_us"] / 1000.0
            x1 = r["callback_end_us"] / 1000.0
            if x1 < ros_x0 or x0 > ros_x1: continue
            x0 -= ros_shift
            x1 -= ros_shift
            if net == "yolov8n" and "backbone_end_us" in r:
                # Faint underlay across the whole inference (backbone +
                # head + any sync between them) — visually unifies the
                # two solid sub-bars as one logical yolo call.
                ax_ros.barh(ly, max(x1 - x0, 0.05), left=x0, height=0.78,
                             color=color, alpha=0.20, edgecolor="none")
                bb_end = r["backbone_end_us"] / 1000.0 - ros_shift
                # backbone solid box.
                ax_ros.barh(ly, max(bb_end - x0, 0.05), left=x0, height=0.7,
                             color=color, edgecolor="black", linewidth=0.3)
                # head solid box.
                ax_ros.barh(ly, max(x1 - bb_end, 0.05), left=bb_end, height=0.7,
                             color=color, edgecolor="black", linewidth=0.3)
            else:
                ax_ros.barh(ly, max(x1 - x0, 0.05), left=x0, height=0.7,
                             color=color, edgecolor="black", linewidth=0.2)

    # XPURT (formerly "MILP") rows. Same underlay treatment for yolov8 —
    # the two segments (backbone seg100 + head seg101) of the same
    # instance get a faint background spanning their full duration.
    milp_rows = extract_trace_csv(milp_log)
    yolov8_by_inst: dict[int, list[dict]] = {}
    for r in milp_rows:
        if r["network"] == "yolov8n" and r["actual_end_ms"] >= 0:
            yolov8_by_inst.setdefault(r["instance"], []).append(r)
    # Underlay first so solid bars draw on top.
    for inst, segs in yolov8_by_inst.items():
        if len(segs) < 1: continue
        x0 = min(s["actual_start_ms"] for s in segs)
        x1 = max(s["actual_end_ms"]   for s in segs)
        # All segs of one yolo instance currently land on the same
        # actual_backend (DSP). Use that lane's y.
        ly = lane_y.get(segs[0]["actual_backend"], 0)
        color = NETWORK_COLOR["yolov8n"]
        ax_milp.barh(ly, max(x1 - x0, 0.05), left=x0, height=0.78,
                      color=color, alpha=0.20, edgecolor="none")
    # Solid per-segment bars on top.
    for r in milp_rows:
        if r["actual_end_ms"] < 0: continue
        ly = lane_y.get(r.get("actual_backend", ""), 0)
        net = r["network"]
        color = NETWORK_COLOR.get(net, (0.5, 0.5, 0.5))
        x0 = r["actual_start_ms"]
        x1 = r["actual_end_ms"]
        ax_milp.barh(ly, max(x1 - x0, 0.05), left=x0, height=0.7,
                      color=color, edgecolor="black", linewidth=0.3)

    for ax in axes:
        ax.set_yticks(list(lane_y.values()))
        ax.set_yticklabels(lane_set)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    if is_snap:
        ax_ros.set_xlim(0, view_width)
        ax_milp.set_xlim(0, view_width)
    else:
        ax_ros.set_xlim(0, window_ms)
        ax_milp.set_xlim(0, window_ms)

    # Compute deadline-miss summary for the ROS panel title.
    miss_summary = []
    for net, _, period in nodes:
        m, n = _ros_deadline_misses(ros_rows[net], period)
        miss_summary.append(f"{net}={m}/{n}")
    ros_window = max(_ros_window_ms(rows) for rows in ros_rows.values())
    ax_ros.set_title(
        f"ROS {scenario} — {len(milp_rows)}-eq workload over {ros_window:.0f} ms,  "
        f"deadline misses: " + ", ".join(miss_summary),
        loc="left", fontsize=10)

    milp_window = max(r["actual_end_ms"] for r in milp_rows) if milp_rows else 0
    milp_misses = 0; milp_total = 0
    periods = {"dronet": 5.0, "mlp_control": 2.0}
    for r in milp_rows:
        if r["network"] not in periods: continue
        T = periods[r["network"]]
        deadline = (r["instance"] + 1) * T
        milp_total += 1
        if r["actual_end_ms"] > deadline:
            milp_misses += 1
    ax_milp.set_title(
        f"XPURT scheduled runtime — {len(milp_rows)} dispatches over {milp_window:.0f} ms,  "
        f"deadline misses: {milp_misses}/{milp_total} periodic",
        loc="left", fontsize=10)

    ax_milp.set_xlabel("time (ms)")
    ax_ros.set_ylabel("backend lane")
    ax_milp.set_ylabel("backend lane")

    handles = [mpatches.Patch(color=NETWORK_COLOR[k], label=k) for k in NETWORK_COLOR]
    fig.legend(handles=handles, loc="upper center", ncol=3,
                bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    if is_snap:
        suptitle = (f"ROS {scenario} vs XPURT — one yolov8 makespan ({view_width:.0f} ms), "
                    f"ROS centered on yolov8n seq={snapshot_yolo_seq} (t≈{center_ms:.0f}ms abs); "
                    "XPURT from t=0")
    else:
        suptitle = f"ROS {scenario} vs XPURT — same workload (dronet@5ms + mlp@2ms + yolov8@33ms)"
    fig.suptitle(suptitle, y=1.06, fontsize=11)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ros-dir",  default="/tmp/ros_baseline")
    ap.add_argument("--scenario", required=True, choices=["B1", "B2"])
    ap.add_argument("--milp-log", default="/tmp/3way_warm_hw_run.log")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--window-ms", type=float, default=2200.0)
    ap.add_argument("--snapshot-yolo-seq", type=int, default=None,
                    help="If set, render a focused snapshot centered on "
                         "the yolov8n callback with this seq, with "
                         "snapshot-pad-ms of context on each side. The "
                         "MILP panel is drawn from t=0 with the same x-width.")
    ap.add_argument("--snapshot-pad-ms", type=float, default=2.0,
                    help="Margin in ms on each side of the chosen yolo "
                         "callback's [cb_start, cb_end]. Default 2 ms — "
                         "the snapshot width is then ~yolo-exec + 4 ms, "
                         "i.e. one yolo makespan with tiny breathing room.")
    args = ap.parse_args()
    render(args.ros_dir, args.scenario, args.milp_log, args.out,
           window_ms=args.window_ms,
           snapshot_yolo_seq=args.snapshot_yolo_seq,
           snapshot_pad_ms=args.snapshot_pad_ms)


if __name__ == "__main__":
    main()
