#!/usr/bin/env python3
"""Safety-boundary feedback for the dynamic walker (--dynamic-walker).

The dynamic walker lets an idle core take a held task only if
``safety_boundary x planned duration`` ends before that core's own next task is
due. A larger value moves fewer tasks; a smaller one moves more but risks a
moved task PUSHING BACK the receiving core's own work. This script reads one
board run and picks the value for the next run:

  * nothing pushed back  -> lower the value by --step (not below --floor)
  * something pushed back -> go back to the last value that was clean, and stop
  * pushed back before any clean value was seen -> raise by --step and retry

PUSHED BACK means: one of a core's own tasks became ready while that core was
running a moved task, and it started more than --threshold-us after it became
ready. "Ready" = its data dependencies and its ordering edge (the task planned
before it on that core) have finished, and, for an instance's first op, its
release time has passed.

State lives in a small JSON file the board run reads
(``MB_XPURT_SAFETY_FILE=<file>`` in run_xpurt_k1.sh)::

    {"safety_boundary": 1.3, "status": "searching", "last_clean": 1.4,
     "step": 0.1, "floor": 1.0, "history": [{"run": "...", "value": 1.5,
     "pushed_back": 0, "moved": 908, ...}, ...]}

usage:
  walker_safety_feedback.py --state walker_safety.json --run-label r1 \\
      --stdout <N>_stdout.txt --trace <N>_trace.csv --schedule <N>.json \\
      --graph mlp_control=<graph.json> --graph dronet=... --graph yolov8_nano=...
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DEFAULT_START = 1.5


# ---- decision rule --------------------------------------------------------

def new_state(start: float = DEFAULT_START, step: float = 0.1, floor: float = 1.0) -> dict:
    return {"safety_boundary": start, "status": "searching", "last_clean": None,
            "step": step, "floor": floor, "history": []}


def next_state(state: dict, value: float, pushed_back: int, record: dict | None = None) -> dict:
    """Fold one run (made at `value`) into a NEW state. Pure function."""
    s = json.loads(json.dumps(state))
    s["history"] = list(s.get("history", [])) + [dict(record or {}, value=value,
                                                      pushed_back=pushed_back)]
    step, floor = float(s["step"]), float(s["floor"])
    if s["status"] == "converged":
        return s                                    # keep recording, stop moving
    if pushed_back == 0:
        s["last_clean"] = value
        if value <= floor + 1e-9:
            s["safety_boundary"], s["status"] = floor, "converged"
        elif s["status"] == "raising":              # first clean value on the way up
            s["safety_boundary"], s["status"] = value, "converged"
        else:
            s["safety_boundary"] = round(max(floor, value - step), 6)
    elif s["last_clean"] is not None:
        s["safety_boundary"], s["status"] = s["last_clean"], "converged"
    else:                                           # never clean yet: back off upward
        s["safety_boundary"], s["status"] = round(value + step, 6), "raising"
    return s


# ---- trace analysis -------------------------------------------------------

def _name_to_id(graph_path: str) -> dict[str, int]:
    g = json.load(open(graph_path))
    return {o["name"]: int(o["dispatch_id"]) for o in g["ops"] if o.get("dispatch_id") is not None}


def pushed_back(trace_csv: str, schedule_json: str, graphs: dict[str, str],
                ticks_per_ms: float = 1e6, threshold_us: float = 50.0) -> dict:
    """Count own tasks pushed back by a moved task (see module docstring)."""
    sched = json.load(open(schedule_json))["dispatches"]
    ids = {net: _name_to_id(p) for net, p in graphs.items()}
    rows = list(csv.DictReader(open(trace_csv)))
    if not rows or "ran_on_hart" not in rows[0]:
        raise SystemExit(f"{trace_csv}: no ran_on_hart column -- not a dynamic-walker trace")
    R = {}
    for t in rows:
        key = f"{t['network']}{t['instance']}_dispatch_{ids[t['network']][t['name']]}"
        R[key] = dict(s=int(t["actual_start_cycles"]) / ticks_per_ms,
                      f=int(t["actual_end_cycles"]) / ticks_per_ms,
                      plan=int(t["hart"]), ran=int(t["ran_on_hart"]), moved=t["moved"] == "1",
                      rel=float(t["predicted_start_ms"]), first=t["dispatch_id"] == "0",
                      zero=int(t["dispatch_id"]) < 0)
    busy_moved: dict[int, list[tuple[float, float]]] = {}
    for r in R.values():
        if r["moved"]:
            busy_moved.setdefault(r["ran"], []).append((r["s"], r["f"]))
    thr = threshold_us / 1000.0
    n_checked = 0
    hits: list[dict] = []
    for key, r in R.items():
        if r["moved"] or r["zero"]:
            continue
        ent = sched[key]
        ready = [R[d]["f"] for d in ent.get("dependencies", []) if d in R]
        if ent.get("time_dependency") in R:
            ready.append(R[ent["time_dependency"]]["f"])
        if r["first"]:
            ready.append(r["rel"])
        t_ready = max(ready, default=0.0)
        n_checked += 1
        if r["s"] - t_ready > thr and any(a <= t_ready < b for a, b in busy_moved.get(r["plan"], [])):
            hits.append({"task": key, "core": r["plan"], "delay_us": round((r["s"] - t_ready) * 1000, 1)})
    return {"pushed_back": len(hits), "checked": n_checked,
            "moved": sum(r["moved"] for r in R.values()), "examples": hits[:10]}


def walker_line(stdout_text: str) -> dict:
    m = re.search(r"dynamic walker safety_boundary=([\d.]+) moved=(\d+) refused_safety=(\d+)", stdout_text)
    if not m:
        raise SystemExit("no 'dynamic walker' line in the run output -- was it a dynamic run?")
    return {"value": float(m.group(1)), "moved": int(m.group(2)), "refused": int(m.group(3))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="walker_safety.json (created if missing)")
    ap.add_argument("--stdout", required=True, help="board run output (<N>_stdout.txt)")
    ap.add_argument("--trace", required=True, help="<N>_trace.csv")
    ap.add_argument("--schedule", required=True)
    ap.add_argument("--graph", action="append", required=True, metavar="NET=GRAPH_JSON")
    ap.add_argument("--run-label", default="")
    ap.add_argument("--start", type=float, default=DEFAULT_START)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--floor", type=float, default=1.0)
    ap.add_argument("--threshold-us", type=float, default=50.0)
    ap.add_argument("--ticks-per-ms", type=float, default=1e6, help="trace clock (ns on hosted Linux)")
    a = ap.parse_args(argv)

    p = Path(a.state)
    state = json.loads(p.read_text()) if p.exists() else new_state(a.start, a.step, a.floor)
    w = walker_line(Path(a.stdout).read_text())
    if abs(w["value"] - float(state["safety_boundary"])) > 1e-6 and state["status"] != "converged":
        print(f"note: run used {w['value']} but state says {state['safety_boundary']}; using the run's value")
    pb = pushed_back(a.trace, a.schedule, dict(g.split("=", 1) for g in a.graph),
                     a.ticks_per_ms, a.threshold_us)
    rec = {"run": a.run_label, "moved": w["moved"], "refused": w["refused"],
           "checked": pb["checked"], "examples": pb["examples"][:3]}
    new = next_state(state, w["value"], pb["pushed_back"], rec)
    p.write_text(json.dumps(new, indent=1) + "\n")
    print(f"[safety] run {a.run_label or '?'} at {w['value']}: moved {w['moved']}, pushed back "
          f"{pb['pushed_back']} of {pb['checked']} own tasks -> next {new['safety_boundary']} "
          f"({new['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
