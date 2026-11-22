"""Safety-boundary feedback for the dynamic walker: step rule and pushed-back check."""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("wsf", ROOT / "scripts" / "walker_safety_feedback.py")
wsf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wsf)


def test_steps_down_while_clean_then_returns_to_last_clean_value():
    s = wsf.new_state(1.5, 0.1, 1.0)
    s = wsf.next_state(s, 1.5, 0)
    assert (s["safety_boundary"], s["status"]) == (1.4, "searching")
    s = wsf.next_state(s, 1.4, 0)
    assert s["safety_boundary"] == pytest.approx(1.3)
    s = wsf.next_state(s, 1.3, 2)                 # pushed back -> last clean, stop
    assert (s["safety_boundary"], s["status"]) == (1.4, "converged")
    s = wsf.next_state(s, 1.4, 5)                 # converged: records, no longer moves
    assert s["safety_boundary"] == 1.4 and len(s["history"]) == 4


def test_stops_at_floor():
    s = wsf.new_state(1.1, 0.1, 1.0)
    s = wsf.next_state(s, 1.1, 0)
    assert s["safety_boundary"] == pytest.approx(1.0) and s["status"] == "searching"
    s = wsf.next_state(s, 1.0, 0)
    assert (s["safety_boundary"], s["status"]) == (1.0, "converged")


def test_raises_when_the_start_value_already_pushes_back():
    s = wsf.new_state(1.5, 0.1, 1.0)
    s = wsf.next_state(s, 1.5, 1)
    assert (s["safety_boundary"], s["status"]) == (1.6, "raising")
    s = wsf.next_state(s, 1.6, 1)
    assert s["safety_boundary"] == pytest.approx(1.7)
    s = wsf.next_state(s, 1.7, 0)
    assert (s["safety_boundary"], s["status"]) == (1.7, "converged")


def test_next_state_does_not_mutate_input():
    s = wsf.new_state()
    snap = json.dumps(s)
    wsf.next_state(s, 1.5, 0, {"run": "x"})
    assert json.dumps(s) == snap


def _trace(path: Path, rows: list[dict]) -> None:
    cols = ["entry_id", "network", "instance", "dispatch_id", "op", "name", "core_kind", "hart",
            "predicted_start_ms", "predicted_duration_ms", "worker_kind_idx",
            "actual_start_cycles", "actual_end_cycles", "ran_on_hart", "moved"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(dict(entry_id=i, op="x", core_kind="k", predicted_duration_ms=1.0,
                            worker_kind_idx=0, **r))


def test_pushed_back_detects_an_own_task_waiting_behind_a_moved_task(tmp_path):
    # Core 1 runs a moved task (n0 from core 0) from 0 to 3 ms. Its own task m0 is
    # ready at 1 ms (release) but starts at 3 ms: pushed back by 2 ms.
    # m1 (core 1, ready at 4 ms when m0 ends... we make it start on time) is not.
    graph = tmp_path / "g.json"
    graph.write_text(json.dumps({"ops": [{"name": "a", "dispatch_id": 0},
                                         {"name": "b", "dispatch_id": 1}]}))
    sched = {"dispatches": {
        "n0_dispatch_0": {"dependencies": []},
        "m0_dispatch_0": {"dependencies": []},
        "m0_dispatch_1": {"dependencies": ["m0_dispatch_0"]},
    }}
    (tmp_path / "s.json").write_text(json.dumps(sched))
    ms = 1_000_000
    _trace(tmp_path / "t.csv", [
        dict(network="n", instance=0, dispatch_id=0, name="a", hart=0, predicted_start_ms=0.0,
             actual_start_cycles=0, actual_end_cycles=3 * ms, ran_on_hart=1, moved=1),
        dict(network="m", instance=0, dispatch_id=0, name="a", hart=1, predicted_start_ms=1.0,
             actual_start_cycles=3 * ms, actual_end_cycles=4 * ms, ran_on_hart=1, moved=0),
        dict(network="m", instance=0, dispatch_id=1, name="b", hart=1, predicted_start_ms=2.0,
             actual_start_cycles=4 * ms + 10_000, actual_end_cycles=5 * ms, ran_on_hart=1, moved=0),
    ])
    out = wsf.pushed_back(str(tmp_path / "t.csv"), str(tmp_path / "s.json"),
                          {"n": str(graph), "m": str(graph)})
    assert out["pushed_back"] == 1 and out["moved"] == 1 and out["checked"] == 2
    assert out["examples"][0]["task"] == "m0_dispatch_0"
    assert out["examples"][0]["delay_us"] == pytest.approx(2000.0)


def test_walker_line_parses_run_output():
    w = wsf.walker_line("x\nxpurt: dynamic walker safety_boundary=1.300 moved=12 refused_safety=4\n")
    assert w == {"value": 1.3, "moved": 12, "refused": 4}
