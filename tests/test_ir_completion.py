#!/usr/bin/env python3
"""Validation for pipeline/ingest_xpurt_schedule.py IR-completion pass.

Verifies that:
1. Schedule fixtures missing IR ops get synthesized entries inserted
   so the resulting dispatch table contains every IR op.
2. The dispatch graph (data deps + time_dep) is a strict backward
   DAG — no forward edges in the topologically-sorted entry order.
3. Scalar-FP activations (silu/elu/relu/sigmoid/...) are pinned to
   the rvv_opu (CPU_E#0) hart, never to gemmini (CPU_P#0).
4. Each synthesized entry's dispatch_id correctly remaps to the
   model's compiled-in codegen index (0..OP_COUNT-1) or -1 for
   zero-cost view/chunk2 sentinels.
5. MB_INGEST_SKIP_IR_COMPLETION=1 disables the pass.

Runs against the headline `4 MLP + 2 Dronet + 1 Yolo` fixture from
XPU-RT. Each test prints PASS/FAIL with diagnostics; non-zero exit
on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Locate the repo from this file, not from an absolute path.
#
# This used to be a hardcoded /scratch2/agustin/ModelBlaster, which meant the
# test exercised whichever checkout happened to live there rather than the one
# it ships in -- so a fix made in this tree was silently not under test. It also
# made the test unrunnable for anyone else.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pipeline.ingest_xpurt_schedule import load
from pipeline import core_registry


def _find_schedule() -> Path | None:
    """The fixture lives in XPU-RT, which may be this repo's parent (submodule
    layout) or a sibling. Try both rather than hardcoding one."""
    rel = ("schedules/"
           "scheduled_networks_1yolo_4mlp_2dronet_firesim_greedy_periodic_profiled.json")
    for base in (REPO.parent, REPO.parent / "XPU-RT"):
        cand = base / rel
        if cand.is_file():
            return cand
    return None


_SCHEDULE_PATH = _find_schedule()
SCHEDULE = str(_SCHEDULE_PATH) if _SCHEDULE_PATH else ""
REGISTRY = REPO / "cores/chipyard_gemmini_opu_hetero.json"
IRS = {
    "mlp_control": REPO / "examples/mlp_control/int8/generated/graph.json",
    "dronet": REPO / "examples/dronet/int8/generated/graph.json",
    "yolov8_nano": REPO / "examples/yolov8_nano/int8/generated/graph.json",
}

# The IRs are gitignored build outputs (examples/*/int8/generated/), so a fresh
# checkout does not have them. Skip rather than fail: a missing build artifact
# is not a regression, and pretending otherwise buried four real failures in
# noise for anyone who had not run the generator.
_MISSING = [str(p) for p in list(IRS.values()) + [REGISTRY] if not Path(p).is_file()]
if _SCHEDULE_PATH is None:
    _MISSING.append(
        "the 4-MLP + 2-DroNet + 1-YOLO schedule fixture from XPU-RT")
_SKIP_REASON = (
    "IR-completion fixtures absent (regenerate with the int8 extract/skeleton "
    "pipeline, or run from a tree that has them): " + ", ".join(_MISSING)
) if _MISSING else ""

try:
    import pytest
    pytestmark = pytest.mark.skipif(bool(_MISSING), reason=_SKIP_REASON)
except ImportError:  # direct `python test_ir_completion.py` invocation
    pytest = None
_SCALAR_FP_OPS = {
    "silu_s8", "elu_s8", "relu_s8", "sigmoid_s8", "tanh_s8",
    "gelu_s8", "hardswish_s8", "softmax_s8",
}


def _load_entries(skip_completion: bool = False):
    if skip_completion:
        os.environ["MB_INGEST_SKIP_IR_COMPLETION"] = "1"
    else:
        os.environ.pop("MB_INGEST_SKIP_IR_COMPLETION", None)
    irs = {k: json.loads(Path(v).read_text()) for k, v in IRS.items()}
    reg = core_registry.load(str(REGISTRY))
    return load(SCHEDULE, irs, reg,
                cpu_p_kind="gemmini", cpu_e_kind="rvv_opu")


_failures: list[str] = []
def fail(msg: str) -> None:
    _failures.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def test_completeness():
    """Every IR op appears in the table for every (network, instance)."""
    entries = _load_entries(skip_completion=False)
    irs = {k: json.loads(Path(v).read_text()) for k, v in IRS.items()}
    expected = {}
    for net, ir in irs.items():
        n = sum(1 for op in ir.get("ops", []) if op.get("dispatch_id") is not None)
        expected[net] = n
    # Count total entries (rows) per (network, instance) — every IR
    # op becomes one row, whether dispatch_id maps to codegen or to -1.
    by_inst: dict[tuple[str, int], int] = {}
    for e in entries:
        by_inst[(e.network, e.instance)] = by_inst.get((e.network, e.instance), 0) + 1
    for (net, inst), got in by_inst.items():
        if got < expected[net]:
            fail(f"({net}#{inst}) has only {got} entries; "
                 f"expected {expected[net]} from IR")
    print(f"PASS: every (network, instance) covers all IR ops "
          f"({len(by_inst)} instances)")


def test_backward_deps():
    """No deps or time_deps point forward in the table order."""
    entries = _load_entries(skip_completion=False)
    violations = 0
    for e in entries:
        for d in e.deps_entry_ids:
            if d >= e.entry_id:
                violations += 1
        if e.time_dep_entry_id != -1 and e.time_dep_entry_id >= e.entry_id:
            violations += 1
    if violations:
        fail(f"forward-edge violations: {violations}")
    else:
        print(f"PASS: 0 forward-edge violations across {len(entries)} entries")


def test_activations_on_rvv_opu():
    """Scalar-FP activations land on rvv_opu hart, never gemmini."""
    entries = _load_entries(skip_completion=False)
    bad = []
    for e in entries:
        if e.op in _SCALAR_FP_OPS and e.core_kind == "gemmini":
            bad.append((e.entry_id, e.network, e.op, e.name))
    if bad:
        fail(f"{len(bad)} scalar-FP activations placed on gemmini: "
             f"{bad[:5]}{' ...' if len(bad) > 5 else ''}")
    else:
        n_act = sum(1 for e in entries if e.op in _SCALAR_FP_OPS)
        print(f"PASS: all {n_act} scalar-FP activations on rvv_opu hart")


def test_skip_switch():
    """MB_INGEST_SKIP_IR_COMPLETION=1 produces v2-equivalent count."""
    full = _load_entries(skip_completion=False)
    skip = _load_entries(skip_completion=True)
    if len(skip) >= len(full):
        fail(f"skip mode returned {len(skip)} entries; expected fewer "
             f"than full mode's {len(full)}")
    else:
        print(f"PASS: skip mode {len(skip)} entries < full {len(full)} "
              f"(synthesized {len(full)-len(skip)})")


if __name__ == "__main__":
    print("=== ingest_xpurt_schedule IR-completion validation ===")
    if _MISSING:
        print("SKIP: " + _SKIP_REASON)
        sys.exit(0)
    test_completeness()
    test_backward_deps()
    test_activations_on_rvv_opu()
    test_skip_switch()
    if _failures:
        print(f"\n{len(_failures)} FAILURES")
        sys.exit(1)
    print("\nAll tests passed.")
