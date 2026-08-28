"""Unit tests for pipeline/apply_fusion_hint.py.

These run on hand-built IR dicts (no torch / no codegen) so they're
fast and don't pull in any heavy deps. The fixtures mirror the shape
of `examples/<model>/<quant>/generated/graph.json` (only the fields
the rewrite actually inspects).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.apply_fusion_hint import (
    FusionHintError,
    apply_hint,
)


def _op(did, op, name, inputs, outputs, depends_on=None, **kw):
    base = {
        "name": name,
        "op": op,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "dispatch_id": did,
        "hardware_target": kw.pop("hardware_target", "any"),
        "depends_on": list(depends_on or []),
    }
    base.update(kw)
    return base


def _mlp3_graph():
    """3-op linear/elu/linear chain: x -> mlp_0 -> mlp_1 -> mlp_2."""
    return {
        "name": "tiny",
        "version": 1,
        "input": {"tensor": "x"},
        "output": {"tensor": "mlp_2", "tensors": ["mlp_2"]},
        "tensors": {
            "x": {"shape": [1, 8], "dtype": "i8"},
            "mlp_0": {"shape": [1, 16], "dtype": "i8"},
            "mlp_1": {"shape": [1, 16], "dtype": "i8"},
            "mlp_2": {"shape": [1, 4], "dtype": "i8"},
        },
        "ops": [
            _op(0, "linear_s8", "lin0", ["x"], ["mlp_0"]),
            _op(1, "elu_s8", "elu0", ["mlp_0"], ["mlp_1"], depends_on=[0]),
            _op(2, "linear_s8", "lin1", ["mlp_1"], ["mlp_2"], depends_on=[1]),
        ],
    }


class FuseTwoOpChainTest(unittest.TestCase):
    """A 2-op fuse_group should collapse to one fused op + the trailing op."""

    def test_basic_shape(self):
        g = _mlp3_graph()
        out = apply_hint(g, [[0, 1]])
        self.assertEqual(len(out["ops"]), 2)

        fused, tail = out["ops"]
        # fused op — when sub_ops are exactly [linear_s8, elu_s8] the
        # rewrite now emits the registered KernelSpec key
        # `linear_s8_elu_s8` (Phase 1d) instead of the synthetic chain
        # name; that routes codegen through the registered kernel
        # (with LLM-codegen seeds) rather than the chained-call fallback.
        self.assertEqual(fused["fused_from"], [0, 1])
        self.assertEqual(fused["op"], "linear_s8_elu_s8")
        self.assertEqual(fused["dispatch_id"], 0)
        self.assertEqual(fused["depends_on"], [])
        self.assertEqual(fused["inputs"], ["x"])
        self.assertEqual(fused["outputs"], ["mlp_1"])
        # mlp_0 produced inside the chain, consumed inside — stack-local
        self.assertEqual(fused["internal_tensors"], ["mlp_0"])
        # sub_ops verbatim
        self.assertEqual([s["op"] for s in fused["sub_ops"]],
                         ["linear_s8", "elu_s8"])

        # trailing op: dispatch_id shifted 2 -> 1, depends_on rewired 1 -> 0
        self.assertEqual(tail["op"], "linear_s8")
        self.assertEqual(tail["dispatch_id"], 1)
        self.assertEqual(tail["depends_on"], [0])

    def test_input_unmutated(self):
        g = _mlp3_graph()
        before = [op["dispatch_id"] for op in g["ops"]]
        _ = apply_hint(g, [[0, 1]])
        after = [op["dispatch_id"] for op in g["ops"]]
        self.assertEqual(before, after)


class FuseFullChainTest(unittest.TestCase):
    """A fuse_group covering the entire chain produces one fused op."""

    def test_all_three(self):
        g = _mlp3_graph()
        out = apply_hint(g, [[0, 1, 2]])
        self.assertEqual(len(out["ops"]), 1)
        fused = out["ops"][0]
        self.assertEqual(fused["inputs"], ["x"])
        # mlp_2 is the model output → must stay in outputs even though
        # no downstream op consumes it inside this graph.
        self.assertEqual(fused["outputs"], ["mlp_2"])
        self.assertEqual(set(fused["internal_tensors"]), {"mlp_0", "mlp_1"})
        self.assertEqual(fused["depends_on"], [])


class FuseMultipleGroupsTest(unittest.TestCase):
    """Two disjoint groups should produce two fused ops."""

    def test_two_pairs(self):
        # 5-op chain: pair up [0,1] and [3,4], leave op 2 alone.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t0", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t0"]),
                _op(1, "elu_s8", "b", ["t0"], ["t1"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t1"], ["t2"], depends_on=[1]),
                _op(3, "elu_s8", "d", ["t2"], ["t3"], depends_on=[2]),
                _op(4, "linear_s8", "e", ["t3"], ["t4"], depends_on=[3]),
            ],
        }
        out = apply_hint(g, [[0, 1], [3, 4]])
        self.assertEqual(len(out["ops"]), 3)

        fused_a, mid, fused_b = out["ops"]
        self.assertEqual(fused_a["dispatch_id"], 0)
        self.assertEqual(fused_a["fused_from"], [0, 1])
        self.assertEqual(fused_a["outputs"], ["t1"])
        self.assertEqual(fused_a["depends_on"], [])

        self.assertEqual(mid["dispatch_id"], 1)
        self.assertEqual(mid["op"], "linear_s8")
        self.assertEqual(mid["depends_on"], [0])  # was [1] → fused_a

        self.assertEqual(fused_b["dispatch_id"], 2)
        self.assertEqual(fused_b["fused_from"], [3, 4])
        self.assertEqual(fused_b["depends_on"], [1])  # was [2] → mid


class FuseRejectsTest(unittest.TestCase):

    def test_unknown_id(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 99]])

    def test_duplicate(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 0, 1]])

    def test_out_of_order(self):
        # 1 depends on 0; [1, 0] is not topo-sorted.
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[1, 0]])

    def test_overlapping_groups(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 1], [1, 2]])

    def test_empty_group(self):
        g = _mlp3_graph()
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[]])


class FuseEmptyHintTest(unittest.TestCase):
    """An empty fuse_groups list is a no-op (copy through)."""

    def test_passthrough(self):
        g = _mlp3_graph()
        out = apply_hint(g, [])
        self.assertEqual(len(out["ops"]), 3)
        self.assertEqual([o["dispatch_id"] for o in out["ops"]], [0, 1, 2])
        self.assertEqual([o["op"] for o in out["ops"]],
                         ["linear_s8", "elu_s8", "linear_s8"])


class FuseBranchingOutputTest(unittest.TestCase):
    """If a group member's output is consumed by an op OUTSIDE the group
    AND the group's tail, the tensor must escape as a fused output."""

    def test_intermediate_consumed_outside(self):
        # 4 ops: 0 -> 1 -> 2, but op 1's output is ALSO consumed by op 3.
        # Fuse [0, 1, 2]. The fused op's outputs must include both
        # `t1` (consumed by op 3, OUTSIDE) and `t2` (model output).
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t3", "tensors": ["t3"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t0", "t1", "t2", "t3"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t0"]),
                _op(1, "elu_s8", "b", ["t0"], ["t1"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t1"], ["t2"], depends_on=[1]),
                _op(3, "linear_s8", "d", ["t1"], ["t3"], depends_on=[1]),
            ],
        }
        out = apply_hint(g, [[0, 1, 2]])
        self.assertEqual(len(out["ops"]), 2)
        fused, tail = out["ops"]
        # `t1` escapes because op 3 consumes it; `t2` does not escape
        # (only the fused op produced it and no one outside consumes
        # it — but if `t2` is unreferenced downstream it's not in
        # outputs at all). Op `c` is the last writer of `t2`; since
        # tail consumes `t1` not `t2`, and `t2` isn't the model output
        # in this fixture, it's purely internal.
        self.assertIn("t1", fused["outputs"])
        self.assertNotIn("t2", fused["outputs"])
        self.assertEqual(set(fused["internal_tensors"]), {"t0", "t2"})
        # tail = original op 3
        self.assertEqual(tail["op"], "linear_s8")
        self.assertEqual(tail["depends_on"], [0])  # rewired


class FuseGroupDependencyClosureTest(unittest.TestCase):
    """A fuse_group must be closed under the dependency paths between
    its members.

    Regression for the silent-cycle bug: `_validate_fuse_group` only
    checked INTRA-group ordering, so a group that skipped over an
    external op was accepted and produced a graph where the fused op
    and the skipped op each listed the other in `depends_on`. Nothing
    downstream rejects that — it's a well-formed graph.json that simply
    deadlocks the scheduler.
    """

    def test_skipping_an_external_op_is_rejected(self):
        # a -> b -> c, fuse [0, 2]. Op 1 (`b`) is left outside but sits
        # on the only path from 0 to 2. Fusing 0 and 2 means the fused
        # op both produces b's input and consumes b's output.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t3", "tensors": ["t3"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t1", "t2", "t3"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t1"]),
                _op(1, "elu_s8", "b", ["t1"], ["t2"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t2"], ["t3"], depends_on=[1]),
            ],
        }
        with self.assertRaises(FusionHintError) as ctx:
            apply_hint(g, [[0, 2]])
        msg = str(ctx.exception)
        # The message must name the offending op so the advisor that
        # emitted the hint can be pointed at the right dispatch.
        self.assertIn("1", msg)
        self.assertIn("'b'", msg)

    def test_skipping_an_external_op_over_a_long_path_is_rejected(self):
        # Same defect, but the trapped ops are two hops away — proves
        # the check follows transitive paths, not just direct edges.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t1"]),
                _op(1, "elu_s8", "b", ["t1"], ["t2"], depends_on=[0]),
                _op(2, "elu_s8", "c", ["t2"], ["t3"], depends_on=[1]),
                _op(3, "linear_s8", "d", ["t3"], ["t4"], depends_on=[2]),
            ],
        }
        with self.assertRaises(FusionHintError):
            apply_hint(g, [[0, 3]])

    def test_non_contiguous_ids_on_parallel_branches_are_accepted(self):
        # Guards against "fix" the cheap wrong way (rejecting any group
        # whose dispatch_ids aren't contiguous). Here 0 and 2 have a gap
        # in id space, but op 1 is a SIBLING branch, not an op between
        # them: it descends from 0 and never reaches 2. Fusing [0, 2] is
        # sound and must stay allowed.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t1"]),
                _op(1, "elu_s8", "b", ["t1"], ["t2"], depends_on=[0]),
                _op(2, "elu_s8", "c", ["t1"], ["t3"], depends_on=[0]),
                _op(3, "linear_s8", "d", ["t2", "t3"], ["t4"],
                    depends_on=[1, 2]),
            ],
        }
        out = apply_hint(g, [[0, 2]])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 3)
        fused = ops[0]
        self.assertEqual(fused["fused_from"], [0, 2])
        # And the result is genuinely acyclic — no op lists a dependency
        # that lists it back.
        deps = {o["dispatch_id"]: set(o["depends_on"]) for o in ops}
        for did, ds in deps.items():
            for d in ds:
                self.assertNotIn(did, deps[d],
                                 f"cycle between {did} and {d}")

    def test_disconnected_ops_are_accepted(self):
        # Two independent chains; fusing one op from each is legal (the
        # scheduler is free to run them as one dispatch) because no op
        # lies on a path between them — there is no path at all.
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t2", "t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "y", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t1"]),
                _op(1, "elu_s8", "b", ["t1"], ["t2"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["y"], ["t3"]),
                _op(3, "elu_s8", "d", ["t3"], ["t4"], depends_on=[2]),
            ],
        }
        out = apply_hint(g, [[0, 2]])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 3)
        self.assertEqual(ops[0]["fused_from"], [0, 2])

    def test_adjacent_group_still_accepted(self):
        # The common case must not regress: a contiguous chain has no
        # external op between its members.
        out = apply_hint(_mlp3_graph(), [[0, 1, 2]])
        ops = [o for o in out["ops"] if o.get("dispatch_id") is not None]
        self.assertEqual(len(ops), 1)


class IdRemapTest(unittest.TestCase):
    """`id_remap` must let a consumer recover any op's new identity.

    Regression for the dropped-mapping bug: ids are reassigned
    contiguously, so fusing ops 0+1 shifts ops 2 and 3 down by one even
    though they were untouched. Any artifact keyed on dispatch_id (the
    profile CSV, the cost DB, SchedulerReport rows, Gantt labels) then
    joins against the wrong op.
    """

    def test_untouched_op_identity_is_recoverable(self):
        g = {
            "name": "tiny",
            "input": {"tensor": "x"},
            "output": {"tensor": "t4", "tensors": ["t4"]},
            "tensors": {n: {"shape": [1, 4], "dtype": "i8"}
                        for n in ["x", "t1", "t2", "t3", "t4"]},
            "ops": [
                _op(0, "linear_s8", "a", ["x"], ["t1"]),
                _op(1, "elu_s8", "b", ["t1"], ["t2"], depends_on=[0]),
                _op(2, "linear_s8", "c", ["t2"], ["t3"], depends_on=[1]),
                _op(3, "elu_s8", "d", ["t3"], ["t4"], depends_on=[2]),
            ],
        }
        out = apply_hint(g, [[0, 1]])
        remap = out["id_remap"]
        by_new = {o["dispatch_id"]: o for o in out["ops"]
                  if o.get("dispatch_id") is not None}
        # Op `c` was id 2 and was NOT fused, yet its id changed.
        self.assertEqual(by_new[remap["2"]]["name"], "c")
        self.assertEqual(by_new[remap["3"]]["name"], "d")
        self.assertNotEqual(remap["2"], 2)

    def test_remap_covers_every_input_id_including_fused_members(self):
        g = _mlp3_graph()
        out = apply_hint(g, [[0, 1]])
        remap = out["id_remap"]
        original_ids = {str(o["dispatch_id"]) for o in g["ops"]
                        if o.get("dispatch_id") is not None}
        self.assertEqual(set(remap), original_ids)
        # Fusion is many-to-one: both members land on the fused op's id.
        self.assertEqual(remap["0"], remap["1"])
        fused = next(o for o in out["ops"] if "fused_from" in o)
        self.assertEqual(fused["dispatch_id"], remap["0"])

    def test_remap_is_json_round_trippable(self):
        # The IR is written with json.dump, so the keys must already be
        # strings — an int-keyed dict would silently stringify on write
        # and mismatch anything comparing keys before/after the file
        # round trip.
        out = apply_hint(_mlp3_graph(), [[0, 1]])
        self.assertEqual(out["id_remap"],
                         json.loads(json.dumps(out))["id_remap"])
        self.assertTrue(all(isinstance(k, str) for k in out["id_remap"]))

    def test_identity_remap_when_no_groups(self):
        out = apply_hint(_mlp3_graph(), [])
        # No rewrite happened, so callers must not be handed a stale or
        # missing mapping; absent is fine, present must be the identity.
        if "id_remap" in out:
            self.assertEqual(out["id_remap"], {"0": 0, "1": 1, "2": 2})


if __name__ == "__main__":
    unittest.main()
