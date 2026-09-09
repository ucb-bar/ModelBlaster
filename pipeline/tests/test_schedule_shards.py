"""The schedule's composite targets reach packed-weight code generation."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from pipeline.schedule_shards import apply_schedule_shards  # noqa: E402


def _ir():
    return {"name": "dronet", "ops": [
        {"dispatch_id": 0, "op": "conv2d_s8", "shape": {"OC": 32}},
        {"dispatch_id": 1, "op": "linear_s8", "shape": {"N": 64}},
    ]}


def _schedule(widths):
    dispatches = {}
    for instance, width in enumerate(widths):
        targets = "+".join(f"CPU_P#{i}" for i in range(width))
        for did in (0, 1):
            dispatches[f"dronet{instance}_dispatch_{did}"] = {
                "job_name": f"dronet{instance}", "id": did,
                "hardware_target": targets,
            }
    return {"dispatches": dispatches}


class ScheduleShardTests(unittest.TestCase):

    def test_annotates_packed_conv_but_leaves_runtime_sliceable_linear(self):
        out, applied = apply_schedule_shards(_ir(), _schedule([4, 4]), "dronet")
        self.assertEqual(out["ops"][0]["shard_factor"], 4)
        self.assertNotIn("shard_factor", out["ops"][1])
        self.assertEqual([x["dispatch_id"] for x in applied], [0])

    def test_refuses_invocation_dependent_packed_weight_width(self):
        with self.assertRaisesRegex(ValueError, "different widths"):
            apply_schedule_shards(_ir(), _schedule([2, 4]), "dronet")

    def test_refuses_a_width_that_cannot_partition_output_channels(self):
        ir = _ir()
        ir["ops"][0]["shape"]["OC"] = 30
        with self.assertRaisesRegex(ValueError, "not divisible"):
            apply_schedule_shards(ir, _schedule([4]), "dronet")

    def test_reads_fused_convolution_shape_from_the_conv_sub_op(self):
        ir = _ir()
        ir["ops"][0] = {
            "dispatch_id": 0, "op": "conv2d_batchnorm2d_s8",
            "sub_ops": [{"op": "conv2d_s8", "shape": {"OC": 32}}],
        }
        out, _ = apply_schedule_shards(ir, _schedule([4]), "dronet")
        self.assertEqual(out["ops"][0]["shard_factor"], 4)


if __name__ == "__main__":
    unittest.main()


class OnlyPackedWeightOpsAreConstrained(unittest.TestCase):
    """A linear may take a different width in each periodic instance.

    Its weights are row-major and sliced at runtime from the entry's own pool width, so
    nothing is baked at codegen time. Refusing it cost a board run: greedy's 5-net
    schedule gives `ffn_block` dispatch 1 -- a `linear_s8` -- width 1 in one instance and
    width 4 in another, and `ffn_block` has no convolution at all, so the packed-weight
    rule could never have applied to it.
    """

    def test_a_linear_may_vary_its_width_across_instances(self):
        ir = {"name": "ffn_block", "ops": [
            {"dispatch_id": 0, "op": "layernorm_s8", "shape": {"N": 256}},
            {"dispatch_id": 1, "op": "linear_s8", "shape": {"N": 1024}},
        ]}
        sched = {"dispatches": {}}
        for inst, width in enumerate((1, 4)):
            targets = "+".join(f"CPU_P#{i}" for i in range(width))
            for did in (0, 1):
                sched["dispatches"][f"ffn_block{inst}_dispatch_{did}"] = {
                    "job_name": f"ffn_block{inst}", "id": did,
                    "hardware_target": targets,
                }
        out, applied = apply_schedule_shards(ir, sched, "ffn_block")
        self.assertEqual(applied, [], "a linear needs no shard annotation")
        for op in out["ops"]:
            self.assertNotIn("shard_factor", op)

    def test_a_conv_still_has_to_commit(self):
        """The rule is narrowed, not removed."""
        ir = {"name": "dronet", "ops": [
            {"dispatch_id": 0, "op": "conv2d_s8", "shape": {"OC": 32}},
        ]}
        sched = {"dispatches": {}}
        for inst, width in enumerate((2, 4)):
            targets = "+".join(f"CPU_P#{i}" for i in range(width))
            sched["dispatches"][f"dronet{inst}_dispatch_0"] = {
                "job_name": f"dronet{inst}", "id": 0,
                "hardware_target": targets,
            }
        with self.assertRaisesRegex(ValueError, "different widths"):
            apply_schedule_shards(ir, sched, "dronet")


class ContractAndCodeAgree(unittest.TestCase):
    """The packed-weight op list has two readers and must have one definition.

    XPU-RT reads `cores/codegen_contract.json` to keep its solver inside what this
    module can build. If the two lists diverge the failure is silent in the worst
    direction: the scheduler believes an op is unconstrained, emits a schedule with
    per-instance widths, and the build refuses it at stage 1 of 5.
    """

    def test_the_module_uses_the_contracts_list(self):
        import json
        from pipeline.schedule_shards import (
            _CONTRACT_PATH, _PACKED_WEIGHT_SHARD_OPS,
            _PACKED_WEIGHT_SHARD_OPS_FALLBACK,
        )
        self.assertTrue(_CONTRACT_PATH.exists(), f"no contract at {_CONTRACT_PATH}")
        rules = json.loads(_CONTRACT_PATH.read_text())["rules"]
        contract_ops = set(rules["uniform_width_across_instances"]["applies_to_ops"])
        self.assertEqual(_PACKED_WEIGHT_SHARD_OPS, contract_ops)
        self.assertEqual(_PACKED_WEIGHT_SHARD_OPS_FALLBACK, contract_ops,
                         "the offline fallback has drifted from the contract")

    def test_runtime_sliceable_ops_are_not_also_packed(self):
        """A contract that lists an op in both halves would be incoherent."""
        import json
        from pipeline.schedule_shards import _CONTRACT_PATH
        rules = json.loads(_CONTRACT_PATH.read_text())["rules"]
        packed = set(rules["uniform_width_across_instances"]["applies_to_ops"])
        sliceable = set(rules["runtime_sliceable_ops"]["ops"])
        self.assertEqual(packed & sliceable, set())
