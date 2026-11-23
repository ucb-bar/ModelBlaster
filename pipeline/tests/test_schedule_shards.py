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


    def test_k3_simple_layer_gets_its_scheduled_width(self):
        # k3-work TASK-017: generate_skeleton splits a simple layer only where it has one
        ir = {"name": "dronet", "ops": [
            {"dispatch_id": 0, "op": "add_s8", "shape": {"n": 6272}},
            {"dispatch_id": 1, "op": "silu_s8", "shape": {"n": 64}},
        ]}
        sched = _schedule([4, 4])
        for e in sched["dispatches"].values():      # dispatch 1 stays on one core
            if e["id"] == 1:
                e["hardware_target"] = "CPU_P#0"
        out, applied = apply_schedule_shards(ir, sched, "dronet", ("ime_k3_x100", "rvv_k3"))
        self.assertEqual(out["ops"][0]["shard_factor"], 4)
        self.assertNotIn("shard_factor", out["ops"][1])
        self.assertEqual([x["dispatch_id"] for x in applied], [0])

    def test_simple_layer_on_several_cores_needs_a_k3_backend(self):
        ir = {"name": "dronet", "ops": [{"dispatch_id": 0, "op": "add_s8", "shape": {"n": 64}},
                                        {"dispatch_id": 1, "op": "relu_s8", "shape": {"n": 64}}]}
        # A K3 build: every backend must have the pool path, or the layer is refused.
        with self.assertRaisesRegex(ValueError, "pool path"):
            apply_schedule_shards(ir, _schedule([2]), "dronet", ("rvv_k3", "rvv_x60"))
        # A K1 build (or no backends given): accepted and left alone, as before
        # the K3 pool paths existed; the layers run on their first core.
        for backends in (("rvv_x60",), ("rvv_x60", "ime_x60"), ()):
            out, applied = apply_schedule_shards(ir, _schedule([2]), "dronet", backends)
            self.assertEqual(out["ops"][:2], ir["ops"], backends)
            self.assertEqual(applied, [], backends)

    def test_k1_conv_records_and_errors_are_unchanged(self):
        # No K3 backend: no split axis in the record, and the pre-K3 OC rule and message.
        _, applied = apply_schedule_shards(_ir(), _schedule([4]), "dronet", ("rvv_x60",))
        self.assertEqual(applied, [{"dispatch_id": 0, "op": "conv2d_s8", "oc": 32, "n_shards": 4}])
        ir = _ir()
        ir["ops"][0]["shape"]["OC"] = 30
        with self.assertRaisesRegex(ValueError, r"not divisible by scheduled width 4; codegen"):
            apply_schedule_shards(ir, _schedule([4]), "dronet", ("rvv_x60",))


if __name__ == "__main__":
    unittest.main()
