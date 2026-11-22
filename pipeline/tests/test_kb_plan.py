"""Tests for cost-ranked selection (kb_plan) and the kernel cache's slot
plan (kb_ingest.cache_plan). Pure python."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline.kb_ingest import CudaCache, cache_plan  # noqa: E402
from modelblaster.pipeline.kb_plan import make_plan, rank, select  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS as K  # noqa: E402
from modelblaster.pipeline.tests.test_graph_hints import conv_block  # noqa: E402


def _rec(name, op, us, shape, entry=None):
    return {"name": name, "op": op, "slot": op, "entry": entry or f"kernel_{op}", "signature_sha": "s",
            "shape": shape, "us": us, "seed": "hand"}


C1, C2, C3 = ({"IC": 64, "OC": 64}, {"IC": 128, "OC": 128}, {"IC": 3, "OC": 64})


def _profiles():
    a = {"ir": "A", "ir_dir": "/a", "partition_us": 100.0, "ops": [
        _rec("a.conv1", "conv2d", 40.0, C1), _rec("a.conv2", "conv2d", 40.0, C1),
        _rec("a.relu", "relu", 5.0, {"n": 8}), _rec("a.stem", "conv2d", 15.0, C3)]}
    b = {"ir": "B", "ir_dir": "/b", "partition_us": 200.0, "ops": [
        _rec("b.conv1", "conv2d", 50.0, C1), _rec("b.conv3", "conv2d", 120.0, C2),
        _rec("b.relu", "relu", 30.0, {"n": 8})]}
    return [a, b]


class PlanTest(unittest.TestCase):
    def test_rank_dedups_across_irs(self):
        items = rank(_profiles())
        top = items[0]
        self.assertEqual((top["op"], top["shape"], top["total_us"], top["count"]), ("conv2d", C1, 130.0, 3))
        self.assertEqual([o["ir"] for o in top["occurrences"]], ["A", "B"])      # A has 80 us of it, B 50
        self.assertEqual([round(it["total_us"]) for it in items], [130, 120, 35, 15])

    def test_select_budget(self):
        p = select(rank(_profiles()), coverage=0.5)
        self.assertEqual([it["selected"] for it in p["items"]], [True, True, False, False])
        self.assertEqual(p["items"][2]["why_not"], "coverage-reached")
        p = select(rank(_profiles()), coverage=1.0, top_k=1)
        self.assertEqual([it["why_not"] for it in p["items"]], [None, "top-k", "top-k", "top-k"])
        p = select(rank(_profiles()), coverage=1.0, min_share=0.1)
        self.assertEqual(p["items"][3]["why_not"], "below-min-share")
        first = rank(_profiles())[0]["key"]
        p = select(rank(_profiles()), coverage=0.5, exclude={first})
        self.assertEqual([it["why_not"] for it in p["items"]][:3], ["sent-before", None, None])

    def test_problems_per_representative_ir(self):
        plan = make_plan(_profiles(), coverage=0.6)
        self.assertEqual(plan["problems"][0]["ir"], "B")                 # b.conv3 (120 us) is B's
        byir = {p["ir"]: p["op_names"] for p in plan["problems"]}
        self.assertEqual(byir, {"A": ["a.conv1", "a.conv2"], "B": ["b.conv3"]})
        self.assertAlmostEqual(plan["selected_share"], 250 / 300)


class CachePlanTest(unittest.TestCase):
    SIG = K["conv2d"].signature

    def _cache(self, d, guards):
        c = CudaCache(Path(d))
        for k, (shapes, us) in enumerate(guards):
            c.add(op="conv2d", entry="kernel_conv2d", signature=self.SIG, arch="sm_75", shapes=shapes,
                  code=f"// winner {k}\n", slot_us=us, speedup=2.0, impl_id=f"w{k}", source={})
        return c

    def test_full_cover_is_one_slot(self):
        g = conv_block(2)
        shapes = [o["shape"] for o in g["ops"] if o["op"] == "conv2d"]
        with tempfile.TemporaryDirectory() as d:
            p = cache_plan(g, self._cache(d, [(shapes, 3.0)]), "sm_75", K)
            self.assertEqual(p["sources"], {"conv2d": "// winner 0\n"})
            self.assertEqual(p["hints"], [])

    def test_partial_cover_splits_the_slot(self):
        g = conv_block(2)
        s0, s1 = [o["shape"] for o in g["ops"] if o["op"] == "conv2d"]
        with tempfile.TemporaryDirectory() as d:
            p = cache_plan(g, self._cache(d, [([s1], 1.0)]), "sm_75", K)
            self.assertEqual(p["hints"][0]["groups"], [{"ops": ["conv1"], "cache": "w0", "as": "slot"},
                                                       {"ops": ["conv0"], "cache": None}])
            self.assertEqual(p["hints"][0]["source"][0]["by"], "mb-cache")
            self.assertEqual(p["sources"], {"conv2d__v0": "// winner 0\n"})
            var = {o["name"]: o.get("slot_variant") for o in p["graph"]["ops"]}
            self.assertEqual((var["conv0"], var["conv1"]), ("v1", "v0"))
            p = cache_plan(g, self._cache(d, []), "sm_75", K, split=False)
            self.assertEqual(p["hints"], [])

    def test_fused_seed_uses_guarded_stage_kernels(self):
        from modelblaster.pipeline import graph_hints as gh
        g = conv_block(1)
        g2, _, _ = gh.apply_hints(g, [gh.fuse_hint("fused__conv2d__batchnorm2d", None, {}, "accepted")], K)
        conv_shape = g["ops"][0]["shape"]
        with tempfile.TemporaryDirectory() as d:
            p = cache_plan(g2, self._cache(d, [([conv_shape], 1.0)]), "sm_75", K)
            self.assertEqual(p["sub_sources"], {"fused__conv2d__batchnorm2d": {"conv2d": "// winner 0\n"}})
            other = dict(conv_shape, IC=99)
            p = cache_plan(g2, self._cache(Path(d) / "x", [([other], 1.0)]), "sm_75", K)
            self.assertEqual(p["sub_sources"], {"fused__conv2d__batchnorm2d": {}})

    def test_fused_slot_split_by_stage_winners(self):
        """Fusing convs tuned per shape keeps each one's kernel: the fused slot
        is split by which conv winner covers its first stage."""
        from modelblaster.pipeline import graph_hints as gh
        g = conv_block(2)
        g2, _, _ = gh.apply_hints(g, [gh.fuse_hint("fused__conv2d__batchnorm2d", None, {}, "accepted")], K)
        s0, s1 = [o["shape"] for o in g["ops"] if o["op"] == "conv2d"]
        with tempfile.TemporaryDirectory() as d:
            p = cache_plan(g2, self._cache(d, [([s0], 1.0), ([s1], 1.0)]), "sm_75", K)
            self.assertEqual([gr["as"] for gr in p["hints"][0]["groups"]], ["stage", "stage"])
            self.assertEqual(p["sub_sources"], {"fused__conv2d__batchnorm2d__v0": {"conv2d": "// winner 0\n"},
                                                "fused__conv2d__batchnorm2d__v1": {"conv2d": "// winner 1\n"}})
            # one winner covering both: no split, it seeds every instance
            p = cache_plan(g2, self._cache(Path(d) / "y", [([s0, s1], 1.0)]), "sm_75", K)
            self.assertEqual(p["hints"], [])
            self.assertEqual(p["sub_sources"], {"fused__conv2d__batchnorm2d": {"conv2d": "// winner 0\n"}})


class CachePlanFp16Test(unittest.TestCase):
    """fp16 entries are keyed by the CUDA-side signature (_Float16 -> __half): the split path must find them
    (it compared the raw signature's sha, so a slot with winners tuned on different shapes was never split and
    the model ran on the seed kernels)."""

    def test_fp16_slot_splits_by_shape_winners(self):
        spec = K["linear_f16"]
        shapes = [{"M": 128, "K": 64, "N": 64}, {"M": 128, "K": 64, "N": 64}, {"M": 128, "K": 64, "N": 32}]
        g = {"name": "mlp16", "ops": [{"name": f"l{i}", "op": "linear_f16", "shape": sh} for i, sh in enumerate(shapes)]}
        with tempfile.TemporaryDirectory() as d:
            c = CudaCache(Path(d))
            for k, sh in enumerate((shapes[0], shapes[2])):
                c.add(op="linear_f16", entry="kernel_linear_f16", signature=spec.signature, arch="sm_89",
                      shapes=[sh], code=f"// winner {k}\n", slot_us=1.0, speedup=3.0, impl_id=f"w{k}", source={})
            p = cache_plan(g, c, "sm_89", K)
        self.assertEqual(p["sources"], {"linear_f16__v0": "// winner 0\n", "linear_f16__v1": "// winner 1\n"})
        self.assertEqual(p["hints"][0]["groups"], [{"ops": ["l0", "l1"], "cache": "w0", "as": "slot"},
                                                   {"ops": ["l2"], "cache": "w1", "as": "slot"}])


class PerItemPlanTest(unittest.TestCase):
    def test_one_problem_per_selected_item(self):
        plan = make_plan(_profiles(), coverage=2.0, top_k=None, min_share=0.0, per_item=True)
        sel = [it for it in plan["items"] if it["selected"]]
        self.assertEqual(len(plan["problems"]), len(sel))
        ids = [p["id"] for p in plan["problems"]]
        self.assertEqual(len(ids), len(set(ids)))                     # unique export dirs / names
        for p in plan["problems"]:
            self.assertEqual(len(p["keys"]), 1)
        # the budgeted plan is unchanged: one problem per IR, no ids
        base = make_plan(_profiles(), coverage=0.8)
        self.assertTrue(all("id" not in p for p in base["problems"]))


class ExportPlanRepeatedNamesTest(unittest.TestCase):
    """49_Mamba2ReturnFinalState: einsum_1 lowers to six ops all named einsum_1 in graph.json; the profile (and the
    plan) call them einsum_1, einsum_1#1..#5. export_plan matched those against the raw names, so the 6.9 ms
    einsum_1#4 bmm went to KernelBlaster as an empty partition (baseline 1.28 us, 0 LLM calls) and the
    einsum_2 problem silently carried all three einsum_2 ops."""

    def _graph(self):
        ops = [{"name": "einsum_1", "op": "strided_copy_f16"}, {"name": "einsum_1", "op": "bmm_f16"},
               {"name": "einsum_1", "op": "bmm_f16"}, {"name": "einsum_2", "op": "strided_copy_f16"},
               {"name": "einsum_2", "op": "bmm_f16"}]
        return {"name": "m", "ops": ops, "tensors": {}}

    def test_export_matches_unique_names(self):
        import json
        from unittest import mock

        from modelblaster.pipeline import kb_export
        from modelblaster.pipeline.kb_plan import export_plan

        plan = {"problems": [{"id": "m__bmm_f16_0", "ir": "m", "op_names": ["einsum_1#2"]},
                             {"id": "m__strided_copy_f16_1", "ir": "m", "op_names": ["einsum_2"]},
                             {"id": "m__bmm_f16_2", "ir": "m", "op_names": ["einsum_1#1", "einsum_2#1"]}]}
        got = {}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "graph.json").write_text(json.dumps(self._graph()))
            for p in plan["problems"]:
                p["ir_dir"] = d

            def fake(ir_dir, out, op_indices=None, **kw):
                got[Path(out).name] = op_indices
                return out
            with mock.patch.object(kb_export, "export_problem", fake):
                export_plan(plan, Path(d) / "out", specs={})
        self.assertEqual(got, {"m__bmm_f16_0": [2], "m__strided_copy_f16_1": [3], "m__bmm_f16_2": [1, 4]})

    def test_unknown_op_name_is_an_error_not_an_empty_problem(self):
        from modelblaster.pipeline import graph_hints as gh
        from modelblaster.pipeline.kb_plan import plan_op_indices

        g = gh.unique_names(self._graph())
        with self.assertRaises(ValueError):
            plan_op_indices(g, ["einsum_1#7"], "p")
        with self.assertRaises(ValueError):
            plan_op_indices(g, ["einsum_1#1", "nope"], "p")
        self.assertEqual(plan_op_indices(g, ["einsum_1"], "p"), [0])


if __name__ == "__main__":
    unittest.main()
