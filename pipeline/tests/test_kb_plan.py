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


if __name__ == "__main__":
    unittest.main()
