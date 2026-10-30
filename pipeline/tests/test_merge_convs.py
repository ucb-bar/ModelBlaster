"""merge_sibling_convs (merge_convs.py): detector, cost gate, applier with derived weights, gate verification, flag plumbing.

CPU only (torch for the walkers). Synthetic branchy IRs (merge_fixtures) and, when the IR is present on this machine, the real
fp16 18_SqueezeNet IR (structural + proposals; the numerical gate on it is a script, see RUNBOOK).
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_edit as ge  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import merge_convs as mc  # noqa: E402
from modelblaster.pipeline import weight_derivations as wd  # noqa: E402
from modelblaster.pipeline.kb_export import _harness  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, write_variant_ir  # noqa: E402
from modelblaster.pipeline.tests import merge_fixtures as MF  # noqa: E402
from modelblaster.pipeline.tests import rewrite_fixtures as F  # noqa: E402

FIRE = MF.FIRE
REAL_SQ = Path("/Users/krisd/Documents/ModelBlaster-clone/l3_farm/graph_rewrite/ir_regress/new/fp16/18_SqueezeNet")


def _graph(branches, quant="fp32", **kw):
    return MF.graph_and_weights(branches, quant, **kw)[0]


def pol(**kw):
    """Tiny synthetic graphs: compute_share 0 (the byte model alone decides the cost gate; test_cost_gate exercises the rest) and a wide
    waste cap (1x1+3x3+5x5 wastes 1.2)."""
    return mc.Policy(**{"compute_share": 0.0, "max_waste": 2.0, **kw})


def _accepted(h):
    return {**h, "status": "accepted"}


def merge_ir(orig: Path, out: Path, policy=None, **kw):
    """Propose + apply on the IR at ``orig`` -> (edited IR dir, graph, hint, prov); None hint when nothing proposed."""
    g0 = json.loads((orig / "graph.json").read_text())
    hints, _ = mc.propose(g0, None, dict(_all_specs()), None, policy=policy or pol(), **kw)
    if not hints:
        return None, g0, None, None
    g1, _, prov = gh.apply_hints(g0, [_accepted(hints[0])], dict(_all_specs()))
    return write_variant_ir(orig, g1, out), g1, hints[0], prov


class VerifyRewriteTest(unittest.TestCase):
    """graph_hints.verify_rewrite (apply --verify) must accept a merged graph: the merged conv is not bit-identical to the original walk
    (zero padding reorders the accumulation), so the op carries approx=True and the check is the fp16 output tolerance. On the first B200
    run `graph_hints apply --verify --build` said ok: false at an abs difference of 3e-5 before this."""

    def test_merged_graph_verifies(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ir = MF.branchy_ir(d / "ir", FIRE, "fp32")
            vir, g1, h, _ = merge_ir(ir, d / "merged")
            self.assertIsNotNone(h)
            self.assertTrue(any(o.get("approx") for o in g1["ops"]))
            res = gh.verify_rewrite(ir, g1, specs=dict(_all_specs()), build=False)
            self.assertTrue(res["golden"]["ok"])
            self.assertTrue(res["ok"], res)


class DetectTest(unittest.TestCase):
    def find(self, branches, quant="fp32", **kw):
        return mc.find(_graph(branches, quant), pol(**kw))

    def test_fire_module(self):
        plans, rep = self.find(FIRE)
        self.assertEqual(len(plans), 1)
        p = plans[0]
        self.assertEqual((p["pattern"], p["convs"], p["acts"], p["cat"]), ("conv1x1+conv3x3+relu", ["c0", "c1"], ["a0", "a1"], "cat"))
        self.assertEqual((p["kh"], p["kw"], p["ph"], p["pw"], p["oc"]), (3, 3, 1, 1, [2, 3]))
        self.assertAlmostEqual(p["waste"], (5 * 4 * 9 - (2 * 4 + 3 * 4 * 9)) / (2 * 4 + 3 * 4 * 9), places=3)

    def test_3x3_plus_5x5_and_three_branches(self):
        plans, _ = self.find([{"k": 3, "oc": 2}, {"k": 5, "oc": 2}])
        self.assertEqual(plans[0]["pattern"], "conv3x3+conv5x5+relu")
        self.assertAlmostEqual(plans[0]["waste"], 50 / 34 - 1, places=3)
        plans, _ = self.find([{"k": 1, "oc": 2}, {"k": 3, "oc": 2}, {"k": 5, "oc": 2}])
        self.assertEqual((plans[0]["pattern"], plans[0]["pos"], plans[0]["kh"]), ("conv1x1+conv3x3+conv5x5+relu", 0, 5))

    def test_stride_two_and_dilation_with_centred_padding(self):
        plans, _ = self.find([{"k": 1, "oc": 2, "stride": 2}, {"k": 3, "oc": 3, "stride": 2}])      # both give 3x3 outputs
        self.assertEqual(len(plans), 1)
        self.assertEqual((plans[0]["ph"], plans[0]["pw"]), (1, 1))
        plans, _ = self.find([{"k": 1, "oc": 2, "dil": 2}, {"k": 3, "oc": 3, "dil": 2}])
        self.assertEqual((len(plans), plans[0]["ph"]), (1, 2))

    def test_mismatches_do_not_merge(self):
        # same output size, different stride (and padding not centred alike): 1x1 s1 -> 6x6, 3x3 s2 p4 -> 6x6
        plans, rep = self.find([{"k": 1, "oc": 2}, {"k": 3, "oc": 3, "stride": 2, "pad": 4}])
        self.assertEqual(plans, [])
        self.assertTrue(any("stride" in r["why"] or "padding" in r["why"] for r in rep), rep)
        # different dilation, same output size (6x6): 3x3 pad 1 vs 3x3 dil 2 pad 2
        plans, rep = self.find([{"k": 3, "oc": 2}, {"k": 3, "oc": 3, "dil": 2, "pad": 2}])
        self.assertEqual(plans, [])
        self.assertTrue(any("dilation" in r["why"] for r in rep), rep)
        # a padding that is not centred: 3x3 with pad 2 on a 4x4... 1x1 pad 1 -> 8x8 vs 3x3 pad 2 -> 8x8, offsets 1 vs 1: centred alike (merges)
        plans, _ = self.find([{"k": 1, "oc": 2, "pad": 1}, {"k": 3, "oc": 3, "pad": 2}])
        self.assertEqual((len(plans), plans[0]["ph"]), (1, 2))
        # 1x1 pad 0 (6x6) vs 3x3 pad 1: fine; 1x1 pad 1 (8x8) vs 3x3 pad 1 (6x6): not concat-able at all
        plans, _ = self.find([{"k": 1, "oc": 2, "pad": 1}, {"k": 3, "oc": 3}])
        self.assertEqual(plans, [])

    def test_grouped_and_depthwise_are_skipped_with_a_reason(self):
        plans, rep = self.find([{"k": 3, "oc": 2, "groups": 2}, {"k": 3, "oc": 2, "groups": 2}])
        self.assertEqual(plans, [])
        self.assertTrue(any("grouped" in r["why"] for r in rep), rep)

    def test_even_kernel_and_non_conv_branch(self):
        plans, rep = self.find([{"k": 2, "oc": 2, "pad": 1}, {"k": 2, "oc": 2, "pad": 1}])
        self.assertEqual(plans, [])
        plans, _ = self.find([{"k": 1, "oc": 2}, {"k": 3, "oc": 3}, {"pool": True, "oc": 4}])
        self.assertEqual((len(plans), plans[0]["pos"]), (1, 0))                      # the pool branch stays in the concat
        plans, _ = self.find([{"pool": True, "oc": 4}, {"k": 1, "oc": 2}, {"k": 3, "oc": 3}])
        self.assertEqual((len(plans), plans[0]["pos"]), (1, 1))

    def test_activation_must_be_the_same_on_every_branch(self):
        plans, rep = self.find([{"k": 1, "oc": 2, "act": "relu"}, {"k": 3, "oc": 3, "act": "silu"}])
        self.assertEqual(plans, [])
        self.assertTrue(any("activation" in r["why"] for r in rep), rep)
        plans, _ = self.find([{"k": 1, "oc": 2, "act": None}, {"k": 3, "oc": 3, "act": None}])
        self.assertEqual((plans[0]["pattern"], plans[0]["acts"]), ("conv1x1+conv3x3+noact", []))
        plans, _ = self.find([{"k": 1, "oc": 2, "act": "silu"}, {"k": 3, "oc": 3, "act": "silu"}])
        self.assertTrue(plans[0]["pattern"].endswith("silu"))

    def test_intermediate_with_another_reader_is_not_merged(self):
        g = _graph(FIRE)
        g["ops"].append({"name": "extra", "op": "relu", "inputs": ["c0_o"], "outputs": ["extra_o"], "shape": {"n": 72}})
        g["tensors"]["extra_o"] = {"shape": [1, 2, 6, 6], "dtype": "f32", "quant": None}
        plans, _ = mc.find(g)
        self.assertEqual(plans, [])

    def test_cost_gate(self):
        plans, rep = mc.find(_graph(FIRE), pol(max_waste=0.3))
        self.assertEqual(plans, [])
        self.assertTrue(any("padding waste" in r["why"] for r in rep), rep)
        plans, _ = mc.find(_graph(FIRE), pol(max_waste=0.6))                                  # waste 0.55
        self.assertEqual(len(plans), 1)
        # a wide 3x3 next to a narrow 1x1 pays almost nothing; a narrow 1x1 next to a wide 3x3 ... 1x1(12)+3x3(1): waste ~ 7x
        plans, rep = mc.find(_graph([{"k": 1, "oc": 12}, {"k": 3, "oc": 1}]), pol(max_waste=1.0))
        self.assertEqual(plans, [])
        # the net estimate: a profile that says the convs are expensive and the concat free -> not proposed
        g = _graph(FIRE)
        plans, rep = mc.find(g, pol(compute_share=1.0), op_us={"c0": 500.0, "c1": 500.0, "cat": 1.0}, total_us=1100.0)
        self.assertEqual(plans, [])
        self.assertTrue(any("net gain" in r["why"] for r in rep), rep)
        plans, _ = mc.find(g, pol(compute_share=0.25), op_us={"c0": 50.0, "c1": 50.0, "cat": 100.0}, total_us=1100.0)
        self.assertEqual(len(plans), 1)
        self.assertGreater(plans[0]["est"]["net_us"], 0)

    def test_hint_record(self):
        g = _graph(FIRE)
        hints, rep = mc.propose(g, {"ops": [{"name": "cat", "us": 40.0}, {"name": "c0", "us": 5.0}, {"name": "c1", "us": 20.0}], "e2e_us": 400.0},
                                dict(_all_specs()), None, min_gain_pct=1.0, ir="fire")
        self.assertEqual(len(hints), 1)
        h = hints[0]
        self.assertEqual((h["id"], h["kind"], h["status"], h["target"]), ("merge:conv1x1+conv3x3+relu", mc.KIND, "proposed", "cuda"))
        self.assertEqual(h["chains"], [["c0", "c1", "a0", "a1", "cat"]])
        self.assertEqual(h["instances"][0], {"ir": "fire", "cat": "cat", "convs": ["c0", "c1"], "acts": ["a0", "a1"]})
        src = h["source"][0]
        self.assertEqual(src["by"], "mb-cost")
        self.assertIn("evidence", src)
        self.assertGreater(src["est_gain_pct"], 1.0)
        json.dumps(h)
        done = {"hints": [{**h, "status": "rejected", "evidence": [{"decision": "rejected"}]}]}
        again, _ = mc.propose(g, None, dict(_all_specs()), done)
        self.assertEqual(again, [])                                                                # decided hints are not re-proposed
        low, rep = mc.propose(g, {"ops": [{"name": "cat", "us": 0.5}], "e2e_us": 1000.0}, dict(_all_specs()), None, min_gain_pct=5.0)
        self.assertEqual(low, [])


class ApplyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_mc_apply_"))
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def ir(self, branches, quant="fp32", **kw):
        ApplyTest.n += 1
        return MF.branchy_ir(self.tmp / f"o{ApplyTest.n}", branches, quant, **kw)

    def merged(self, branches, quant="fp32", **kw):
        o = self.ir(branches, quant, **kw)
        e, g1, h, prov = merge_ir(o, self.tmp / f"e{ApplyTest.n}")
        self.assertIsNotNone(h)
        return o, e, g1, h, prov

    def check_ok(self, o, e, prov, **kw):
        v = gv.verify(o, e, seeds=(0, 1), tensor_map=prov["applied"][0]["tensor_map"], **kw)
        self.assertTrue(v.ok, v.summary())
        return v

    def test_fire_in_every_dtype_verifies(self):
        for q in ("fp32", "fp16", "bf16", "native"):
            with self.subTest(q=q):
                o, e, g1, h, prov = self.merged(FIRE, q)
                self.assertEqual(len(g1["ops"]), 6)                           # sq, sq_relu, merged conv, merged act, pool, fc
                self.assertEqual([x["kind"] for x in prov["applied"]], [mc.KIND])
                v = self.check_ok(o, e, prov)
                self.assertTrue(v.checks["exact64"]["ok"] if "exact64" in v.checks else True)
                # the merged conv: one 3x3, OC 5, centred padding, derived weights, dtype kept
                m = next(x for x in g1["ops"] if x["name"].startswith("merge_") and x["op"].startswith("conv"))
                self.assertEqual((m["shape"]["OC"], m["shape"]["KH"], m["shape"]["PH"]), (5, 3, 1))
                self.assertEqual(sorted(wd.derivations(g1)), sorted([m["weight"], m["bias"]]))
                self.assertEqual(wd.validate(g1, set(np.load(o / "weights.npz").files)), [])
                self.assertTrue(wd.is_exact(g1))
                suf = {"fp32": "", "fp16": "_f16", "bf16": "_bf16", "native": "_f16"}[q]
                self.assertEqual(m["op"], "conv2d" + suf)
                ldt = {"fp32": "f32", "fp16": "f16", "bf16": "bf16", "native": "f16"}[q]
                self.assertEqual({d["dtype"] for d in wd.derivations(g1).values()}, {ldt})
                if q == "native":
                    self.assertEqual({g1["weight_dtypes"][k] for k in wd.derivations(g1)}, {"f16"})
                else:
                    self.assertNotIn("weight_dtypes", g1)

    def test_op_and_tensor_bookkeeping(self):
        o, e, g1, h, prov = self.merged(FIRE)
        g0 = json.loads((o / "graph.json").read_text())
        self.assertEqual([x["name"] for x in g1["ops"]], ["sq", "sq_relu", "merge_" + g1["ops"][2]["name"][6:], g1["ops"][3]["name"], "pool", "fc"])
        self.assertEqual(ge.io_names(g1), ge.io_names(g0))
        for t in ("c0_o", "c1_o", "a0_o", "a1_o"):
            self.assertNotIn(t, g1["tensors"])
        self.assertEqual(g1["tensors"]["cat_out"], g0["tensors"]["cat_out"])               # the concat's name, shape, dtype survive
        self.assertEqual(g1["ops"][3]["outputs"], ["cat_out"])
        self.assertEqual(g1["ops"][3]["shape"]["n"], 5 * 36)
        self.assertEqual(len({x["name"] for x in g1["ops"]}), len(g1["ops"]))
        self.assertEqual([x["dispatch_id"] for x in g1["ops"]], list(range(6)))              # renumbered
        self.assertEqual(g1["ops"][2]["hint"], h["id"])
        self.assertEqual(sorted(prov["renamed"]), ["a0", "a1", "c0", "c1", "cat"])
        self.assertEqual(sorted(prov["applied"][0]["tensor_map"]), ["a0_o", "a1_o"])
        # apply never modifies its input graph, and every op's inputs are produced before it
        self.assertEqual(len(g0["ops"]), 9)
        seen = {"x"}
        for x in g1["ops"]:
            self.assertTrue(set(x["inputs"]) <= seen, x["name"])
            seen |= set(x["outputs"])

    def test_weights_are_the_padded_stack_and_equal_the_hand_written_merge(self):
        o = F.fire_ir(self.tmp / "fo", "fp16")                                              # the existing fixture + its hand-written rewrite
        e0, tmap = F.merge_expand(o, self.tmp / "fe")
        e, g1, h, prov = merge_ir(o, self.tmp / "fm")[0:4]
        z, z0 = np.load(e / "weights.npz"), np.load(e0 / "weights.npz")
        m = next(x for x in g1["ops"] if x["op"].startswith("conv2d") and x["name"].startswith("merge_"))
        self.assertTrue(np.array_equal(z[m["weight"]], z0["d.fire.w"]))
        self.assertTrue(np.array_equal(z[m["bias"]], z0["d.fire.b"]))
        self.assertEqual(z[m["weight"]].dtype, np.float16)
        mbir, walk = _harness()
        import torch
        outs = []
        for d in (e0, e):
            with torch.no_grad():
                outs.append(walk.walk(mbir.load_ir(d), dev="cpu")[0])
        self.assertTrue(torch.equal(outs[0], outs[1]))                                      # identical ops on identical weights

    def test_3x3_plus_5x5_stride2_partial_concat_noact_mixed_bias(self):
        cases = {"3x3+5x5": [{"k": 3, "oc": 3}, {"k": 5, "oc": 2}],
                 "stride2": [{"k": 1, "oc": 2, "stride": 2}, {"k": 3, "oc": 3, "stride": 2}],
                 "dilation": [{"k": 1, "oc": 2, "dil": 2}, {"k": 3, "oc": 3, "dil": 2}],
                 "three": [{"k": 1, "oc": 2}, {"k": 3, "oc": 3}, {"k": 5, "oc": 2}],
                 "partial_pool_last": [{"k": 1, "oc": 2}, {"k": 3, "oc": 3}, {"pool": True, "oc": 4}],
                 "partial_pool_first": [{"pool": True, "oc": 4}, {"k": 1, "oc": 2}, {"k": 3, "oc": 3}],
                 "noact": [{"k": 1, "oc": 2, "act": None}, {"k": 3, "oc": 3, "act": None}],
                 "silu": [{"k": 1, "oc": 2, "act": "silu"}, {"k": 3, "oc": 3, "act": "silu"}],
                 "mixed_bias": [{"k": 1, "oc": 2, "bias": False}, {"k": 3, "oc": 3}],
                 "nonsquare": [{"k": (1, 3), "oc": 2}, {"k": (3, 1), "oc": 3}]}
        for name, br in cases.items():
            for q in ("fp32", "fp16"):
                with self.subTest(case=name, q=q):
                    o, e, g1, h, prov = self.merged(br, q)
                    self.check_ok(o, e, prov)
                    self.assertTrue(any(x["op"].startswith("conv2d") and x["name"].startswith("merge_") for x in g1["ops"]))
                    self.assertEqual(wd.validate(g1, set(np.load(o / "weights.npz").files)), [])
        o, e, g1, h, prov = self.merged(cases["partial_pool_last"])
        cat = next(x for x in g1["ops"] if x["name"] == "cat")
        self.assertEqual((cat["op"], cat["shape"]["C_inputs"], cat["shape"]["C_total"]), ("cat2_c1", [5, 4], 9))
        o, e, g1, h, prov = self.merged(cases["mixed_bias"])
        bk = next(k for k in wd.derivations(g1) if k.endswith(".b"))
        self.assertIn("const", json.dumps(g1["weight_derivations"][bk]))                    # zeros stand in for the missing bias

    def test_numerical_equivalence_on_random_weights_and_inputs(self):
        import torch
        mbir, walk = _harness()
        for seed in (1, 2, 3):
            for q in ("fp32", "fp16"):
                o = self.ir([{"k": 1, "oc": 3}, {"k": 3, "oc": 4}, {"k": 5, "oc": 2}], q, seed=seed, hw=(7, 9))
                e, g1, h, prov = merge_ir(o, self.tmp / f"rn{seed}{q}")[0:4]
                a, b = mbir.load_ir(o), mbir.load_ir(e)
                rng = np.random.default_rng(seed)
                for _ in range(2):
                    x = rng.standard_normal(a.golden_in.size).astype(np.float32)
                    if q == "fp16":
                        x = x.astype(np.float16).astype(np.float32)
                    a.golden_in = b.golden_in = x
                    with torch.no_grad():
                        ra, rb = walk.walk(a, dev="cpu")[0], walk.walk(b, dev="cpu")[0]
                    tol = (1e-5 if q == "fp32" else 2e-2) * float(ra.abs().max()) + 1e-5
                    self.assertLessEqual(float((ra - rb).abs().max()), tol, (seed, q))

    def test_float64_walk_agrees_to_1e9(self):
        o, e, g1, h, prov = self.merged([{"k": 1, "oc": 3}, {"k": 3, "oc": 4}], "fp32", seed=5)
        v = self.check_ok(o, e, prov, exact="on", conditioning="off")
        self.assertIn("exact64", v.checks)
        self.assertTrue(v.checks["exact64"]["ok"], v.checks["exact64"])

    def test_derivations_replay_and_a_non_derivable_weight_is_rejected(self):
        o, e, g1, h, prov = self.merged(FIRE, "fp16")
        z0 = np.load(o / "weights.npz")
        base = {k: z0[k] for k in z0.files}
        rep = wd.replay(g1, base)                                                            # rebuilt from the ORIGINAL parameters only
        z1 = np.load(e / "weights.npz")
        for k, a in rep.items():
            self.assertTrue(np.array_equal(a, z1[k]), k)
        # a weight computed outside the derivation machinery: an array with no recorded lineage
        bad = self.tmp / "bad_lineage"
        shutil.copytree(e, bad, symlinks=False)
        arrs = {k: np.load(bad / "weights.npz")[k] for k in np.load(bad / "weights.npz").files}
        arrs["rogue"] = (base["e1.weight"] if "e1.weight" in base else base["c0.weight"]) * np.float16(2)
        np.savez(bad / "weights.npz", **arrs)
        v = gv.verify(o, bad, structural_only=True)
        self.assertFalse(v.ok)
        self.assertIn("weight_not_mapped", {r["code"] for r in v.reasons})
        # a merged op naming a weight that exists nowhere
        g = json.loads((e / "graph.json").read_text())
        next(x for x in g["ops"] if x["name"].startswith("merge_") and x["op"].startswith("conv"))["weight"] = "made.up"
        bad2 = self.tmp / "bad_key"
        shutil.copytree(e, bad2)
        (bad2 / "graph.json").write_text(json.dumps(g))
        v = gv.verify(o, bad2, structural_only=True)
        self.assertIn("weight_not_mapped", {r["code"] for r in v.reasons})
        # a tampered derived array
        bad3 = self.tmp / "bad_val"
        shutil.copytree(e, bad3)
        arrs = {k: np.load(bad3 / "weights.npz")[k] for k in np.load(bad3 / "weights.npz").files}
        k = next(k for k in arrs if k.startswith("d.merge") and k.endswith(".w"))
        arrs[k] = arrs[k] + np.float16(1)
        np.savez(bad3 / "weights.npz", **arrs)
        v = gv.verify(o, bad3, structural_only=True)
        self.assertIn("derivation_mismatch", {r["code"] for r in v.reasons})

    def test_a_wrong_merge_is_caught_by_the_gate(self):
        o, e, g1, h, prov = self.merged(FIRE)
        g = json.loads((e / "graph.json").read_text())
        for k, d in g["weight_derivations"].items():
            if k.endswith(".w"):
                d["expr"]["args"].reverse()                                                   # stacked in the wrong order
        bad = self.tmp / "wrong_order"
        shutil.copytree(e, bad)
        (bad / "graph.json").write_text(json.dumps(g))
        z = np.load(o / "weights.npz")
        wd.write_weights(o / "weights.npz", g, bad / "weights.npz")
        v = gv.verify(o, bad, seeds=(0,), tensor_map=prov["applied"][0]["tensor_map"])
        self.assertFalse(v.ok)

    def test_stale_hint_is_skipped_with_a_reason_and_apply_is_idempotent(self):
        o = self.ir(FIRE)
        g0 = json.loads((o / "graph.json").read_text())
        h = mc.propose(g0, None, dict(_all_specs()), None, policy=pol())[0][0]
        g1, _, p1 = gh.apply_hints(g0, [_accepted(h)], dict(_all_specs()))
        g2, _, p2 = gh.apply_hints(g1, [_accepted(h)], dict(_all_specs()))                   # twice: the ops are gone
        self.assertEqual(ge.graph_digest(g1), ge.graph_digest(g2))
        self.assertTrue(p2["skipped"] and "not in the graph" in p2["skipped"][0]["why"], p2["skipped"])
        g = copy.deepcopy(g0)
        next(x for x in g["ops"] if x["name"] == "c1")["shape"]["SH"] = 2                      # the base changed
        g3, _, p3 = gh.apply_hints(g, [_accepted(h)], dict(_all_specs()))
        self.assertEqual(p3["applied"], [])
        self.assertTrue(p3["skipped"], p3)
        self.assertEqual(mc.scope_ops(h, g0, {}), ["c0", "c1", "a0", "a1", "cat"])
        self.assertEqual(mc.scope_ops(h, g1, {}), [])

    def test_composes_with_fuse_hints(self):
        o = self.ir(FIRE)
        g0 = json.loads((o / "graph.json").read_text())
        h = _accepted(mc.propose(g0, None, dict(_all_specs()), None, policy=pol())[0][0])
        fuse = {"id": "fuse:cr", "kind": "fuse", "target": "cuda", "fused_kind": "fused__conv2d__relu", "status": "accepted"}
        g, _, prov = gh.apply_hints(g0, [fuse, h], dict(_all_specs()))
        self.assertEqual([x["kind"] for x in prov["applied"]], [mc.KIND, "fuse", "fuse"])        # merge first, then the fusions see the new ops
        self.assertTrue(any(x["name"].startswith("merge_") and "+" in x["name"] for x in g["ops"]))
        # a chain-scoped fusion that names an op the merge removed is skipped, not applied wrongly
        fuse2 = {**fuse, "id": "fuse:c0", "chains": [["c0", "a0"]]}
        g, _, prov = gh.apply_hints(g0, [fuse2, h], dict(_all_specs()))
        self.assertEqual([x["kind"] for x in prov["applied"]], [mc.KIND])
        self.assertTrue(any(s["id"] == "fuse:c0" for s in prov["skipped"]))

    def test_accepted_hint_applies_wherever_the_pattern_matches_and_proposed_only_to_its_instances(self):
        o = self.ir(FIRE, "fp16", modules=2)
        g0 = json.loads((o / "graph.json").read_text())
        plans, _ = mc.find(g0, pol())
        self.assertEqual(len(plans), 2)
        h = mc.propose(g0, None, dict(_all_specs()), None, policy=pol())[0][0]
        self.assertEqual((len(h["instances"]), h["policy"]["max_waste"]), (2, 2.0))
        h["instances"] = h["instances"][:1]                                                     # the hint was measured on the first module only
        gp, _, pp = gh.apply_hints(g0, [h], dict(_all_specs()), statuses=None)                  # proposed: its recorded instance only
        self.assertEqual(len(pp["applied"]), 1)
        ga, _, pa = gh.apply_hints(g0, [_accepted(h)], dict(_all_specs()))                      # accepted: every match of the pattern
        self.assertEqual(len(pa["applied"]), 2, pa)
        self.assertEqual(sum(x["name"].startswith("merge_") and x["op"].startswith("conv") for x in ga["ops"]), 2)
        e = write_variant_ir(o, ga, self.tmp / "two_mod")
        v = gv.verify(o, e, seeds=(0,), tensor_map={k: m for a_ in pa["applied"] for k, m in a_["tensor_map"].items()})
        self.assertTrue(v.ok, v.summary())

    def test_a_second_ir_extends_the_instances_of_a_known_hint(self):
        g = _graph(FIRE)
        hk = mc.MergeHook(go.Config(kb=Path("."), merge_sibling_convs="auto", merge_sibling_max_waste="2", min_gain_pct=0.0))
        doc = gh.new_doc()
        prof = lambda n: {"ir": n, "ops": [{"name": "cat", "us": 30.0}], "e2e_us": 100.0}  # noqa: E731
        hk.propose({"irA": g}, [prof("irA")], doc, dict(_all_specs()))
        hk.propose({"irB": copy.deepcopy(g)}, [prof("irB")], doc, dict(_all_specs()))
        self.assertEqual(len(doc["hints"]), 1)
        self.assertEqual([i["ir"] for i in doc["hints"][0]["instances"]], ["irA"])               # the same ops (same names) are one instance
        g2 = copy.deepcopy(g)
        for o_ in g2["ops"]:
            if o_["name"] == "cat":
                o_["name"] = "cat_b"
        hk.propose({"irC": g2}, [prof("irC")], doc, dict(_all_specs()))
        self.assertEqual(sorted(i["cat"] for i in doc["hints"][0]["instances"]), ["cat", "cat_b"])
        self.assertEqual(len(doc["hints"][0]["source"]), 3)

    def test_cli_then_graph_hints_apply_verify(self):
        import subprocess
        o = self.ir(FIRE, "fp16")
        hints = self.tmp / "cli_hints.json"
        r = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.merge_convs", str(o), "--out", str(hints), "--accept"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        rep = json.loads(r.stdout)
        self.assertEqual([h["id"] for h in rep["hints"]], ["merge:conv1x1+conv3x3+relu"])
        self.assertEqual(json.loads(hints.read_text())["hints"][0]["status"], "accepted")
        out = self.tmp / "cli_ir"
        r = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.graph_hints", "apply", str(o), str(hints), str(out), "--verify"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(mc.KIND, r.stdout)
        g = json.loads((out / "graph.json").read_text())
        self.assertEqual(len(g["ops"]), 6)
        r = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.merge_convs", str(o), "--max-waste", "0.1"], capture_output=True, text=True, timeout=300)
        rep = json.loads(r.stdout)
        self.assertEqual(rep["hints"], [])
        self.assertTrue(any("padding waste" in x["why"] for x in rep["skipped"]))

    def test_export_of_the_merged_ir(self):
        from modelblaster.pipeline.kb_export import export_problem
        for q in ("fp16", "native", "fp32"):
            o, e, g1, h, prov = self.merged(FIRE, q)
            prob = export_problem(e, self.tmp / f"p_{q}", strategy="whole")
            man = json.loads((prob / "manifest.json").read_text())
            bufs = {b["name"]: b for b in man["buffers"]}
            w = next(k for k in wd.derivations(g1) if k.endswith(".w"))
            ldt = "f32" if q == "fp32" else "f16"
            self.assertEqual((bufs["w:" + w]["numel"], bufs["w:" + w]["dtype"]), (5 * 4 * 3 * 3, ldt))
            self.assertNotIn("w:c0.weight", bufs)                                           # the replaced convs' weights are not exported
            self.assertEqual(sum(1 for s in man["slots"] for _ in s["instances"]), 6)
            convs = [s for s in man["slots"] if s["op"].startswith("conv2d")]
            shapes = sorted(i["shape"]["OC"] for s in convs for i in s["instances"])
            self.assertEqual(shapes, [4, 5])                                                # the stem and the merged conv (a new shape of the slot)


class FlagTest(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.pop(k, None) for k in (mc.ENV, mc.ENV_WASTE)}

    def tearDown(self):
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def cfg(self, **kw):
        return go.Config(kb=Path("."), **kw)

    def test_off_is_inert(self):
        hk = mc.MergeHook(self.cfg())
        self.assertEqual((hk.mode, hk.active(), hk.round_key(), hk.decide_kinds()), ("off", False, {}, ()))
        doc = gh.new_doc()
        g = _graph(FIRE)
        self.assertEqual(hk.propose({"fire": g}, [{"ir": "fire", "ops": []}], doc, {}), [])
        self.assertEqual(doc["hints"], [])
        self.assertEqual(hk.log, [])

    def test_resolution_env_and_validation(self):
        self.assertEqual(mc.resolve_mode(self.cfg(merge_sibling_convs="auto")), "auto")
        os.environ[mc.ENV] = "auto"
        self.assertEqual(mc.resolve_mode(self.cfg()), "auto")
        self.assertEqual(mc.resolve_mode(self.cfg(merge_sibling_convs="off")), "off")           # the flag wins over the env
        os.environ[mc.ENV] = "0"
        self.assertEqual(mc.resolve_mode(self.cfg()), "off")
        with self.assertRaises(ValueError):
            mc.resolve_mode(self.cfg(merge_sibling_convs="sometimes"))
        os.environ[mc.ENV_WASTE] = "0.25"
        self.assertEqual(mc.Policy.from_cfg(self.cfg()).max_waste, 0.25)
        self.assertEqual(mc.Policy.from_cfg(self.cfg(merge_sibling_max_waste="2")).max_waste, 2.0)

    def test_on_proposes_through_the_hook_and_registers_the_kind(self):
        hk = mc.MergeHook(self.cfg(merge_sibling_convs="auto", min_gain_pct=0.0))
        self.assertEqual((hk.round_key(), hk.decide_kinds()), ({"merge_sibling_convs": "auto"}, (mc.KIND,)))
        self.assertIn(mc.KIND, gh.CUDA_KINDS)
        doc = gh.new_doc()
        g = _graph(FIRE)
        prof = {"ir": "fire", "ops": [{"name": "cat", "us": 30.0}, {"name": "c1", "us": 10.0}, {"name": "c0", "us": 5.0}], "e2e_us": 200.0}
        new = hk.propose({"fire": g}, [prof], doc, dict(_all_specs()))
        self.assertEqual([h["id"] for h in new], ["merge:conv1x1+conv3x3+relu"])
        self.assertEqual(doc["hints"][0]["status"], "proposed")
        self.assertEqual(hk.summary()["log"][0]["proposed"], ["merge:conv1x1+conv3x3+relu"])
        hk2 = mc.MergeHook(self.cfg(merge_sibling_convs="auto", merge_sibling_max_waste="0.1"))
        self.assertEqual(hk2.round_key()["merge_max_waste"], 0.1)

    def test_kb_graph_opt_scope_and_footprint_know_the_kind(self):
        g = _graph(FIRE)
        h = mc.propose(g, None, dict(_all_specs()), None, policy=pol())[0][0]
        specs = dict(_all_specs())
        self.assertEqual(go._scope_ops(h, g, specs), ["c0", "c1", "a0", "a1", "cat"])
        fp = go.footprint(h, "fire", g, specs)
        self.assertIn("kernel_conv2d", fp["writes"])
        self.assertEqual({a for _, a in fp["ops"]}, {"c0", "c1", "a0", "a1", "cat"})

    def test_cli_flags_exist(self):
        import subprocess
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", "--help"], capture_output=True, text=True, timeout=120)
        self.assertIn("--merge-sibling-convs", out.stdout)
        self.assertIn("--merge-sibling-max-waste", out.stdout)


@unittest.skipUnless((REAL_SQ / "graph.json").exists(), "the real fp16 18_SqueezeNet IR is not on this machine")
class RealSqueezeNetTest(unittest.TestCase):
    def test_eight_fire_modules_are_found_and_applied(self):
        g0 = json.loads((REAL_SQ / "graph.json").read_text())
        hints, rep = mc.propose(g0, None, dict(_all_specs()), None, ir="18_SqueezeNet")
        self.assertEqual(len(hints), 1, rep)
        h = hints[0]
        self.assertEqual((h["id"], len(h["instances"])), ("merge:conv1x1+conv3x3+relu", 8))
        g1, _, prov = gh.apply_hints(g0, [_accepted(h)], dict(_all_specs()))
        self.assertEqual((len(g0["ops"]), len(g1["ops"])), (65, 65 - 8 * 3))
        self.assertEqual(len(prov["applied"]), 8)
        self.assertEqual(prov["skipped"], [])
        self.assertTrue(all(d["dtype"] == "f16" for d in wd.derivations(g1).values()))
        self.assertEqual(wd.validate(g1, set(np.load(REAL_SQ / "weights.npz").files)), [])
        self.assertTrue(wd.is_exact(g1))
        # structural gate on the real weights (no walk: the 64x3x512x512 walk is a script, RUNBOOK)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            vir = write_variant_ir(REAL_SQ, g1, Path(d) / "v")
            v = gv.verify(REAL_SQ, vir, structural_only=True)
            self.assertTrue(v.ok, v.summary())
            # a hint measured on ONE module: proposed it applies to that instance only; accepted it applies wherever the pattern matches
            h1 = {**h, "instances": h["instances"][:1]}
            g2, _, _ = gh.apply_hints(g0, [h1], dict(_all_specs()), statuses=None)
            self.assertEqual(len(g2["ops"]), 65 - 3)
            g3, _, _ = gh.apply_hints(g0, [_accepted(h1)], dict(_all_specs()))
            self.assertEqual(len(g3["ops"]), 65 - 8 * 3)


TITAN = {"host": None, "label": "0", "index": 0, "uuid": "GPU-titan", "name": "NVIDIA TITAN RTX", "arch": "sm_75"}


class LoopTest(unittest.TestCase):
    """kb_graph_opt._run with every GPU step faked (as test_graph_rewrite.LoopTest): the merge hint is proposed by the hook, exported as an
    original and a variant problem, decided by the (fake) paired timing, and applies again in the next round."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_mc_loop_"))
        cls.o = MF.branchy_ir(cls.tmp / "ir", FIRE, "fp16")
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        import modelblaster.pipeline.kb_model_run as mr
        from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB
        self.mr = mr
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr = mr.model_runtime
        self.env = {k: os.environ.get(k) for k in ("KERNELBLASTER_GPU_FLOCK", mc.ENV, mc.ENV_WASTE)}
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        os.environ.pop(mc.ENV, None)
        os.environ.pop(mc.ENV_WASTE, None)
        self.profiles = []

        def profile(irdir, work, *, hints_doc=None, kbeval=None, gpu_model=None, allow_gpu_mismatch=False, **kw):
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            g = json.loads((Path(d) / "graph.json").read_text())
            us = {"cat": 30.0}
            ops = [{"name": o["name"], "op": o["op"], "us": us.get(o["name"], 4.0 if o["op"].startswith(("conv", "fused__conv")) else 1.0),
                    "shape": o.get("shape")} for o in g["ops"]]
            tot = sum(o["us"] for o in ops)
            p = {"ir": g["name"], "ir_dir": str(d), "source_ir": str(irdir), "passed": True, "partition_us": tot, "ops": ops, "e2e_us": tot,
                 "launch_us": 0.0, "device": TITAN}
            self.profiles.append(p)
            return p

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 80.0] * reps
                    for d, _ in problems]
        go.profile_ir = profile
        go.verify_ir = lambda ir, cache, **kw: {"ir": "branchy", "passed": True, "partition_us": 5.0, "device": TITAN, "slots": {}}
        mr.model_runtime = lambda ir, **kw: {"device": TITAN, "passed": True, "model_e2e_us": 4.0}
        go.remeasure, go.KBRunner = remeasure, FakeKB
        go.make_plan = lambda *a, **k: {"problems": [], "selected_share": 0.0}
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}], "remote": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        self.mr.model_runtime = self.saved_mr
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _run(self, name, **kw):
        cfg = go.Config(kb=self.tmp / "kb", arch="75", rounds=3, max_variants=3, remeasure=3, accept_gain=0.03, **kw)
        s = go.run([self.o], self.tmp / name, cfg)
        return s, gh.load(self.tmp / name / "graph_hints.json"), json.loads((self.tmp / name / "state.json").read_text())

    def test_on_end_to_end(self):
        s, doc, st = self._run("on", merge_sibling_convs="auto")
        h = next(x for x in doc["hints"] if x["kind"] == mc.KIND)
        self.assertEqual((h["id"], h["status"]), ("merge:conv1x1+conv3x3+relu", "accepted"))
        ev = h["evidence"][-1]
        self.assertEqual((ev["decision"], ev["original_us"], ev["variant_us"]), ("accepted", 100.0, 80.0))
        self.assertEqual(ev["applied"][0]["kind"], mc.KIND)
        self.assertEqual(len(ev["applied"][0]["derived_weights"]), 2)
        self.assertEqual(s["merge_sibling_convs"]["mode"], "auto")
        self.assertIn(h["id"], s["accepted"])
        var = next((self.tmp / "on" / "r00" / "variants").glob("merge*/variant/manifest.json"))
        man = json.loads(var.read_text())
        self.assertTrue(any(n.startswith("merge_") for n in man["partition"]["op_names"]), man["partition"]["op_names"])
        self.assertNotIn("c0", man["partition"]["op_names"])
        orig = json.loads(next((self.tmp / "on" / "r00" / "variants").glob("merge*/orig/manifest.json")).read_text())
        self.assertIn("c0", orig["partition"]["op_names"])
        last = self.profiles[-1]                                                                  # the next round profiles the merged graph
        g = json.loads((Path(last["ir_dir"]) / "graph.json").read_text())
        self.assertTrue(any(o["name"].startswith("merge_") for o in g["ops"]))
        self.assertTrue(any(k.startswith("d.merge.") for k in np.load(Path(last["ir_dir"]) / "weights.npz").files))

    def test_a_losing_variant_is_rejected_with_evidence(self):
        go.remeasure = lambda problems, *, arch, reps, lease=None, **kw: [[100.0] * reps for _ in problems]
        s, doc, st = self._run("lose", merge_sibling_convs="auto")
        h = next(x for x in doc["hints"] if x["kind"] == mc.KIND)
        self.assertEqual(h["status"], "rejected")
        self.assertEqual(h["evidence"][-1]["decision"], "rejected")
        self.assertEqual(s["accepted"], [])

    def test_waste_threshold_flag_blocks_the_proposal(self):
        s, doc, st = self._run("waste", merge_sibling_convs="auto", merge_sibling_max_waste="0.1")
        self.assertEqual([h for h in doc["hints"] if h["kind"] == mc.KIND], [])
        self.assertTrue(any("padding waste" in r["why"] for r in s["merge_sibling_convs"]["log"][0]["skipped"]))

    def test_off_equals_unset_and_leaves_no_trace(self):
        a = self._run("a")
        b = self._run("b", merge_sibling_convs="off")
        os.environ[mc.ENV] = "off"
        c = self._run("c")
        os.environ.pop(mc.ENV)

        def norm(x):
            s, doc, st = x
            doc = json.loads(json.dumps(doc))
            for h in doc["hints"]:
                for e in h.get("evidence", []):
                    e.pop("ts", None)

            def strip(o):
                if isinstance(o, dict):
                    return {k: strip(v) for k, v in o.items() if k not in ("seconds", "decide_seconds", "ts", "telemetry", "key")}
                if isinstance(o, list):
                    return [strip(v) for v in o]
                if isinstance(o, str):
                    return o.replace(str(self.tmp), "<T>")
                return o
            return json.dumps(strip({"hints": doc, "summary": s, "rounds": st["rounds"]}), sort_keys=True, default=str)
        self.assertEqual(norm(a), norm(b))
        self.assertEqual(norm(a), norm(c))
        self.assertNotIn("merge_sibling_convs", a[0])
        self.assertTrue(all(h["kind"] != mc.KIND for h in a[1]["hints"]))
        key = lambda n: json.loads((self.tmp / n / "state.json").read_text())["rounds"][0]["key"]  # noqa: E731
        self.assertEqual(key("a"), key("b"))
        self.assertNotIn("merge_sibling_convs", json.dumps(key("a")))
        self._run("k_on", merge_sibling_convs="auto")
        self.assertNotEqual(key("a"), key("k_on"))


if __name__ == "__main__":
    unittest.main()
