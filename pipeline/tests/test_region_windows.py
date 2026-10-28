"""Region windows (region_windows.py, --region-windows; docs/region_windows.md).

CPU only. Synthetic chain / unrolled-RNN / dense-block IRs (window_fixtures), the real final graphs of 34_VanillaRNNHidden and
15_DenseNet121 when the farm results are present (skipped otherwise), the configurable member cap, window validity (the one
authority is ``region_slots.fuse_region``), the ``groups`` form of the ``fuse_region`` hint, the walker equivalence of a windowed IR,
``export_problem`` of a windowed IR, the Caps hook (flag off = inert, on = proposals), and the whole round loop with fake GPU steps.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import modelblaster.pipeline.kb_model_run as mr  # noqa: E402
from modelblaster.pipeline import capabilities as capm  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import region_slots as rs  # noqa: E402
from modelblaster.pipeline import region_windows as rw  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, with_fused_specs  # noqa: E402
from modelblaster.pipeline.tests import window_fixtures as wf  # noqa: E402
from modelblaster.pipeline.tests.test_graph_rewrite import TITAN, _profile  # noqa: E402
from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB  # noqa: E402

FARM = Path(__file__).resolve().parents[3].parent / "l3_farm"
if not FARM.is_dir():
    FARM = Path.home() / "Documents" / "ModelBlaster-clone" / "l3_farm"


def _final(p):
    return FARM / "results_headline" / p / "work" / "final" / p / "ir" / "graph.json"


def _env(**kw):
    return mock.patch.dict(os.environ, {k: v for k, v in kw.items() if v is not None}, clear=False)


def _sp(g):
    return with_fused_specs(g, dict(_all_specs()))


def _cfg(**kw):
    return rw.WindowConfig(mode="auto", **kw)


class ConfigTest(unittest.TestCase):
    def test_mode(self):
        self.assertEqual(rw.resolve_mode(None), "off")
        for v, want in (("off", "off"), ("auto", "auto"), ("1", "auto"), ("0", "off"), ("", "off"), (" AUTO ", "auto")):
            self.assertEqual(rw.resolve_mode(v), want)
        with self.assertRaises(ValueError):
            rw.resolve_mode("always")
        with _env(MB_REGION_WINDOWS="auto"):
            self.assertEqual(rw.resolve_mode(None), "auto")
            self.assertEqual(rw.resolve_mode("off"), "off")          # the flag wins over the env

    def test_from_cfg(self):
        c = rw.WindowConfig.from_cfg(None, env={})
        self.assertEqual((c.mode, c.sizes, c.max_candidates, c.hard_cap, c.min_ops), ("off", (256,), 3, 1100, 4))
        self.assertFalse(c.on)
        ns = mock.Mock(region_windows="auto", region_window_sizes="64, 128;256", region_window_candidates=2, region_window_hard_cap=500,
                       region_window_min_ops=0)
        c = rw.WindowConfig.from_cfg(ns, env={"MB_REGION_WINDOW_MIN": "6"})
        self.assertEqual((c.on, c.sizes, c.max_candidates, c.hard_cap, c.min_ops), (True, (64, 128, 256), 2, 500, 6))
        self.assertEqual(c.member_cap(), 500)
        self.assertEqual(rw.WindowConfig.from_cfg(None, env={"MB_REGION_WINDOW_SIZES": "512"}).member_cap(), 1100)
        for bad in ("a,b", "1", ""):
            if bad:
                with self.assertRaises(ValueError):
                    rw._int_list(bad, "x")

    def test_member_cap_default_and_override(self):
        with _env(MB_REGION_MAX_MEMBERS=""):
            os.environ.pop("MB_REGION_MAX_MEMBERS", None)
            self.assertEqual(rs.max_members(), 256)
            self.assertEqual(rs.MAX_MEMBERS, 256)
        with _env(MB_REGION_MAX_MEMBERS="600"):
            self.assertEqual(rs.max_members(), 600)
        for bad in ("x", "1", "-5"):
            with _env(MB_REGION_MAX_MEMBERS=bad), self.assertRaises(ValueError):
                rs.max_members()

    def test_fuse_region_cap(self):
        g = wf.mlp_chain(150)                                         # 300 ops
        sp = _sp(g)
        names = [o["name"] for o in g["ops"]]
        os.environ.pop("MB_REGION_MAX_MEMBERS", None)
        with self.assertRaisesRegex(ValueError, r"region needs 2\.\.256 ops"):
            rs.fuse_region(g, names, sp)                              # unchanged default message and cap
        rs.fuse_region(g, names[:256], sp)                            # 256 members still fine
        with _env(MB_REGION_MAX_MEMBERS="300"):
            g2, _ = rs.fuse_region(g, names, sp)
            self.assertEqual(len(g2["ops"]), 1)
        with _env(MB_REGION_MAX_MEMBERS="64"), self.assertRaisesRegex(ValueError, r"region needs 2\.\.64 ops"):
            rs.fuse_region(g, names[:65], sp)


class StructureTest(unittest.TestCase):
    def test_cut_costs_chain_and_aliases(self):
        g = wf.mlp_chain(6)
        run = rw.eligible_runs(g)[0]
        cost = rw.cut_costs(g, run)
        self.assertEqual(cost[1:-1], [1] * (len(run) - 1))            # a chain: one tensor crosses every cut
        self.assertEqual(set(rw.alias_reach(g, run)), {-1})
        d = wf.dense_blocks((3, 3))
        run = rw.eligible_runs(d)[0]
        reach = rw.alias_reach(d, run)
        glegal = rw.global_legal(reach)
        cost = rw.cut_costs(d, run)
        names = [d["ops"][i]["name"] for i in run]
        for b in (0, 1):
            first, last = names.index(f"conv{b}_0"), names.index(f"tcat{b}")
            for c in range(first + 2, last + 1):          # after the first layer's view .. before the transition concat
                self.assertFalse(glegal[c], (b, c, names[c]))      # no cut inside a dense block: every later concat reads its views
            self.assertTrue(glegal[last + 1])             # right after the block's concat: legal ...
            self.assertEqual(cost[last + 1], 1)           # ... and single-frontier (one tensor crosses)
        self.assertEqual(cost[names.index("conv1_0")], 1)

    def test_runs_split_at_regions_and_skip_dead(self):
        g = wf.mlp_chain(6)
        sp = _sp(g)
        g2, _ = rs.fuse_region(g, [o["name"] for o in g["ops"][4:8]], sp)
        runs = rw.eligible_runs(g2)
        self.assertEqual([len(r) for r in runs], [4, 4])              # the region op is not a member and ends the run
        g3 = copy.deepcopy(g)
        g3["ops"].append({"name": "dead", "op": "relu_f16", "inputs": ["fc0_o"], "outputs": ["dead_o"], "shape": {"n": 32}})
        g3["tensors"]["dead_o"] = {"shape": [2, 16], "dtype": "f16", "quant": None}
        self.assertEqual(len(rw.eligible_runs(g3)[0]), len(g["ops"]))

    def test_motifs(self):
        toks = list("abc") + list("wxyz") * 10 + list("q")
        ms = rw.detect_motifs(toks)
        self.assertEqual([(m.start, m.period, m.reps) for m in ms], [(3, 4, 10)])      # the primitive period, not 8 / 12 / ...
        self.assertEqual(rw.detect_motifs(list("abcdefghij" * 1)), [])
        self.assertEqual(rw.detect_motifs(list("ab") * 2), [])                        # fewer than 3 repeats
        two = rw.detect_motifs(list("p" * 30 + "xy" * 20))
        self.assertEqual(sorted((m.start, m.period) for m in two), [(0, 1), (30, 2)])
        g = wf.rnn_unrolled(24)
        run = rw.eligible_runs(g)[0]
        m = rw.detect_motifs([rw.kind_token(g["ops"][i]) for i in run])
        self.assertTrue(any(x.period == 4 and x.reps >= 23 for x in m), m)

    def test_kind_token(self):
        k = rw.kind_token
        self.assertEqual(k({"op": "fused__cat3_c1_f16__batchnorm2d_f16__relu_f16"}), "fused__cat__batchnorm2d__relu")
        self.assertEqual(k({"op": "catn_f16"}), "cat")
        self.assertEqual(k({"op": "conv2d_f16"}), "conv2d")


class WindowTest(unittest.TestCase):
    def _check_valid(self, g, cands, size):
        sp = _sp(g)
        for c in cands:
            ok, bad = rw.validate_groups(g, sp, c.groups)
            self.assertEqual((len(ok), bad), (len(c.groups), []), c.label)
            cur = g
            for grp in c.groups:                                       # applied one after the other on the progressively fused graph
                cur, spec = rs.fuse_region(cur, gh._locate(cur, grp), sp)
            sizes = [len(w["sub_ops"]) for w in cur["ops"] if w["op"].startswith("region__")]
            if c.label != "whole":
                self.assertLessEqual(max(sizes), size, c.label)

    def test_rnn_windows_are_motif_aligned(self):
        g = wf.rnn_unrolled(24)                                               # 1 + 24*4 + 4 = 101 ops
        cands, rep = rw.propose_windows(g, _sp(g), _cfg(sizes=(40,), min_ops=2))
        labels = {c.label: c for c in cands}
        self.assertIn("chunk40", labels)
        self.assertIn("whole", labels)
        self.assertEqual(rep["motifs"][0]["period"], 4)
        c = labels["chunk40"]
        for grp in c.groups[:-1]:
            ops = [next(o for o in g["ops"] if rw._atoms(o["name"])[0] == a) for a in grp if a.startswith(("getitem", "cat_", "i2h", "h2o"))]
            self.assertEqual(len(ops) % 4, 0, "a window cuts a step in two")
        self._check_valid(g, cands, 40)
        covered = sorted(a for grp in c.groups for a in grp)
        self.assertEqual(len(covered), len(set(covered)), "windows overlap")

    def test_chain_windows_and_blocks(self):
        g = wf.mlp_chain(30)                                           # 60 ops, single frontier everywhere
        cands, rep = rw.propose_windows(g, _sp(g), _cfg(sizes=(16,), min_ops=2))
        self.assertTrue(cands)
        c = next(c for c in cands if c.label == "chunk16")
        self.assertTrue(all(len(grp) <= 16 for grp in c.groups))
        self.assertEqual(sum(len(grp) for grp in c.groups), 60)       # a chain is covered completely
        self._check_valid(g, cands, 16)
        self.assertEqual(len({tuple(map(tuple, c.groups)) for c in cands}), len(cands))     # no duplicate sets

    def test_dense_blocks_never_cut(self):
        g = wf.dense_blocks((6, 7, 5), growth=4)
        sp = _sp(g)
        for size in (24, 40, 200):
            cands, rep = rw.propose_windows(g, sp, _cfg(sizes=(size,), min_ops=2, hard_cap=10))
            self.assertEqual(rep["dropped"], [], (size, rep["dropped"]))                # nothing cut across a block's views
            self._check_valid(g, cands, size)
        cands, rep = rw.propose_windows(g, sp, _cfg(sizes=(40,), min_ops=2, hard_cap=10))
        c = cands[0]
        # every window boundary is a block boundary: no window ends strictly inside a block (conv{b}_{l} .. tcat{b})
        ends = [grp[-1] for grp in c.groups]
        self.assertTrue(all(e.startswith(("trans", "tcat")) for e in ends), ends)

    def test_whole_respects_the_hard_cap(self):
        g = wf.mlp_chain(20)
        cands, _ = rw.propose_windows(g, _sp(g), _cfg(sizes=(16,), hard_cap=30, min_ops=2))
        self.assertNotIn("whole", [c.label for c in cands])
        cands, _ = rw.propose_windows(g, _sp(g), _cfg(sizes=(16,), hard_cap=100, min_ops=2))
        self.assertIn("whole", [c.label for c in cands])

    def test_tiny_graph_and_budget(self):
        g = wf.mlp_chain(1)
        self.assertEqual(rw.propose_windows(g, _sp(g), _cfg(min_ops=4))[0], [])     # 2 ops: nothing worth a region
        g = wf.rnn_unrolled(24)
        cands, _ = rw.propose_windows(g, _sp(g), _cfg(sizes=(20, 40, 60), min_ops=2, max_candidates=2))
        self.assertEqual(len(cands), 2)
        ests = [c.est_gain_pct for c in cands]
        self.assertEqual(ests, sorted(ests, reverse=True))

    def test_ranking_uses_the_profile(self):
        g = wf.mlp_chain(20)
        us = {o["name"]: (10.0 if o["name"].startswith("fc1") else 0.1) for o in g["ops"]}
        cands, _ = rw.propose_windows(g, _sp(g), _cfg(sizes=(16,), min_ops=2), op_us=us, total_us=sum(us.values()))
        whole = next(c for c in cands if c.label == "whole")
        self.assertAlmostEqual(whole.est_gain_pct, 8.0, places=1)
        part = next(c for c in cands if c.label == "chunk16")
        self.assertLessEqual(part.est_gain_pct, whole.est_gain_pct)

    def test_nonconvex_group_is_refused(self):
        """A group that skips an op reading its member's output would be run before that reader: refused, never proposed."""
        g = wf.mlp_chain(4)
        names = [o["name"] for o in g["ops"]]
        sp = _sp(g)
        ok, bad = rw.validate_groups(g, sp, [[names[0], names[2], names[3]]])        # fc0, fc1, relu1 without relu0
        self.assertEqual(ok, [])
        self.assertIn("reads a region output before the region ends", bad[0]["why"])


class HintTest(unittest.TestCase):
    def test_groups_hint_applies_every_window(self):
        g = wf.rnn_unrolled(24)
        sp = _sp(g)
        cands, _ = rw.propose_windows(g, sp, _cfg(sizes=(40,), min_ops=2))
        c = next(c for c in cands if c.label == "chunk40")
        h = rw.hint_for(c, "m")
        self.assertEqual((h["id"], h["kind"], h["pattern"]), ("region:win:chunk40", "fuse_region", "ops"))
        h["status"] = "accepted"
        g2, added, prov = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
        regs = [o for o in g2["ops"] if o["op"].startswith("region__")]
        self.assertEqual(len(regs), len(c.groups))
        self.assertEqual(prov["skipped"], [])
        self.assertEqual(g2["output"], g["output"])
        for r in regs:                                                  # explicit contract: inputs / outputs / internal tensors
            self.assertTrue(r["inputs"] and r["outputs"] and r["internal"] is not None)
            self.assertEqual(r["hint"], "region:win:chunk40")
        self.assertEqual({a["kind"] for a in prov["applied"]}, {"fuse_region"})
        self.assertEqual(set(added), {r["op"] for r in regs})
        # the same windows applied to a graph with other names fail softly (skipped, nothing raised)
        g3 = copy.deepcopy(g)
        g3["ops"] = g3["ops"][:30]
        _, _, prov3 = gh.apply_hints(g3, [h], sp, statuses=("accepted",))
        self.assertTrue(prov3["skipped"])

    def test_members_and_pattern_forms_unchanged(self):
        g = wf.mlp_chain(6)
        sp = _sp(g)
        names = [o["name"] for o in g["ops"]]
        h = {"id": "region:x", "kind": "fuse_region", "target": "cuda", "status": "accepted", "pattern": "ops", "members": names[:6]}
        g2, _, prov = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
        self.assertEqual(sum(o["op"].startswith("region__") for o in g2["ops"]), 1)
        h2 = {"id": "region:all", "kind": "fuse_region", "target": "cuda", "status": "accepted", "pattern": "all"}
        g3, _, _ = gh.apply_hints(copy.deepcopy(g), [h2], sp, statuses=("accepted",))
        self.assertEqual(len(g3["ops"]), 1)

    def test_windowed_ir_is_numerically_the_same_model(self):
        """The torch walker (sub_ops recursed) gives the same output for the original and the windowed IR; export_problem accepts it."""
        import torch
        from modelblaster.pipeline import kb_export
        from modelblaster.pipeline.kb_export import _harness
        tmp = Path(tempfile.mkdtemp(prefix="mb_win_"))
        try:
            g = wf.rnn_unrolled(24)
            sp = _sp(g)
            cands, _ = rw.propose_windows(g, sp, _cfg(sizes=(40,), min_ops=2))
            c = next(c for c in cands if c.label == "chunk40")
            h = rw.hint_for(c, "m")
            h["status"] = "accepted"
            g2, added, _ = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
            a = wf.ir_dir_from_graph(tmp / "a", g)
            b = wf.ir_dir_from_graph(tmp / "b", g2)
            mbir, walk_torch = _harness()
            outs = []
            for d in (a, b):
                with torch.no_grad():
                    o, _ = walk_torch.walk(mbir.load_ir(d), dev="cpu")
                outs.append(o.numpy())
            np.testing.assert_array_equal(outs[0], outs[1])
            # structural gate of the rewrite infrastructure accepts the windowed graph
            from modelblaster.pipeline import graph_verify as gv
            w = {k: np.zeros(1, np.float32) for k in gv._graph_weight_keys(g2)}
            v = gv.Verdict()
            gv.structural(v, g2, copy.deepcopy(g2), w, w, None, check_base_values=False)
            self.assertEqual([r for r in v.reasons if r["code"] != "shape_guard"], [])
            # export: one region slot per window, explicit signature, the hand composite seeds (fused members from sub_sources)
            lib = kb_export.CudaLibrary(kb_export._hand_source(kb_export.DEFAULT_HAND_CU, "fp16"))
            ssrc = {k: sp[k].cuda_seed(lib, sub_sources={}) for k in {o["op"] for o in g["ops"] if o["op"].startswith("fused__")}}
            out = kb_export.export_problem(b, tmp / "prob", specs=with_fused_specs(g2, sp), name="win", slot_sources=ssrc)
            man = json.loads((Path(out) / "manifest.json").read_text())
            self.assertEqual(sum(s["id"].startswith("region__") for s in man["slots"]), len(c.groups))
            for s in man["slots"]:
                if s["id"].startswith("region__"):
                    self.assertTrue((Path(out) / s["file"]).read_text().strip())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CapsTest(unittest.TestCase):
    def setUp(self):
        self.g = wf.rnn_unrolled(24)
        self.profile = {"ir": "m", "e2e_us": 100.0, "ops": [{"name": o["name"], "us": 1.0} for o in self.g["ops"]]}

    def test_off_is_inert(self):
        os.environ.pop("MB_REGION_WINDOWS", None)
        caps = capm.Caps(mock.Mock(spec=[]))
        self.assertFalse(caps.windows.on)
        self.assertEqual(caps.round_key(), {})
        self.assertEqual(caps.summary(), dict(capm.DEFAULTS))
        self.assertEqual(caps.decide_kinds(), ())
        doc = gh.new_doc()
        self.assertEqual(caps.propose({"m": self.g}, [self.profile], doc, dict(_all_specs())), [])
        self.assertEqual(doc["hints"], [])
        with _env():
            before = os.environ.get("MB_REGION_MAX_MEMBERS")
            caps.apply_env()
            self.assertEqual(os.environ.get("MB_REGION_MAX_MEMBERS"), before)     # the cap is untouched when the flag is off

    def test_off_with_other_caps_keeps_the_summary(self):
        caps = capm.Caps(mock.Mock(spec=[], region_slots="all"))
        self.assertEqual(caps.summary()["region_slots"], "all")
        self.assertNotIn("region_windows", caps.summary())
        self.assertEqual(caps.round_key(), {"caps_region_slots": "all"})

    def test_on_proposes_window_hints(self):
        cfg = mock.Mock(spec=[], region_windows="auto", region_window_sizes="40", region_window_candidates=0,
                        region_window_hard_cap=0, region_window_min_ops=2)
        caps = capm.Caps(cfg)
        self.assertTrue(caps.windows.on)
        self.assertEqual(caps.round_key()["caps_region_windows"], "auto")
        self.assertIn("sizes=40", caps.round_key()["caps_region_windows_cfg"])
        self.assertEqual(caps.decide_kinds(), ("fuse_region",))
        doc = gh.new_doc()
        specs = dict(_all_specs())
        new = caps.propose({"m": self.g}, [self.profile], doc, specs)
        ids = {h["id"] for h in new}
        self.assertTrue(ids >= {"region:win:whole", "region:win:chunk40"}, ids)
        for h in new:
            self.assertEqual((h["kind"], h["status"], h["source"][0]["by"]), ("fuse_region", "proposed", rw.SOURCE))
            self.assertTrue(gh.is_region(h) if hasattr(gh, "is_region") else True)
        n = len(new)
        again = caps.propose({"m": self.g}, [self.profile], doc, specs)                 # already in the registry: nothing new
        self.assertEqual(again, [])
        self.assertEqual(len(doc["hints"]), n)
        with _env(MB_REGION_MAX_MEMBERS=None):
            os.environ.pop("MB_REGION_MAX_MEMBERS", None)
            caps.apply_env()
            self.assertEqual(int(os.environ["MB_REGION_MAX_MEMBERS"]), caps.windows.member_cap())
            os.environ.pop("MB_REGION_MAX_MEMBERS")

    def test_proposer_errors_do_not_break_a_round(self):
        cfg = mock.Mock(spec=[], region_windows="auto")
        caps = capm.Caps(cfg)
        with mock.patch.object(rw, "propose_windows", side_effect=RuntimeError("boom")):
            self.assertEqual(caps.propose({"m": self.g}, [self.profile], gh.new_doc(), dict(_all_specs())), [])


@unittest.skipUnless(_final("34_VanillaRNNHidden").is_file() and _final("15_DenseNet121").is_file(), "no final IRs of 34 / 15")
class RealGraphsTest(unittest.TestCase):
    def _load(self, p):
        return gh.unique_names(json.loads(_final(p).read_text()))

    def test_34(self):
        g = self._load("34_VanillaRNNHidden")
        self.assertEqual(len(g["ops"]), 1062)
        cfg = _cfg(sizes=(256,))
        with _env(MB_REGION_MAX_MEMBERS=str(cfg.member_cap())):
            t = time.time()
            cands, rep = rw.propose_windows(g, _sp(g), cfg)
            self.assertLess(time.time() - t, 30)
        by = {c.label: c for c in cands}
        self.assertEqual(rep["dropped"], [])
        self.assertEqual(rep["motifs"][0]["period"], 4)                  # (view, cat, linear+tanh, linear): one recurrence step
        self.assertEqual(rep["motifs"][0]["reps"], 256)
        sizes = [len([a for a in grp]) for grp in by["chunk256"].groups]
        self.assertEqual(len(sizes), 5)                                  # 4 recurrence windows of 64 steps + the 37-op stack
        sp = _sp(g)
        for c in cands:                                                  # each is applicable at once
            h = rw.hint_for(c, "m")
            h["status"] = "accepted"
            with _env(MB_REGION_MAX_MEMBERS=str(cfg.member_cap())):
                g2, _, prov = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
            self.assertEqual(prov["skipped"], [], c.label)
            self.assertEqual(sum(o["op"].startswith("region__") for o in g2["ops"]), len(c.groups))
        # without the lifted cap the same hint is refused (the old behaviour: 805 / 1062 ops cannot be one region)
        h = rw.hint_for(by["chunk256"], "m")
        h["status"] = "accepted"
        os.environ.pop("MB_REGION_MAX_MEMBERS", None)
        _, _, prov = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
        self.assertEqual(prov["skipped"], [])                            # 256-op windows fit the default cap
        h = rw.hint_for(by["whole"], "m")
        h["status"] = "accepted"
        _, _, prov = gh.apply_hints(copy.deepcopy(g), [h], sp, statuses=("accepted",))
        self.assertTrue(prov["skipped"] and "2..256" in prov["skipped"][0]["why"])

    def test_15(self):
        g = self._load("15_DenseNet121")
        cfg = _cfg(sizes=(256,))
        cands, rep = rw.propose_windows(g, _sp(g), cfg)
        self.assertEqual(rep["dropped"], [])
        by = {c.label: c for c in cands}
        self.assertIn("whole", by)
        self.assertEqual(len(by["chunk256"].groups), 2)                  # whole dense blocks, never cut across their views
        self.assertEqual(by["whole"].stats["largest"], 305)
        self.assertEqual(rs.max_members(), 256)                          # the old cap: 305 ops were un-regionable


# ------------------------------------------------------------------------------ the round loop
class LoopTest(unittest.TestCase):
    """kb_graph_opt._run, every GPU step faked, on the unrolled-RNN fixture: windows proposed, exported as original + variant, decided."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_win_loop_"))
        cls.g = wf.rnn_unrolled(24, fused=False)
        cls.ir = wf.ir_dir_from_graph(cls.tmp / "ir", cls.g)
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr = mr.model_runtime
        self.env = {k: os.environ.get(k) for k in ("KERNELBLASTER_GPU_FLOCK", "MB_REGION_WINDOWS", "MB_REGION_MAX_MEMBERS")}
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        os.environ.pop("MB_REGION_WINDOWS", None)
        os.environ.pop("MB_REGION_MAX_MEMBERS", None)

        def profile(irdir, work, *, hints_doc=None, **kw):
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            return _profile(d, irdir)

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 70.0] * reps
                    for d, _ in problems]
        go.profile_ir = profile
        go.verify_ir = lambda ir, cache, **kw: {"ir": "synthetic", "passed": True, "partition_us": 5.0, "device": TITAN, "slots": {}}
        mr.model_runtime = lambda ir, **kw: {"device": TITAN, "passed": True, "model_e2e_us": 4.0}
        go.remeasure, go.KBRunner = remeasure, FakeKB
        go.make_plan = lambda *a, **k: {"problems": [], "selected_share": 0.0}
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}], "remote": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        mr.model_runtime = self.saved_mr
        for k, v in self.env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def _run(self, name, **kw):
        cfg = go.Config(kb=self.tmp / "kb", arch="75", rounds=2, max_variants=2, remeasure=3, accept_gain=0.03, **kw)
        s = go.run([self.ir], self.tmp / name, cfg)
        return s, gh.load(self.tmp / name / "graph_hints.json"), json.loads((self.tmp / name / "state.json").read_text())

    def test_windows_end_to_end(self):
        s, doc, st = self._run("on", region_windows="auto", region_window_sizes="40", region_window_min_ops=2)
        wins = [h for h in doc["hints"] if h["id"].startswith("region:win:")]
        self.assertTrue(wins)
        acc = [h for h in wins if h["status"] == "accepted"]
        self.assertTrue(acc, [(h["id"], h["status"]) for h in wins])
        h = acc[0]
        ev = h["evidence"][-1]
        self.assertEqual((ev["decision"], ev["original_us"], ev["variant_us"]), ("accepted", 100.0, 70.0))
        self.assertTrue(any(a["kind"] == "fuse_region" for a in ev["applied"]))
        self.assertEqual(s["capabilities"]["region_windows"].split(";")[0], "sizes=40")
        # the variant problem was exported with one region slot per window
        man = next(iter((self.tmp / "on" / "r00" / "variants").glob(f"{h['id'].replace(':', '_')}*/variant/manifest.json")), None) \
            or next(iter((self.tmp / "on" / "r00" / "variants").glob("*/variant/manifest.json")))
        slots = json.loads(man.read_text())["slots"]
        self.assertTrue(any(x["id"].startswith("region__") for x in slots))

    def test_off_equals_unset(self):
        a = self._run("a")
        b = self._run("b", region_windows="off")
        os.environ["MB_REGION_WINDOWS"] = "off"
        c = self._run("c")
        os.environ.pop("MB_REGION_WINDOWS")

        def norm(x):
            s, doc, st = x

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
        self.assertTrue(all(not h["id"].startswith("region:win:") for h in a[1]["hints"]))
        self.assertNotIn("region_windows", json.dumps(a[0]))
        k = lambda n: json.loads((self.tmp / n / "state.json").read_text())["rounds"][0]["key"]      # noqa: E731
        self.assertEqual(k("a"), k("b"))
        self.assertEqual(k("a"), k("c"))

    def test_the_flag_changes_the_round_key_only_when_on(self):
        self._run("k_off")
        self._run("k_on", region_windows="auto", region_window_min_ops=2)
        k = lambda n: json.loads((self.tmp / n / "state.json").read_text())["rounds"][0]["key"]      # noqa: E731
        self.assertNotEqual(k("k_off"), k("k_on"))


if __name__ == "__main__":
    unittest.main()
