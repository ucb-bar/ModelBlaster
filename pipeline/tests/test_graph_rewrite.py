"""Graph-rewrite plumbing (graph_edit patches, the ``rewrite`` hint, graph_rewrite.RewriteHook, the kb_graph_opt flag).

CPU only (torch for the walkers): the whole round loop runs with fake GPU steps (as test_graphopt_remote.RunWiringTest), a fake
KernelBlaster and a fake paired re-timing, with a deterministic proposer (the SqueezeNet expand-merge rewrite on the fire fixture).
Flag off: the hook is inert and the work dir is what it was without the feature.
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

import modelblaster.pipeline.kb_model_run as mr  # noqa: E402
from modelblaster.pipeline import graph_edit as ge  # noqa: E402
from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_rewrite as grm  # noqa: E402
from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, write_variant_ir  # noqa: E402
from modelblaster.pipeline.tests import rewrite_fixtures as F  # noqa: E402
from modelblaster.pipeline.tests.test_kb_graph_opt_parallel import FakeKB  # noqa: E402

TITAN = {"host": None, "label": "0", "index": 0, "uuid": "GPU-titan", "name": "NVIDIA TITAN RTX", "arch": "sm_75"}


def _profile(ir_dir: Path, source_ir: Path) -> dict:
    g = json.loads((Path(ir_dir) / "graph.json").read_text())
    ops = [{"name": o["name"], "op": o["op"], "us": 4.0 if o["op"].startswith(("conv", "fused__conv")) else 1.0, "shape": o.get("shape")}
           for o in g["ops"]]
    tot = sum(o["us"] for o in ops)
    return {"ir": g["name"], "ir_dir": str(ir_dir), "source_ir": str(source_ir), "passed": True, "partition_us": tot, "ops": ops,
            "e2e_us": tot, "launch_us": 0.0, "device": TITAN}


class _Prog:
    """RoundProgress.once without the files."""

    def __init__(self):
        self.d = {}

    def once(self, name, fn):
        if name not in self.d:
            self.d[name] = json.loads(json.dumps(fn(), default=str))
        return self.d[name]


class _Wrong(grm.RewriteProposer):
    name = "wrong"

    def propose(self, ir, profile, budget):
        e, tmap = F.merge_expand(Path(ir["dir"]), Path(budget["work"]) / "wrong", wrong="order")
        return [grm.RewriteCandidate(ir=ir["name"], edited_ir_dir=e, tensor_map=tmap, description="stacks the filters in the wrong order",
                                     est_gain_pct=9.0)]


class _Raises(grm.RewriteProposer):
    name = "raises"

    def propose(self, ir, profile, budget):
        raise RuntimeError("proposer crashed")


class _NewOp(grm.RewriteProposer):
    name = "newop"

    def propose(self, ir, profile, budget):
        return [grm.RewriteCandidate(ir=ir["name"], edited_ir_dir=Path(ir["dir"]), edit_type="new_op", description="a fused gate kernel",
                                     new_ops=[grm.NewOpSpec(kind="fused_gate", signature="void kernel_fused_gate(float*)", semantics="x")])]


class PatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gr_patch_"))
        cls.o = F.fire_ir(cls.tmp / "o", "fp32")
        cls.e, cls.tmap = F.merge_expand(cls.o, cls.tmp / "e")
        cls.g0 = json.loads((cls.o / "graph.json").read_text())
        cls.g1 = json.loads((cls.e / "graph.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_diff_apply_round_trip(self):
        p = ge.diff(self.g0, self.g1, tensor_map=self.tmap)
        p = json.loads(json.dumps(p))                                # a patch is plain JSON
        self.assertEqual(sorted(p["remove_ops"]), ["cat", "e1", "e1_relu", "e3", "e3_relu"])
        self.assertEqual(sorted(o["name"] for o in p["add_ops"]), ["merged", "merged_relu"])
        self.assertEqual(sorted(p["weight_derivations"]), ["d.fire.b", "d.fire.e1_3x3", "d.fire.w"])
        self.assertEqual(p["changed_tensors"], [])
        g2, info = ge.apply_patch(self.g0, p)
        self.assertEqual(ge.graph_digest(g2), ge.graph_digest(self.g1))
        self.assertEqual(sorted(info["added"]), ["merged", "merged_relu"])
        self.assertEqual(ge.new_op_kinds(p), ["conv2d", "relu"])
        # the original is untouched, and the order is topological
        self.assertEqual(len(self.g0["ops"]), 9)
        seen = set(ge.io_names(g2)[0]) | {t for t, m in g2["tensors"].items() if m.get("const")}
        for o in g2["ops"]:
            self.assertTrue(set(o["inputs"]) <= seen, o["name"])
            seen |= set(o["outputs"])

    def test_a_stale_patch_is_refused_with_the_reason(self):
        p = ge.diff(self.g0, self.g1)
        g = copy.deepcopy(self.g0)
        next(o for o in g["ops"] if o["name"] == "e3")["shape"]["OC"] = 99          # another change touched an op the patch removes
        with self.assertRaises(ge.PatchError) as cm:
            ge.apply_patch(g, p)
        self.assertEqual(cm.exception.code, "stale")
        self.assertIn("e3", str(cm.exception))
        g = copy.deepcopy(self.g0)
        g["ops"] = [o for o in g["ops"] if o["name"] != "cat"]
        with self.assertRaises(ge.PatchError) as cm:
            ge.apply_patch(g, p)
        self.assertIn("not in this graph", str(cm.exception))
        g, _ = ge.apply_patch(self.g0, p)
        with self.assertRaises(ge.PatchError):                                       # twice: the removed ops are gone
            ge.apply_patch(g, p)

    def test_slot_variant_annotations_do_not_make_a_patch_stale(self):
        g = copy.deepcopy(self.g0)
        next(o for o in g["ops"] if o["name"] == "e3")["slot_variant"] = "k3"
        g2, _ = ge.apply_patch(g, ge.diff(self.g0, self.g1))
        self.assertEqual(len(g2["ops"]), 6)

    def test_editor_refuses_cycles_and_dangling_inputs(self):
        ed = ge.Editor(self.g0, {})
        ed.op("sq")["inputs"] = ["cat_out"]
        with self.assertRaises(ge.PatchError) as cm:
            ed.result()
        self.assertEqual(cm.exception.code, "cycle")
        ed = ge.Editor(self.g0, {})
        ed.op("e1")["inputs"] = ["nothing"]
        with self.assertRaises(ge.PatchError):
            ed.result()


class HintApplyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gr_hint_"))
        cls.o = {q: F.fire_ir(cls.tmp / f"o_{q}", q) for q in ("fp32", "fp16", "native")}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _hint(self, q, after=()):
        e, tmap = F.merge_expand(self.o[q], self.tmp / f"e_{q}")
        g0, g1 = (json.loads((d / "graph.json").read_text()) for d in (self.o[q], e))
        c = grm.RewriteCandidate(ir="fire", edited_ir_dir=e, tensor_map=tmap, est_gain_pct=5.0)
        c.id = grm.candidate_id(c, ge.diff(g0, g1))
        v = gv.verify(self.o[q], e, tensor_map=tmap, structural_only=True)
        return grm.make_hint(c, ge.diff(g0, g1, tensor_map=tmap), v, 2.0, "t", after_hints=list(after)), g0, g1, e

    def test_apply_hints_gives_the_edited_graph_and_derived_weights_flow_to_the_variant_ir(self):
        for q in ("fp32", "fp16", "native"):
            h, g0, g1, e = self._hint(q)
            self.assertEqual(h["kind"], "rewrite")
            self.assertEqual(h["source"][0]["by"], "agent-rewrite")
            g, added, prov = gh.apply_hints(g0, [h], dict(_all_specs()), statuses=None)
            self.assertEqual(ge.graph_digest(g), ge.graph_digest(g1), q)
            self.assertEqual([a["kind"] for a in prov["applied"]], ["rewrite"])
            self.assertEqual(sorted(prov["added"]), ["merged", "merged_relu"])
            self.assertIn(prov["renamed"]["e1"], ("merged", "merged_relu"))      # removed ops map onto the added ones (the variant partition holds both)
            self.assertIn("weight_derivations", g)
            vir = write_variant_ir(self.o[q], g, self.tmp / f"v_{q}")
            z = np.load(vir / "weights.npz")
            self.assertEqual(z["d.fire.w"].dtype, np.float16 if q != "fp32" else np.float32)
            # what the flow's round 2+ sees is the same as the candidate the gate verified
            v = gv.verify(self.o[q], vir)
            self.assertTrue(v.ok, (q, v.reasons))
            if q == "native":
                self.assertEqual(g["weight_dtypes"]["d.fire.w"], "f16")
            # a graph WITHOUT the hint is not touched: no derivations, nothing added
            g_plain, _, prov2 = gh.apply_hints(g0, [], dict(_all_specs()))
            self.assertNotIn("weight_derivations", g_plain)

    def test_a_rewrite_on_the_wrong_base_is_skipped_not_applied(self):
        h, g0, g1, e = self._hint("fp32")
        g = copy.deepcopy(g0)
        next(o for o in g["ops"] if o["name"] == "e1")["shape"]["OC"] = 7
        g2, _, prov = gh.apply_hints(g, [h], dict(_all_specs()), statuses=None)
        self.assertEqual(prov["applied"], [])
        self.assertIn("does not apply", prov["skipped"][0]["why"])
        self.assertEqual(ge.graph_digest(g2), ge.graph_digest(g))

    def test_ordering_after_the_hints_it_was_proposed_on(self):
        _, g0, g1, e = self._hint("fp32")
        fuse = {"id": "fuse:sq", "kind": "fuse", "target": "cuda", "fused_kind": "fused__conv2d__relu", "chains": [["sq", "sq_relu"]],
                "status": "accepted"}
        specs = dict(_all_specs())
        # a rewrite proposed AFTER the fusion was accepted sits after it, in whatever order the hints file lists them
        gf, _, _ = gh.apply_hints(g0, [fuse], specs)
        h, _, _, _ = self._hint("fp32", after=["fuse:sq"])
        a, _, pa = gh.apply_hints(g0, [h, fuse], specs, statuses=None)
        b, _, pb = gh.apply_hints(g0, [fuse, h], specs, statuses=None)
        self.assertEqual(ge.graph_digest(a), ge.graph_digest(b))
        self.assertEqual([x["kind"] for x in pa["applied"]], ["fuse", "rewrite"])
        self.assertTrue(any(o["op"] == "fused__conv2d__relu" for o in a["ops"]))
        # proposed with nothing accepted yet: it goes first, so a later fusion that names an op it removed is skipped
        fuse2 = {"id": "fuse:e1", "kind": "fuse", "target": "cuda", "fused_kind": "fused__conv2d__relu", "chains": [["e1", "e1_relu"]],
                 "status": "accepted"}
        h0, _, _, _ = self._hint("fp32", after=[])
        c, _, pc = gh.apply_hints(g0, [fuse2, h0], specs, statuses=None)
        self.assertEqual([x["kind"] for x in pc["applied"]], ["rewrite"])
        self.assertTrue(pc["skipped"] and pc["skipped"][0]["id"] == "fuse:e1")
        self.assertEqual(ge.graph_digest(c), ge.graph_digest(g1))
        # and a pattern-level accepted fusion applies to the NEW ops
        fuse3 = {"id": "fuse:cr", "kind": "fuse", "target": "cuda", "fused_kind": "fused__conv2d__relu", "status": "accepted"}
        d, _, pd = gh.apply_hints(g0, [fuse3, h0], specs, statuses=None)
        self.assertEqual([x["kind"] for x in pd["applied"]], ["rewrite", "fuse", "fuse"])         # conv -> relu pairs: sq's and the merged one
        self.assertIn("merged+merged_relu", [o["name"] for o in d["ops"]])

    def test_export_of_the_rewritten_ir_carries_the_derived_weight_as_an_ordinary_buffer(self):
        from modelblaster.pipeline.kb_export import export_problem
        for q in ("fp16", "native"):
            h, g0, g1, e = self._hint(q)
            g, added, _ = gh.apply_hints(g0, [h], dict(_all_specs()), statuses=None)
            vir = write_variant_ir(self.o[q], g, self.tmp / f"x_{q}")
            prob = export_problem(vir, self.tmp / f"p_{q}", strategy="whole")
            man = json.loads((prob / "manifest.json").read_text())
            bufs = {b["name"]: b for b in man["buffers"]}
            self.assertIn("w:d.fire.w", bufs)
            self.assertEqual((bufs["w:d.fire.w"]["numel"], bufs["w:d.fire.w"]["dtype"]), (5 * 3 * 3 * 3, "f16"))
            self.assertNotIn("w:e1.weight", bufs)                         # the replaced ops' weights are not exported
            self.assertEqual(sum(1 for s in man["slots"] for _ in s["instances"]), 6)
            slots = {s["op"] for s in man["slots"]}
            self.assertTrue(all(k.endswith("_f16") for k in slots), slots)   # the per-op / per-slot dtype contract holds


class HookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gr_hook_"))
        cls.o = F.fire_ir(cls.tmp / "o", "fp16")
        cls.g0 = json.loads((cls.o / "graph.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self._env = os.environ.pop(grm.ENV, None)

    def tearDown(self):
        os.environ.pop(grm.ENV, None)
        if self._env is not None:
            os.environ[grm.ENV] = self._env

    def _cfg(self, **kw):
        return go.Config(kb=Path("."), **kw)

    def _propose(self, proposer, name, doc=None, prog=None):
        cfg = self._cfg(graph_rewrite="agent", min_gain_pct=0.5)
        hook = grm.RewriteHook(cfg, self.tmp / name, proposer=proposer)
        doc = doc if doc is not None else gh.new_doc()
        prof = _profile(self.o, self.o)
        new = hook.propose(0, self.tmp / name, prog, {"fire": self.g0}, [prof], doc, dict(_all_specs()))
        return hook, doc, new

    def test_flag_resolution(self):
        self.assertEqual(grm.resolve_mode(self._cfg()), "off")
        os.environ[grm.ENV] = "agent"
        self.assertEqual(grm.resolve_mode(self._cfg()), "agent")
        self.assertEqual(grm.resolve_mode(self._cfg(graph_rewrite="off")), "off")      # the flag wins over the environment
        os.environ[grm.ENV] = "0"
        self.assertEqual(grm.resolve_mode(self._cfg()), "off")
        with self.assertRaises(ValueError):
            grm.resolve_mode(self._cfg(graph_rewrite="on"))
        # --graph-rewrite is a command line flag
        import subprocess
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", "--help"], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[2], env={**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).resolve().parents[3]), os.environ.get("PYTHONPATH", "")])})
        self.assertIn("--graph-rewrite", out.stdout, out.stderr[-500:])
        self.assertIn("--graph-rewrite-proposer", out.stdout)

    def test_off_is_inert(self):
        hook = grm.RewriteHook(self._cfg(), self.tmp / "off")
        self.assertFalse(hook.active())
        self.assertEqual((hook.round_key(), hook.decide_kinds()), ({}, ()))
        doc = gh.new_doc()
        before = json.dumps(doc)
        self.assertEqual(hook.propose(0, self.tmp / "off", None, None, None, doc, {}), [])     # not even the arguments are looked at
        self.assertEqual(json.dumps(doc), before)
        self.assertFalse((self.tmp / "off").exists())
        self.assertIsNone(hook.proposer)

    def test_agent_mode_with_the_llm_stub_raises_not_implemented(self):
        hook = grm.RewriteHook(self._cfg(graph_rewrite="agent"), self.tmp / "stub")
        self.assertIsInstance(hook.proposer, grm.AgentProposer)
        self.assertEqual(hook.round_key(), {"graph_rewrite": "agent"})
        with self.assertRaises(NotImplementedError) as cm:
            hook.propose(0, self.tmp / "stub", None, {"fire": self.g0}, [_profile(self.o, self.o)], gh.new_doc(), dict(_all_specs()))
        self.assertIn("not implemented", str(cm.exception))

    def test_deterministic_proposer_through_the_gate(self):
        hook, doc, new = self._propose(grm.MergeExpandProposer(), "ok")
        self.assertEqual(len(new), 1)
        h = new[0]
        self.assertEqual((h["kind"], h["status"], h["target"]), ("rewrite", "proposed", "cuda"))
        s = h["source"][0]
        self.assertEqual((s["by"], s["proposer"], s["ir"], s["edit_type"]), ("agent-rewrite", "test-merge-expand", "fire", "rewire"))
        self.assertGreater(s["est_gain_pct"], 0)
        self.assertEqual(s["est_cost"]["ops_before"], 9)
        self.assertEqual(s["est_cost"]["ops_after"], 6)
        self.assertTrue(s["verify"]["ok"] and all(s["verify"]["checks"].values()))
        self.assertEqual(len(h["rewrite"]["derived_weights"]), 3)
        self.assertTrue(all(k.startswith("d.merge.") for k in h["rewrite"]["derived_weights"]))
        self.assertEqual(go._est_gain(h), s["est_gain_pct"])                             # what the round loop ranks by
        # asking again (a resumed round) merges the same hint: no duplicate
        hook2, doc2, new2 = self._propose(grm.MergeExpandProposer(), "ok", doc=doc)
        self.assertEqual(len([x for x in doc2["hints"] if x["id"] == h["id"]]), 1)
        self.assertEqual(new2, [])

    def test_the_round_progress_makes_proposals_resumable(self):
        calls = []

        class Counting(grm.MergeExpandProposer):
            def propose(self, *a):
                calls.append(1)
                return super().propose(*a)
        prog = _Prog()
        self._propose(Counting(), "res", prog=prog)
        _, doc, new = self._propose(Counting(), "res", prog=prog)
        self.assertEqual(len(calls), 1)                                                   # the second call replays the saved decisions
        self.assertEqual(len(new), 1)

    def test_a_wrong_rewrite_is_a_rejected_hypothesis_and_never_timed(self):
        hook, doc, new = self._propose(_Wrong(), "wrong")
        self.assertEqual(new, [])
        (h,) = doc["hints"]
        self.assertEqual((h["status"], h["source"][0]["by"]), ("rejected", "agent-rewrite"))
        (ev,) = h["evidence"]
        self.assertEqual((ev["stage"], ev["decision"]), ("graph_verify", "rejected"))
        self.assertIn("output_mismatch", [r["code"] for r in ev["reasons"]])
        self.assertIn("graph_verify", ev["why"])
        # decided hypotheses are not proposed again
        hook2, doc2, new2 = self._propose(_Wrong(), "wrong", doc=doc)
        self.assertEqual((new2, len(doc2["hints"]), len(doc2["hints"][0]["evidence"])), ([], 1, 1))

    def test_region_candidates_become_fuse_region_hints_with_explicit_members(self):
        hook, doc, new = self._propose(grm.RegionProposer(["e1", "e1_relu", "e3", "e3_relu", "cat"], est_gain_pct=4.0), "region")
        self.assertEqual(len(new), 1, doc)
        h = new[0]
        self.assertEqual((h["kind"], h["members"]), ("fuse_region", ["e1", "e1_relu", "e3", "e3_relu", "cat"]))
        self.assertTrue(h["id"].startswith("region:agent:"))
        self.assertEqual(h["source"][0]["est_gain_pct"], 4.0)
        # the flow's own applier gives the region op the walkers evaluate
        g, added, prov = gh.apply_hints(self.g0, [h], dict(_all_specs()), statuses=None)
        self.assertEqual([a["kind"] for a in prov["applied"]], ["fuse_region"])
        self.assertTrue(any(o["op"].startswith("region__") for o in g["ops"]))

    def test_new_op_kinds_are_a_later_phase(self):
        hook, doc, new = self._propose(_NewOp(), "newop")
        self.assertEqual(new, [])
        self.assertIn("new_op_kind", [r["code"] for r in doc["hints"][0]["evidence"][0]["reasons"]])

    def test_a_crashing_proposer_loses_only_its_candidates(self):
        hook, doc, new = self._propose(_Raises(), "raises")
        self.assertEqual((new, doc["hints"]), ([], []))
        self.assertIn("proposer crashed", hook.log[0]["error"])


# ------------------------------------------------------------------------------ the round loop
class LoopTest(unittest.TestCase):
    """kb_graph_opt._run with every GPU step faked and the deterministic proposer."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gr_loop_"))
        cls.o = F.fire_ir(cls.tmp / "ir", "fp16")
        (cls.tmp / "kb" / "scripts").mkdir(parents=True)
        (cls.tmp / "kb" / "scripts" / "time_problems.py").write_text("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan", "remeasure", "KBRunner")}
        self.saved_mr = mr.model_runtime
        self.env = {k: os.environ.get(k) for k in ("KERNELBLASTER_GPU_FLOCK", grm.ENV)}
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        os.environ.pop(grm.ENV, None)
        self.profiles = []

        def profile(irdir, work, *, hints_doc=None, kbeval=None, gpu_model=None, allow_gpu_mismatch=False, **kw):
            d, specs, prov = gh.apply_to_ir(Path(irdir), hints_doc or gh.new_doc(), Path(work) / "ir")
            p = _profile(d, irdir)
            self.profiles.append(p)
            return p

        def remeasure(problems, *, arch, reps, lease=None, **kw):
            return [[100.0 if json.loads((Path(d) / "manifest.json").read_text())["name"].endswith("__orig") else 80.0] * reps
                    for d, _ in problems]
        go.profile_ir = profile
        go.verify_ir = lambda ir, cache, **kw: {"ir": "fire", "passed": True, "partition_us": 5.0, "device": TITAN, "slots": {}}
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
        cfg = go.Config(kb=self.tmp / "kb", arch="75", rounds=3, max_variants=3, remeasure=3, accept_gain=0.03, **kw)
        s = go.run([self.o], self.tmp / name, cfg)
        return s, gh.load(self.tmp / name / "graph_hints.json"), json.loads((self.tmp / name / "state.json").read_text())

    def test_agent_mode_end_to_end(self):
        s, doc, st = self._run("on", graph_rewrite="agent", graph_rewrite_proposer="test-merge-expand")
        h = next(x for x in doc["hints"] if x["kind"] == "rewrite")
        self.assertEqual(h["status"], "accepted")
        self.assertEqual(h["source"][0]["by"], "agent-rewrite")
        ev = h["evidence"][-1]
        self.assertEqual((ev["decision"], ev["original_us"], ev["variant_us"]), ("accepted", 100.0, 80.0))     # the paired timing
        self.assertAlmostEqual(ev["gain"], 0.2)
        self.assertEqual(ev["applied"][0]["kind"], "rewrite")
        self.assertEqual(len(ev["applied"][0]["derived_weights"]), 3)
        self.assertIn(h["id"], s["accepted"])
        self.assertEqual(s["graph_rewrite"]["mode"], "agent")
        self.assertEqual(s["graph_rewrite"]["log"][0]["ok"], True)
        self.assertEqual(st["rounds"][0]["decisions"][-1]["hint"], h["id"])
        # the paired problems were exported from the original and the rewritten graph, the variant holding the new ops
        var = next((self.tmp / "on" / "r00" / "variants").glob("rewrite*/variant/manifest.json"))
        man = json.loads(var.read_text())
        self.assertTrue(any(n.startswith("mrg_") for n in man["partition"]["op_names"]), man["partition"]["op_names"])
        self.assertNotIn("e1", man["partition"]["op_names"])
        buf = {b["name"] for b in man["buffers"]}
        self.assertTrue(any(b.startswith("w:d.merge.") and b.endswith(".w") for b in buf))
        # from the next round on the rewrite applies to the profiled IR (derived weights materialised in its dir)
        last = self.profiles[-1]
        g = json.loads((Path(last["ir_dir"]) / "graph.json").read_text())
        self.assertEqual(len(g["ops"]), 4 + 0 if any("+" in o["name"] for o in g["ops"]) else 6)    # + the pattern-level fusion, if any
        self.assertTrue(any(k.startswith("d.merge.") and k.endswith(".w") for k in np.load(Path(last["ir_dir"]) / "weights.npz").files))
        # the candidate went through the gate before it was timed
        self.assertTrue(h["source"][0]["verify"]["ok"])
        self.assertIn("graph_rewrite", st["rounds"][0]["key"] if isinstance(st["rounds"][0].get("key"), dict) else "graph_rewrite")

    def test_off_equals_unset_equals_no_feature(self):
        a = self._run("a")                                          # the flag untouched
        b = self._run("b", graph_rewrite="off")
        os.environ[grm.ENV] = "off"
        c = self._run("c")
        os.environ.pop(grm.ENV)

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
        sa, _, sta = a
        self.assertNotIn("graph_rewrite", sa)
        self.assertEqual(sta["rounds"][0]["key"], self._key_of("b"))                   # same round key as without the feature
        self.assertFalse((self.tmp / "a" / "r00" / "rewrite").exists())
        self.assertTrue(all(h["kind"] != "rewrite" for h in a[1]["hints"]))

    def _key_of(self, name):
        return json.loads((self.tmp / name / "state.json").read_text())["rounds"][0]["key"]

    def test_the_flag_changes_the_round_key_only_when_on(self):
        self._run("k_off")
        self._run("k_on", graph_rewrite="agent", graph_rewrite_proposer="test-merge-expand")
        self.assertNotEqual(self._key_of("k_off"), self._key_of("k_on"))


if __name__ == "__main__":
    unittest.main()
