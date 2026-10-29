"""The A6 flow mechanisms on ``--target opencl`` (docs/opencl_a6.md), without a device.

Supported: ``--round-controller`` (flow level; its candidate counting follows the flow's target), ``--merge-sibling-convs`` (a graph
rewrite: the hook stamps its hints with the target, the applier and the OpenCL export handle the merged conv as an ordinary conv2d
shape). Refused loudly (never a silent no-op): ``--region-slots`` / ``--region-windows`` / ``--region-loop`` (CUDA composite seed,
CUDA loop backend), ``--view-fusion``, ``--graph-rewrite``, ``--sol-gate``, ``--multiturn``. The CUDA-only cache routing of the search
fixes is off on OpenCL. CPU only (torch for the walkers and the export).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline import merge_convs as mc  # noqa: E402
from modelblaster.pipeline import search_fixes as sfx  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, write_variant_ir  # noqa: E402
from modelblaster.pipeline.tests import merge_fixtures as MF  # noqa: E402

ENVS = ("MB_REGION_WINDOWS", "MB_REGION_LOOP", "MB_GRAPH_REWRITE", "MB_SOL_GATE", "MB_MERGE_SIBLING_CONVS", "MB_ROUND_CONTROLLER")
A6_CUDA = dict(round_controller="auto", round_controller_budget_usd=40.0, merge_sibling_convs="auto", region_windows="auto",
               region_loop="on", region_loop_budget="40,120,4,3600", rounds=3)


class _Env(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.pop(k, None) for k in ENVS}

    def tearDown(self):
        for k, v in self._env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def cfg(**kw):
    kw.setdefault("target", "opencl")
    return go.Config(kb=Path("."), arch="adreno650", board_host="root@rb5", remote_dir="/root/mb", **kw)


class RefusalTest(_Env):
    def test_cuda_never_refuses(self):
        self.assertEqual(go.opencl_refusals(go.Config(kb=Path("."), **A6_CUDA, region_slots="all", sol_gate="final")), [])

    def test_the_a6_set_without_regions_is_accepted_on_opencl(self):
        a6 = {k: v for k, v in A6_CUDA.items() if not k.startswith("region_")}
        self.assertEqual(go.opencl_refusals(cfg(**a6)), [])

    def test_each_cuda_only_flag_is_refused_with_its_name(self):
        for kw, name in (({"region_windows": "auto"}, "--region-windows"), ({"region_loop": "on"}, "--region-loop"),
                         ({"region_slots": "all"}, "--region-slots"), ({"view_fusion": "on"}, "--view-fusion"),
                         ({"graph_rewrite": "agent"}, "--graph-rewrite"), ({"sol_gate": "final"}, "--sol-gate"),
                         ({"multiturn": "on"}, "--multiturn")):
            with self.subTest(flag=name):
                why = go.opencl_refusals(cfg(**kw))
                self.assertEqual(len(why), 1)
                self.assertIn(name, why[0])

    def test_environment_switches_are_refused_too(self):
        os.environ["MB_REGION_LOOP"] = "on"
        os.environ["MB_REGION_WINDOWS"] = "auto"
        os.environ["MB_SOL_GATE"] = "accept"
        why = go.opencl_refusals(cfg())
        self.assertEqual(len(why), 3)
        os.environ["MB_REGION_LOOP"] = "off"
        self.assertEqual(len(go.opencl_refusals(cfg())), 2)

    def test_full_a6_on_opencl_raises_before_any_work(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError) as cm:
                go.run([Path(t)], Path(t) / "w", cfg(**A6_CUDA))
            m = str(cm.exception)
            self.assertIn("--region-windows", m)
            self.assertIn("--region-loop", m)
            self.assertIn("--round-controller, --merge-sibling-convs", m)         # what is supported
            self.assertFalse((Path(t) / "w").exists())                          # nothing was created

    def test_cli_exits_2_naming_the_flag(self):
        with tempfile.TemporaryDirectory() as t:
            r = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", t, "--work", t + "/w", "--kb", t,
                                "--target", "opencl", "--board-host", "root@rb5", "--remote-dir", "/root/mb", "--arch", "adreno650",
                                "--region-loop", "on", "--region-windows", "auto"], capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])})
        self.assertEqual(r.returncode, 2, r.stderr[-500:])
        self.assertIn("--target opencl does not support", r.stderr)
        self.assertIn("--region-loop", r.stderr)


class MergeOpenCLTest(_Env):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_ocl_a6_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _hints(self, target):
        o = MF.branchy_ir(self.tmp / f"o_{target}", MF.FIRE, "fp16")
        g = json.loads((o / "graph.json").read_text())
        hk = mc.MergeHook(cfg(merge_sibling_convs="auto", min_gain_pct=0.0, **({} if target == "opencl" else {"target": "cuda"})))
        doc = gh.new_doc()
        prof = {"ir": "fire", "ops": [{"name": "cat", "us": 30.0}, {"name": "c1", "us": 10.0}, {"name": "c0", "us": 5.0}], "e2e_us": 200.0}
        hk.propose({"fire": g}, [prof], doc, dict(_all_specs()))
        return o, g, doc

    def test_the_hook_stamps_the_flows_target(self):
        _, _, doc = self._hints("opencl")
        self.assertEqual([h["target"] for h in doc["hints"]], ["opencl"])
        _, _, doc = self._hints("cuda")
        self.assertEqual([h["target"] for h in doc["hints"]], ["cuda"])

    def test_the_applier_takes_only_this_targets_hint(self):
        o, g, doc = self._hints("opencl")
        h = {**doc["hints"][0], "status": "accepted"}
        g1, _, prov = gh.apply_hints(g, [h], dict(_all_specs()), target="opencl")
        self.assertEqual(len(prov["applied"]), 1, prov)
        self.assertLess(len(g1["ops"]), len(g["ops"]))
        g2, _, prov2 = gh.apply_hints(g, [h], dict(_all_specs()), target="cuda")
        self.assertEqual(prov2["applied"], [])
        self.assertIn("decided for target opencl", prov2["skipped"][0]["why"])

    def test_the_merged_ir_exports_for_opencl(self):
        from modelblaster.pipeline.kb_export import export_problem
        o, g, doc = self._hints("opencl")
        h = {**doc["hints"][0], "status": "accepted"}
        g1, _, prov = gh.apply_hints(g, [h], dict(_all_specs()), target="opencl")
        e = write_variant_ir(o, g1, self.tmp / "merged_ir")
        prob = export_problem(e, self.tmp / "merged_problem", strategy="whole", target="opencl")
        man = json.loads((prob / "manifest.json").read_text())
        self.assertEqual(man["target"], "opencl")
        convs = [s for s in man["slots"] if s["op"].startswith("conv2d")]
        self.assertEqual(sorted(i["shape"]["OC"] for s in convs for i in s["instances"]), [4, 5])   # the stem + the merged conv
        for s in convs:
            src = (prob / s["file"]).read_text()
            self.assertIn("__kernel", src)                                                         # an OpenCL C program, not CUDA
            self.assertIn("mb_launch", src)
            self.assertNotIn("__global__", src)
        self.assertTrue((prob / "driver.cpp").exists())
        tail = (prob / man["walker_tail"]).read_text() if "walker_tail" in man else next(prob.glob("walker_tail.*")).read_text()
        self.assertIn("mb_any{b[", tail)                                                           # the call sites use it ...
        self.assertIn("struct mb_any", tail)                                                       # ... so the OpenCL tail defines it (board build: 'mb_any was not declared')


class FixesTest(_Env):
    def test_cache_routing_is_cuda_only(self):
        f = sfx.Fixes(cfg(), Path("/tmp"))
        self.assertEqual((f.fallback, f.routing, f.requeueing), ("on", False, True))
        self.assertIsNone(f.ingest_requeued([], None, {}))
        self.assertTrue(sfx.Fixes(go.Config(kb=Path(".")), Path("/tmp")).routing)                  # CUDA as before


class ControllerOnOpenCLTest(_Env):
    """kb_graph_opt._run with every timed step faked: the round controller counts the merge hint (stamped ``opencl``) as a graph candidate."""

    def test_controller_sees_the_opencl_merge_candidate(self):
        import modelblaster.pipeline.kb_model_run as mr
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "kb" / "scripts").mkdir(parents=True)
            (t / "kb" / "scripts" / "time_problems.py").write_text("")
            o = MF.branchy_ir(t / "ir" / "fire", MF.FIRE, "fp16")
            ADRENO = {"host": "root@rb5", "label": "root@rb5", "index": None, "uuid": None,
                      "name": "QUALCOMM Adreno(TM) | OpenCL 2.0 Adreno(TM) 650", "arch": "adreno650"}
            saved = (go.profile_ir, go.verify_ir, mr.model_runtime, go.evaluate_hints, go.make_plan)
            seen = {}

            def profile(ir, work, *, kbeval=None, target="cuda", cache=None, arch=None, **kw):
                Path(work).mkdir(parents=True, exist_ok=True)
                shutil.copy(Path(ir) / "graph.json", Path(work) / "graph.json")
                g = json.loads((Path(ir) / "graph.json").read_text())
                ops = [{"name": x["name"], "op": x["op"], "us": 100.0 if x["name"] in ("c0", "c1", "cat") else 1.0} for x in g["ops"]]
                return {"ir": "fire", "ir_dir": str(ir), "source_ir": str(ir), "passed": True, "partition_us": sum(x["us"] for x in ops),
                        "ops": ops, "e2e_us": 400.0, "launch_us": 20.0, "device": ADRENO}

            def decide(items, *a, **kw):
                seen["items"] = [(h["id"], h["target"]) for h, _, _ in items]
                return [{"hint": h["id"], "status": "rejected", "hint_status": "rejected", "gain": 0.0} for h, _, _ in items]

            go.profile_ir = profile
            go.verify_ir = lambda ir, cache, **kw: {"ir": "fire", "passed": True, "partition_us": 5.0, "device": ADRENO, "slots": {}}
            mr.model_runtime = lambda ir, **kw: {"device": ADRENO, "passed": True, "model_e2e_us": 4.0, "seed_e2e_us": 8.0, "speedup": 2.0}
            go.evaluate_hints = decide
            go.make_plan = lambda profiles, **kw: {"problems": [], "selected_share": 0.0, "items": [], "profiles": []}
            try:
                c = go.Config(kb=t / "kb", target="opencl", arch="adreno650", board_host="root@rb5", remote_dir="/root/mb",
                                            rounds=2, round_controller="graph-first", merge_sibling_convs="auto", min_gain_pct=0.0)
                s = go.run([o], t / "w", c)
            finally:
                go.profile_ir, go.verify_ir, mr.model_runtime, go.evaluate_hints, go.make_plan = saved
                from modelblaster.pipeline import region_loop as rlm
                rlm.set_round(None)                                   # the controller's per-round switch is module state
            rc = s["round_controller"]
            self.assertEqual(rc["policy"], "graph-first")
            self.assertIn("2 candidate(s)", rc["rounds"][0]["reasons"][0])             # the fuse + the merge hint, both stamped opencl, counted
            self.assertEqual(rc["rounds"][0]["move"], "graph")
            self.assertEqual(s["merge_sibling_convs"]["mode"], "auto")
            self.assertTrue(seen.get("items"), "the merge hint was never decided")
            self.assertEqual(seen["items"][0][1], "opencl")


if __name__ == "__main__":
    unittest.main()
