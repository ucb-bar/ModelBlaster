"""The graph optimiser on the OpenCL (Qualcomm Adreno) target, without a device.

- the OpenCL kernel cache (slot target, ``.cpp`` sources, device-label arch), ingest per target;
- hints decided per slot target; the flow stamps its own hints;
- N-stage OpenCL fused seeds from the current-best stage kernels (g++ syntax check where the
  OpenCL headers are installed);
- the driver's replay line (``MB_E2E_US basis=replay``) and its parse;
- KernelBlaster commands (run_multislot, time_problems) for the board;
- profile / verify / model run through a fake board evaluator: OpenCL slots exported, the jobs
  sent, the device recorded, the seed model as the baseline (no torch on the board): needs torch
  (CPU) for the export;
- ``kb_graph_opt._run`` wiring with every timed step faked.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_eval as ke  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline.kb_ingest import CudaCache, arch_key, ingest  # noqa: E402
from modelblaster.pipeline.kb_proposals import _all_specs, make_chain_spec  # noqa: E402
from modelblaster.pipeline.kernel_abi import parse_signature  # noqa: E402
from modelblaster.pipeline.tests.test_kb_model_run_eval import _have_torch, _tiny_ir  # noqa: E402

ADRENO = {"host": "root@rb5", "label": "root@rb5", "index": None, "uuid": None,
          "name": "QUALCOMM Adreno(TM) | OpenCL 2.0 Adreno(TM) 650", "arch": "adreno650"}


def _add(cache, op="relu", arch="adreno650", code="// k\n", shape=None):
    spec = _all_specs()[op]
    return cache.add(op=op, entry=parse_signature(spec.signature)[0], signature=spec.signature, arch=arch,
                     shapes=[shape or {"n": 8}], code=code, slot_us=1.0, speedup=2.0, impl_id=f"t:{op}", source={})


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_arch_key(self):
        self.assertEqual((arch_key("75"), arch_key("sm_89"), arch_key("adreno650", "opencl")),
                         ("sm_75", "sm_89", "adreno650"))

    def test_targets_never_mix(self):
        ocl = CudaCache(self.d / "c", target="opencl")
        e, st = _add(ocl, code="// opencl relu\n")
        ocl.save()
        self.assertEqual((st, e["target"], e["arch"]), ("added", "opencl", "adreno650"))
        self.assertTrue(e["file"].endswith(".cpp") and (self.d / "c" / e["file"]).exists())
        spec = _all_specs()["relu"]
        entry = parse_signature(spec.signature)[0]
        self.assertIsNotNone(ocl.lookup(entry, spec.signature, "adreno650", [{"n": 8}]))
        # the same index read as a CUDA cache: the entry is not its own, even under the same arch string
        cu = CudaCache(self.d / "c")
        self.assertIsNone(cu.lookup(entry, spec.signature, "adreno650", [{"n": 8}]))
        e2, st2 = _add(cu, arch="adreno650", code="// cuda relu\n")
        self.assertEqual(st2, "added")                     # not "kept-existing": a separate entry
        self.assertNotIn("target", e2)                     # CUDA entries as before
        self.assertTrue(e2["file"].endswith(".cu"))
        with self.assertRaises(ValueError):
            CudaCache(self.d / "x", target="metal")

    def test_ingest_takes_only_the_caches_target(self):
        spec = _all_specs()["relu"]
        from modelblaster.pipeline.kb_ingest import signature_sha
        from modelblaster.pipeline.cuda_dtypes import cuda_signature
        w = self.d / "w"
        (w / "code").mkdir(parents=True)
        (w / "code" / "a.cu").write_text("// cl\n")
        rows = [{"id": f"i{t}", "op": "relu", "signature_sha": signature_sha(cuda_signature(spec.signature)),
                 "arch": a, "target": t, "shapes": [{"n": 8}], "file": "code/a.cu", "slot_us": 1.0,
                 "speedup_vs_seed": 2.0} for t, a in (("opencl", "adreno650"), ("cuda", "sm_75"))]
        (w / "winners.json").write_text(json.dumps({"format": "kb-winners/1", "winners": rows}))
        ocl = CudaCache(self.d / "c2", target="opencl")
        rep = ingest(w / "winners.json", ocl, dict(_all_specs()))
        self.assertEqual([r["status"] for r in rep], ["added", "other-target"])
        self.assertEqual([(e["arch"], e["target"]) for e in ocl.entries], [("adreno650", "opencl")])
        cu = CudaCache(self.d / "c3")
        self.assertEqual([r["status"] for r in ingest(w / "winners.json", cu, dict(_all_specs()))],
                         ["other-target", "added"])


class HintTargetTest(unittest.TestCase):
    def test_hints_apply_on_their_own_target(self):
        g = {"name": "m", "ops": [], "tensors": {}, "output": {"tensor": "y"}}
        hs = [{"id": "split:a", "kind": "split_slot", "target": "cuda", "slot": "a", "groups": [], "status": "accepted"},
              {"id": "split:b", "kind": "split_slot", "target": "opencl", "slot": "b", "groups": [],
               "status": "accepted"}]
        _, _, prov = gh.apply_hints(g, hs, target="opencl")
        why = {s["id"]: s["why"] for s in prov["skipped"]}
        self.assertIn("decided for target cuda", why.get("split:a", ""))
        self.assertNotIn("decided for target", why.get("split:b", ""))      # reaches the split applier
        _, _, prov = gh.apply_hints(g, hs, target="cuda")
        self.assertIn("decided for target opencl", {s["id"]: s["why"] for s in prov["skipped"]}.get("split:b", ""))

    def test_the_flow_stamps_its_target(self):
        cfg = go.Config(kb=Path("."), target="opencl")
        hs = go._targeted([{"id": "fuse:x", "target": "cuda"}, {"id": "xpurt:y", "target": "riscv"}], cfg)
        self.assertEqual([h["target"] for h in hs], ["opencl", "riscv"])
        self.assertEqual(go._targeted([{"id": "fuse:x", "target": "cuda"}], go.Config(kb=Path(".")))[0]["target"],
                         "cuda")


def _have_cl_headers() -> bool:
    if not shutil.which("g++"):
        return False
    return subprocess.run(["g++", "-fsyntax-only", "-x", "c++", "-"], input="#include <CL/cl.h>\n",
                          capture_output=True, text=True).returncode == 0


class FusedSeedTest(unittest.TestCase):
    def test_three_stage_seed_from_current_bests(self):
        from modelblaster.pipeline.kb_export_opencl import HAND_LIB, RUNTIME_H
        from modelblaster.pipeline.kb_export import wrap_slot
        from modelblaster.pipeline.opencl_seeds import OpenCLSeeds

        specs = _all_specs()
        spec = make_chain_spec(["conv2d", "relu", "sigmoid"], ["input", "input"], specs)
        lib = OpenCLSeeds(HAND_LIB)
        best_relu = lib.slot_source("kernel_relu").replace("k_relu", "k_relu_best")
        src = lib.fused_seed(spec, {"relu": best_relu})
        for k in ("conv2d", "relu", "sigmoid"):
            self.assertIn(f"namespace stage_{k} {{", src)
        self.assertIn("k_relu_best", src)                                  # the current best stage
        self.assertIn("// current best relu kernel", src)
        entry = parse_signature(spec.signature)[0]
        self.assertIn(f"void {entry}(", src)
        self.assertIn("stage_relu::kernel_relu(a_output", src)             # stage 1 reads stage 0's output
        self.assertIn("stage_sigmoid::kernel_sigmoid(b_output", src)
        if _have_cl_headers():
            tu = RUNTIME_H.read_text() + "\n" + wrap_slot("f", src)
            cp = subprocess.run(["g++", "-std=c++11", "-w", "-fsyntax-only", "-x", "c++", "-"], input=tu,
                                capture_output=True, text=True)
            self.assertEqual(cp.returncode, 0, cp.stderr[-2000:])


class DriverReplayTest(unittest.TestCase):
    def test_driver_has_the_replay_and_it_parses(self):
        from modelblaster.pipeline.kb_export import parse_driver_output
        from modelblaster.pipeline.kb_export_opencl import _DRIVER_TEMPLATE

        for line in ("MB_E2E_US %.3f basis=replay", "MB_ATTR_SUM_US %.3f basis=events", "MB_LAUNCH_OVERHEAD_US eager="):
            self.assertIn(line, _DRIVER_TEMPLATE)
        out = ("MB_PARTITION_US 100.0 reps=3\nMB_ATTR_SUM_US 90.0 basis=events reps=3 launches=6.0\n"
               "MB_E2E_US 120.0 basis=replay mean=121.0 reps=5\nMB_LAUNCH_OVERHEAD_US eager=30.0 frac_eager=0.25\n"
               "passed\n")
        r = parse_driver_output(out)
        self.assertEqual((r["passed"], r["e2e_us"], r["e2e_basis"]), (True, 120.0, "replay"))


class CommandsTest(unittest.TestCase):
    def cfg(self, **kw):
        return go.Config(kb=Path("/kb"), target="opencl", arch="adreno650", board_host="root@rb5",
                         remote_dir="/root/mb", board_lock="/root/mb_board.lock", ssh_key="/k",
                         board_env=["A=1"], gpu_model="X", **kw)

    def test_kb_runs_on_the_board(self):
        cmd = " ".join(go.KBRunner(self.cfg(), Path("/w/knowledge")).command([Path("/p")], Path("/o")))
        for part in ("--evaluator board", "--board-host root@rb5", "--remote-dir /root/mb", "--ssh-key /k",
                     "--board-lock /root/mb_board.lock", "--board-env A=1", "--arch adreno650"):
            self.assertIn(part, cmd)
        self.assertNotIn("--gpu-model", cmd)                    # CUDA pool flags stay off
        self.assertNotIn("--ssh-host", cmd)
        local = go.KBRunner(go.Config(kb=Path("/kb"), target="opencl", arch="titanrtx"), Path("/k"))
        self.assertIn("--evaluator local", " ".join(local.command([Path("/p")], Path("/o"))))

    def test_mb_timings_through_the_board(self):
        cfg = self.cfg()
        self.assertTrue(go.use_kb_eval(cfg))
        with self.assertRaises(ValueError):
            go.use_kb_eval(go.Config(kb=Path("."), target="opencl", mb_evaluator="builtin"))
        ev = ke.KBEval(evaluator="board", kb=Path("/kb"), board_host="root@rb5", remote_dir="/root/mb",
                       board_lock="/root/mb_board.lock", board_env=["A=1"])
        cmd = " ".join(ev.command(Path("j"), Path("o"), arch="adreno650", work=Path("w"), gpu_model="Adreno"))
        for part in ("--evaluator board", "--board-host root@rb5", "--remote-dir /root/mb", "--arch adreno650",
                     "--board-lock /root/mb_board.lock", "--board-env A=1", "--gpu-model Adreno"):
            self.assertIn(part, cmd)
        self.assertIs(ev.for_device({"host": "root@rb5", "ident": "opencl"}), ev)
        self.assertEqual(go.detect_gpus(cfg)["remote"][0]["host"], "root@rb5")


class FakeBoard(ke.KBEval):
    """Answers like time_problems.py --evaluator board: one device, replay e2e."""

    def __init__(self, ok=True):
        super().__init__(evaluator="board", board_host="root@rb5", remote_dir="/root/mb")
        self.ok, self.calls = ok, []

    def available(self):
        return True

    def run(self, jobs, *, arch, work, gpu_model=None, name="time", driver_env=None):
        self.calls.append({"jobs": jobs, "arch": arch, "gpu_model": gpu_model, "name": name})
        out = {}
        for j in jobs:
            man = json.loads((Path(j["problem"]) / "manifest.json").read_text())
            self.calls[-1].setdefault("targets", []).append(man["target"])
            ops = {o["name"]: 1.0 + k for k, o in enumerate(man["ops"])}
            row = {"ok": self.ok, "stage": "ok", "partition_us": 50.0, "e2e_us": 60.0 if j["id"] != "seed" else 90.0,
                   "e2e_basis": "replay", "op_us": ops, "attr_sum_us": sum(ops.values()),
                   "attr_launches": float(len(ops)), "slot_checks": {s["id"]: {"rel_err": 0.0, "ok": True}
                                                                    for s in man["slots"]},
                   "slot_us": {}, "outputs": {}, "errors": [], "runtime": "process", "gpu": ADRENO}
            refs = [dict(row, e2e_us=90.0)] * j["reps"] if j.get("incumbent") else []
            out[j["id"]] = {"id": j["id"], "runs": [row] * j["reps"], "ref_runs": refs, "gpu": ADRENO,
                            "gpus": [ADRENO["label"]], "paired": bool(j.get("incumbent")),
                            "pairing": "interleaved" if j.get("incumbent") else None}
        return {"jobs": out, "seconds": 1.0, "models": [ADRENO["name"]], "pool": []}


@unittest.skipUnless(_have_torch(), "needs torch (CPU) for the export")
class BoardStepsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.ir = _tiny_ir(cls.d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_profile_exports_opencl_and_times_on_the_board(self):
        from modelblaster.pipeline.kb_plan import profile_ir

        ev = FakeBoard()
        p = profile_ir(self.ir, self.d / "prof", arch="adreno650", runs=2, kbeval=ev, target="opencl",
                       cache=CudaCache(self.d / "pc", target="opencl"))
        c = ev.calls[0]
        self.assertEqual((c["name"], c["arch"], c["targets"]), ("profile", "adreno650", ["opencl"]))
        self.assertTrue(p["passed"])
        self.assertEqual((p["arch"], p["target"], p["device"]), ("adreno650", "opencl", ADRENO))
        self.assertEqual(p["op_time_basis"], "events")
        self.assertGreater(p["launch_us"], 0.0)                 # replay minus kernel time, per launch
        with self.assertRaises(ValueError):                    # no builtin OpenCL runs
            profile_ir(self.ir, self.d / "prof2", arch="adreno650", target="opencl")
        with self.assertRaises(ValueError):                    # a CUDA cache for an OpenCL profile
            profile_ir(self.ir, self.d / "prof3", arch="adreno650", target="opencl", kbeval=ev,
                       cache=CudaCache(self.d / "cc"))

    def test_verify_and_model_run_with_an_opencl_cache(self):
        from modelblaster.pipeline.kb_export import export_problem
        from modelblaster.pipeline.kb_ingest import verify_ir
        from modelblaster.pipeline.kb_model_run import model_runtime

        prob = export_problem(self.ir, self.d / "seed", strategy="whole", target="opencl")
        man = json.loads((prob / "manifest.json").read_text())
        s = man["slots"][0]
        self.assertTrue(s["file"].endswith(".cpp"))
        cache = CudaCache(self.d / "vc", target="opencl")
        spec = _all_specs()[s["op"]]
        cache.add(op=s["op"], entry=parse_signature(spec.signature)[0], signature=spec.signature, arch="adreno650",
                  shapes=[i["shape"] for i in s["instances"]], code=(prob / s["file"]).read_text() + "// best\n",
                  slot_us=1.0, speedup=2.0, impl_id="t:" + s["id"], source={})
        cache.save()
        ev = FakeBoard()
        v = verify_ir(self.ir, cache, arch="adreno650", specs=dict(_all_specs()), work=self.d / "ver", kbeval=ev)
        self.assertTrue(v["passed"])
        self.assertEqual((ev.calls[0]["targets"], v["device"]), (["opencl"], ADRENO))
        self.assertTrue(any(e["model_verified"] for e in cache.entries))
        m = model_runtime(self.ir, work=self.d / "model", cache=cache, arch="adreno650", kbeval=ev,
                          torch_baseline="auto")
        self.assertEqual((m["target"], m["arch"], m["passed"]), ("opencl", "adreno650", True))
        self.assertEqual((m["model_e2e_us"], m["seed_e2e_us"], m["model_basis"]), (60.0, 90.0, "replay"))
        self.assertAlmostEqual(m["speedup"], 1.5)
        self.assertEqual(m["torch_status"], "skipped")         # no torch on the board: the seed is the baseline
        self.assertIn("seed model", m["torch_reason"])
        self.assertEqual(set(ev.calls[-1]["targets"]), {"opencl"})
        with self.assertRaises(ValueError):
            model_runtime(self.ir, work=self.d / "m2", cache=cache, arch="adreno650", kbeval=None)


class RunWiringTest(unittest.TestCase):
    """kb_graph_opt._run on the OpenCL target with every timed step faked."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "kb" / "scripts").mkdir(parents=True)
        (self.t / "kb" / "scripts" / "time_problems.py").write_text("")
        (self.t / "ir").mkdir()
        (self.t / "ir" / "graph.json").write_text(json.dumps({"ops": [], "output": {"tensor": "y"}}))
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir")}
        import modelblaster.pipeline.kb_model_run as mr
        self.mr, self.saved_mr = mr, mr.model_runtime
        self.calls = {"profile": [], "verify": [], "model": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        self.mr.model_runtime = self.saved_mr
        self.tmp.cleanup()

    def test_every_step_on_the_board(self):
        def profile(ir, work, *, kbeval=None, target="cuda", cache=None, arch=None, **kw):
            self.calls["profile"].append({"kbeval": kbeval, "target": target, "cache": cache.target, "arch": arch})
            Path(work).mkdir(parents=True, exist_ok=True)
            (Path(work) / "graph.json").write_text(json.dumps({"ops": [], "output": {"tensor": "y"}}))
            return {"ir": "tiny", "ir_dir": str(work), "source_ir": str(ir), "passed": True, "partition_us": 10.0,
                    "ops": [], "e2e_us": 9.0, "launch_us": 0.0, "device": ADRENO}

        def verify(ir, cache, *, kbeval=None, **kw):
            self.calls["verify"].append({"kbeval": kbeval, "cache": cache.target})
            return {"ir": "tiny", "passed": True, "partition_us": 5.0, "device": ADRENO, "slots": {}}

        def model(ir, *, kbeval=None, target="cuda", gpu_model=None, **kw):
            self.calls["model"].append({"kbeval": kbeval, "target": target, "gpu_model": gpu_model})
            return {"device": ADRENO, "passed": True, "model_e2e_us": 4.0, "seed_e2e_us": 8.0, "speedup": 2.0}

        go.profile_ir, go.verify_ir, self.mr.model_runtime = profile, verify, model
        cfg = go.Config(kb=self.t / "kb", target="opencl", arch="adreno650", board_host="root@rb5",
                        remote_dir="/root/mb", rounds=1)
        s = go.run([self.t / "ir"], self.t / "w", cfg)
        p, v, m = self.calls["profile"][0], self.calls["verify"][0], self.calls["model"][0]
        self.assertEqual((p["target"], p["cache"], p["arch"], p["kbeval"].evaluator), ("opencl", "opencl",
                                                                                        "adreno650", "board"))
        self.assertEqual((v["cache"], v["kbeval"].evaluator), ("opencl", "board"))
        self.assertEqual((m["target"], m["kbeval"].evaluator, m["gpu_model"]), ("opencl", "board", ADRENO["name"]))
        self.assertEqual(s["gpu"]["gpu_model"], ADRENO["name"])
        self.assertEqual(s["irs"]["tiny"]["model_speedup"], 2.0)
        st = json.loads((self.t / "w" / "state.json").read_text())
        self.assertEqual(st["gpus"]["mb_evaluator"]["evaluator"], "board")
        key = go.kb_resume.round_key(cfg, [self.t / "ir"])
        eff = st["efficiency"]                          # the E7 efficiency rules (on by default) are part of this flow's key
        key["efficiency"] = {"rules": eff["rules"], "stop_patience": eff["stop_patience"],
                             "plan_min_share_late": eff["plan_min_share_late"]}
        self.assertEqual(st["rounds"][0]["key"], go.kb_resume.key_hash(key))
        self.assertEqual(go.kb_resume.round_key(cfg, [])["target"], "opencl")
        self.assertNotIn("target", go.kb_resume.round_key(go.Config(kb=Path(".")), []))   # CUDA keys unchanged


if __name__ == "__main__":
    unittest.main()
