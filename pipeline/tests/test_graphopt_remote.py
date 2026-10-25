"""The graph optimiser's own GPU steps -- per-round profiles (kb_plan.profile_ir) and the final
verification (kb_ingest.verify_ir) -- through KernelBlaster's evaluators, and one GPU model for
every GPU step of a kb_graph_opt run.

- result mapping, median choice, device checks, the flow's GPU model, leases, flags: everywhere;
- profile_ir / verify_ir with a fake evaluator (the job they send, the fields, the device): needs
  torch (CPU) for the export;
- the wiring of kb_graph_opt._run with fake profile / KernelBlaster / verify / model steps;
- parity of the builtin path and KernelBlaster's local evaluator on one problem: needs nvcc, a CUDA
  GPU and a KernelBlaster checkout with scripts/time_problems.py (``MB_KB_ROOT``).
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

from modelblaster.pipeline import kb_eval as ke  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402
from modelblaster.pipeline.tests.test_kb_model_run_eval import L40S, _have_torch, _tiny_ir  # noqa: E402

TITAN = {"host": None, "label": "0", "index": 0, "uuid": "GPU-titan", "name": "NVIDIA TITAN RTX", "arch": "sm_75"}


def _row(part, *, ok=True, gpu=L40S, ops=None, slots=None):
    return {"ok": ok, "stage": "ok" if ok else "check", "partition_us": part, "partition_graph_us": part - 1,
            "e2e_us": part - 1, "e2e_basis": "graph", "op_us": dict(ops or {}), "attr_op_us": dict(ops or {}),
            "attr_sum_us": sum((ops or {}).values()), "attr_launches": float(len(ops or {})),
            "launch_overhead_frac": 0.1, "slot_checks": dict(slots or {}), "slot_us": {}, "outputs": {},
            "errors": [], "runtime": "host", "gpu": gpu, **({} if ok else {"log": "MB_ERROR bad"})}


class MappingTest(unittest.TestCase):
    def test_driver_result_matches_the_builtin_keys(self):
        r = ke.driver_result(_row(10.0, ops={"a": 1.0}, slots={"s0": {"rel_err": 1e-7, "ok": True}}))
        self.assertTrue(r["passed"])
        self.assertEqual(r["slots"], {"s0": {"rel_err": 1e-7, "ok": True}})      # slot_checks -> slots
        self.assertEqual((r["attr_op_us"], r["op_us"], r["partition_us"]), ({"a": 1.0}, {"a": 1.0}, 10.0))
        self.assertEqual(r["device"], L40S)
        # every key profile_ir / verify_ir read from build_and_run is there
        for k in ("passed", "partition_us", "op_us", "attr_op_us", "e2e_us", "e2e_basis", "attr_sum_us",
                  "attr_launches", "launch_overhead_frac", "slots", "slot_us", "log"):
            self.assertIn(k, r)

    def test_median_run_like_build_and_run(self):
        m = ke.median_run([_row(30.0), _row(10.0), _row(5.0, ok=False), _row(20.0)])
        self.assertEqual(m["partition_us"], 20.0)                    # median of the passing runs
        self.assertEqual(m["partition_us_runs"], [30.0, 10.0, 5.0, 20.0])
        bad = ke.median_run([_row(5.0, ok=False)])
        self.assertFalse(bad["passed"])
        self.assertIn("MB_ERROR", bad["log"])
        self.assertFalse(ke.median_run([])["passed"])

    def test_check_device(self):
        self.assertIsNone(ke.check_device(L40S, None, what="x"))      # nothing required
        self.assertIsNone(ke.check_device({"name": "default"}, "NVIDIA L40S", what="x"))   # unknown device
        self.assertTrue(ke.check_device(L40S, "l40s", what="x"))
        with self.assertRaises(ke.HardwareMismatch):
            ke.check_device(TITAN, "NVIDIA L40S", what="profile")
        w = []
        self.assertFalse(ke.check_device(TITAN, "NVIDIA L40S", what="profile", allow_mismatch=True, warnings=w))
        self.assertTrue(w and "TITAN" in w[0])


class FakeEval(ke.KBEval):
    """Records the jobs; answers like time_problems.py with ``device`` (raises NoDevice for
    ``fail_model`` when pinned to a model)."""

    def __init__(self, device=L40S, fail_model=False, ok=True, slots_bad=()):
        super().__init__(evaluator="ssh", ssh_hosts=["u@gpuhost"], remote_dir="/r")
        self.device, self.fail_model, self.ok, self.slots_bad, self.calls = device, fail_model, ok, slots_bad, []

    def available(self):
        return True

    def run(self, jobs, *, arch, work, gpu_model=None, name="time"):
        self.calls.append({"jobs": jobs, "arch": arch, "gpu_model": gpu_model, "name": name})
        if self.fail_model and gpu_model:
            raise ke.HardwareMismatch(f"NoDevice: no {gpu_model} GPU among the selected devices")
        out = {}
        for j in jobs:
            man = json.loads((Path(j["problem"]) / "manifest.json").read_text())
            ops = {o["name"]: 1.0 + k for k, o in enumerate(man["ops"])}
            slots = {s["id"]: {"rel_err": 1e-7, "ok": s["id"] not in self.slots_bad} for s in man["slots"]}
            ok = self.ok and not self.slots_bad
            runs = [_row(20.0 + 10 * k, ok=ok, gpu=self.device, ops=ops, slots=slots) for k in range(j["reps"])]
            out[j["id"]] = {"id": j["id"], "runs": runs, "ref_runs": [], "gpu": self.device,
                            "gpus": [self.device["label"]]}
        return {"jobs": out, "seconds": 1.0, "models": [self.device["name"]], "pool": []}


def _make_cache(ir: Path, d: Path, name: str, arch: str = "89"):
    """A cache with one winner (the hand seed of the first slot) for the tiny model."""
    from modelblaster.pipeline.kb_export import export_problem
    from modelblaster.pipeline.kb_ingest import CudaCache
    from modelblaster.pipeline.kernel_abi import parse_signature
    from modelblaster.pipeline.kb_proposals import _all_specs

    specs = _all_specs()
    prob = export_problem(ir, d / f"{name}_seed", strategy="whole")
    man = json.loads((prob / "manifest.json").read_text())
    s = man["slots"][0]
    spec = specs[s["op"]]
    cache = CudaCache(d / f"{name}_cache")
    cache.add(op=s["op"], entry=parse_signature(spec.signature)[0], signature=spec.signature, arch=f"sm_{arch}",
              shapes=[i.get("shape", {}) for i in s["instances"]], code=(prob / s["file"]).read_text(),
              slot_us=1.0, speedup=1.0, impl_id="t:" + s["id"], dtype=man.get("dtype"), source={})
    cache.save()
    shutil.rmtree(prob, ignore_errors=True)
    return cache, specs, s["id"]


@unittest.skipUnless(_have_torch(), "needs torch (CPU) for the export")
class FakeEvalStepsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.ir = _tiny_ir(cls.d)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_profile_through_the_evaluator(self):
        from modelblaster.pipeline.kb_plan import profile_ir

        ev = FakeEval()
        p = profile_ir(self.ir, self.d / "prof", arch="89", runs=3, kbeval=ev, gpu_model="NVIDIA L40S")
        self.assertEqual(len(ev.calls), 1)
        c = ev.calls[0]
        self.assertEqual((c["name"], c["arch"], c["gpu_model"], c["jobs"][0]["reps"]), ("profile", "89", "NVIDIA L40S", 3))
        self.assertIsNone(c["jobs"][0]["sources"])                     # the exported problem's own slots
        self.assertTrue(p["passed"])
        self.assertEqual(p["partition_us"], 30.0)                      # the median run, as build_and_run
        self.assertEqual(p["partition_us_runs"], [20.0, 30.0, 40.0])
        self.assertEqual(p["op_time_basis"], "cupti")
        self.assertEqual([r["us"] for r in p["ops"]], [1.0 + k for k in range(len(p["ops"]))])
        self.assertEqual(p["device"], L40S)
        self.assertEqual(p["evaluator"]["evaluator"], "ssh")
        self.assertEqual(json.loads((self.d / "prof" / "profile.json").read_text())["device"], L40S)
        self.assertFalse((self.d / "prof" / "problem" / "data.bin").exists())    # still cleaned up

    def test_profile_hardware_mismatch(self):
        from modelblaster.pipeline.kb_plan import profile_ir

        with self.assertRaises(ke.HardwareMismatch):
            profile_ir(self.ir, self.d / "prof_err", arch="89", runs=1, kbeval=FakeEval(fail_model=True),
                       gpu_model="NVIDIA L40S")
        ev = FakeEval(TITAN, fail_model=True)
        p = profile_ir(self.ir, self.d / "prof_allow", arch="89", runs=1, kbeval=ev, gpu_model="NVIDIA L40S",
                       allow_gpu_mismatch=True)
        self.assertEqual([c["gpu_model"] for c in ev.calls], ["NVIDIA L40S", None])
        self.assertEqual(p["device"]["name"], "NVIDIA TITAN RTX")
        self.assertTrue(p["hw_warnings"])

    def test_verify_through_the_evaluator(self):
        from modelblaster.pipeline.kb_ingest import verify_ir

        cache, specs, sid = _make_cache(self.ir, self.d, "v")
        ev = FakeEval()
        v = verify_ir(self.ir, cache, arch="sm_89", specs=specs, work=self.d / "ver", kbeval=ev,
                      gpu_model="NVIDIA L40S")
        c = ev.calls[0]
        self.assertEqual((c["name"], c["arch"], c["gpu_model"], c["jobs"][0]["reps"]), ("verify", "89", "NVIDIA L40S", 1))
        self.assertTrue(v["passed"])
        self.assertEqual(v["device"], L40S)
        self.assertTrue(all(x["ok"] for x in v["slots"].values()))
        self.assertTrue(any(e["model_verified"] for e in cache.entries))
        self.assertFalse((self.d / "ver" / "kb_time" / "verify_kbwork").exists())

    def test_verify_failure_marks_the_entry(self):
        from modelblaster.pipeline.kb_ingest import verify_ir

        cache, specs, sid = _make_cache(self.ir, self.d, "vf")
        v = verify_ir(self.ir, cache, arch="sm_89", specs=specs, work=self.d / "ver_bad",
                      kbeval=FakeEval(slots_bad=(sid,)))
        self.assertFalse(v["passed"])
        self.assertTrue(any(not x["ok"] for x in v["slots"].values()))
        self.assertTrue(any(e["rejected"] for e in cache.entries))


class FlowHardwareTest(unittest.TestCase):
    def cfg(self, **kw):
        return go.Config(kb=Path("."), **kw)

    def test_first_device_fixes_the_model(self):
        hw = go.FlowHardware(self.cfg(), {})
        self.assertIsNone(hw.model)
        hw.check("profile", L40S, "r00/a")
        self.assertEqual(hw.model, "NVIDIA L40S")
        hw.check("kb", {**L40S, "label": "u@gpuhost:3"}, "run")   # another GPU of the model is fine
        s = hw.summary()
        self.assertEqual((s["gpu_model"], s["models"], s["same_model"], s["hosts"], s["local_gpu_used"]),
                         ("NVIDIA L40S", ["NVIDIA L40S"], True, ["u@gpuhost"], False))
        with self.assertRaises(ke.HardwareMismatch):
            hw.check("verify", TITAN, "a")
        s = hw.summary()                                              # the offending device is recorded
        self.assertEqual((s["same_model"], s["local_gpu_used"]), (False, True))
        self.assertEqual(list(s["steps"]), ["profile", "kb", "verify"])

    def test_explicit_and_resumed_models_and_allow(self):
        self.assertEqual(go.FlowHardware(self.cfg(gpu_model="l40s"), {"gpu_model": "x"}).model, "l40s")
        self.assertEqual(go.FlowHardware(self.cfg(), {"gpu_model": "NVIDIA L40S"}).model, "NVIDIA L40S")
        hw = go.FlowHardware(self.cfg(allow_gpu_mismatch=True), {"gpu_model": "NVIDIA L40S"})
        hw.check("verify", TITAN, "a")
        s = hw.summary()
        self.assertFalse(s["same_model"])
        self.assertTrue(s["local_gpu_used"])
        self.assertTrue(s["warnings"])

    def test_pin(self):
        hw = go.FlowHardware(self.cfg(), {"gpu_model": "NVIDIA L40S"})
        self.assertEqual(hw.pin(self.cfg()).gpu_model, "NVIDIA L40S")
        # with --allow-gpu-mismatch only an explicit model pins the KernelBlaster runs
        hw = go.FlowHardware(self.cfg(allow_gpu_mismatch=True), {"gpu_model": "NVIDIA L40S"})
        self.assertIsNone(hw.pin(self.cfg(allow_gpu_mismatch=True)).gpu_model)
        self.assertIsNone(go.FlowHardware(self.cfg(), {}).pin(self.cfg()).gpu_model)

    def test_restrict_leases(self):
        leases = go.GpuLeases([{"ident": "a", "name": "NVIDIA TITAN RTX"}, {"ident": "b", "name": "NVIDIA L40S"}])
        go._restrict_leases(leases, "l40s", False)
        self.assertEqual([d["ident"] for d in leases.devices], ["b"])
        with self.assertRaises(ke.HardwareMismatch):
            go._restrict_leases(go.GpuLeases([{"ident": "a", "name": "NVIDIA TITAN RTX"}]), "l40s", False)
        unknown = go.GpuLeases([])                                   # unpinned default device: left alone
        go._restrict_leases(unknown, "l40s", False)
        self.assertEqual(unknown.devices, [{"ident": None, "name": "default"}])


class FlagsTest(unittest.TestCase):
    def test_mb_evaluator_selection(self):
        with tempfile.TemporaryDirectory() as t:
            kb = Path(t)
            (kb / "scripts").mkdir()
            (kb / "scripts" / "time_problems.py").write_text("")
            ssh = go.Config(kb=kb, ssh_hosts=["u@h"], cuda_visible_devices=["2"], ssh_key="/k", remote_dir="/r",
                            runtime="host")
            ev = go.mb_kb_eval(ssh)                                   # auto + --ssh-host: KernelBlaster's ssh
            self.assertEqual((ev.evaluator, ev.ssh_hosts, ev.cuda_visible_devices, ev.ssh_key, ev.remote_dir,
                              ev.runtime, ev.devices), ("ssh", ["u@h"], ["2"], "/k", "/r", "host", 1))
            self.assertIsNone(go.mb_kb_eval(go.Config(kb=kb)))         # auto locally: builtin (as before)
            loc = go.mb_kb_eval(go.Config(kb=kb, mb_evaluator="kb", gpus="1"))
            self.assertEqual((loc.evaluator, loc.gpus), ("local", "1"))
            from dataclasses import replace
            self.assertIsNone(go.mb_kb_eval(replace(ssh, mb_evaluator="builtin")))
        with tempfile.TemporaryDirectory() as t:                      # asked for, but no time_problems.py
            with self.assertRaises(RuntimeError):
                go.mb_kb_eval(go.Config(kb=Path(t), ssh_hosts=["u@h"], remote_dir="/r"))
        with self.assertRaises(ValueError):
            go.use_kb_eval(go.Config(kb=Path("."), mb_evaluator="nope"))

    def test_cli_flag(self):
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", "--help"],
                             cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=120)
        self.assertIn("--mb-evaluator", out.stdout)
        for mod in ("kb_plan", "kb_ingest"):
            args = ["profile", "--help"] if mod == "kb_plan" else ["--help"]
            out = subprocess.run([sys.executable, "-m", f"modelblaster.pipeline.{mod}", *args],
                                 cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=120)
            for f in ("--ssh-host", "--remote-dir", "--gpu-model", "--evaluator"):
                self.assertIn(f, out.stdout, mod)


class RemeasureDeviceTest(unittest.TestCase):
    def test_local_remeasure_records_the_leased_device(self):
        seen = {}

        def fake_remeasure(pair, *, arch, reps, lease=None, **kw):
            with lease as env:
                seen["env"] = env
            return [[10.0], [9.0]]

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            os.environ["KERNELBLASTER_GPU_LOCK_DIR"] = str(t)
            for side in ("o", "v"):
                (t / side / "best" / "slots").mkdir(parents=True)
            leases = go.GpuLeases([{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}])
            job = {"orig": t / "orig", "variant": t / "variant"}
            old = go.remeasure
            go.remeasure = fake_remeasure
            try:
                times = go.measure_hint(job, {"final_ok": True, "_dir": str(t / "o")},
                                        {"final_ok": True, "_dir": str(t / "v")},
                                        go.Config(kb=t, remeasure=3), leases)
            finally:
                go.remeasure = old
                os.environ.pop("KERNELBLASTER_GPU_LOCK_DIR", None)
        self.assertEqual(times, [[10.0], [9.0]])
        self.assertEqual(seen["env"], {"CUDA_VISIBLE_DEVICES": "GPU-titan"})
        self.assertEqual((job["remeasure_device"]["name"], job["remeasure_device"]["uuid"]),
                         ("NVIDIA TITAN RTX", "GPU-titan"))

    def test_kb_local_remeasure_leases_without_the_lock(self):
        """--mb-evaluator kb locally: the KernelBlaster evaluator pinned to the leased GPU takes its
        lock per run, so the lease must not hold it (it would wait on this process)."""
        seen = {}

        def fake_remeasure(pair, *, arch, reps, kbeval=None, gpu_model=None, info=None, work=None, lease=None):
            seen.update(kbeval=kbeval, gpu_model=gpu_model)
            lock = go.gpu_lock_path("GPU-titan")
            with go._flock(lock, blocking=False):                     # free: the lease does not hold it
                pass
            info["device"] = TITAN
            return [[10.0], [9.0]]

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            os.environ["KERNELBLASTER_GPU_LOCK_DIR"] = str(t)
            (t / "kb" / "scripts").mkdir(parents=True)
            (t / "kb" / "scripts" / "time_problems.py").write_text("")
            for side in ("o", "v"):
                (t / side / "best" / "slots").mkdir(parents=True)
                (t / side / "result.json").write_text(json.dumps({"final_ok": True, "arch": "sm_75", "gpu": TITAN}))
            leases = go.GpuLeases([{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}])
            job = {"orig": t / "orig", "variant": t / "variant"}
            old = go.remeasure
            go.remeasure = fake_remeasure
            try:
                go.measure_hint(job, {"final_ok": True, "_dir": str(t / "o")}, {"final_ok": True, "_dir": str(t / "v")},
                                go.Config(kb=t / "kb", remeasure=3, mb_evaluator="kb"), leases)
            finally:
                go.remeasure = old
                os.environ.pop("KERNELBLASTER_GPU_LOCK_DIR", None)
        self.assertEqual((seen["kbeval"].evaluator, seen["kbeval"].gpus), ("local", "GPU-titan"))
        self.assertEqual(seen["gpu_model"], "NVIDIA TITAN RTX")
        self.assertEqual(job["remeasure_device"], TITAN)


class RunWiringTest(unittest.TestCase):
    """kb_graph_opt._run with every GPU step faked: what each step is given and what the summary says."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "kb" / "scripts").mkdir(parents=True)
        (self.t / "kb" / "scripts" / "time_problems.py").write_text("")
        (self.t / "ir").mkdir()
        (self.t / "ir" / "graph.json").write_text(json.dumps({"ops": [], "output": {"tensor": "y"}}))
        self.saved = {k: getattr(go, k) for k in ("profile_ir", "verify_ir", "detect_gpus", "make_plan")}
        import modelblaster.pipeline.kb_model_run as mr
        self.mr, self.saved_mr = mr, mr.model_runtime
        self.calls = {"profile": [], "verify": [], "model": []}

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(go, k, v)
        self.mr.model_runtime = self.saved_mr
        self.tmp.cleanup()

    def install(self, devices):
        """``devices``: the device each successive profile reports."""
        seq = list(devices)

        def profile(ir, work, *, kbeval=None, gpu_model=None, allow_gpu_mismatch=False, **kw):
            self.calls["profile"].append({"kbeval": kbeval, "gpu_model": gpu_model})
            Path(work).mkdir(parents=True, exist_ok=True)
            (Path(work) / "graph.json").write_text(json.dumps({"ops": [], "output": {"tensor": "y"}}))
            return {"ir": "tiny", "ir_dir": str(work), "source_ir": str(ir), "passed": True, "partition_us": 10.0,
                    "ops": [], "e2e_us": 9.0, "launch_us": 0.0, "device": seq.pop(0) if seq else L40S}

        def verify(ir, cache, *, kbeval=None, gpu_model=None, allow_gpu_mismatch=False, **kw):
            self.calls["verify"].append({"kbeval": kbeval, "gpu_model": gpu_model})
            return {"ir": "tiny", "passed": True, "partition_us": 5.0, "device": L40S, "slots": {}}

        def model(ir, *, kbeval=None, gpu_model=None, **kw):
            self.calls["model"].append({"kbeval": kbeval, "gpu_model": gpu_model})
            return {"device": L40S, "passed": True, "model_e2e_us": 4.0}

        go.profile_ir, go.verify_ir, self.mr.model_runtime = profile, verify, model
        go.detect_gpus = lambda cfg: {"local": [], "remote": [{"host": "u@gpuhost", "ident": "GPU-l40s-2",
                                                               "index": 2, "name": "NVIDIA L40S"}]}

    def cfg(self, **kw):
        return go.Config(kb=self.t / "kb", ssh_hosts=["u@gpuhost"], remote_dir="/r", arch="89", rounds=2, **kw)

    def test_every_step_on_the_first_profiles_model(self):
        self.install([L40S])
        s = go.run([self.t / "ir"], self.t / "w", self.cfg())
        # the first profile had no model to match; it fixed the model for the rest
        self.assertEqual([c["gpu_model"] for c in self.calls["profile"]], [None])
        self.assertEqual(self.calls["profile"][0]["kbeval"].evaluator, "ssh")
        self.assertEqual(self.calls["verify"][0]["gpu_model"], "NVIDIA L40S")
        self.assertEqual(self.calls["verify"][0]["kbeval"].evaluator, "ssh")
        self.assertEqual(self.calls["model"][0]["gpu_model"], "NVIDIA L40S")
        g = s["gpu"]
        self.assertEqual((g["gpu_model"], g["same_model"], g["local_gpu_used"], g["hosts"]),
                         ("NVIDIA L40S", True, False, ["u@gpuhost"]))
        self.assertEqual(sorted(g["steps"]), ["model_run", "profile", "verify"])
        self.assertEqual(s["irs"]["tiny"]["verify_gpu"], "NVIDIA L40S")
        st = json.loads((self.t / "w" / "state.json").read_text())
        self.assertEqual(st["gpu_model"], "NVIDIA L40S")
        self.assertEqual(st["gpus"]["mb_evaluator"]["evaluator"], "ssh")
        # the KernelBlaster runs of a resumed call are pinned to it
        kb = go.KBRunner(go.FlowHardware(self.cfg(), st).pin(self.cfg()), self.t / "k")
        self.assertIn("--gpu-model NVIDIA L40S", " ".join(kb.command([self.t / "p"], self.t / "o")))
        # and a later call's profile is pinned too (--rounds is the work dir's total: one more round)
        self.install([L40S])
        go.run([self.t / "ir"], self.t / "w", self.cfg(more_rounds=1))
        self.assertEqual(self.calls["profile"][-1]["gpu_model"], "NVIDIA L40S")

    def test_a_profile_on_another_model_is_an_error(self):
        self.install([TITAN])
        with self.assertRaises(ke.HardwareMismatch):
            go.run([self.t / "ir"], self.t / "w2", self.cfg(gpu_model="NVIDIA L40S"))
        self.install([TITAN])
        s = go.run([self.t / "ir"], self.t / "w3", self.cfg(gpu_model="NVIDIA L40S", allow_gpu_mismatch=True))
        self.assertFalse(s["gpu"]["same_model"])
        self.assertTrue(s["gpu"]["warnings"])

    def test_builtin_locally_by_default(self):
        self.install([TITAN])
        go.detect_gpus = lambda cfg: {"local": [{"ident": "GPU-titan", "index": 0, "name": "NVIDIA TITAN RTX"}],
                                      "remote": []}
        cfg = go.Config(kb=self.t / "kb", arch="75", rounds=1)
        try:
            go.run([self.t / "ir"], self.t / "w4", cfg)
        except ke.HardwareMismatch:
            pass                                    # the fake verify / model answer L40S: refused, as they should be
        self.assertIsNone(self.calls["profile"][0]["kbeval"])         # no remote flags: ModelBlaster's own run
        self.assertEqual(self.calls["verify"][0]["gpu_model"], "NVIDIA TITAN RTX")
        self.assertIsNone(self.calls["verify"][0]["kbeval"])


def _gpu_ready() -> str | None:
    if not shutil.which("nvcc"):
        return "needs nvcc"
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=30).returncode != 0:
            return "needs a CUDA GPU"
    except (OSError, subprocess.SubprocessError):
        return "needs a CUDA GPU"
    if not ke.KBEval().available():
        return "needs a KernelBlaster checkout with scripts/time_problems.py (MB_KB_ROOT)"
    if not _have_torch():
        return "needs torch"
    return None


@unittest.skipIf(_gpu_ready() is not None, _gpu_ready() or "")
class LocalParityTest(unittest.TestCase):
    """The same profile / verification through ModelBlaster's builtin run and through
    KernelBlaster's local evaluator: the same checks, ops, launches and device."""

    @classmethod
    def setUpClass(cls):
        base = os.environ.get("TMPDIR")
        cls.tmp = tempfile.TemporaryDirectory(dir=base if base and Path(base).is_dir() else None)
        cls.d = Path(cls.tmp.name)
        cls.ir = _tiny_ir(cls.d)
        cls.arch = os.environ.get("MB_TEST_ARCH", "75")
        cls.ev = ke.KBEval(evaluator="local", python=os.environ.get("MB_KB_PYTHON", sys.executable), timeout=600)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_profile_parity(self):
        from modelblaster.pipeline.kb_plan import profile_ir

        a = profile_ir(self.ir, self.d / "pa", arch=self.arch, runs=2)
        b = profile_ir(self.ir, self.d / "pb", arch=self.arch, runs=2, kbeval=self.ev)
        self.assertTrue(a["passed"] and b["passed"], b.get("log"))
        self.assertEqual([r["name"] for r in a["ops"]], [r["name"] for r in b["ops"]])
        self.assertEqual(a["op_time_basis"], b["op_time_basis"])
        self.assertEqual(a["launches"], b["launches"])
        self.assertTrue(all(r["us"] for r in b["ops"]))
        self.assertEqual(len(b["partition_us_runs"]), 2)
        self.assertEqual(a["device"]["name"], b["device"]["name"])
        self.assertEqual(b["evaluator"]["evaluator"], "local")

    def test_verify_parity(self):
        from modelblaster.pipeline.kb_ingest import verify_ir

        res = []
        for k, ev in enumerate((None, self.ev)):
            cache, specs, _ = _make_cache(self.ir, self.d, f"par{k}", arch=self.arch)
            res.append(verify_ir(self.ir, cache, arch=self.arch, specs=specs, work=self.d / f"vpar{k}", kbeval=ev))
        a, b = res
        self.assertTrue(a["passed"] and b["passed"], b.get("log"))
        self.assertEqual({k: v["ok"] for k, v in a["slots"].items()}, {k: v["ok"] for k, v in b["slots"].items()})
        self.assertEqual(a["device"]["name"], b["device"]["name"])


if __name__ == "__main__":
    unittest.main()
