"""The model's result through KernelBlaster's evaluators (pipeline/kb_eval.py, kb_model_run).

- the hardware the model must be timed on (the GPU model / arch the KernelBlaster runs recorded),
  the evaluator flags and their pass-through (kb_model_run, kb_graph_opt, remeasure): everywhere;
- model_runtime with a fake evaluator (the jobs it sends, the paired fields, device matching):
  needs torch (CPU) for the export;
- model_runtime through KernelBlaster's local evaluator: needs nvcc, a CUDA GPU and a
  KernelBlaster checkout with scripts/time_problems.py (``MB_KB_ROOT``, default
  third_party/KernelBlaster).
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
L40S = {"host": "u@gpuhost", "label": "u@gpuhost:2", "index": 2, "uuid": "GPU-l40s-2", "name": "NVIDIA L40S",
        "arch": "sm_89"}

MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 32)
    def forward(self, x):
        return torch.relu(self.fc2(torch.relu(self.fc(x))))
def get_inputs():
    return [torch.randn(4, 64)]
def get_init_inputs():
    return []
'''


def _rec(name, arch="sm_89"):
    return {"run": "r", "gpu": {"name": name, "arch": arch} if name else None, "arch": arch}


class HardwareTest(unittest.TestCase):
    def test_model_and_arch_from_the_runs(self):
        hw = ke.required_hardware([_rec("NVIDIA L40S"), _rec("NVIDIA L40S")])
        self.assertEqual((hw["gpu_model"], hw["arch"], hw["verified"]), ("NVIDIA L40S", "89", True))
        self.assertEqual(hw["warnings"], [])

    def test_runs_without_a_gpu_record_warn(self):
        hw = ke.required_hardware([_rec(None, "sm_75")])
        self.assertIsNone(hw["gpu_model"])
        self.assertEqual(hw["arch"], "75")             # the arch still comes from the runs
        self.assertFalse(hw["verified"])
        self.assertTrue(any("cannot be matched" in w for w in hw["warnings"]))

    def test_mismatches_are_errors_unless_allowed(self):
        with self.assertRaises(ke.HardwareMismatch):
            ke.required_hardware([_rec("NVIDIA L40S"), _rec("NVIDIA TITAN RTX", "sm_75")])
        with self.assertRaises(ke.HardwareMismatch):
            ke.required_hardware([_rec("NVIDIA L40S")], gpu_model="NVIDIA TITAN RTX")
        with self.assertRaises(ke.HardwareMismatch):
            ke.required_hardware([_rec("NVIDIA L40S")], arch="75")
        hw = ke.required_hardware([_rec("NVIDIA L40S")], gpu_model="NVIDIA TITAN RTX", allow_mismatch=True)
        self.assertEqual(hw["gpu_model"], "NVIDIA TITAN RTX")
        self.assertFalse(hw["verified"])
        self.assertTrue(hw["warnings"])
        # the same model spelled differently is no mismatch
        self.assertEqual(ke.required_hardware([_rec("NVIDIA L40S")], gpu_model="l40s")["gpu_model"], "l40s")
        self.assertTrue(ke.same_gpu_model("NVIDIA  L40S", "l40s"))
        self.assertFalse(ke.same_gpu_model("default", "default"))      # unknown models never match

    def test_recorded_gpus_reads_result_json(self):
        with tempfile.TemporaryDirectory() as t:
            r = Path(t) / "run"
            r.mkdir()
            (r / "result.json").write_text(json.dumps({"arch": "sm_89", "gpu": L40S}))
            self.assertEqual(ke.recorded_gpus([r]), [{"run": str(r), "gpu": L40S, "arch": "sm_89"}])


class FlagsTest(unittest.TestCase):
    def parse(self, *argv):
        ap = argparse.ArgumentParser()
        ke.add_args(ap)
        return ap.parse_args(list(argv))

    def test_ssh_flags_pass_through(self):
        a = self.parse("--ssh-host", "u@h1", "--ssh-host", "u@h2", "--cuda-visible-devices", "2,3",
                       "--ssh-key", "/k", "--remote-dir", "/r", "--runtime", "host", "--devices", "2",
                       "--kb", str(ROOT))
        ev = ke.from_args(a)
        self.assertEqual(ev.evaluator, "ssh")                          # auto: ssh with --ssh-host
        cmd = ev.command(Path("j.json"), Path("o.json"), arch="sm_89", work=Path("w"), gpu_model="NVIDIA L40S")
        s = " ".join(cmd)
        for want in ("--evaluator ssh", "--ssh-host u@h1", "--ssh-host u@h2", "--cuda-visible-devices 2,3",
                     "--ssh-key /k", "--remote-dir /r", "--runtime host", "--devices 2", "--arch 89",
                     "--gpu-model NVIDIA L40S"):
            self.assertIn(want, s)
        self.assertTrue(cmd[1].endswith("scripts/time_problems.py"))

    def test_local_and_builtin(self):
        ev = ke.from_args(self.parse("--evaluator", "local", "--gpus", "0,2"))
        s = " ".join(ev.command(Path("j"), Path("o"), arch="75", work=Path("w")))
        self.assertIn("--evaluator local", s)
        self.assertIn("--gpus 0,2", s)
        self.assertNotIn("--gpu-model", s)
        self.assertIsNone(ke.from_args(self.parse("--evaluator", "builtin")))
        with tempfile.TemporaryDirectory() as t:                     # auto without a checkout: builtin
            self.assertIsNone(ke.from_args(self.parse("--kb", t)))

    def test_for_device(self):
        ev = ke.KBEval(evaluator="ssh", ssh_hosts=["u@a", "u@b"], remote_dir="/r", devices=3)
        one = ev.for_device({"host": "u@b", "ident": "GPU-x", "name": "NVIDIA L40S"})
        self.assertEqual((one.evaluator, one.ssh_hosts, one.cuda_visible_devices, one.devices),
                         ("ssh", ["u@b"], ["GPU-x"], 1))
        loc = ke.KBEval().for_device({"host": None, "ident": "GPU-y"})
        self.assertEqual((loc.evaluator, loc.gpus, loc.devices), ("local", "GPU-y", 1))
        self.assertIs(ev.for_device(None), ev)


class FakeEval(ke.KBEval):
    """Records the jobs; answers like time_problems.py: each job timed on ``device``,
    the candidate 2x faster than the incumbent (paired on the host)."""

    def __init__(self, device=L40S, fail_model=False):
        super().__init__(evaluator="ssh", ssh_hosts=["u@gpuhost"], remote_dir="/r")
        self.device, self.fail_model, self.calls = device, fail_model, []

    def available(self):
        return True

    def run(self, jobs, *, arch, work, gpu_model=None, name="time"):
        self.calls.append({"jobs": jobs, "arch": arch, "gpu_model": gpu_model, "name": name})
        if self.fail_model and gpu_model:
            raise ke.HardwareMismatch(f"NoDevice: no {gpu_model} GPU among the selected devices")
        out = {}
        for j in jobs:
            paired = j.get("incumbent") is not None
            runs = [{"ok": True, "e2e_us": 50.0 + k, "e2e_basis": "graph", "partition_us": 60.0 + k,
                     "partition_graph_us": 50.0 + k, "ref_e2e_us": 100.0 + 2 * k if paired else None,
                     "e2e_ratio": {"med": 0.5, "basis": "graph"} if paired else None, "runtime": "host",
                     "attr_sum_us": 40.0, "slot_checks": {}, "outputs": {}, "errors": [], "gpu": self.device}
                    for k in range(j.get("reps", 1))]
            out[j["id"]] = {"id": j["id"], "runs": runs, "ref_runs": [], "paired": paired,
                            "pairing": "host" if paired else None, "gpu": self.device, "gpus": [self.device["label"]]}
        return {"jobs": out, "seconds": 1.0, "models": [self.device["name"]], "pool": [], "host_stats": {}}


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _tiny_ir(d: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    (d / "tiny.py").write_text(MODEL)
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(d / "tiny.py"), "--out-dir", str(d / "ir"), "--quant", "fp32"],
                        cwd=root, capture_output=True, text=True, timeout=600,
                        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])})
    assert cp.returncode == 0, cp.stderr[-2000:]
    return d / "ir"


def _window_runs(ir: Path, d: Path, gpu: dict | None, arch: str) -> Path:
    """Two windows exported and given KernelBlaster-run directories whose best slots are the
    seeds with a marker; result.json records ``gpu`` (as KernelBlaster now does)."""
    from modelblaster.pipeline.kb_export import export_problem, partition_ir

    g = json.loads((ir / "graph.json").read_text())
    for k, idx in enumerate(partition_ir(g, "window", max_ops=2)):
        pdir = export_problem(ir, d / "win" / f"p{k}", op_indices=idx, strategy="window")
        man = json.loads((pdir / "manifest.json").read_text())
        run = d / "runs" / man["name"]
        (run / "best" / "slots").mkdir(parents=True)
        for sl in man["slots"]:
            (run / "best" / "slots" / f"{sl['id']}.cu").write_text((pdir / sl["file"]).read_text() + "\n// win\n")
        (run / "result.json").write_text(json.dumps({"problem": man["name"], "problem_dir": str(pdir),
                                                     "final_ok": True, "run_id": "t", "slots": {}, "arch": f"sm_{arch}",
                                                     **({"gpu": gpu} if gpu else {})}))
    return d / "runs"


@unittest.skipUnless(_have_torch(), "needs torch (CPU) for the export")
class FakeEvalModelRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.ir = _tiny_ir(cls.d)
        cls.runs = _window_runs(cls.ir, cls.d, L40S, "89")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_and_partitions_on_the_runs_gpu_model(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        ev = FakeEval()
        r = model_runtime(self.ir, work=self.d / "mr", runs=[self.runs], reps=2, time_partitions=True, kbeval=ev)
        self.assertEqual(len(ev.calls), 1)                             # one evaluator for all of it
        c = ev.calls[0]
        self.assertEqual((c["gpu_model"], c["arch"]), ("NVIDIA L40S", "89"))   # inferred from the runs
        ids = [j["id"] for j in c["jobs"]]
        self.assertEqual(ids, ["model", "part00", "part01"])         # paired: no separate seed job
        self.assertIsNotNone(c["jobs"][0]["incumbent"])              # the hand-seeded model, same request
        for j in c["jobs"][1:]:                                       # partitions: their bests vs the hand seeds
            self.assertTrue(all(t.endswith("// win\n") for t in j["sources"].values()))
            self.assertFalse(any(t.endswith("// win\n") for t in j["incumbent"].values()))
        self.assertTrue(r["passed"])
        self.assertEqual((r["model_e2e_us"], r["seed_e2e_us"], r["paired_ratio"]), (50.5, 101.0, 0.5))
        self.assertAlmostEqual(r["speedup_paired"], 2.0)
        self.assertEqual(r["arch"], "sm_89")
        self.assertEqual((r["gpu_name"], r["gpu_uuid"], r["gpu_host"]), ("NVIDIA L40S", "GPU-l40s-2", "u@gpuhost"))
        self.assertTrue(r["hardware"]["match"])
        self.assertTrue(r["hardware"]["verified"])
        self.assertEqual(r["evaluator"]["evaluator"], "ssh")
        self.assertEqual(len(r["partitions"]), 2)
        self.assertTrue(all(p["same_device"] and p["kb_gpu"] == "NVIDIA L40S" for p in r["partitions"]))
        self.assertTrue(all(p["seed_kind"] == "hand" for p in r["partitions"]))
        self.assertAlmostEqual(r["sum_partitions_e2e_us"], 101.0)
        # winners from the runs carry their GPU into the working cache
        idx = json.loads((self.d / "mr" / "cache" / "index.json").read_text())
        self.assertTrue(all(e["source"]["gpu"]["name"] == "NVIDIA L40S" for e in idx["entries"]))

    def test_no_gpu_of_the_model_is_an_error(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        with self.assertRaises(ke.HardwareMismatch):
            model_runtime(self.ir, work=self.d / "mr_err", runs=[self.runs], reps=1, kbeval=FakeEval(fail_model=True))
        # a timing that lands on another model (an evaluator not honouring gpu_model) is refused too
        titan = {**L40S, "name": "NVIDIA TITAN RTX", "host": None, "label": "0"}
        with self.assertRaises(ke.HardwareMismatch):
            model_runtime(self.ir, work=self.d / "mr_err2", runs=[self.runs], reps=1, kbeval=FakeEval(titan))
        with self.assertRaises(ke.HardwareMismatch):                  # --arch contradicting the runs
            model_runtime(self.ir, work=self.d / "mr_err3", runs=[self.runs], reps=1, arch="75", kbeval=FakeEval())

    def test_allow_gpu_mismatch_times_anyway_and_says_so(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        titan = {**L40S, "name": "NVIDIA TITAN RTX", "host": None, "label": "0"}
        ev = FakeEval(titan, fail_model=True)
        r = model_runtime(self.ir, work=self.d / "mr_allow", runs=[self.runs], reps=1, kbeval=ev,
                          allow_gpu_mismatch=True)
        self.assertEqual([c["gpu_model"] for c in ev.calls], ["NVIDIA L40S", None])   # retried unpinned
        self.assertFalse(r["hardware"]["match"])
        self.assertTrue(any("another GPU model" in w for w in r["hardware"]["warnings"]))

    def test_explicit_gpu_model_and_extra_records(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        ev = FakeEval()
        r = model_runtime(self.ir, work=self.d / "mr_rec", runs=[], reps=1, kbeval=ev, recorded=[_rec("NVIDIA L40S")])
        self.assertEqual(ev.calls[0]["gpu_model"], "NVIDIA L40S")     # kb_graph_opt's rounds' records
        self.assertNotIn("partitions", r)


class GraphOptPassThroughTest(unittest.TestCase):
    def cfg(self, **kw):
        from modelblaster.pipeline.kb_graph_opt import Config
        return Config(**{"kb": ROOT, **kw})

    def test_kb_runner_flags(self):
        from modelblaster.pipeline.kb_graph_opt import KBRunner

        cmd = " ".join(KBRunner(self.cfg(ssh_hosts=["u@h"], remote_dir="/r", runtime="host", gpu_model="NVIDIA L40S"),
                                Path("k")).command([Path("p")], Path("o")))
        for want in ("--evaluator ssh", "--ssh-host u@h", "--remote-dir /r", "--runtime host",
                     "--gpu-model NVIDIA L40S"):
            self.assertIn(want, cmd)

    def test_kb_eval_for_uses_the_kb_pool(self):
        from modelblaster.pipeline import kb_graph_opt as go

        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "scripts").mkdir()
            (Path(t) / "scripts" / "time_problems.py").write_text("")
            ev = go.kb_eval_for(self.cfg(kb=Path(t), ssh_hosts=["u@h"], cuda_visible_devices=["2"], ssh_key="/k",
                                         remote_dir="/r", runtime="host"))
            self.assertEqual((ev.evaluator, ev.ssh_hosts, ev.cuda_visible_devices, ev.ssh_key, ev.remote_dir,
                              ev.runtime), ("ssh", ["u@h"], ["2"], "/k", "/r", "host"))
            self.assertEqual(go.kb_eval_for(self.cfg(kb=Path(t), gpus="0")).evaluator, "local")
            self.assertIsNone(go.kb_eval_for(self.cfg(kb=Path(t) / "nope")))         # an older checkout

    def test_remote_remeasure_on_a_leased_remote_gpu(self):
        """With --ssh-host the hint re-measurement leases one of the KernelBlaster runs' remote GPUs
        and times through the ssh evaluator pinned to it, on the runs' GPU model."""
        from modelblaster.pipeline import kb_graph_opt as go

        seen = {}

        def fake_remeasure(pair, *, arch, reps, kbeval=None, gpu_model=None, info=None, work=None, lease=None):
            seen.update(kbeval=kbeval, gpu_model=gpu_model, lease=lease, arch=arch, reps=reps)
            info["device"] = L40S
            return [[10.0], [9.0]]

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            (t / "kb" / "scripts").mkdir(parents=True)
            (t / "kb" / "scripts" / "time_problems.py").write_text("")
            for side in ("o", "v"):
                (t / side / "best" / "slots").mkdir(parents=True)
                (t / side / "result.json").write_text(json.dumps({"final_ok": True, "arch": "sm_89", "gpu": L40S}))
            leases = go.GpuLeases([{"host": "u@gpuhost", "ident": "GPU-l40s-2", "index": 2, "name": "NVIDIA L40S"}])
            job = {"orig": t / "orig", "variant": t / "variant"}
            old = go.remeasure
            go.remeasure = fake_remeasure
            try:
                cfg = self.cfg(kb=t / "kb", ssh_hosts=["u@gpuhost"], remote_dir="/r", remeasure=3, arch="89")
                times = go.measure_hint(job, {"final_ok": True, "_dir": str(t / "o")},
                                        {"final_ok": True, "_dir": str(t / "v")}, cfg, leases)
            finally:
                go.remeasure = old
        self.assertEqual(times, [[10.0], [9.0]])
        ev = seen["kbeval"]
        self.assertEqual((ev.evaluator, ev.ssh_hosts, ev.cuda_visible_devices, ev.devices),
                         ("ssh", ["u@gpuhost"], ["GPU-l40s-2"], 1))
        self.assertEqual(seen["gpu_model"], "NVIDIA L40S")
        self.assertIsNone(seen["lease"])                               # no local GPU lease / lock
        self.assertEqual(job["remeasure_device"], L40S)
        self.assertEqual(len(leases.log), 1)

    def test_remote_lease_takes_no_local_lock(self):
        from modelblaster.pipeline import kb_graph_opt as go

        with tempfile.TemporaryDirectory() as t:
            os.environ["KERNELBLASTER_GPU_LOCK_DIR"] = t
            try:
                leases = go.GpuLeases([{"host": "u@h", "ident": "GPU-r", "name": "NVIDIA L40S"}])
                with leases.lease_device() as dev:
                    self.assertEqual(dev["ident"], "GPU-r")
                    self.assertEqual(list(Path(t).iterdir()), [])
                local = go.GpuLeases([{"ident": "GPU-l", "name": "x"}])
                with local.lease() as env:
                    self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "GPU-l"})
                    self.assertTrue(list(Path(t).iterdir()))           # the local KernelBlaster lock is held
            finally:
                os.environ.pop("KERNELBLASTER_GPU_LOCK_DIR", None)

    def test_remeasure_through_an_evaluator(self):
        from modelblaster.pipeline.kb_proposals import remeasure

        class Ev(FakeEval):
            pass
        ev = Ev()
        info = {}
        with tempfile.TemporaryDirectory() as t:
            times = remeasure([(Path(t) / "a", {"s": "x"}), (Path(t) / "b", {"s": "y"})], arch="89", reps=3,
                              kbeval=ev, gpu_model="NVIDIA L40S", info=info, work=Path(t) / "w")
        self.assertEqual(times, [[50.0, 51.0, 52.0], [50.0, 51.0, 52.0]])
        self.assertEqual(ev.calls[0]["gpu_model"], "NVIDIA L40S")
        self.assertEqual([j["id"] for j in ev.calls[0]["jobs"]], ["p0", "p1"])
        self.assertEqual(info["device"], L40S)


def _have_gpu_and_kb() -> bool:
    if shutil.which("nvcc") is None or shutil.which("nvidia-smi") is None or not _have_torch():
        return False
    if not ke.KBEval().available():
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_have_gpu_and_kb(), "needs torch, nvcc, a CUDA GPU and a KernelBlaster checkout "
                     "with scripts/time_problems.py (MB_KB_ROOT)")
class LocalEvaluatorModelRunTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.ir = _tiny_ir(cls.d)
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.splitlines()[0].split(",")
        cls.name, cls.arch = q[0].strip(), q[1].strip().replace(".", "")
        cls.runs = _window_runs(cls.ir, cls.d, {"name": cls.name, "arch": f"sm_{cls.arch}"}, cls.arch)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_model_and_partitions_on_one_local_gpu(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        ev = ke.KBEval(evaluator="local")
        r = model_runtime(self.ir, work=self.d / "mr", runs=[self.runs], reps=2, time_partitions=True, kbeval=ev)
        self.assertTrue(r["passed"], r["model"].get("log"))
        self.assertEqual(r["model_basis"], "graph")
        self.assertTrue(r["paired"])
        self.assertTrue(r["seed_e2e_us"] and r["model_e2e_us"] and r["speedup_paired"])
        self.assertEqual(r["model"]["runtime"], "host")
        self.assertTrue(ke.same_gpu_model(r["gpu_name"], self.name))
        self.assertTrue(r["gpu_uuid"].startswith("GPU-"))
        self.assertTrue(r["hardware"]["match"])
        self.assertEqual(len(r["partitions"]), 2)
        self.assertTrue(all(p["passed"] and p["same_device"] and p["seed_e2e_us"] for p in r["partitions"]))
        self.assertTrue((self.d / "mr" / "kb_time" / "model.result.json").exists())

    def test_runs_from_another_gpu_model_are_refused(self):
        from modelblaster.pipeline.kb_model_run import model_runtime

        d = self.d / "other"
        d.mkdir()
        for run in self.runs.iterdir():
            dst = d / run.name
            shutil.copytree(run, dst)
            res = json.loads((dst / "result.json").read_text())
            res["gpu"] = {"name": "NVIDIA H100 80GB HBM3", "arch": f"sm_{self.arch}"}
            (dst / "result.json").write_text(json.dumps(res))
        with self.assertRaises(ke.HardwareMismatch) as cm:
            model_runtime(self.ir, work=self.d / "mr_other", runs=[d], reps=1, kbeval=ke.KBEval(evaluator="local"))
        self.assertIn("H100", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
