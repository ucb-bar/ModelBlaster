"""Parallel hint decisions in the graph optimiser (``kb_graph_opt``).

- footprints / conflicts / waves on hand-made hints;
- the decide phase of a round with several hints, one by one
  (``width=1``) and in waves, on a real exported IR with a fake
  KernelBlaster (writes each problem's seeds back as its bests, tagged) and a
  deterministic fake re-timing (a function of the problem and its sources):
  both must make the same decisions with the same evidence and leave the
  same cache -- including a hint seeded from a kernel that an earlier,
  conflicting hint's acceptance cached;
- the locks: GPU leases (in-process exclusivity, KernelBlaster's lock file
  across processes), the work-dir lock, atomic hints-file writes.

Needs torch (CPU) for the IR extraction; no GPU, nvcc or KernelBlaster.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import kb_graph_opt as go  # noqa: E402

MODEL = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(4, 4, 3, padding=1)
        self.b1 = nn.BatchNorm2d(4)
        self.c2 = nn.Conv2d(4, 4, 1)
        self.c3 = nn.Conv2d(4, 4, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        a = self.relu(self.b1(self.c1(x)))
        return torch.cat([a, torch.sigmoid(self.c2(x)), torch.tanh(self.c3(x))], dim=1)
def get_inputs():
    return [torch.randn(1, 4, 6, 6)]
def get_init_inputs():
    return []
'''


def _hint(hid, kind, est, ir, **kw):
    return {"id": hid, "kind": kind, "target": "cuda", "status": "proposed", "evidence": [],
            "source": [{"by": "test", "ir": ir, "est_gain_pct": est}], **kw}


def hints_for(ir: str) -> list[dict]:
    """In rank order: a conv split (conflicts with every hint reading conv
    kernels), two fusions sharing b1, two independent conv fusions."""
    return [
        _hint("split:conv2d", "split_slot", 9.0, ir, slot="conv2d", by="shape_bucket",
              groups=[{"bucket": "k3", "ops": ["c1", "c3"]}, {"bucket": "k1", "ops": ["c2"]}]),
        _hint("fuse:fused__conv2d__batchnorm2d", "fuse", 8.0, ir, fused_kind="fused__conv2d__batchnorm2d",
              chains=[["c1", "b1"]]),
        _hint("fuse:fused__conv2d__sigmoid", "fuse", 7.0, ir, fused_kind="fused__conv2d__sigmoid",
              chains=[["c2", "sigmoid"]]),
        _hint("fuse:fused__batchnorm2d__relu", "fuse", 6.0, ir, fused_kind="fused__batchnorm2d__relu",
              chains=[["b1", "relu"]]),
        _hint("fuse:fused__conv2d__tanh", "fuse", 5.0, ir, fused_kind="fused__conv2d__tanh",
              chains=[["c3", "tanh"]]),
    ]


class _FakeProc:
    def __init__(self, thread):
        self.t = thread
        self.returncode = None

    def poll(self):
        if self.t.is_alive():
            return None
        self.returncode = 0
        return 0

    def wait(self):
        self.t.join()
        self.returncode = 0
        return 0

    def kill(self):
        pass


class FakeKB(go.KBRunner):
    """Writes OUT/<problem>/best/slots (each seed + a tag naming the run)
    and result.json, one problem after another in a thread (last first, so
    results arrive while the run is still going)."""

    def __init__(self, cfg, knowledge):
        super().__init__(cfg, knowledge)
        self.runs: list[list[str]] = []

    def start(self, problems, out, **kw):
        out.mkdir(parents=True, exist_ok=True)
        self.runs.append([go.problem_name(Path(p)) for p in problems])

        def work():
            for p in reversed(problems):
                p = Path(p)
                man = json.loads((p / "manifest.json").read_text())
                d = out / man["name"]
                (d / "best" / "slots").mkdir(parents=True, exist_ok=True)
                slots = {}
                for s in man["slots"]:
                    code = (p / s["file"]).read_text() + f"\n// kb {man['name']}\n"
                    (d / "best" / "slots" / f"{s['id']}.cu").write_text(code)
                    slots[s["id"]] = {"baseline_us": 10.0, "best_us": 5.0, "final_us": 5.0}
                (d / "result.json").write_text(json.dumps(
                    {"problem": man["name"], "run_id": "fake", "arch": "sm_75", "final_ok": True,
                     "baseline_partition_us": 100.0, "final_partition_us": 90.0, "slots": slots,
                     "llm_calls": 0}))
                time.sleep(0.02)

        t = threading.Thread(target=work)
        t.start()
        return _FakeProc(t)


def fake_remeasure(problems, *, arch, reps, lease=None, **kw):
    """Original 100 us; the variant's gain is a function of the problem and
    every source it runs (seeds included), so a hint evaluated from a
    different cache state gets a different number. The split and the
    conv->bn fusion always win, so their acceptance changes later seeds."""
    out = []
    with (lease if lease is not None else go.contextlib.nullcontext({})):
        for d, src in problems:
            name = json.loads((Path(d) / "manifest.json").read_text())["name"]
            if name.endswith("__orig"):
                out.append([100.0] * reps)
                continue
            h = hashlib.sha1((name + json.dumps(src, sort_keys=True)).encode()).digest()
            g = 0.05 if ("split" in name or "conv2d__batchnorm2d__variant" in name) else (h[0] / 255.0 - 0.5) / 5
            out.append([100.0 * (1 - g)] * reps)
    return out


@unittest.skipIf(importlib.util.find_spec("torch") is None, "needs torch")
class DecideParallelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_gopar_", dir=base))
        (cls.tmp / "m.py").write_text(MODEL)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT.parent), os.environ.get("PYTHONPATH", "")])}
        cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                             str(cls.tmp / "m.py"), "--out-dir", str(cls.tmp / "ir"), "--quant", "fp32"],
                            cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
        assert cp.returncode == 0, cp.stderr[-2000:]
        cls.graph = gh.unique_names(json.loads((cls.tmp / "ir" / "graph.json").read_text()))
        cls.ir = cls.graph["name"]

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MB_TEST_KEEP"):
            import shutil
            shutil.rmtree(cls.tmp, ignore_errors=True)

    def _specs(self):
        from modelblaster.pipeline.kb_proposals import _all_specs
        return dict(_all_specs())

    def test_footprints_and_waves(self):
        specs = self._specs()
        hs = hints_for(self.ir)
        fps = [go.footprint(h, self.ir, self.graph, specs) for h in hs]
        split, cbn, csig, bnr, ctanh = fps
        self.assertEqual(split["writes"], {"kernel_conv2d"})
        self.assertEqual(cbn["ops"], {(self.ir, "c1"), (self.ir, "b1")})
        self.assertTrue(go.conflicts(split, cbn) and go.conflicts(split, csig) and go.conflicts(split, ctanh))
        self.assertFalse(go.conflicts(split, bnr))              # bn -> relu reads no conv kernel
        self.assertTrue(go.conflicts(cbn, bnr))                 # share b1
        self.assertFalse(go.conflicts(cbn, csig) or go.conflicts(csig, ctanh) or go.conflicts(cbn, ctanh))
        # bn->relu does not conflict with the split, but with conv->bn, which
        # ranks before it: it waits for conv->bn's wave
        self.assertEqual(go.schedule(fps, 8), [[0], [1, 2, 4], [3]])
        fps2 = [fps[1], fps[3], fps[0], fps[2], fps[4]]         # conv-bn, bn-relu, split, conv-sig, conv-tanh
        self.assertEqual(go.schedule(fps2, 8), [[0], [1, 2], [3, 4]])
        self.assertEqual(go.schedule(fps, 1), [[0], [1], [2], [3], [4]])
        self.assertEqual(go.schedule(fps, 2), [[0], [1, 2], [3, 4]])     # conv->tanh spills over
        # conflicting pairs keep their rank order in every width
        for w in (1, 2, 3, 8):
            wave = {i: k for k, ws in enumerate(go.schedule(fps, w)) for i in ws}
            for i in range(5):
                for j in range(i):
                    if go.conflicts(fps[i], fps[j]):
                        self.assertLess(wave[j], wave[i], (w, i, j))
        # deterministic
        self.assertEqual(go.schedule(fps, 3), go.schedule(fps, 3))

    def _decide(self, width, tag):
        from modelblaster.pipeline.kb_ingest import CudaCache
        work = self.tmp / tag
        cfg = go.Config(kb=Path("."), arch="75", remeasure=3, accept_gain=0.0, parallel_hints=width)
        kb = FakeKB(cfg, work / "knowledge")
        cache, doc = CudaCache(work / "cache"), gh.new_doc()
        hs = hints_for(self.ir)
        gh.merge(doc, hs)
        prof = {"ir": self.ir, "ir_dir": str(self.tmp / "ir"), "source_ir": str(self.tmp / "ir")}
        orig = go.remeasure
        go.remeasure = fake_remeasure
        try:
            waves, decisions = go.decide_hints([(h, prof) for h in hs], {self.ir: self.graph}, work / "variants",
                                               width, kb=kb, cache=cache, doc=doc, cfg=cfg, specs=self._specs(),
                                               leases=go.GpuLeases([]), hints_path=work / "graph_hints.json")
        finally:
            go.remeasure = orig
        return work, waves, decisions, doc, cache, kb

    def test_parallel_equals_sequential(self):
        ws, wav_s, dec_s, doc_s, cache_s, kb_s = self._decide(1, "seq")
        wp, wav_p, dec_p, doc_p, cache_p, kb_p = self._decide(8, "par")
        self.assertEqual(len(wav_s), 5)
        self.assertEqual(wav_p, [["split:conv2d"],
                                 ["fuse:fused__conv2d__batchnorm2d", "fuse:fused__conv2d__sigmoid",
                                  "fuse:fused__conv2d__tanh"],
                                 ["fuse:fused__batchnorm2d__relu"]])
        self.assertEqual([len(r) for r in kb_p.runs], [2, 6, 2])   # one KernelBlaster run per wave
        self.assertEqual([len(r) for r in kb_s.runs], [2] * 5)

        def strip(d):
            return {k: v for k, v in d.items() if k != "cached"}
        self.assertEqual([strip(d) for d in dec_s], [strip(d) for d in dec_p])
        self.assertEqual([d["hint"] for d in dec_p], [h["id"] for h in hints_for(self.ir)])   # rank order

        def ev(doc):
            return {h["id"]: [(e["decision"], e["gain"], e["remeasured"]) for e in h["evidence"]]
                    for h in doc["hints"]}
        self.assertEqual(ev(doc_s), ev(doc_p))
        self.assertEqual({h["id"]: h["status"] for h in doc_s["hints"]},
                         {h["id"]: h["status"] for h in doc_p["hints"]})
        self.assertEqual(gh.get(doc_p, "split:conv2d")["status"], "accepted")

        def entries(c):
            return sorted((e["op"], e["entry"], tuple(e["shape_guard"]), c.code(e).split("// kb ")[-1])
                          for e in c.entries)
        self.assertEqual(entries(cache_s), entries(cache_p))
        self.assertTrue(any(e["op"] == "conv2d" for e in cache_p.entries))
        # the conv fusions after the split were seeded from its cached conv kernels, in both modes
        for w in (ws, wp):
            man = json.loads((w / "variants" / "fuse_fused__conv2d__sigmoid" / "orig" / "manifest.json").read_text())
            tags = {s["op"]: s["seed_source"] for s in man["slots"]}
            self.assertTrue(tags["conv2d"].startswith("cache:"), tags)
        self.assertEqual(json.loads((wp / "graph_hints.json").read_text())["hints"], doc_p["hints"])

    def test_failed_wave_leaves_hints_undecided_instead_of_aborting(self):
        """A wave's KernelBlaster run that exits non-zero (the B200 pilot: `problem names collide`) used to raise and kill the whole problem;
        now its hints stay undecided (proposed) and the round goes on."""
        from modelblaster.pipeline.kb_ingest import CudaCache

        class BrokenKB(FakeKB):
            def start(self, problems, out, **kw):
                out.mkdir(parents=True, exist_ok=True)
                self.runs.append([go.problem_name(Path(p)) for p in problems])
                return _FakeProc(threading.Thread(target=lambda: None))

            def collect(self, problems, out, proc):
                raise RuntimeError("KernelBlaster failed (1); see wave.log")

        work = self.tmp / "broken"
        cfg = go.Config(kb=Path("."), arch="75", remeasure=3, accept_gain=0.0, parallel_hints=8)
        kb = BrokenKB(cfg, work / "knowledge")
        cache, doc = CudaCache(work / "cache"), gh.new_doc()
        hs = hints_for(self.ir)
        gh.merge(doc, hs)
        prof = {"ir": self.ir, "ir_dir": str(self.tmp / "ir"), "source_ir": str(self.tmp / "ir")}
        orig = go.remeasure
        go.remeasure = fake_remeasure
        try:
            waves, decisions = go.decide_hints([(h, prof) for h in hs], {self.ir: self.graph}, work / "variants", 8, kb=kb, cache=cache,
                                               doc=doc, cfg=cfg, specs=self._specs(), leases=go.GpuLeases([]),
                                               hints_path=work / "graph_hints.json")
        finally:
            go.remeasure = orig
        self.assertEqual(len(decisions), len(hs))
        self.assertTrue(all(d["status"] == "undecided" and d["hint_status"] == "proposed" for d in decisions), decisions)
        self.assertTrue(all(h["status"] == "proposed" for h in doc["hints"]))


class LockTest(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_gopar_lock_", dir=base))
        self._env = {k: os.environ.get(k) for k in ("KERNELBLASTER_GPU_LOCK_DIR", "KERNELBLASTER_GPU_FLOCK")}
        os.environ["KERNELBLASTER_GPU_LOCK_DIR"] = str(self.tmp)
        os.environ.pop("KERNELBLASTER_GPU_FLOCK", None)

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_leases_are_exclusive_per_device(self):
        leases = go.GpuLeases([{"ident": "GPU-test-a"}, {"ident": "GPU-test-b"}])
        seen = []

        def job(k):
            with leases.lease() as env:
                seen.append(env["CUDA_VISIBLE_DEVICES"])
                time.sleep(0.05)

        ts = [threading.Thread(target=job, args=(k,)) for k in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.assertEqual(sorted(set(seen)), ["GPU-test-a", "GPU-test-b"])
        by_dev: dict[int, list] = {}
        for x in leases.log:
            by_dev.setdefault(x["device"], []).append((x["t0"], x["t1"]))
        self.assertEqual(sum(len(v) for v in by_dev.values()), 8)
        for spans in by_dev.values():
            spans.sort()
            for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
                self.assertLessEqual(a1, b0)                    # never two runs on one GPU
        self.assertTrue((self.tmp / "kernelblaster_gpu_GPU-test-a.lock").exists())

    def test_lease_waits_for_another_process_holding_the_gpu_lock(self):
        """KernelBlaster (any process) holds /dev/shm/kernelblaster_gpu_<uuid>.lock
        while it times: a lease must wait for it."""
        lock = go.gpu_lock_path("GPU-test-c")
        self.assertEqual(lock, self.tmp / "kernelblaster_gpu_GPU-test-c.lock")
        holder = subprocess.Popen([sys.executable, "-c",
                                   "import fcntl, os, sys, time\n"
                                   "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
                                   "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); time.sleep(1.0)\n",
                                   str(lock)], stdout=subprocess.PIPE, text=True)
        self.assertEqual(holder.stdout.readline().strip(), "held")
        t0 = time.time()
        with go.GpuLeases([{"ident": "GPU-test-c"}]).lease():
            waited = time.time() - t0
        holder.wait()
        self.assertGreater(waited, 0.5)
        # and with the lock off (KERNELBLASTER_GPU_FLOCK=0) it does not wait
        os.environ["KERNELBLASTER_GPU_FLOCK"] = "0"
        holder = subprocess.Popen([sys.executable, "-c",
                                   "import fcntl, os, sys, time\n"
                                   "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
                                   "fcntl.flock(fd, fcntl.LOCK_EX); print('held', flush=True); time.sleep(1.0)\n",
                                   str(lock)], stdout=subprocess.PIPE, text=True)
        holder.stdout.readline()
        t0 = time.time()
        with go.GpuLeases([{"ident": "GPU-test-c"}]).lease():
            self.assertLess(time.time() - t0, 0.5)
        holder.wait()

    def test_work_dir_has_one_owner(self):
        work = self.tmp / "work"
        code = ("import sys; sys.path.insert(0, sys.argv[2])\n"
                "from pathlib import Path\n"
                "from modelblaster.pipeline import kb_graph_opt as go\n"
                "try:\n"
                "    with go.work_lock(Path(sys.argv[1])): print('got')\n"
                "except RuntimeError as e: print('busy')\n")
        with go.work_lock(work):
            out = subprocess.run([sys.executable, "-c", code, str(work), str(ROOT.parent)],
                                 capture_output=True, text=True).stdout.strip()
            self.assertEqual(out, "busy")
            with self.assertRaises(RuntimeError):
                with go.work_lock(work):
                    pass
        out = subprocess.run([sys.executable, "-c", code, str(work), str(ROOT.parent)],
                             capture_output=True, text=True).stdout.strip()
        self.assertEqual(out, "got")

    def test_hints_save_is_atomic(self):
        path = self.tmp / "graph_hints.json"
        docs = [{"format": gh.FORMAT, "hints": [{"id": f"h{k}", "pad": "x" * 200_000}]} for k in range(4)]
        gh.save(docs[0], path)
        stop = threading.Event()
        bad = []

        def writer(k):
            while not stop.is_set():
                gh.save(docs[k], path)

        def reader():
            while not stop.is_set():
                try:
                    d = json.loads(path.read_text())
                    if d not in docs:
                        bad.append("mixed")
                except ValueError as e:
                    bad.append(repr(e))

        ts = [threading.Thread(target=writer, args=(k,)) for k in range(4)] + [threading.Thread(target=reader)]
        for t in ts:
            t.start()
        time.sleep(1.0)
        stop.set()
        for t in ts:
            t.join()
        self.assertEqual(bad, [])
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.endswith(".tmp")], [])


class CommandTest(unittest.TestCase):
    def test_pool_flags_pass_through(self):
        cfg = go.Config(kb=Path("/kb"), python="py", ssh_hosts=["a", "b"], cuda_visible_devices=["auto"],
                        ssh_key="k")
        cmd = go.KBRunner(cfg, Path("/k")).command([Path("/p")], Path("/o"), parallel_problems=6)
        s = " ".join(cmd)
        self.assertIn("--parallel-problems 6", s)
        self.assertIn("--evaluator ssh --ssh-host a --ssh-host b --cuda-visible-devices auto --ssh-key k", s)
        cmd = go.KBRunner(go.Config(kb=Path("/kb"), gpus="2,3"), Path("/k")).command([Path("/p")], Path("/o"))
        self.assertIn("--gpus 2,3", " ".join(cmd))
        self.assertNotIn("--parallel-problems", " ".join(cmd))
        self.assertIn("--knowledge v2 --ncu off --graph-hints /graph_hints.json", " ".join(cmd))
        self.assertNotIn("--model-hard", " ".join(cmd))
        cfg = go.Config(kb=Path("/kb"), knowledge="v1", ncu="best", model_simple="s", model_hard="h",
                        hetero_policy="category", upkeep_model="u", reward="slot", kb_args=["--explore-frac", "0.2"])
        s = " ".join(go.KBRunner(cfg, Path("/w/knowledge")).command([Path("/p")], Path("/o")))
        self.assertIn("--knowledge v1 --ncu best --reward slot --model-simple s --model-hard h", s)
        self.assertIn("--hetero-policy category --upkeep-model u", s)
        self.assertNotIn("--graph-hints", s)
        self.assertTrue(s.endswith("--explore-frac 0.2"))
        self.assertEqual(go.auto_parallel_hints(0), 4)
        self.assertEqual(go.auto_parallel_hints(4), 8)


if __name__ == "__main__":
    unittest.main()
