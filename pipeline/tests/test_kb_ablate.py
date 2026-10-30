"""kb_ablate (ablation runner) against a fake entry point.

The fake stands in for KernelBlaster's run_multislot.py (or, with ``--work``,
kb_graph_opt): it writes plausible result.json / summary.json / telemetry.jsonl
with seeded random times and costs, and records what it saw (environment,
knowledge dir, seed dir) so isolation can be checked. No GPU, no LLM.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_ablate as K  # noqa: E402

FAKE = r'''
import argparse, json, math, os, random, signal, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("problems", nargs="+")
ap.add_argument("--out"); ap.add_argument("--work"); ap.add_argument("--knowledge-dir"); ap.add_argument("--kb")
ap.add_argument("--python"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model")
ap.add_argument("--gpu-model"); ap.add_argument("--knowledge"); ap.add_argument("--steps", type=int, default=2)
ap.add_argument("--trace")
RESUME_FLAG
a, rest = ap.parse_known_args()
signal.signal(signal.SIGTERM, lambda *_: sys.exit(75))
graph = a.work is not None
out = Path(a.work or a.out); out.mkdir(parents=True, exist_ok=True)
kdir = Path(a.knowledge_dir) if a.knowledge_dir else out / "knowledge"
kdir.mkdir(parents=True, exist_ok=True)
seen = {"argv": sys.argv[1:], "cwd": os.getcwd(), "tmpdir": os.environ.get("TMPDIR"),
        "seed_dir": os.environ.get("KB_KNOWLEDGE_SEED_DIR"),
        "mined": (Path(os.environ.get("KB_KNOWLEDGE_SEED_DIR", "/nonexistent")) / "mined.json").exists(),
        "knowledge_files": sorted(p.name for p in kdir.iterdir()), "knowledge_dir": str(kdir),
        "secret_seen": "AWS_BEARER_TOKEN_BEDROCK" in os.environ, "pythonpath": os.environ.get("PYTHONPATH"),
        "kb_knowledge_env": os.environ.get("KB_KNOWLEDGE"), "resumed": bool(getattr(a, "resume", False)),
        "cache": (out / "cache").exists(), "cache_files": sorted(p.name for p in (out / "cache").iterdir())
        if (out / "cache").exists() else None, "rest": rest}
(out / "seen.json").write_text(json.dumps(seen))
(kdir / "learned.txt").write_text("written by the run")      # the run mutates its knowledge
(out / "partial.json").write_text("{}")
flag = os.environ.get("FAKE_SLEEP_UNLESS")
if flag and not Path(flag).exists():
    time.sleep(60)
if os.environ.get("FAKE_FAIL") == "all":
    sys.exit(1)
eff = float(os.environ.get("FAKE_EFFECT", "1")); salt = os.environ.get("FAKE_SALT", "")
usd_call = float(os.environ.get("FAKE_USD_PER_CALL", "0.01")); gpu = os.environ.get("FAKE_GPU", "NVIDIA TITAN RTX")
tel = open(out / "telemetry.jsonl", "a"); tel.write(json.dumps({"t": "meta", "run": "fake"}) + "\n")
res, irs = [], {}
for i, p in enumerate(a.problems):
    name = Path(p).name
    rng = random.Random(f"{name}|{a.seed}|{salt}")
    base = 100.0 * (i + 1) * math.exp(rng.gauss(0, 0.02))
    final = base / 2.0 / eff * math.exp(rng.gauss(0, 0.05))
    calls, tin, tout = 3, 3000, 2000
    pname = name + "__r00" if graph else name
    for c in range(calls):
        tel.write(json.dumps({"t": "llm", "run": "fake", "span": i * 10 + c, "ts": time.time(), "role": "codegen_simple",
                              "model": a.model, "problem": pname, "calls": 1, "input_tokens": tin // calls,
                              "output_tokens": tout // calls, "cost_usd": 0.0, "est_usd": usd_call,
                              "truncated": int(c == 0 and i == 0)}) + "\n")
    tel.write(json.dumps({"t": "count", "run": "fake", "name": "win", "n": 1, "ts": time.time(), "problem": pname}) + "\n")
    r = {"problem": pname, "baseline_partition_us": base, "final_partition_us": final, "speedup": base / final,
         "final_ok": os.environ.get("FAKE_FAIL") != name, "llm_calls": calls,
         "llm_usage": {"prompt_tokens": tin, "completion_tokens": tout, "cost_usd": 0.0},
         "llm_usage_by_model": [{"model": a.model, "calls": calls, "prompt_tokens": tin, "completion_tokens": tout,
                                 "cost_usd": 0.0, "est_usd": usd_call * calls}],
         "gpu": {"name": gpu, "uuid": "GPU-fake-" + gpu.replace(" ", "_")}, "seconds": 1.0,
         "baseline_e2e_us": base * 0.9, "final_e2e_us": final * 0.9}
    d = out / ("r00/run" if graph else ".") / pname; d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(r))
    res.append({k: r[k] for k in ("problem", "baseline_partition_us", "final_partition_us", "speedup", "final_ok",
                                   "llm_calls", "llm_usage")})
    irs[name] = {"first_us": base, "final_us": final, "passed": r["final_ok"], "speedup": base / final,
                 "model_e2e_us": final * 0.8, "seed_e2e_us": base * 0.8, "model_speedup": base / final,
                 "model_passed": True, "model_gpu": gpu, "hints": ["fuse:x"]}
tel.close()
if graph:
    (out / "summary.json").write_text(json.dumps({"irs": irs, "rounds": 1, "accepted": ["fuse:x"],
                                                  "gpu": {"gpu_model": gpu, "models": [gpu]}}))
else:
    (out / "summary.json").write_text(json.dumps(res + [{"models": {}}]))
'''


def fake_script(path: Path, resume: bool = True) -> Path:
    path.write_text(FAKE.replace("RESUME_FLAG", 'ap.add_argument("--resume", action="store_true")' if resume else ""))
    return path


def git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t"})


class Base(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("MB_TEST_TMP", tempfile.gettempdir()))
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(prefix="mb_ablate_", dir=base))
        # a fake KernelBlaster checkout (git, so the code identity and freezing work)
        self.kb = self.tmp / "kb"
        kd = self.kb / "data" / "kernelblaster" / "knowledge"
        kd.mkdir(parents=True)
        (kd / "mined.json").write_text('{"outcomes": []}')
        (kd / "targets.json").write_text("{}")
        (self.kb / "scripts").mkdir()
        fake_script(self.kb / "scripts" / "run_multislot.py")
        git("init", "-q", cwd=self.kb)
        git("add", "-A", cwd=self.kb)
        git("commit", "-qm", "init", cwd=self.kb)
        self.inputs = []
        for i in range(3):
            d = self.tmp / "problems" / f"p{i}"
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps({"name": f"p{i}"}))
            self.inputs.append(str(d))
        self.ledger = self.tmp / "ledger.jsonl"
        os.environ["MB_ABLATE_POLL_S"] = "0.05"

    def tearDown(self):
        if not os.environ.get("MB_TEST_KEEP"):
            shutil.rmtree(self.tmp, ignore_errors=True)

    def spec(self, **kw) -> Path:
        s = {"name": "t", "entry": "multislot", "kb": str(self.kb), "mb": str(ROOT), "python": sys.executable,
             "inputs": self.inputs, "flags": {"--model": "stub:x", "--steps": 2}, "seeds": [0, 1],
             "gpu_model": "NVIDIA TITAN RTX", "freeze_code": True, "trace": None,
             "arms": {"A": {"env": {"FAKE_SALT": "a"}}, "B": {"env": {"FAKE_SALT": "b"}}}}
        s.update(kw)
        p = self.tmp / "spec.json"
        p.write_text(json.dumps(s))
        return p

    def run_cli(self, *a) -> int:
        return K.main(list(map(str, a)))


class SpecTest(Base):
    def test_parse_defaults_and_errors(self):
        p = self.spec(arms={"A": {}, "B": {"flags": {"--knowledge": "v2"}, "knowledge": {"mined": False}}})
        s = K.normalize_spec(K.load_spec_file(p), p.parent)
        self.assertEqual(s["arms"]["B"]["flags"], {"--model": "stub:x", "--steps": 2, "--knowledge": "v2"})
        self.assertFalse(s["arms"]["B"]["knowledge"]["mined"])
        self.assertEqual(s["report"]["baseline"], "A")
        self.assertEqual(s["interleave"], "counterbalance")
        self.assertNotEqual(K.arm_hash(s, "A"), K.arm_hash(s, "B"))
        # bookkeeping keys do not change the base hash; result-affecting ones do
        s2 = K.normalize_spec({**K.load_spec_file(p), "max_parallel": 4, "budget": {"usd": 3}}, p.parent)
        self.assertEqual(K.base_hash(s), K.base_hash(s2))
        s3 = K.normalize_spec({**K.load_spec_file(p), "gpu_model": "NVIDIA L40S"}, p.parent)
        self.assertNotEqual(K.base_hash(s), K.base_hash(s3))
        bad = [{"arms": {"A": {"flags": {"--seed": 3}}}},                   # runner-owned flag
               {"arms": {"A": {"env": {"AWS_BEARER_TOKEN_BEDROCK": "x"}}}},  # secrets stay out of specs
               {"flags": {}, "arms": {"A": {}}},                             # no explicit --model
               {"arms": {"A": {"inputs": ["x"]}}},                           # inputs are shared
               {"bogus": 1}]
        for b in bad:
            with self.subTest(b=b), self.assertRaises(K.AblateError):
                K.normalize_spec({**K.load_spec_file(p), **b}, p.parent)

    def test_yaml_spec(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("no PyYAML")
        p = self.tmp / "s.yaml"
        p.write_text(yaml.safe_dump(json.loads(self.spec().read_text())))
        self.assertEqual(set(K.normalize_spec(K.load_spec_file(p), p.parent)["arms"]), {"A", "B"})

    def test_interleaving(self):
        p = self.spec(seeds=[0, 1, 2, 3], arms={"A": {}, "B": {}, "C": {}})
        raw = K.load_spec_file(p)
        ids = lambda s: [r["id"] for r in K.plan_runs(K.normalize_spec(s, p.parent))]  # noqa: E731
        self.assertEqual(ids(raw)[:6], ["A__r00", "B__r00", "C__r00", "C__r01", "B__r01", "A__r01"])
        self.assertEqual(ids({**raw, "interleave": "abab"})[:6],
                         ["A__r00", "B__r00", "C__r00", "A__r01", "B__r01", "C__r01"])
        self.assertEqual(ids({**raw, "interleave": "sequential"})[:4], ["A__r00", "A__r01", "A__r02", "A__r03"])
        r1, r2 = ids({**raw, "interleave": "random", "order_seed": 7}), ids({**raw, "interleave": "random",
                                                                            "order_seed": 7})
        self.assertEqual(r1, r2)                                   # reproducible
        for b in range(4):                                         # every block has each arm once
            self.assertEqual(sorted(x[0] for x in r1[3 * b:3 * b + 3]), ["A", "B", "C"])
            self.assertTrue(all(x.endswith(f"r{b:02d}") for x in r1[3 * b:3 * b + 3]))
        per = K.plan_runs(K.normalize_spec({**raw, "per_input": True}, p.parent))
        self.assertEqual(len(per), 3 * 4 * 3)
        self.assertEqual(per[0]["inputs"], [self.inputs[0]])


class RunTest(Base):
    def test_isolation_manifest_and_scrubbing(self):
        snap = self.tmp / "snap"
        snap.mkdir()
        (snap / "claims.json").write_text("[]")
        p = self.spec(arms={"A": {"env": {"FAKE_SALT": "a"}},
                            "B": {"env": {"FAKE_SALT": "b"}, "knowledge": {"mined": False, "snapshot": str(snap)},
                                  "flags": {"--knowledge": "v2"}}})
        secret = "s3cr3t-value-for-test"
        old = {k: os.environ.get(k) for k in ("AWS_BEARER_TOKEN_BEDROCK", "KB_KNOWLEDGE", "PYTHONPATH")}
        os.environ.update(AWS_BEARER_TOKEN_BEDROCK=secret, KB_KNOWLEDGE="v2", PYTHONPATH="/leak")
        try:
            rc = self.run_cli("run", p, "--root", self.tmp / "abl", "--max-parallel", 2)
        finally:
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        self.assertEqual(rc, 0)
        root = self.tmp / "abl"
        runs = sorted((root / "runs").iterdir())
        self.assertEqual([r.name for r in runs], ["A__r00", "A__r01", "B__r00", "B__r01"])
        seen = {r.name: json.loads((r / "out" / "seen.json").read_text()) for r in runs}
        for name, s in seen.items():
            rdir = root / "runs" / name
            self.assertEqual(s["tmpdir"], str(rdir / "tmp"))
            self.assertEqual(s["seed_dir"], str(rdir / "seed"))
            self.assertEqual(s["knowledge_dir"], str(rdir / "knowledge"))
            self.assertIsNone(s["kb_knowledge_env"])                  # inherited knob dropped
            self.assertIsNone(s["pythonpath"])
            self.assertTrue(s["secret_seen"])                          # the run still gets its credentials
            self.assertIn("--gpu-model", s["argv"])
            self.assertIn("--seed", s["argv"])
            self.assertEqual(s["mined"], name.startswith("A"))
            self.assertEqual(s["knowledge_files"], ["claims.json"] if name.startswith("B") else [])
            self.assertFalse((rdir / "tmp").exists())                  # cleaned after the run
        self.assertFalse((snap / "learned.txt").exists())              # the snapshot is copied, not shared
        self.assertTrue((root / "runs" / "B__r00" / "knowledge" / "learned.txt").exists())
        # seeds: the same in both arms per repeat
        self.assertEqual(seen["A__r01"]["argv"][seen["A__r01"]["argv"].index("--seed") + 1], "1")
        self.assertEqual(seen["B__r01"]["argv"][seen["B__r01"]["argv"].index("--seed") + 1], "1")
        man = json.loads((root / "runs" / "B__r00" / "manifest.json").read_text())
        self.assertEqual(man["env"]["set"]["AWS_BEARER_TOKEN_BEDROCK"] if "AWS_BEARER_TOKEN_BEDROCK" in
                         man["env"]["set"] else "<redacted>", "<redacted>")
        self.assertIn("KB_KNOWLEDGE", man["env"]["dropped"])
        self.assertTrue(man["code"]["frozen"])
        self.assertEqual(man["code"]["kb"]["commit"], K.git_identity(self.kb)["commit"])
        self.assertFalse(man["code"]["kb"]["dirty"])
        self.assertIsNotNone(man["isolation"]["seed_hash"])
        self.assertEqual(man["isolation"]["knowledge_hash"], K.tree_hash(snap))
        self.assertFalse(man["isolation"]["mined"])
        self.assertEqual(man["gpu_recorded"]["models"], ["NVIDIA TITAN RTX"])
        self.assertIn("python", man["versions"])
        self.assertTrue(man["command"][1].startswith(str(root / "code" / "kb")))   # the frozen copy ran
        man_a = json.loads((root / "runs" / "A__r00" / "manifest.json").read_text())
        self.assertNotEqual(man_a["isolation"]["seed_hash"], man["isolation"]["seed_hash"])
        self.assertNotEqual(man_a["config_hash"], man["config_hash"])
        self.assertEqual(man_a["spec_hash"], man["spec_hash"])
        for f in root.rglob("*"):                                      # the secret is stored nowhere
            if f.is_file() and f.stat().st_size < 5_000_000:
                self.assertNotIn(secret, f.read_text(errors="replace"), f)
        rep = json.loads((root / "report.json").read_text())
        self.assertEqual(rep["per_arm"]["A"]["results_ok"], 6)
        self.assertEqual(rep["per_arm"]["A"]["llm_calls"], 18)
        self.assertEqual(rep["per_arm"]["A"]["wins"], 6)
        self.assertAlmostEqual(rep["per_arm"]["A"]["usd"], 0.18, places=6)
        self.assertAlmostEqual(rep["per_arm"]["A"]["truncation_rate"], 2 / 18)
        self.assertEqual(rep["rows"][0]["basis"], "e2e")

    def test_graph_opt_entry_and_cache(self):
        cache = self.tmp / "cache_snap"
        cache.mkdir()
        (cache / "index.json").write_text("{}")
        p = self.spec(entry="graph_opt", script=str(fake_script(self.tmp / "fake_go.py")), freeze_code=False,
                      seeds=[0], arms={"fresh": {}, "warm": {"mb_cache": {"snapshot": str(cache)}},
                                       "shared": {"mb_cache": {"shared": str(self.tmp / "shared_cache")}}})
        self.assertEqual(self.run_cli("run", p, "--root", self.tmp / "abl"), 0)
        root = self.tmp / "abl"
        seen = {a: json.loads((root / "runs" / f"{a}__r00" / "work" / "seen.json").read_text())
                for a in ("fresh", "warm", "shared")}
        self.assertFalse(seen["fresh"]["cache"])                       # no cache unless the arm opts in
        self.assertEqual(seen["warm"]["cache_files"], ["index.json"])
        self.assertTrue((root / "runs" / "shared__r00" / "work" / "cache").is_symlink())
        self.assertEqual(seen["fresh"]["pythonpath"], str(ROOT.parent))
        self.assertIn("--kb", seen["fresh"]["argv"])
        man = json.loads((root / "runs" / "shared__r00" / "manifest.json").read_text())
        self.assertFalse(man["isolation"]["cache_isolated"])
        rep = json.loads((root / "report.json").read_text())
        self.assertEqual(rep["per_arm"]["fresh"]["results_ok"], 3)
        row = rep["rows"][0]
        self.assertEqual(row["basis"], "model")
        self.assertEqual(row["llm_calls"], 3)                           # attributed via p0__r00
        self.assertEqual(row["wins"], 1)

    def test_budget_stop_ledger_and_resume(self):
        p = self.spec(flags={"--model": "us.anthropic.claude-haiku-5-5"},
                      budget={"usd": 0.1, "est_usd_per_run": 0.09, "ledger": str(self.ledger), "who": "ablate-test"},
                      env={"FAKE_USD_PER_CALL": "0.01"})                 # 9 calls x $0.01 per run
        root = self.tmp / "abl"
        self.assertEqual(self.run_cli("run", p, "--root", root), 4)    # stopped on the budget
        st = json.loads((root / "state.json").read_text())
        done = [r for r in st["runs"].values() if r["status"] == "done"]
        self.assertEqual(len(done), 1)
        self.assertTrue(all(r.get("reason") == "budget" for r in st["runs"].values() if r["status"] == "pending"))
        rows = [json.loads(x) for x in self.ledger.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["who"], "ablate-test")
        self.assertEqual(rows[0]["calls"], 9)
        self.assertAlmostEqual(rows[0]["est_usd"], 0.09)
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_ablate", "status", str(root)],
                             capture_output=True, text=True, cwd=ROOT.parent,
                             env={**os.environ, "PYTHONPATH": str(ROOT.parent)})
        self.assertIn("budget", out.stdout)
        # the ablation continues with a bigger cap: done runs are skipped
        self.assertEqual(self.run_cli("run", p, "--root", root, "--budget-usd", 1), 0)
        st = json.loads((root / "state.json").read_text())
        self.assertTrue(all(r["status"] == "done" and r["attempts"] == 1 for r in st["runs"].values()))
        self.assertEqual(len(self.ledger.read_text().splitlines()), 4)
        self.assertAlmostEqual(st["spent_usd"], 0.36, places=6)

    def test_dry_run_creates_nothing(self):
        p = self.spec(flags={"--model": "us.anthropic.claude-haiku-5-5", "--rollouts": 2, "--steps": 3})
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(self.run_cli("run", p, "--root", self.tmp / "abl", "--dry-run"), 0)
        self.assertFalse((self.tmp / "abl").exists())
        txt = buf.getvalue()
        self.assertIn("4 run(s)", txt)
        self.assertIn("estimated cost: $", txt)
        self.assertIn("--resume True", txt)
        est = K.estimate_run_usd(K.normalize_spec(K.load_spec_file(p), p.parent), "A", 3)
        self.assertAlmostEqual(est, 3 * 2 * 3 * 1.5 * (4000 * 1 + 3000 * 5) / 1e6)

    def _start(self, p, root):
        return subprocess.Popen([sys.executable, "-m", "modelblaster.pipeline.kb_ablate", "run", str(p), "--root",
                                 str(root)], cwd=ROOT.parent, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": str(ROOT.parent)}, start_new_session=True)

    def _wait_running(self, root, n=1, timeout=60):
        t = time.time() + timeout
        while time.time() < t:
            st = K._read_json(root / "state.json") or {"runs": {}}
            run = [r for r in st["runs"].values() if r["status"] == "running"]
            if len(run) >= n and all((root / "runs" / r["id"] / "out" / "partial.json").exists() for r in run):
                return run
            time.sleep(0.1)
        self.fail("no run started")

    def test_resume_after_kill(self):
        for resume_flag in (True, False):
            with self.subTest(resume_flag=resume_flag):
                fake_script(self.kb / "scripts" / "run_multislot.py", resume=resume_flag)
                git("commit", "-qam", f"resume {resume_flag}", "--allow-empty", cwd=self.kb)
                flag = self.tmp / f"go_{resume_flag}"
                p = self.spec(env={"FAKE_SLEEP_UNLESS": str(flag)}, seeds=[0])
                root = self.tmp / f"abl_{resume_flag}"
                proc = self._start(p, root)
                running = self._wait_running(root)
                # a hard kill of the runner and of the run (machine reboot, OOM killer, ...)
                os.killpg(proc.pid, signal.SIGKILL)
                os.killpg(running[0]["pid"], signal.SIGKILL)
                proc.wait()
                time.sleep(0.2)
                flag.write_text("go")
                self.assertEqual(self.run_cli("run", p, "--root", root), 0)
                st = json.loads((root / "state.json").read_text())
                r = st["runs"][running[0]["id"]]
                self.assertEqual((r["status"], r["attempts"]), ("done", 2))
                self.assertEqual(r["mode"], "resume-flag" if resume_flag else "fresh")
                seen = json.loads((root / "runs" / r["id"] / "out" / "seen.json").read_text())
                self.assertEqual(seen["resumed"], resume_flag)
                other = [x for x in st["runs"].values() if x["id"] != r["id"]]
                self.assertTrue(all(x["attempts"] == 1 for x in other))

    def test_sigterm_stops_cleanly(self):
        flag = self.tmp / "go"
        p = self.spec(env={"FAKE_SLEEP_UNLESS": str(flag)}, seeds=[0], max_parallel=2)
        root = self.tmp / "abl"
        proc = self._start(p, root)
        self._wait_running(root, n=2)
        os.kill(proc.pid, signal.SIGTERM)
        self.assertEqual(proc.wait(timeout=60), 75)
        st = json.loads((root / "state.json").read_text())
        self.assertEqual(sorted(r["status"] for r in st["runs"].values()), ["interrupted", "interrupted"])
        flag.write_text("go")
        self.assertEqual(self.run_cli("run", p, "--root", root), 0)
        st = json.loads((root / "state.json").read_text())
        self.assertTrue(all(r["status"] == "done" and r["mode"] == "resume-flag" for r in st["runs"].values()))


class ReportTest(Base):
    def _ablation(self, **arms):
        self.inputs = []
        for i in range(6):
            d = self.tmp / "problems6" / f"q{i}"
            d.mkdir(parents=True)
            self.inputs.append(str(d))
        p = self.spec(seeds=[0, 1, 2, 3], max_parallel=4, arms=arms)
        root = self.tmp / "abl"
        self.assertEqual(self.run_cli("run", p, "--root", root), 0)
        return root, json.loads((root / "report.json").read_text())

    def test_effect_detected_null_not(self):
        root, rep = self._ablation(A={"env": {"FAKE_SALT": "a"}},
                                   B={"env": {"FAKE_SALT": "b", "FAKE_EFFECT": "1.25"}},
                                   C={"env": {"FAKE_SALT": "c"}})
        c = {x["b"]: x for x in rep["comparisons"]}
        b, n = c["B"]["final_us"], c["C"]["final_us"]
        self.assertEqual(b["n"], 24)
        self.assertAlmostEqual(b["ratio"], 0.8, delta=0.04)
        self.assertLess(b["ci95"][1], 1.0)
        self.assertLess(b["p_perm"], 0.001)
        self.assertLess(b["p_wilcoxon"], 0.001)
        self.assertGreater(n["p_perm"], 0.05)
        self.assertGreater(n["p_wilcoxon"], 0.05)
        self.assertLess(n["ci95"][0], 1.0)
        self.assertGreater(n["ci95"][1], 1.0)
        self.assertGreater(c["B"]["speedup"]["ratio"], 1.15)
        md = (root / "report.md").read_text()
        self.assertIn("| B vs A | final time | 24 (6) |", md)
        self.assertTrue((root / "report.csv").read_text().startswith("arm,repeat,seed,problem"))
        # re-render from the runs alone
        (root / "report.md").unlink()
        self.assertEqual(self.run_cli("report", root), 0)
        self.assertTrue((root / "report.md").exists())

    def test_refuses_mismatched_code_and_gpus(self):
        root, rep = self._ablation(A={"env": {"FAKE_SALT": "a"}}, B={"env": {"FAKE_SALT": "b"}})
        self.assertFalse(rep["refused"])
        mf = root / "runs" / "B__r01" / "manifest.json"
        man = json.loads(mf.read_text())
        man["code"]["kb"]["commit"] = "0" * 40
        mf.write_text(json.dumps(man))
        rep = K.build_report(root)
        self.assertTrue(rep["refused"])
        self.assertEqual(rep["comparisons"], [])
        self.assertTrue(any("different KB code" in i for i in rep["issues"]))
        self.assertEqual(self.run_cli("report", root), 3)
        rep = K.build_report(root, force=True)
        self.assertTrue(rep["forced"])
        self.assertEqual(len(rep["comparisons"]), 1)
        # different GPU models recorded
        man["code"]["kb"]["commit"] = json.loads((root / "runs" / "A__r00" / "manifest.json").read_text())["code"]["kb"]["commit"]
        man["gpu_recorded"] = {"models": ["NVIDIA L40S"]}
        mf.write_text(json.dumps(man))
        rep = K.build_report(root)
        self.assertTrue(rep["refused"])
        self.assertTrue(any("GPU models" in i for i in rep["issues"]))

    def test_moved_checkout_and_changed_spec_refused(self):
        p = self.spec(freeze_code=False, seeds=[0], arms={"A": {}})
        root = self.tmp / "abl"
        self.assertEqual(self.run_cli("run", p, "--root", root), 0)
        # more repeats of the same configuration may be added ...
        p = self.spec(freeze_code=False, seeds=[0, 1], arms={"A": {}})
        (self.kb / "scripts" / "extra.py").write_text("# new code\n")
        git("add", "-A", cwd=self.kb)
        git("commit", "-qm", "moved", cwd=self.kb)
        # ... but not on moved code (live checkouts), nor with a changed base configuration
        self.assertEqual(self.run_cli("run", p, "--root", root), 2)
        p2 = self.spec(freeze_code=False, seeds=[0, 1], arms={"A": {}}, gpu_model="NVIDIA L40S")
        self.assertEqual(self.run_cli("run", p2, "--root", root), 2)
        p3 = self.spec(freeze_code=False, seeds=[0, 1], arms={"A": {"flags": {"--steps": 9}}})
        self.assertEqual(self.run_cli("run", p3, "--root", root), 2)
        self.assertEqual(self.run_cli("run", p, "--root", root, "--force"), 0)
        rep = K.build_report(root)
        self.assertTrue(any("different KB code" in i for i in rep["issues"]))


class StatsTest(unittest.TestCase):
    def test_exact_tests(self):
        d = [0.1 * (i + 1) for i in range(10)]
        self.assertAlmostEqual(K.wilcoxon_p(d), 2 / 1024)
        self.assertAlmostEqual(K.sign_flip_p(d), 2 / 1024)
        self.assertAlmostEqual(K.wilcoxon_p([1.0, -1.0]), 1.0)
        # R wilcox.test example (depression scores, n = 9, V = 40): exact two-sided p = 0.03906
        x = [1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30]
        y = [0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29]
        self.assertAlmostEqual(K.wilcoxon_p([b - a for a, b in zip(x, y)]), 0.0390625, places=6)
        lo, hi = K.bootstrap_ci([math.log(0.8)] * 5 + [math.log(0.9)] * 5)
        self.assertTrue(0.79 < lo < hi < 0.91)

    def test_scrub(self):
        env = K.scrub_env({"AWS_BEARER_TOKEN_BEDROCK": "x", "KB_LLM_SERVER_TOKEN": "y", "ANTHROPIC_API_KEY": "z",
                           "KB_LLM_SERVER_TOKEN_FILE": "/p", "PATH": "/bin"})
        self.assertEqual(env, {"AWS_BEARER_TOKEN_BEDROCK": "<redacted>", "KB_LLM_SERVER_TOKEN": "<redacted>",
                               "ANTHROPIC_API_KEY": "<redacted>", "KB_LLM_SERVER_TOKEN_FILE": "/p", "PATH": "/bin"})
        self.assertEqual(K.scrub_argv(["a", "--api-key", "k", "--token=t", "--ssh-key", "/f"]),
                         ["a", "--api-key", "<redacted>", "--token=<redacted>", "--ssh-key", "/f"])


if __name__ == "__main__":
    unittest.main()
