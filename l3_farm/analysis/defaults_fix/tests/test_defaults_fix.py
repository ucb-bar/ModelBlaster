"""Offline tests (fakes only: no ssh, no Slurm, no key file) of farm_defaults.py and its wiring in farm.py.
Run:  python3 -m pytest l3_farm/analysis/defaults_fix/tests -q      (or: python3 l3_farm/analysis/defaults_fix/tests/test_defaults_fix.py)"""
import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import farm_defaults as FD  # noqa: E402
import farm  # noqa: E402  (reads config.json, no side effects; state is never written here)

CMD = "python -m x.llm_server --port 23456 --token-file /t --max-concurrency {c} {q}--allow-models azure/.* --ledger /l --budget-usd {b} --log-file /y"


class Relay(unittest.TestCase):
    def test_defaults_ignore_legacy_kb(self):
        w = FD.relay_conf({"kb": {"relay_concurrency": 12, "relay_budget_usd": 150}})
        self.assertEqual((w["conc"], w["queue"], w["budget"]), (96, 1024, 10000))
        self.assertEqual(len(FD.legacy_relay_differs({"kb": {"relay_concurrency": 12, "relay_budget_usd": 150}})), 2)

    def test_config_block_wins(self):
        w = FD.relay_conf({"relay": {"concurrency": 48, "queue": 64, "budget_usd": 5}})
        self.assertEqual((w["conc"], w["queue"], w["budget"]), (48, 64, 5))

    def test_parse_and_diffs(self):
        want = FD.relay_conf({})
        have = FD.parse_relay_info("conc=12 budget=150 port=23456")
        self.assertEqual(have, {"conc": 12, "budget": 150.0, "port": 23456})
        self.assertEqual(len(FD.relay_diffs(want, have)), 2)          # queue not on the cmdline: not compared
        self.assertEqual(FD.relay_diffs(want, {"conc": 96, "queue": 1024, "budget": 10000.0, "port": 1}), [])
        self.assertEqual(FD.relay_diffs(want, {}), [])                # unknown -> no action

    def test_decision(self):
        want = FD.relay_conf({})
        bad = {"conc": 12, "port": 5}
        self.assertEqual(FD.relay_decision(want, bad, {}, busy=False)[0], "restart")
        self.assertEqual(FD.relay_decision(want, bad, {}, busy=True)[0], "busy")
        self.assertEqual(FD.relay_decision(dict(want, restart_busy=True), bad, {}, busy=True)[0], "restart")
        self.assertEqual(FD.relay_decision(want, bad, {"tries": 3}, busy=False)[0], "giveup")
        self.assertEqual(FD.relay_decision(want, {"conc": 12}, {}, busy=False)[0], "noport")
        self.assertEqual(FD.relay_decision(want, {}, {}, busy=False)[0], "unknown")

    def test_check_due(self):
        self.assertTrue(FD.check_due({}, 1000, 10))
        self.assertFalse(FD.check_due({"t": 1000}, 1100, 10))
        self.assertTrue(FD.check_due({"t": 1000, "diff": True}, 1061, 10))


class FakeFarm:
    """Patch farm.srun / farm.detach / farm.log and CFG for one test."""
    def __enter__(self):
        self.cfg, self.srun, self.detach, self.log = copy.deepcopy(farm.CFG), farm.srun, farm.detach, farm.log
        self.calls, self.logs, self.info = [], [], b""
        farm.srun = lambda job, script, args=(), inp=b"", key=False, timeout=300: (self.calls.append(("srun", job, script)) or (0, self.info, b""))
        farm.detach = lambda job, script, args=(), inp=b"", key=False: (self.calls.append(("detach", job, script, list(args), key)) or True)
        farm.log = lambda *a: self.logs.append(" ".join(map(str, a)))
        return self

    def __exit__(self, *e):
        farm.CFG.clear(); farm.CFG.update(self.cfg)
        farm.srun, farm.detach, farm.log = self.srun, self.detach, self.log


class Wiring(unittest.TestCase):
    def test_bring_up_uses_defaults(self):
        with FakeFarm() as f:
            nd = {"gpu": "L40S", "tree": "T", "jobs": {}, "setup": "done", "relay": "down"}
            self.assertEqual(farm.bring_up(1, nd, {"tree": "T"}), "relay")
            self.assertEqual(f.calls[-1], ("detach", 1, "relay.sh", [96, 10000, 1024], True))   # key=True: over stdin, never argv

    def test_guard_restarts_wrong_relay(self):
        with FakeFarm() as f:
            f.info = b" conc=12 budget=150 port=23456\n"
            s = {}
            nd = {"ok": True, "relay": "ok"}
            self.assertTrue(farm.relay_guard(s, 7, nd, busy=False))
            self.assertEqual(f.calls[-1], ("detach", 7, "relay_restart.sh", [96, 1024, 10000, 23456], True))
            self.assertTrue(any("restarting" in l for l in f.logs))
            n = len(f.calls)
            self.assertFalse(farm.relay_guard(s, 7, nd, busy=False))       # rate-limited: no second probe/restart right away
            self.assertEqual(len(f.calls), n)

    def test_guard_noop_when_matching_or_busy(self):
        with FakeFarm() as f:
            f.info = b" conc=96 queue=1024 budget=10000 port=1\n"
            self.assertFalse(farm.relay_guard({}, 7, {"ok": True, "relay": "ok"}, False))
            self.assertFalse([c for c in f.calls if c[0] == "detach"])
            f.info = b" conc=12 budget=150 port=1\n"
            self.assertFalse(farm.relay_guard({}, 8, {"ok": True, "relay": "ok"}, True))   # busy: not restarted
            self.assertFalse([c for c in f.calls if c[0] == "detach"])
            self.assertFalse(farm.relay_guard({}, 9, {"ok": True, "relay": "down"}, False))  # relay not ok: bring_up's job

    def test_guard_disabled_and_errors_swallowed(self):
        with FakeFarm() as f:
            farm.CFG["relay"] = {"enforce": False}
            self.assertFalse(farm.relay_guard({}, 7, {"ok": True, "relay": "ok"}, False))
            self.assertEqual(f.calls, [])
            farm.CFG["relay"] = {}
            farm.srun = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
            self.assertFalse(farm.relay_guard({}, 7, {"ok": True, "relay": "ok"}, False))


class Alloc(unittest.TestCase):
    def test_default_unchanged(self):
        self.assertEqual(FD.alloc_plan({"alloc_time": "10:00:00"}, "driver"), dict(mode="crun", time="10:00:00", note=None, role="driver"))
        self.assertEqual(FD.alloc_plan({}, "pool")["time"], "10:00:00")

    def test_24h_and_clamp(self):
        self.assertEqual(FD.norm_time("24:00:00"), ("1-00:00:00", None))
        self.assertEqual(FD.norm_time("1-00:00:00"), ("1-00:00:00", None))
        t, note = FD.norm_time("1-12:00:00")
        self.assertEqual(t, "1-00:00:00"); self.assertIn("clamped", note)
        self.assertEqual(FD.norm_time("3-00:00:00")[0], "1-00:00:00")
        with self.assertRaises(ValueError):
            FD.norm_time("soon")

    def test_per_role(self):
        cfg = {"alloc_time": "10:00:00", "alloc": {"driver_mode": "batch", "driver_time": "24:00:00"}}
        self.assertEqual(FD.alloc_plan(cfg, "driver")["mode"], "batch")
        self.assertEqual(FD.alloc_plan(cfg, "driver")["time"], "1-00:00:00")
        self.assertEqual(FD.alloc_plan(cfg, "pool"), dict(mode="crun", time="10:00:00", note=None, role="pool"))
        with self.assertRaises(ValueError):
            FD.alloc_plan({"alloc_mode": "slurm"}, "pool")

    def test_need_driver_and_pick(self):
        self.assertTrue(FD.need_driver(None, {}, {}, 120))
        self.assertFalse(FD.need_driver(5, {}, {5: 400}, 120))                 # untagged live driver with time left
        self.assertTrue(FD.need_driver(5, {}, {5: 60}, 120))                   # about to end: successor needed
        self.assertFalse(FD.need_driver(5, {"9": "driver"}, {5: 60, 9: None}, 120))   # a pending driver holder exists
        self.assertEqual(FD.pick_driver([3, 9, 4], {"9": "driver"}), 9)
        self.assertEqual(FD.pick_driver([3, 9, 4], {}), 3)                     # old rule
        self.assertIsNone(FD.pick_driver([], {}))

    def test_alloc_batch_calls_submit_holder_not_crun(self):
        import batch_nodes as B
        with FakeFarm() as f:
            farm.CFG["alloc"] = {"driver_mode": "batch", "driver_time": "24:00:00"}
            got = {}
            orig, popen = B.submit_holder, subprocess.Popen
            B.submit_holder = lambda **kw: (got.update(kw) or 4242)
            subprocess.Popen = lambda *a, **k: self.fail("crun must not run")
            sc = dict(farm.SC)
            try:
                farm.SC["mode"] = "pooled"
                s = {"last_alloc": 0}
                farm.alloc(s, [])
            finally:
                B.submit_holder, subprocess.Popen = orig, popen
                farm.SC.clear(); farm.SC.update(sc)
            self.assertEqual(got["time"], "1-00:00:00"); self.assertEqual(got["warn_s"], 240)
            self.assertEqual(got["account"], "cuopt_fallback"); self.assertTrue(got["name"].startswith("mb-"))
            self.assertEqual(s["alloc_roles"], {"4242": "driver"})

    def test_alloc_crun_default_command(self):
        with FakeFarm() as f:
            import tempfile
            seen = {}
            popen, root = subprocess.Popen, farm.ROOT
            subprocess.Popen = lambda argv, **k: seen.update(argv=argv)
            with tempfile.TemporaryDirectory() as d:            # alloc() appends to <ROOT>/state/alloc.log: keep it off the real state/
                (Path(d) / "state").mkdir()
                farm.ROOT = Path(d)
                try:
                    farm.alloc({"last_alloc": 0}, [])
                finally:
                    subprocess.Popen, farm.ROOT = popen, root
            self.assertIn("-t 10:00:00", seen["argv"][-1]); self.assertTrue(seen["argv"][-1].startswith("crun -q"))

    def test_sbatch_test_cmd_is_test_only(self):
        c = FD.sbatch_test_cmd({})
        self.assertIn("--test-only", c); self.assertIn("-t 1-00:00:00", c); self.assertIn("-A cuopt_fallback", c)
        self.assertNotIn("srun", c)


class Left(unittest.TestCase):
    def test_margin(self):
        self.assertEqual(FD.effective_left(600, {}), 585)
        self.assertEqual(FD.effective_left(600, {"left_safety_min": 0}), 600)
        self.assertEqual(FD.effective_left(10, {}), 0)

    def test_jobs_conservative(self):
        with FakeFarm():
            orig = farm.ssh
            farm.ssh = lambda cmd, inp=None, timeout=300: (0, b"11|RUNNING|n1|9:59:10|l40s@x\n12|PENDING||10:00:00|l40s@x\n", b"")
            try:
                r = farm.jobs()
            finally:
                farm.ssh = orig
            self.assertEqual((r[0]["left_raw"], r[0]["left"]), (599, 584))

    def test_expiry_warning_once(self):
        w = {}
        self.assertFalse(FD.expiry_due(120, w, 5, {}))
        self.assertTrue(FD.expiry_due(30, w, 5, {}))
        w["5"] = 1
        self.assertFalse(FD.expiry_due(10, w, 5, {}))
        self.assertTrue(FD.expiry_due(10, w, 6, {}))
        self.assertTrue(FD.expiry_due(45, {}, 5, {"driver_warn_min": 60}))


class PooledDriver(unittest.TestCase):
    def run_tick(self, left_raw, f, busy=False):
        s = {"tree": "T", "problems": {"p1": {"status": "running" if busy else "done", "node": 7}}, "sched": {"driver": 7}}
        live = {7: dict(job=7, host="h7", left=left_raw - 15, left_raw=left_raw, state="RUNNING")}
        probes = {7: {"ok": True, "relay": "ok", "setup": "done", "tree": "T", "gpu": "L40S", "foreign": "0", "jobs": {}}}
        sc = dict(farm.SC)
        farm.SC["mode"] = "pooled"
        try:
            farm.pooled_tick(s, probes, live, [7])
        finally:
            farm.SC.clear(); farm.SC.update(sc)
        return s

    def test_expiry_warning_event_once(self):
        with FakeFarm() as f:
            f.info = b" conc=96 queue=1024 budget=10000 port=1\n"
            s = self.run_tick(100, f)
            self.assertFalse(s.get("warnings"))
            s = self.run_tick(29, f)
            self.assertEqual(len(s["warnings"]), 1)
            self.assertTrue(any(l.startswith("WARNING driver 7") for l in f.logs))
            self.assertEqual(s["sched"]["decisions"][-1]["event"], "driver_expiry_warning")

    def test_driver_relay_restarted_when_idle_not_when_busy(self):
        with FakeFarm() as f:
            f.info = b" conc=12 budget=150 port=9\n"
            self.run_tick(300, f, busy=True)
            self.assertFalse([c for c in f.calls if c[0] == "detach"])
        with FakeFarm() as f:
            f.info = b" conc=12 budget=150 port=9\n"
            self.run_tick(300, f, busy=False)
            self.assertEqual([c for c in f.calls if c[0] == "detach"], [("detach", 7, "relay_restart.sh", [96, 1024, 10000, 9], True)])


class Scripts(unittest.TestCase):
    def test_remote_scripts_parse(self):
        for n in ("relay.sh", "relay_info.sh", "relay_restart.sh", "holder.sh"):
            self.assertEqual(subprocess.run(["bash", "-n", str(ROOT / "remote" / n)]).returncode, 0, n)

    def test_relay_info_parses_cmdline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pgrep"
            p.write_text("#!/bin/bash\necho '1 " + CMD.format(c=12, q="", b=150) + "'\n"); p.chmod(0o755)
            out = subprocess.run(["bash", str(ROOT / "remote" / "relay_info.sh")], capture_output=True, text=True,
                                 env=dict(os.environ, PATH=d + ":" + os.environ["PATH"])).stdout
            self.assertEqual(FD.parse_relay_info(out), {"conc": 12, "budget": 150.0, "port": 23456})
            p.write_text("#!/bin/bash\necho '1 " + CMD.format(c=96, q="--max-queue 1024 ", b=10000) + "'\n")
            out = subprocess.run(["bash", str(ROOT / "remote" / "relay_info.sh")], capture_output=True, text=True,
                                 env=dict(os.environ, PATH=d + ":" + os.environ["PATH"])).stdout
            self.assertEqual(FD.relay_diffs(FD.relay_conf({}), FD.parse_relay_info(out)), [])


if __name__ == "__main__":
    unittest.main()
