"""Graph search redesign (graph_search / variant_policy / graph_priors), GPU-, torch- and LLM-free.

- priors: keys, the bootstrap table reproduces the sweep's acceptance counts, partition scoping, eligibility,
  exploration, this work dir's measured decisions, A5's ``kb_global.struct_prior`` adapter;
- variant_policy: checkpoint (accepted / promising / every rejection test), successive-halving rungs, allocation,
  survival weights and steps, trajectories;
- the interleaved engine with a stub KernelBlaster process / re-timing: decisions are made while the run is still
  going, in completion order; an acceptance stops exactly the plan problems whose ops it absorbs; a resumed round
  does not decide twice; a hint whose re-timing lost to the infrastructure is re-timed after the run;
- decide_checkpoint: promising is non-voting and keeps the rung state; telemetry carries budget, evidence, prior;
- pre-apply from priors (exploration draw, uncertain kinds untouched), survival-weighted plan budgets;
- graph-diverse beam: members, member docs, one plan item per op signature across members (shared kernels),
  successive halving, commit;
- flags / round key: defaults add nothing to the round key (existing work dirs resume), flags do.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_priors as gp  # noqa: E402
from modelblaster.pipeline import graph_search as gsm  # noqa: E402
from modelblaster.pipeline import kb_resume  # noqa: E402
from modelblaster.pipeline import variant_policy as vp  # noqa: E402

DECISIONS = Path(__file__).resolve().parents[3] / "l3_farm" / "analysis" / "e7_data" / "decisions.csv"
PART = "cuda:sm_89|fp16"


def _cfg(**kw):
    base = dict(arch="89", seed=0, steps=4, variant_steps=2, accept_gain=0.03, min_gain_pct=2.0, max_variants=3,
                remeasure=5, plan_parallel=0, stop_patience=8, knowledge_global_part=PART)
    base.update(gsm.CFG_DEFAULTS)
    base.update(kw)
    return SimpleNamespace(**base)


def _fuse(kind, status="proposed", chains=None, est=5.0, ir="m"):
    h = gh.fuse_hint(kind, chains, {"by": "test", "ir": ir, "est_gain_pct": est})
    h["status"] = status
    return h


class _Prog:
    """kb_resume.RoundProgress stand-in (in memory)."""

    def __init__(self, phases=None):
        self.data = {"phases": {k: {"value": v} for k, v in (phases or {}).items()}}

    def get(self, k, d=None):
        return self.data["phases"].get(k, {}).get("value", d)

    def mark(self, k, v=None):
        self.data["phases"][k] = {"value": json.loads(json.dumps(v, default=str))}

    def once(self, k, fn):
        if k in self.data["phases"]:
            return self.get(k)
        v = fn()
        self.mark(k, v)
        return v


# ------------------------------------------------------------------ priors

class PriorsTest(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(gp.fusion_keys("fused__conv2d_f16__batchnorm2d_f16__add_f16__a"),
                         ["conv2d+batchnorm2d+add", "conv+bn+ewise", "conv+*2"])
        self.assertEqual(gp.fusion_keys({"kind": "fuse_attention", "id": "a"}), ["fuse_attention"])
        self.assertEqual(gp.fusion_keys("fused__cat3_c1_f16__batchnorm2d_f16")[1], "cat+bn")

    @unittest.skipUnless(DECISIONS.exists(), "sweep decisions.csv not present")
    def test_bootstrap_reproduces_the_sweep_counts(self):
        t = gp.table_from_decisions(DECISIONS)
        c = t["counts"]
        want = {"batchnorm2d+relu": [6, 0], "linear+relu": [4, 0], "batchnorm2d+add": [3, 0],
                "ewise2+ewise2": [5, 0], "conv2d_dw+batchnorm2d+relu6": [4, 0], "catn+batchnorm2d": [3, 0],
                "conv2d+batchnorm2d": [9, 5], "conv2d+relu": [4, 4], "conv2d+batchnorm2d+add": [1, 2]}
        for k, v in want.items():
            self.assertEqual(c[k], v, k)

    def test_eligibility_partition_and_backoff(self):
        tab = {"part": PART, "source": "t", "counts": {"batchnorm2d+relu": [6, 0], "conv2d+batchnorm2d": [9, 5],
                                                       "catn+batchnorm2d": [3, 0], "cat+bn": [9, 0],
                                                       "cat2_c1+batchnorm2d": [2, 0]}}
        p = gp.Priors(PART, tables=[tab], use_a5=False)
        hi = p.prior("fused__batchnorm2d_f16__relu_f16")
        self.assertGreaterEqual(hi["lb"], 0.55)
        self.assertLess(p.prior("fused__conv2d_f16__batchnorm2d_f16")["lb"], 0.55)
        self.assertGreaterEqual(p.prior("fused__catn_f16__batchnorm2d_f16")["lb"], 0.55)      # 3/0 qualifies
        b = p.prior("fused__cat2_c1_f16__batchnorm2d_f16")                                  # 2/0 -> backs off
        self.assertEqual((b["key"], b["level"]), ("cat+bn", 1))
        other = gp.Priors("cuda:sm_100|fp16", tables=[tab], use_a5=False).prior("fused__batchnorm2d_f16__relu_f16")
        self.assertEqual(other["source"], "seed")                     # another partition: no transfer
        self.assertAlmostEqual(gp.beta_quantile(0.1, 7, 1), 0.1 ** (1 / 7), places=4)

    def test_work_decisions_count_only_measured(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t) / "graph_decisions.jsonl"
            rows = [{"event": "decision", "hint": "fuse:fused__conv2d_f16__relu_f16", "status": "accepted",
                     "measured": True}] * 3 + \
                   [{"event": "decision", "hint": "fuse:fused__conv2d_f16__relu_f16", "status": "accepted",
                     "measured": False}] * 5
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            p = gp.Priors(PART, local=f, use_a5=False)
            r = p.prior("fused__conv2d_f16__relu_f16")
            self.assertEqual((r["accepted"], r["rejected"], r["source"]), (3.0, 0.0, "work"))

    def test_a5_adapter(self):
        from modelblaster.pipeline import kb_global
        calls = []

        def struct_prior(kind, context, *, work=None, store=None, part=None):
            calls.append((kind, context, part))
            return SimpleNamespace(accepted=10, rejected=0, n=10, rate=1.0)
        old = getattr(kb_global, "struct_prior", None)
        kb_global.struct_prior = struct_prior
        try:
            tab = {"part": PART, "source": "t", "counts": {"conv2d+relu": [4, 4]}}
            r = gp.Priors(PART, tables=[tab]).prior({"kind": "fuse", "fused_kind": "fused__conv2d_f16__relu_f16"})
        finally:
            if old is None:
                del kb_global.struct_prior
            else:
                kb_global.struct_prior = old
        self.assertEqual(calls[0], ("fuse", ("conv2d", "relu"), PART))
        self.assertEqual((r["accepted"], r["rejected"], r["source"]), (10.0, 0.0, "a5"))   # table not added

    def test_exploration_is_deterministic(self):
        draws = [gp.explore(0, f"h{i}", 0.1) for i in range(400)]
        self.assertEqual(draws, [gp.explore(0, f"h{i}", 0.1) for i in range(400)])
        self.assertTrue(20 <= sum(draws) <= 60)
        self.assertFalse(any(gp.explore(0, f"h{i}", 0.0) for i in range(50)))


# ------------------------------------------------------------------ variant_policy

class PolicyTest(unittest.TestCase):
    def test_checkpoint_paths(self):
        acc = vp.checkpoint(orig_us=100, var_us=90, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8)
        self.assertEqual(acc["decision"], "accepted")
        prom = vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8,
                             traj=[200, 120, 101], prior={"p_mean": 0.6, "n": 10})
        self.assertEqual(prom["decision"], "promising")
        flat = vp.checkpoint(orig_us=100, var_us=120, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8,
                             traj=[120, 120])
        self.assertTrue(flat["decision"] == "rejected" and flat["why"].startswith("plateau"))
        near = vp.checkpoint(orig_us=100, var_us=97.5, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8,
                             traj=[97.5, 97.5])                     # within 2% of the target, parity seed
        self.assertEqual(near["decision"], "promising")
        cap = vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=2, steps_used=8, cap_steps=8,
                            traj=[200, 101])
        self.assertTrue(cap["why"].startswith("budget cap"))
        bound = vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8,
                              traj=[200, 101], hr={"incumbent_us": 10, "optimistic_us": 9})
        self.assertTrue(bound["why"].startswith("bound falsified"))
        prior = vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=0, steps_used=1, cap_steps=8,
                              traj=[200, 101], prior={"p_mean": 0.05, "n": 20, "source": "t"})
        self.assertTrue(prior["why"].startswith("prior"))
        slow = vp.checkpoint(orig_us=100, var_us=300, accept_gain=0.03, rung=0, steps_used=1, cap_steps=3,
                             traj=[400, 300], prior={"p_mean": 0.5})
        self.assertTrue(slow["why"].startswith("projected"))
        later = vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=1, steps_used=3, cap_steps=8,
                              traj=[101, 101], history=[{"var_us": 101.5, "orig_us": 100}])
        self.assertTrue(later["why"].startswith("plateau"))          # later rung: < 2% over the previous
        self.assertEqual(vp.checkpoint(orig_us=100, var_us=101, accept_gain=0.03, rung=0, steps_used=1,
                                       cap_steps=8, defer=False)["decision"], "rejected")

    def test_rungs_and_allocation(self):
        bc = vp.BudgetCfg(base_steps=1, max_steps=2, cap_steps=8, eta=2.0)
        self.assertEqual([vp.rung_steps(k, 1, bc) for k in range(4)], [1, 2, 4, 8])
        self.assertEqual(vp.rung_steps(3, 1, bc, used=3), 5)          # never past the cap
        al = vp.allocate([{"id": "a", "rank": 1, "rung": 0}, {"id": "b", "rank": 5, "rung": 1, "used": 1}],
                         vp.BudgetCfg(base_steps=1, max_steps=2, cap_steps=8, round_budget=2))
        self.assertEqual([(c["id"], c["steps"]) for c in al], [("b", 2), ("a", 0)])

    def test_survival(self):
        pend = [{"p": 0.8, "absorb": {"bn1", "relu1"}, "heads": {"conv1"}}]
        self.assertAlmostEqual(vp.survival(["relu1"], pend), 0.2)
        self.assertAlmostEqual(vp.survival(["conv1"], pend, head_weight=0.5), 0.6)
        self.assertEqual(vp.survival(["fc"], pend), 1.0)
        self.assertEqual(vp.survival_steps(4, 0.2), 1)
        self.assertEqual(vp.survival_steps(4, 0.6), 2)
        self.assertEqual(vp.survival_steps(4, 1.0), 4)

    def test_trajectory(self):
        r = {"baseline_e2e_us": 100, "steps": [{"e2e_after": 90, "status": "improved"},
                                                {"e2e_after": 95, "status": "regressed"},
                                                {"e2e_after": 80, "status": "improved"}]}
        self.assertEqual(vp.trajectory(r), [100, 90, 90, 80])


# ------------------------------------------------------------------ interleaved engine

class _Proc:
    def __init__(self, t_end):
        self.t_end = t_end

    def poll(self):
        return 0 if time.time() >= self.t_end else None

    def kill(self):
        self.t_end = 0


def _problem(root: Path, name: str, ops: list[str], ir="m") -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"name": name, "ir_name": ir, "ops": [{"name": n} for n in ops],
                                                 "slots": []}))
    return d


class InterleavedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gs = gsm.GraphSearch(_cfg(graph_search="interleaved"), self.root, [])

    def tearDown(self):
        self.tmp.cleanup()

    def _jobs(self):
        jobs = []
        for k, hid in enumerate(("fuse:A", "fuse:B")):
            o = _problem(self.root / "v", f"{hid[-1]}__orig", [])
            v = _problem(self.root / "v", f"{hid[-1]}__variant", [])
            jobs.append({"hint": {"id": hid}, "orig": o, "variant": v, "ev": {"ir": "m"}, "names": [],
                         "gs": {"rung": 0, "steps": 1, "used": 0}})
        return jobs

    def _engine(self, results, t_end, decisions, times, prog=None, measure=None):
        run_dir = self.root / "run"

        def read(d):
            name = json.loads((Path(d) / "manifest.json").read_text())["name"]
            t = results.get(name)
            return {"final_ok": True, "llm_calls": 1} if t is not None and time.time() >= t else None

        def finish(j, ro, rv, tm):
            decisions.append((j["hint"]["id"], time.time()))
            st = "accepted" if j["hint"]["id"] == "fuse:A" else "rejected"
            return {"hint": j["hint"]["id"], "status": st, "hint_status": st}
        return gsm.Interleaved(self.gs, run_dir=run_dir, prog=prog or _Prog(), start=lambda probs: _Proc(t_end),
                               read=read, measure=measure or (lambda j, ro, rv, attempts=None: times),
                               finish=finish, collect=lambda probs, proc: [{"problem": str(p)} for p in probs],
                               scope_after=lambda j: {"bn1", "relu1"}, poll_s=0.01)

    def test_decisions_mid_run_in_completion_order_and_stops(self):
        t0 = time.time()
        plan = [_problem(self.root / "p", "bn_r00", ["bn1"]), _problem(self.root / "p", "relu_r00", ["relu1"]),
                _problem(self.root / "p", "conv_r00", ["conv1", "relu1"]),
                _problem(self.root / "p", "other_ir", ["bn1"], ir="n")]
        results = {"B__orig": t0 + 0.05, "B__variant": t0 + 0.05, "A__orig": t0 + 0.15, "A__variant": t0 + 0.2}
        decisions = []
        eng = self._engine(results, t0 + 0.6, decisions, [[10.0], [9.0]])
        out = eng.run(plan, ["m", "m", "m", "n"], self._jobs())
        self.assertEqual([d[0] for d in decisions], ["fuse:B", "fuse:A"])        # completion order
        self.assertTrue(all(t < t0 + 0.6 for _, t in decisions))                  # before the run ended
        self.assertEqual(sorted(out["stopped"]), ["bn_r00", "relu_r00"])          # absorbed only, same IR only
        self.assertTrue((self.root / "run" / "bn_r00" / "mb_stop.json").exists())
        self.assertFalse((self.root / "run" / "conv_r00" / "mb_stop.json").exists())
        self.assertFalse((self.root / "run" / "other_ir" / "mb_stop.json").exists())
        self.assertEqual(len(out["plan_results"]), 4)
        self.assertEqual([e["event"] for e in self.gs.events], ["stop"])

    def test_union_of_accepted_hints_stops_a_shared_problem(self):
        t0 = time.time()
        plan = [_problem(self.root / "p", "bn_pair", ["bn2", "ds1"]), _problem(self.root / "p", "other", ["fc"])]
        results = {"A__orig": t0, "A__variant": t0, "B__orig": t0 + 0.1, "B__variant": t0 + 0.1}
        eng = self._engine(results, t0 + 0.4, [], [[1.0], [0.5]])
        eng.finish = lambda j, ro, rv, tm: {"hint": j["hint"]["id"], "status": "accepted", "hint_status": "accepted"}
        eng.scope_after = lambda j: {"bn2"} if j["hint"]["id"] == "fuse:A" else {"ds1"}
        out = eng.run(plan, ["m", "m"], self._jobs())
        self.assertEqual(out["stopped"], ["bn_pair"])                              # only once both are accepted
        stops = [e for e in self.gs.events if e["event"] == "stop"]
        self.assertEqual([e["hint"] for e in stops], ["fuse:B"])

    def test_absorbed_uses_measured_chains_on_a_partly_fused_graph(self):
        import modelblaster.pipeline.kb_graph_opt as go
        g = {"ops": [{"name": "conv1", "op": "conv2d_f16"}, {"name": "bn1+relu", "op": "fused__batchnorm2d_f16__relu_f16"},
                     {"name": "fc", "op": "linear_f16"}]}
        h = {"id": "fuse:fused__conv2d_f16__batchnorm2d_f16__relu_f16", "kind": "fuse", "status": "accepted",
             "fused_kind": "fused__conv2d_f16__batchnorm2d_f16__relu_f16", "tried_chains": [["conv1", "bn1", "relu"]]}
        doc = {"hints": [h]}
        old = go._scope_ops
        go._scope_ops = lambda *a, **k: []                                         # the kind's pattern: not found
        try:
            got = self.gs.absorbed({"hint": h, "ev": {"ir": "m"}}, doc, {"m": g}, {})
        finally:
            go._scope_ops = old
        self.assertEqual(got, {"conv1", "bn1", "relu"})

    def test_resume_does_not_decide_twice(self):
        t0 = time.time()
        prog = _Prog({"gs_inter": {"decided": {"fuse:A": {"hint": "fuse:A", "status": "accepted"}}}})
        decisions = []
        results = {k: t0 for k in ("A__orig", "A__variant", "B__orig", "B__variant")}
        out = self._engine(results, t0 + 0.05, decisions, [[1.0], [2.0]], prog=prog).run([], [], self._jobs())
        self.assertEqual([d[0] for d in decisions], ["fuse:B"])
        self.assertEqual([d["status"] for d in out["decisions"]], ["accepted", "rejected"])
        self.assertEqual(set(prog.get("gs_inter")["decided"]), {"fuse:A", "fuse:B"})

    def test_infra_failure_is_retimed_after_the_run(self):
        t0 = time.time()
        calls = []
        jobs = self._jobs()

        def measure(j, ro, rv, attempts=None):
            calls.append((j["hint"]["id"], attempts, time.time()))
            if attempts is None:
                j["remeasure_infra"] = True
                return None
            j["remeasure_infra"] = False
            return [[1.0], [2.0]]
        results = {k: t0 for k in ("A__orig", "A__variant", "B__orig", "B__variant")}
        decisions = []
        self._engine(results, t0 + 0.2, decisions, None, measure=measure).run([], [], jobs)
        retries = [c for c in calls if c[1] == 1]
        self.assertEqual(len(retries), 2)
        self.assertTrue(all(t >= t0 + 0.2 for _, _, t in retries))               # after the run finished
        self.assertEqual(len(decisions), 2)


class DecideCheckpointTest(unittest.TestCase):
    def test_promising_is_non_voting_and_keeps_the_rung(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved"), Path(t), [])
            doc = gh.new_doc()
            h = _fuse("fused__conv2d_f16__relu_f16")
            gh.merge(doc, [h])
            job = {"hint": h, "ev": {"ir": "m"}, "names": [], "gs": {"rung": 0, "steps": 1, "used": 0}}
            ro = {"final_ok": True, "llm_calls": 2, "llm_usage": {"prompt_tokens": 1000, "completion_tokens": 100}}
            rv = {"final_ok": True, "llm_calls": 2, "_dir": str(Path(t) / "v"),
                  "baseline_e2e_us": 200, "steps": [{"e2e_after": 101, "status": "improved"}]}
            std = []
            res = gsm.decide_checkpoint(gs, job, ro, rv, [[100.0], [101.0]], cfg=_cfg(), doc=doc,
                                        finish_std=lambda *a: std.append(a) or {"status": "x"})
            self.assertEqual(res["status"], "promising")
            self.assertEqual(std, [])
            rec = gh.get(doc, h["id"])
            self.assertEqual(rec["status"], "proposed")                      # no vote
            self.assertEqual(rec["gs_deferred"]["rung"], 0)
            self.assertEqual(rec["gs_deferred"]["steps_used"], 1)
            e = gs.events[-1]
            self.assertEqual((e["event"], e["status"], e["steps"], e["llm_calls"]), ("decision", "promising", 1, 4))
            self.assertIn("prior", e)
            self.assertGreater(e["usd"], 0)
            # the next rung: twice the steps; the seeds continue from the rung's best
            b = gs.hint_budget(rec)
            self.assertEqual((b["rung"], b["steps"], b["used"]), (1, 2, 1))
            # accepted at the next rung: the standard path decides, the deferred state is closed
            job2 = dict(job, hint=rec, gs=b)
            res2 = gsm.decide_checkpoint(gs, job2, ro, rv, [[100.0], [90.0]], cfg=_cfg(), doc=doc,
                                         finish_std=lambda *a: (gh.record_decision(doc, h["id"], {"decision": "accepted"}),
                                                                {"status": "accepted", "hint_status": "accepted"})[1])
            self.assertEqual(res2["status"], "accepted")
            self.assertIn("gs_deferred_done", gh.get(doc, h["id"]))

    def test_seed_override_from_the_last_rung(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "best"
            d.mkdir()
            (d / "fused__x.cu").write_text("// best of rung 0")
            (d / "other.cu").write_text("// not in the variant")
            h = {"gs_deferred": {"sources_dir": str(d), "rung": 0}}
            vplan = {"graph": {"ops": [{"op": "fused__x"}]}}
            seeds, tags = {}, {}
            import modelblaster.pipeline.kb_export as ke
            old = ke.slot_id_of
            ke.slot_id_of = lambda o, specs: o["op"]
            try:
                got = gsm.variant_seed_override(h, vplan, seeds, tags, {})
            finally:
                ke.slot_id_of = old
            self.assertEqual(got, ["fused__x"])
            self.assertEqual(seeds, {"fused__x": "// best of rung 0"})
            self.assertEqual(gsm.variant_seed_override({}, vplan, seeds, tags, {}), [])


# ------------------------------------------------------------------ pre-apply, survival budgets

class PreapplyTest(unittest.TestCase):
    def test_preapply_high_prior_only(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved", graph_explore=0.0), Path(t), [])
            gs.priors = gp.Priors(PART, tables=[{"part": PART, "counts": {"batchnorm2d+relu": [6, 0],
                                                                         "conv2d+relu": [4, 4]}}], use_a5=False)
            doc = gh.new_doc()
            gh.merge(doc, [_fuse("fused__batchnorm2d_f16__relu_f16", chains=[["bn1", "relu1"]]),
                           _fuse("fused__conv2d_f16__relu_f16", chains=[["conv1", "relu2"]])])
            done = gs.preapply(0, doc, {}, [], {})
            self.assertEqual(done, ["fuse:fused__batchnorm2d_f16__relu_f16"])
            h = gh.get(doc, done[0])
            self.assertEqual(h["status"], "accepted")
            self.assertEqual(h["evidence"][-1]["by"], "graph-prior")
            self.assertFalse(h["evidence"][-1]["measured"])
            self.assertEqual(gh.get(doc, "fuse:fused__conv2d_f16__relu_f16")["status"], "proposed")
            ev = [e for e in gs.events if e["event"] == "decision"]
            self.assertEqual((ev[0]["phase"], ev[0]["measured"], ev[0]["steps"]), ("preapply", False, 0))
            # the pre-applied decision is not evidence for the prior (only measured ones are)
            self.assertEqual(gp.Priors(PART, local=gs.log_path, use_a5=False)
                             .prior("fused__batchnorm2d_f16__relu_f16")["source"], "seed")

    def test_exploration_leaves_it_to_measurement(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved", graph_explore=1.0), Path(t), [])
            gs.priors = gp.Priors(PART, tables=[{"part": PART, "counts": {"batchnorm2d+relu": [6, 0]}}], use_a5=False)
            doc = gh.new_doc()
            gh.merge(doc, [_fuse("fused__batchnorm2d_f16__relu_f16")])
            self.assertEqual(gs.preapply(0, doc, {}, [], {}), [])
            self.assertEqual(doc["hints"][0]["status"], "proposed")
            self.assertEqual(gs.events[-1]["event"], "explore")

    def test_rounds_mode_does_not_preapply(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_priors="rank"), Path(t), [])
            doc = gh.new_doc()
            gh.merge(doc, [_fuse("fused__batchnorm2d_f16__relu_f16")])
            self.assertEqual(gs.preapply(0, doc, {}, [], {}), [])

    def test_survival_budgets(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved"), root, [])
            gs.priors = gp.Priors(PART, tables=[{"part": PART, "counts": {"conv2d+batchnorm2d": [9, 1]}}],
                                  use_a5=False)
            doc = gh.new_doc()
            gh.merge(doc, [_fuse("fused__conv2d_f16__batchnorm2d_f16", chains=[["conv1", "bn1"]])])
            dirs = [_problem(root, "bn", ["bn1"]), _problem(root, "conv", ["conv1"]), _problem(root, "fc", ["fc"])]
            plan = {"problems": [{"ir": "m", "op_names": ["bn1"]}, {"ir": "m", "op_names": ["conv1"]},
                                 {"ir": "m", "op_names": ["fc"]}]}
            rep = gs.plan_budgets(dirs, plan, doc, {"m": {}}, {}, lambda h, g, s: ["conv1+bn1"] if h else [],
                                  steps=4, stop_patience=8)
            self.assertEqual(rep["bn"]["steps"], 1)                         # p_accept ~0.83: absorbed
            self.assertEqual(rep["conv"]["steps"], 3)                       # head: weight 0.25 (replay-tuned)
            self.assertEqual(rep["fc"]["steps"], 4)
            man = lambda n: json.loads((root / n / "manifest.json").read_text())   # noqa: E731
            self.assertEqual(man("bn")["mb_budget"], {"steps": 1, "stop_patience": 8})
            self.assertNotIn("mb_budget", man("fc"))                         # untouched: same problem hash
            self.assertFalse(gsm.set_budget(root / "bn", {"steps": 1, "stop_patience": 8}))   # idempotent


# ------------------------------------------------------------------ beam

class BeamTest(unittest.TestCase):
    def _gs(self, t, n=5):
        gs = gsm.GraphSearch(_cfg(graph_diverse_beam=n), Path(t), [])
        gs.priors = gp.Priors(PART, tables=[{"part": PART, "counts": {"batchnorm2d+relu": [6, 0],
                                                                     "conv2d+batchnorm2d": [9, 5]}}], use_a5=False)
        return gs

    def test_members(self):
        with tempfile.TemporaryDirectory() as t:
            doc = gh.new_doc()
            gh.merge(doc, [_fuse("fused__batchnorm2d_f16__relu_f16"), _fuse("fused__conv2d_f16__batchnorm2d_f16"),
                           {"id": "layout:fold_strided_gemm", "kind": "fold_layout", "status": "proposed",
                            "source": [{}], "evidence": []}])
            ms = gsm.beam_members(doc, self._gs(t), 5)
            self.assertEqual([m["origin"] for m in ms], ["baseline", "high-prior", "uncertain", "layout-fold"])
            self.assertEqual(ms[1]["hints"], ["fuse:fused__batchnorm2d_f16__relu_f16"])
            self.assertEqual(len(ms[2]["hints"]), 2)
            self.assertEqual(len(gsm.beam_members(doc, self._gs(t), 2)), 2)
            md = gsm.member_doc(doc, ms[1])
            self.assertEqual(gh.get(md, ms[1]["hints"][0])["status"], "accepted")
            self.assertEqual(gh.get(doc, ms[1]["hints"][0])["status"], "proposed")     # the run's doc untouched

    def test_one_plan_item_per_signature_across_members(self):
        from modelblaster.pipeline.kb_plan import make_plan
        op = lambda n, k, sh: {"name": n, "op": k, "entry": k, "signature_sha": "s" + k, "shape": sh, "us": 10.0,  # noqa
                               "seed": "seed"}
        base = {"ir": "m", "ir_dir": "/a", "partition_us": 30, "ops": [op("conv1", "conv", {"n": 1}),
                                                                        op("bn1", "bn", {"n": 1}),
                                                                        op("fc", "linear", {"n": 2})]}
        fused = {"ir": "m", "ir_dir": "/b", "partition_us": 20, "ops": [op("conv1+bn1", "fused__conv__bn", {"n": 1}),
                                                                         op("fc", "linear", {"n": 2})]}
        prof = [gsm.relabel(base, "m0"), gsm.relabel(fused, "m1")]
        plan = make_plan(prof, coverage=2.0, top_k=None, per_item=True, min_share=0.0)
        kinds = sorted(it["op"] for it in plan["items"] if it["selected"])
        self.assertEqual(kinds, ["bn", "conv", "fused__conv__bn", "linear"])  # linear once: shared by both members
        lin = next(it for it in plan["items"] if it["op"] == "linear")
        self.assertEqual(len(lin["occurrences"]), 2)
        self.assertEqual(len(plan["problems"]), 4)

    def test_halving_and_commit(self):
        self.assertEqual(gsm.halve({"m0": 10.0, "m1": 8.0, "m2": None, "m3": 9.0}, ["m0", "m1", "m2", "m3"]),
                         ["m1", "m3"])
        self.assertEqual(gsm.halve({"m1": 8.0, "m3": 7.0}, ["m1", "m3"]), ["m3"])
        doc = gh.new_doc()
        gh.merge(doc, [_fuse("fused__a_f16__relu_f16"), _fuse("fused__b_f16__relu_f16")])
        ms = [{"id": "m0", "hints": []}, {"id": "m1", "hints": ["fuse:fused__a_f16__relu_f16"]},
              {"id": "m2", "hints": ["fuse:fused__a_f16__relu_f16", "fuse:fused__b_f16__relu_f16"]}]
        lost = gsm.beam_commit(doc, ms[1], ms, {"beam_times_us": {}}, 1)
        self.assertEqual(lost, ["fuse:fused__b_f16__relu_f16"])
        self.assertEqual([h["status"] for h in doc["hints"]], ["accepted", "rejected"])


# ------------------------------------------------------------------ flags, round key, winners

class FlagsTest(unittest.TestCase):
    def test_defaults_are_inert_and_keep_the_round_key(self):
        from modelblaster.pipeline.kb_graph_opt import Config
        with tempfile.TemporaryDirectory() as t:
            cfg = Config(kb=Path("."))
            gs = gsm.GraphSearch(cfg, Path(t), [])
            self.assertFalse(gs.active())
            self.assertEqual(gs.round_key(), {})
            key = kb_resume.round_key(cfg, [])
            k2 = dict(key, **gs.round_key())
            self.assertEqual(kb_resume.key_hash(key), kb_resume.key_hash(k2))
            self.assertEqual((gs.priors_mode, gs.char_steps, gs.survival_on, gs.inter), ("off", 0, False, False))
            self.assertEqual(gs.plan_input([1]), [1])
            self.assertEqual(gs.decide_kinds(), ())
            self.assertFalse(gs.blocks_fixpoint(0, gh.new_doc()))
            spec = {"x": 1}
            self.assertIs(gs.ingest_specs(spec, {}), spec)
            self.assertEqual(gs.winners(Path(t)), Path(t) / "winners" / "winners.json")
            self.assertFalse((Path(t) / "graph_decisions.jsonl").exists())

    def test_flags_change_the_key_and_resolve(self):
        from modelblaster.pipeline.kb_graph_opt import Config
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved"), Path(t), [])
            self.assertTrue(gs.active())
            k = gs.round_key()["graph_search"]
            self.assertEqual((k["mode"], k["priors"], k["characterize"][0], k["survival"][0]),
                             ("interleaved", "preapply", 1, True))
            gs2 = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="off",
                                         graph_characterize=0, graph_survival="off"), Path(t), [])
            self.assertNotEqual(gs2.round_key(), gs.round_key())
            with self.assertRaises(ValueError):
                gsm.GraphSearch(Config(kb=Path("."), graph_search="beam"), Path(t), [])

    def test_cli_exposes_the_flags(self):
        import subprocess
        out = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.kb_graph_opt", "--help"],
                             capture_output=True, text=True, cwd=str(ROOT.parent))
        for f in ("--graph-search", "--graph-priors", "--graph-diverse-beam", "--graph-characterize",
                  "--graph-survival", "--graph-rung-steps", "--graph-layout-fold"):
            self.assertIn(f, out.stdout)

    def test_filter_winners(self):
        with tempfile.TemporaryDirectory() as t:
            w = Path(t) / "winners.json"
            w.write_text(json.dumps({"format": "kb-winners/1", "winners": [
                {"op": "a", "problem_id": "m__fuse_x__variant"}, {"op": "b", "problem_id": "m__fuse_x__orig"},
                {"op": "c", "problem_id": "m__conv_0__r00"}]}))
            f = gsm.filter_winners(w, Path(t) / "w2.json")
            self.assertEqual([r["op"] for r in json.loads(f.read_text())["winners"]], ["b", "c"])

    def test_exec_mode_recorded_and_mismatch_flagged(self):
        import os
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved", graph_concurrency="auto"), Path(t), [])
            gs.start_round(0)
            old = os.environ.get("MB_GRAPH_CONCURRENCY")
            os.environ["MB_GRAPH_CONCURRENCY"] = "on"              # auto starts on ...
            try:
                gs.event("decision", hint="h", status="accepted", phase="rung0", exec_mode=gs.exec_mode("partition"))
                e = gs.events[-1]["exec_mode"]
                self.assertEqual((e["concurrency"], e["sm_budget_policy"], e["graph_replay"], e["scope"]),
                                 ("on", "split", "cuda_graph", "partition"))
                os.environ["MB_GRAPH_CONCURRENCY"] = "off"         # ... and the round's comparison switched it off
                mm = gs.mode_mismatches()
                self.assertEqual(mm[0]["diff"]["concurrency"], ["on", "off"])
                self.assertIn("exec_mode_mismatch", gs.summary())
                gs.event("decision", hint="h2", status="accepted", phase="beam", exec_mode=gs.exec_mode("whole"))
                self.assertEqual(len(gs.mode_mismatches()), 1)          # same mode, whole graph: not flagged
            finally:
                if old is None:
                    os.environ.pop("MB_GRAPH_CONCURRENCY", None)
                else:
                    os.environ["MB_GRAPH_CONCURRENCY"] = old

    def test_event_log(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(_cfg(graph_search="interleaved"), Path(t), [])
            gs.start_round(2)
            gs.event("decision", hint="h", status="accepted", phase="rung0", steps=1, measured=True)
            rows = [json.loads(x) for x in (Path(t) / "graph_decisions.jsonl").read_text().splitlines()]
            self.assertEqual((rows[0]["round"], rows[0]["hint"]), (2, "h"))
            self.assertIn("t_run_s", rows[0])
            s = gs.round_summary(2)
            self.assertEqual(s["by_status"], {"accepted": 1})


if __name__ == "__main__":
    unittest.main()
