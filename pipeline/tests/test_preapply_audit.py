"""Unapply audit of pre-applied fusions (preapply_audit / graph_search flags), GPU-, torch- and LLM-free.

- class filtering: ``--graph-preapply-classes`` high / all / none / auto (class table and sweep priors; conv / GEMM heads
  never pre-applied under ``high``);
- audit decision logic: revert / keep / tie, the margin boundary, several IRs;
- the audit with an injected build + timer on a real (pure python) graph: alternative graph = incumbent with the hint
  unfused (equal to the deployed revert), keep / revert / tie, several hints (independent verdicts, greedy combination,
  combination that does not hold), absorbed hints, failures (timer, export, alternative without a passing run) never
  change the incumbent;
- learning: hint evidence, ``graph_decisions.jsonl`` decision (measured, phase ``preapply_audit``), ``state.json`` row;
- flags off: byte-identical round key, no file, no event; round key entries only for non-default flags.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modelblaster.pipeline import graph_hints as gh  # noqa: E402
from modelblaster.pipeline import graph_priors as gp  # noqa: E402
from modelblaster.pipeline import graph_search as gsm  # noqa: E402
from modelblaster.pipeline import preapply_audit as pa  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config  # noqa: E402
from test_graph_hints import conv_block  # noqa: E402

PART = "cuda:sm_89|fp16"


def fuse(kind):
    return {"id": "fuse:" + kind, "kind": "fuse", "fused_kind": kind}


def chain_graph() -> dict:
    """x -> sigmoid s1 -> relu r1 -> relu r2 -> sigmoid s2 -> z (two independent elementwise chains)."""
    t = {k: {"shape": [8]} for k in ("x", "a", "b", "c", "z")}
    ops = [{"name": n, "op": op, "inputs": [i], "outputs": [o], "shape": {"n": 8}, "dispatch_id": k,
            "depends_on": [k - 1] if k else []}
           for k, (n, op, i, o) in enumerate([("s1", "sigmoid", "x", "a"), ("r1", "relu", "a", "b"),
                                              ("r2", "relu", "b", "c"), ("s2", "sigmoid", "c", "z")])]
    return {"name": "m", "input": {"tensor": "x"}, "output": {"tensor": "z"}, "tensors": t, "ops": ops,
            "dispatches": list(range(4))}


H1, H2 = "fuse:fused__sigmoid__relu", "fuse:fused__relu__sigmoid"


def pre_applied(kind, chains, status="accepted", audited=None):
    h = gh.fuse_hint(kind, chains, {"by": "test"}, status)
    h["evidence"] = [{"decision": "accepted", "by": "graph-prior", "measured": False, "round": 0}]
    if audited:
        h["evidence"].append({"decision": audited, "by": pa.AUDIT_BY, "measured": True})
    return h


def two_hint_doc():
    return {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]]),
                                           pre_applied("fused__relu__sigmoid", [["r2", "s2"]])]}


def cfg(**kw):
    base = dict(arch="89", seed=0, steps=4, variant_steps=2, accept_gain=0.03, min_gain_pct=2.0, max_variants=3,
                remeasure=5, plan_parallel=0, stop_patience=8, knowledge_global_part=PART)
    base.update(gsm.CFG_DEFAULTS)
    base.update(kw)
    return SimpleNamespace(**base)


def fake_compose(ir_dir, graph, specs, cache, work, arch):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    (work / "names.json").write_text(json.dumps([o["name"] for o in graph["ops"]]))
    return work


def fake_timer(cost, reps=5):
    """cost(frozenset of op names) -> us (None: the problem has no passing run)."""
    calls = []

    def timer(problems, work):
        out = []
        for p in problems:
            names = frozenset(json.loads((Path(p) / "names.json").read_text()))
            c = cost(names)
            out.append([] if c is None else [c * (1 + 0.001 * k) for k in range(reps)])
        calls.append([sorted(json.loads((Path(p) / "names.json").read_text())) for p in problems])
        return out, {"device": {"label": "0"}}
    timer.calls = calls
    return timer


def run(doc, cost, tmp, margin=0.03, timer=None, compose=fake_compose, ids=None, graph=None):
    ir = Path(tmp) / "ir"
    ir.mkdir(exist_ok=True)
    (ir / "graph.json").write_text(json.dumps(graph or chain_graph()))
    timer = timer or fake_timer(cost)
    rep = pa.run_audit(irs=[ir], doc=doc, ids=ids or pa.pending_ids(doc), specs={}, cache=None, cfg=cfg(),
                       work=Path(tmp) / "w", margin=margin, timer=timer, compose=compose, router=None)
    return rep, timer


def costs(fused1, fused2, un1=50.0, un2=50.0, other=0.0, both_penalty=0.0):
    """Cost model: a fused chain costs ``fusedN``, its two stages together ``unN``; ``both_penalty`` extra when both are
    unfused (an interaction)."""
    def cost(names):
        c = other
        c += fused1 if "s1+r1" in names else un1
        c += fused2 if "r2+s2" in names else un2
        if "s1" in names and "r2" in names and "s2" in names:
            c += both_penalty
        return c
    return cost


# ------------------------------------------------------------------ class filtering

def gs_for(classes="auto", priors="standard", **kw):
    t = tempfile.mkdtemp()
    return gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors=priors,
                                  graph_preapply_classes=classes, **kw), Path(t), [])


class ClassFilter(unittest.TestCase):
    def test_standard_auto_is_high(self):
        gs = gs_for()
        self.assertEqual(gs.pre_classes, "high")
        for k, ok in (("fused__batchnorm2d_f16__relu_f16", True), ("fused__batchnorm2d_f16__add_f16__a__relu_f16", True),
                      ("fused__conv2d_f16__batchnorm2d_f16", False), ("fused__conv2d_f16__batchnorm2d_f16__relu_f16", False),
                      ("fused__linear_f16__relu_f16", False), ("fused__conv2d_f16__maxpool2d_f16", False)):
            self.assertEqual(gs.eligible(gs.prior(fuse(k)), fuse(k)), ok, k)

    def test_all_restores_a4_with_the_class_table(self):
        gs = gs_for("all")
        self.assertEqual(gs.pre_classes, "all")
        self.assertTrue(gs.eligible(gs.prior(fuse("fused__conv2d_f16__batchnorm2d_f16")), fuse("fused__conv2d_f16__batchnorm2d_f16")))
        self.assertFalse(gs.eligible(gs.prior(fuse("fused__conv2d_f16__maxpool2d_f16"))))     # unknown stays measured

    def test_none_disables_preapply(self):
        gs = gs_for("none", graph_explore=0.0)
        self.assertFalse(gs.eligible(gs.prior(fuse("fused__batchnorm2d_f16__relu_f16"))))
        doc = gh.new_doc()
        gh.merge(doc, [gh.fuse_hint("fused__batchnorm2d_f16__relu_f16", [["bn1", "relu1"]], {"by": "t"})])
        self.assertEqual(gs.preapply(0, doc, {}, [], {}), [])
        self.assertEqual(doc["hints"][0]["status"], "proposed")
        self.assertEqual(gs.audit_mode, "off")

    def test_heavy_head_guard_beats_a_table_that_says_high(self):
        with tempfile.TemporaryDirectory() as t:
            tab = json.loads(gp.STANDARD.read_text())
            tab["patterns"]["conv+bn"] = "high"
            f = Path(t) / "tab.json"
            f.write_text(json.dumps(tab))
            gs = gs_for("high")
            gs.priors = gp.Priors.standard(PART, table=f)
            h = fuse("fused__conv2d_f16__batchnorm2d_f16")
            self.assertEqual(gs.prior(h)["class"], "high")
            self.assertFalse(gs.eligible(gs.prior(h), h))

    def test_sweep_priors_auto_is_a4_and_high_restricts(self):
        counts = {"conv2d+batchnorm2d": [6, 0], "batchnorm2d+relu": [6, 0]}
        tab = {"part": PART, "counts": counts}
        for classes, conv_ok in (("auto", True), ("all", True), ("high", False)):
            gs = gs_for(classes, priors="preapply")
            gs.priors = gp.Priors(PART, tables=[tab], use_a5=False)
            self.assertEqual(gs.pre_classes, "all" if classes == "auto" else classes)
            c, b = fuse("fused__conv2d_f16__batchnorm2d_f16"), fuse("fused__batchnorm2d_f16__relu_f16")
            self.assertEqual(gs.eligible(gs.prior(c), c), conv_ok, classes)
            self.assertEqual(gs.eligible(gs.prior(c)), conv_ok, classes)             # by the prior's key alone
            self.assertTrue(gs.eligible(gs.prior(b), b), classes)

    def test_preapply_skips_conv_head_with_standard_high(self):
        gs = gs_for(graph_explore=0.0)
        doc = gh.new_doc()
        gh.merge(doc, [gh.fuse_hint("fused__conv2d_f16__batchnorm2d_f16", [["c", "b"]], {"by": "t"}),
                       gh.fuse_hint("fused__batchnorm2d_f16__relu_f16", [["b", "r"]], {"by": "t"})])
        done = gs.preapply(0, doc, {}, [], {})
        self.assertEqual(done, ["fuse:fused__batchnorm2d_f16__relu_f16"])
        self.assertEqual(gh.get(doc, "fuse:fused__conv2d_f16__batchnorm2d_f16")["status"], "proposed")

    def test_preapply_all_forces_conv_bn(self):
        gs = gs_for("all", graph_explore=0.0)
        doc = gh.new_doc()
        gh.merge(doc, [gh.fuse_hint("fused__conv2d_f16__batchnorm2d_f16", [["c", "b"]], {"by": "t"})])
        self.assertEqual(gs.preapply(0, doc, {}, [], {}), ["fuse:fused__conv2d_f16__batchnorm2d_f16"])

    def test_bad_flag_values(self):
        with self.assertRaises(ValueError):
            gs_for("some")
        with self.assertRaises(ValueError):
            gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_preapply_audit="x"), Path("."), [])


# ------------------------------------------------------------------ decision rule

class Decide(unittest.TestCase):
    def test_revert_keep_tie(self):
        self.assertEqual(pa.decide(100, 90, 0.03)["verdict"], "revert")
        d = pa.decide(100, 99, 0.03)
        self.assertEqual((d["verdict"], d["reason"]), ("keep", "tie"))
        d = pa.decide(100, 120, 0.03)
        self.assertEqual((d["verdict"], d["reason"]), ("keep", "fusion_faster"))
        self.assertEqual(pa.decide(100, 100, 0.03)["reason"], "tie")
        self.assertEqual(pa.decide(None, 90, 0.03)["verdict"], "error")

    def test_margin_boundary_is_the_flows_rule(self):
        self.assertEqual(pa.decide(100, 97, 0.03)["verdict"], "revert")            # gain == margin: like gain >= accept_gain
        self.assertEqual(pa.decide(100, 97.01, 0.03)["verdict"], "keep")
        self.assertEqual(pa.decide(100, 100, 0.0)["verdict"], "keep")              # a tie never reverts, even at margin 0

    def test_several_irs(self):
        self.assertEqual(pa.decide_multi([(100, 80), (100, 98)], 0.03)["verdict"], "revert")
        d = pa.decide_multi([(100, 60), (100, 120)], 0.03)                          # summed gain 10%, one IR 20% worse
        self.assertEqual((d["verdict"], d["reason"]), ("keep", "slower_in_an_ir"))


# ------------------------------------------------------------------ the audit

class Audit(unittest.TestCase):
    def test_pending_ids(self):
        doc = two_hint_doc()
        doc["hints"].append(pre_applied("fused__sigmoid__sigmoid", None, status="proposed"))      # not accepted
        doc["hints"].append(pre_applied("fused__relu__relu", None, audited="accepted"))          # already audited
        m = gh.fuse_hint("fused__x", None, {"by": "t"}, "accepted")
        m["evidence"] = [{"decision": "accepted", "by": "kb", "measured": True}]
        doc["hints"].append(m)                                                                   # measured, not pre-applied
        self.assertEqual(pa.pending_ids(doc), [H1, H2])

    def test_alternative_graph_is_the_unfused_incumbent_and_matches_the_deployed_revert(self):
        g0, doc = chain_graph(), two_hint_doc()
        inc, added, _ = gh.apply_hints(g0, doc["hints"], {})
        self.assertEqual([o["name"] for o in inc["ops"]], ["s1+r1", "r2+s2"])
        fused = [o["name"] for o in inc["ops"] if o.get("hint") == H1]
        alt, prov = pa.unfuse(inc, fused, dict(added), H1)
        self.assertEqual([o["name"] for o in alt["ops"]], ["s1", "r1", "r2+s2"])
        self.assertTrue(pa.deployed_match(g0, doc, H1, alt, dict(added)))
        self.assertIsNot(pa.deployed_match(g0, doc, H2, alt, dict(added)), True)       # the other hint's revert differs

    def test_keep_when_the_fusion_is_faster(self):
        with tempfile.TemporaryDirectory() as t:
            doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
            rep, timer = run(doc, costs(40, 0, un2=0), t)
            hr = rep["hints"][H1]
            self.assertEqual((hr["verdict"], hr["reason"]), ("keep", "fusion_faster"))
            self.assertEqual(rep["reverted"], [])
            self.assertEqual(len(timer.calls), 1)                    # incumbent + the alternative in ONE paired call
            self.assertEqual(timer.calls[0], [["r2", "s1+r1", "s2"], ["r1", "r2", "s1", "s2"]])

    def test_revert_when_the_alternative_is_faster_by_the_margin(self):
        with tempfile.TemporaryDirectory() as t:
            doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
            rep, _ = run(doc, costs(80, 0, un2=0, un1=50), t)
            hr = rep["hints"][H1]
            self.assertEqual(hr["verdict"], "revert")
            self.assertAlmostEqual(hr["gain_revert"], (80 - 50) / 80, places=2)
            self.assertEqual(rep["reverted"], [H1])
            x = hr["irs"]["ir"]
            self.assertEqual((x["status"], x["fused_ops"], x["stage_ops"]), ("timed", ["s1+r1"], ["s1", "r1"]))
            self.assertEqual(len(x["inc_runs"]), 5)

    def test_tie_keeps(self):
        with tempfile.TemporaryDirectory() as t:
            doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
            rep, _ = run(doc, costs(100, 0, un2=0, un1=99), t)                      # alternative 1% faster < 3%
            self.assertEqual((rep["hints"][H1]["verdict"], rep["hints"][H1]["reason"]), ("keep", "tie"))
            rep, _ = run(doc, costs(100, 0, un2=0, un1=90), t, margin=0.15)         # margin is a parameter
            self.assertEqual(rep["hints"][H1]["verdict"], "keep")

    def test_several_hints_independent_verdicts(self):
        with tempfile.TemporaryDirectory() as t:
            rep, timer = run(two_hint_doc(), costs(80, 30, un1=50, un2=50), t)       # H1 slower fused, H2 faster fused
            self.assertEqual((rep["hints"][H1]["verdict"], rep["hints"][H2]["verdict"]), ("revert", "keep"))
            self.assertEqual(rep["reverted"], [H1])
            self.assertEqual(rep["kept"], [H2])
            self.assertIsNone(rep["combined"])                                        # a single revert: nothing to combine
            self.assertEqual(len(timer.calls), 1)                                     # 1 incumbent + 2 alternatives, one paired call
            self.assertEqual(len(timer.calls[0]), 3)

    def test_greedy_combination_is_retimed_once_and_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            rep, timer = run(two_hint_doc(), costs(80, 90, un1=50, un2=50), t)
            self.assertEqual(sorted(rep["reverted"]), sorted([H1, H2]))
            self.assertEqual(rep["combined"]["verdict"], "revert")
            self.assertEqual(len(timer.calls), 2)                                     # + one combined re-timing
            self.assertEqual(timer.calls[1][1], ["r1", "r2", "s1", "s2"])             # both unfused in the combined graph
            self.assertEqual(len(timer.calls[1]), 2)

    def test_combination_that_does_not_hold_keeps_only_the_best_single(self):
        with tempfile.TemporaryDirectory() as t:
            rep, _ = run(two_hint_doc(), costs(80, 90, un1=50, un2=50, both_penalty=500), t)
            self.assertEqual(rep["combined"]["verdict"], "keep")
            self.assertEqual(rep["reverted"], [H2])                                   # H2 has the larger gain (90 -> 50)
            self.assertEqual((rep["hints"][H1]["verdict"], rep["hints"][H1]["reason"]), ("keep", "combined_interaction"))

    def test_absorbed_hint_is_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
            # a longer hint absorbed the chain: its op carries the other hint id
            big = gh.fuse_hint("fused__sigmoid__relu__relu", [["s1", "r1", "r2"]], {"by": "test"}, "accepted")
            big["evidence"] = [{"decision": "accepted", "by": "kb", "measured": True}]
            doc["hints"].append(big)
            rep, timer = run(doc, costs(1, 1), t, ids=[H1])
            self.assertEqual(rep["hints"][H1]["verdict"], "skip")
            self.assertEqual(rep["hints"][H1]["irs"]["ir"]["status"], "not_applicable")
            self.assertEqual(timer.calls, [])

    def test_failures_keep_the_incumbent(self):
        doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
        with tempfile.TemporaryDirectory() as t:                  # the alternative has no passing run
            rep, _ = run(doc, lambda n: None if "s1" in n else 50.0, t)
            self.assertEqual((rep["hints"][H1]["verdict"], rep["hints"][H1]["reason"]), ("keep", "alt_failed"))
            self.assertEqual(rep["reverted"], [])
        with tempfile.TemporaryDirectory() as t:                  # the incumbent has none
            rep, _ = run(doc, lambda n: None if "s1+r1" in n else 50.0, t)
            self.assertEqual(rep["hints"][H1]["verdict"], "error")
            self.assertIn("incumbent", json.dumps(rep["ir_errors"]))
        with tempfile.TemporaryDirectory() as t:                  # the timer itself raises
            def boom(problems, work):
                raise RuntimeError("ssh: connect to host failed")
            rep, _ = run(doc, None, t, timer=boom)
            self.assertEqual(rep["hints"][H1]["verdict"], "error")
            self.assertEqual(rep["reverted"], [])
        with tempfile.TemporaryDirectory() as t:                  # exporting the alternative fails
            def compose(ir_dir, graph, *a):
                if "s1" in [o["name"] for o in graph["ops"]]:
                    raise ValueError("cannot export")
                return fake_compose(ir_dir, graph, *a)
            rep, _ = run(doc, costs(1, 1), t, compose=compose)
            self.assertEqual(rep["hints"][H1]["verdict"], "error")
            self.assertIn("export failed", rep["hints"][H1]["why"])


# ------------------------------------------------------------------ hook: doc, events, state, learning

class Hook(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp()
        self.work = Path(self.t) / "work"
        self.work.mkdir()
        self.gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="standard"), self.work, [])
        self.ir = Path(self.t) / "ir"
        self.ir.mkdir()
        (self.ir / "graph.json").write_text(json.dumps(chain_graph()))
        self.hp = self.work / "graph_hints.json"
        self.state = {"rounds": [{"round": 0, "decisions": []}]}
        self.sp = self.work / "state.json"
        (self.work / "r00").mkdir()

    def hook(self, doc, cost, **kw):
        return pa.audit_hook(self.gs, "final", irs=[self.ir], doc=doc, hints_path=self.hp, cache=None, specs={},
                             cfg=cfg(), leases=None, recorded=[], state=self.state, state_path=self.sp,
                             compose=fake_compose, timer=fake_timer(cost), router=None, **kw)

    def test_revert_is_recorded_everywhere(self):
        doc = two_hint_doc()
        out = self.hook(doc, costs(80, 30, un1=50, un2=50))
        self.assertEqual(out["reverted"], [H1])
        h1, h2 = gh.get(doc, H1), gh.get(doc, H2)
        self.assertEqual((h1["status"], h2["status"]), ("rejected", "accepted"))
        ev = h1["evidence"][-1]
        self.assertEqual((ev["by"], ev["decision"], ev["measured"], ev["phase"]), (pa.AUDIT_BY, "rejected", True, "final"))
        self.assertGreater(ev["original_us"], 0)                                 # original = unfused, variant = fused
        self.assertGreater(ev["variant_us"], ev["original_us"])                  # the fused (incumbent) one is slower
        self.assertLess(ev["gain"], 0)                                           # fused is the slower one: negative fusion gain
        self.assertEqual(gh.load(self.hp)["hints"][0]["status"], "rejected")     # doc saved
        # graph_decisions.jsonl
        evs = [json.loads(x) for x in self.gs.log_path.read_text().splitlines()]
        dec = {e["hint"]: e for e in evs if e["event"] == "decision"}
        self.assertEqual((dec[H1]["phase"], dec[H1]["status"], dec[H1]["measured"]), ("preapply_audit", "rejected", True))
        self.assertEqual((dec[H2]["status"], dec[H2]["measured"]), ("accepted", True))
        self.assertEqual(dec[H1]["exec_mode"]["scope"], "whole")
        self.assertEqual([e["event"] for e in evs if e["event"].startswith("audit")], ["audit_start", "audit_end"])
        # state.json: rows like a measured hint decision (what knowledge.v2.priors.rows_from_state reads)
        st = json.loads(self.sp.read_text())
        rows = {d["hint"]: d for d in st["rounds"][-1]["decisions"]}
        self.assertEqual((rows[H1]["status"], rows[H1]["hint_status"], rows[H1]["phase"]), ("rejected", "rejected", "preapply_audit"))
        self.assertEqual(rows[H2]["status"], "accepted")
        self.assertIn("original_us", rows[H1])
        # summary
        self.assertEqual(self.gs.summary()["preapply_audit"][0]["reverted"], [H1])
        # the work dir's measured decisions are evidence for the structural statistics
        p = gp.Priors(PART, local=self.gs.log_path, use_a5=False, min_n=1)
        pr = p.prior(fuse("fused__sigmoid__relu"))
        self.assertEqual((pr["accepted"], pr["rejected"]), (0, 1))

    def test_rows_are_read_by_the_knowledge_struct_priors(self):
        kb = ROOT / "third_party" / "KernelBlaster" / "src"
        sys.path.insert(0, str(kb))
        try:
            from kernelblaster.knowledge.v2 import priors as kp
        except Exception as e:  # noqa: BLE001 -- the KernelBlaster checkout / its deps are optional here
            self.skipTest(f"knowledge.v2.priors not importable: {e}")
        doc = two_hint_doc()
        self.hook(doc, costs(80, 30, un1=50, un2=50))
        rows = kp.rows_from_state(self.state, part=PART, problem="m", run_uid="u")
        by = {r["hint"]: r for r in rows}
        self.assertFalse(by[H1]["accepted"])
        self.assertTrue(by[H2]["accepted"])

    def test_kept_is_accepted_evidence_and_idempotent(self):
        doc = {"format": gh.FORMAT, "hints": [pre_applied("fused__sigmoid__relu", [["s1", "r1"]])]}
        out = self.hook(doc, costs(40, 0, un2=0))
        self.assertEqual((out["kept"], out["reverted"]), ([H1], []))
        self.assertEqual(gh.get(doc, H1)["status"], "accepted")
        self.assertEqual(gh.get(doc, H1)["evidence"][-1]["decision"], "accepted")
        self.assertIsNone(self.hook(doc, costs(40, 0, un2=0)))                    # audited: not again (resume safe)

    def test_error_is_recorded_and_never_raises(self):
        doc = two_hint_doc()
        def boom(problems, work):
            raise RuntimeError("no GPU")
        out = pa.audit_hook(self.gs, "final", irs=[self.ir], doc=doc, hints_path=self.hp, cache=None, specs={}, cfg=cfg(),
                            leases=None, recorded=[], state=self.state, state_path=self.sp, compose=fake_compose,
                            timer=boom, router=None)
        self.assertEqual(out["reverted"], [])
        self.assertEqual([gh.get(doc, h)["status"] for h in (H1, H2)], ["accepted", "accepted"])
        evs = [json.loads(x) for x in self.gs.log_path.read_text().splitlines()]
        self.assertTrue([e for e in evs if e["event"] == "audit_error"])
        self.assertEqual(self.state["rounds"][-1]["decisions"], [])
        # the GraphSearch wrapper turns an exception of the module itself into audit_error, too
        orig = pa.audit_hook
        try:
            pa.audit_hook = lambda *a, **k: (_ for _ in ()).throw(KeyError("x"))
            r = self.gs.audit("final", irs=[self.ir], doc=doc, hints_path=self.hp, cache=None, specs={})
        finally:
            pa.audit_hook = orig
        self.assertIn("KeyError", r["audit_error"])
        self.assertEqual(gh.get(doc, H1)["status"], "accepted")

    def test_remeasure_zero_cannot_time(self):
        doc = two_hint_doc()
        out = pa.audit_hook(self.gs, "final", irs=[self.ir], doc=doc, hints_path=self.hp, cache=None, specs={},
                            cfg=cfg(remeasure=0), leases=None, recorded=[], state=None, state_path=None)
        self.assertIn("remeasure 0", out["audit_error"])
        self.assertEqual(gh.get(doc, H1)["status"], "accepted")

    def test_no_pending_hint_is_a_noop(self):
        doc = {"format": gh.FORMAT, "hints": []}
        self.assertIsNone(self.hook(doc, costs(1, 1)))
        self.assertFalse((self.work / pa.DIRNAME).exists())
        self.assertFalse(self.gs.log_path.exists())


# ------------------------------------------------------------------ record at pre-apply time

class Record(unittest.TestCase):
    def test_preapply_records_stage_ops_and_the_pre_fusion_graph(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="standard",
                                        graph_explore=0.0), Path(t), [])
            doc = gh.new_doc()
            gh.merge(doc, [gh.fuse_hint("fused__batchnorm2d__relu", [["bn0", "relu0"]], {"by": "t", "ir": "blk"})])
            g = conv_block(1, reuse_relu=False)
            done = gs.preapply(0, doc, {"blk": g}, [], {})
            self.assertEqual(done, ["fuse:fused__batchnorm2d__relu"])
            rec = pa.load_records(Path(t))[done[0]]
            self.assertEqual((rec["round"], rec["class"], rec["chains"]), (0, "high", [["bn0", "relu0"]]))
            self.assertEqual([s["name"] for s in rec["stage_ops"]["blk"]], ["bn0", "relu0"])
            self.assertEqual([s["op"] for s in rec["stage_ops"]["blk"]], ["batchnorm2d", "relu"])
            self.assertEqual(json.loads(Path(rec["pre_graph"]["blk"]).read_text())["ops"][1]["name"], "bn0")
            self.assertEqual(rec["pre_graph"].keys(), {"blk"})
            self.assertEqual(pa.pending_ids(doc), ["fuse:fused__batchnorm2d__relu"])   # the doc alone finds it

    def test_a_failing_record_never_fails_the_preapply(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="standard",
                                        graph_explore=0.0), Path(t), [])
            doc = gh.new_doc()
            gh.merge(doc, [gh.fuse_hint("fused__batchnorm2d__relu", [["bn0", "relu0"]], {"by": "t"})])
            orig = pa.record_preapplied
            try:
                pa.record_preapplied = lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
                done = gs.preapply(0, doc, {}, [], {})
            finally:
                pa.record_preapplied = orig
            self.assertEqual(done, ["fuse:fused__batchnorm2d__relu"])
            self.assertTrue([e for e in gs.events if e["event"] == "audit_error"])


# ------------------------------------------------------------------ flags: off = identical, key only when non-default

class Flags(unittest.TestCase):
    def key(self, **kw):
        with tempfile.TemporaryDirectory() as t:
            return gsm.GraphSearch(Config(kb=Path("."), **kw), Path(t), []).round_key()

    def test_defaults_add_nothing(self):
        self.assertEqual(self.key(), {})                                          # not interleaved: no key at all
        k = self.key(graph_search="interleaved")["graph_search"]                  # the pre-audit key, byte for byte
        self.assertEqual(sorted(k), ["beam", "characterize", "explore", "layout_fold", "mode", "prior_conf", "prior_min_gain",
                                     "prior_min_n", "priors", "rungs", "survival"])
        ks = self.key(graph_search="interleaved", graph_priors="standard")["graph_search"]
        self.assertEqual(ks["priors_table"], "standard")
        self.assertFalse([x for x in ks if x.startswith("preapply")])

    def test_only_non_default_flags_add_entries(self):
        base = self.key(graph_search="interleaved", graph_priors="standard")["graph_search"]
        k = self.key(graph_search="interleaved", graph_priors="standard", graph_preapply_classes="all")["graph_search"]
        self.assertEqual({x: k[x] for x in k if x not in base}, {"preapply_classes": "all"})
        for kw in ({"graph_preapply_audit": "final"}, {"graph_preapply_audit": "off"}, {"graph_preapply_margin": 0.05},
                   {"graph_preapply_route": "off"}):                              # end-of-run knobs: no round-key change
            kk = self.key(graph_search="interleaved", graph_priors="standard", **kw)["graph_search"]
            self.assertEqual(kk, base, kw)
        k = self.key(graph_search="interleaved", graph_priors="standard", graph_preapply_audit="rounds",
                     graph_preapply_margin=0.05)["graph_search"]
        self.assertEqual(k["preapply_audit"], ["rounds", 0.05, True])

    def test_resolution(self):
        with tempfile.TemporaryDirectory() as t:
            r = lambda **kw: gsm.GraphSearch(Config(kb=Path("."), **kw), Path(t), [])      # noqa: E731
            d = r()
            self.assertEqual((d.audit_mode, d.pre_classes, d.priors_mode), ("off", "all", "off"))
            self.assertIsNone(d.audit("final", irs=[], doc={"hints": []}, hints_path=Path(t) / "h.json", cache=None, specs={}))
            i = r(graph_search="interleaved")
            self.assertEqual((i.audit_mode, i.pre_classes, i.audit_margin), ("final", "all", 0.03))
            self.assertEqual(r(graph_search="interleaved", graph_preapply_audit="off").audit_mode, "off")
            self.assertEqual(r(graph_search="interleaved", graph_preapply_margin=0.07).audit_margin, 0.07)
            self.assertEqual(r(graph_search="interleaved", graph_priors="rank").audit_mode, "off")
            self.assertEqual(r(graph_priors="preapply").audit_mode, "final")

    def test_round_hook_is_off_unless_rounds(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="standard"), Path(t), [])
            self.assertIsNone(gs.audit("round", irs=[], doc={"hints": []}, hints_path=Path(t) / "h.json", cache=None, specs={}))
            self.assertFalse((Path(t) / pa.DIRNAME).exists())

    def test_cli_flags_exist(self):
        import argparse
        from dataclasses import asdict
        names = set(asdict(Config(kb=Path("."))))
        for k in ("graph_preapply_classes", "graph_preapply_audit", "graph_preapply_margin", "graph_preapply_route"):
            self.assertIn(k, names)
            self.assertIn(k, gsm.CFG_DEFAULTS)
            self.assertEqual(getattr(Config(kb=Path(".")), k), gsm.CFG_DEFAULTS[k])
        del argparse


if __name__ == "__main__":
    unittest.main()
