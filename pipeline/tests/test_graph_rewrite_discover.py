"""Discovery mode (graph_rewrite_discover.py): stub tool loop + fake paired evaluator end to end, the proposals file, the CLI gate. CPU only."""
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

from modelblaster.pipeline import graph_rewrite as grm  # noqa: E402
from modelblaster.pipeline import graph_rewrite_discover as gd  # noqa: E402
from modelblaster.pipeline import graph_verify as gv  # noqa: E402
from modelblaster.pipeline.tests import merge_fixtures as MF  # noqa: E402
from modelblaster.pipeline.tests.test_graph_rewrite_sandbox import fire_script  # noqa: E402

BAD_IMPORT = "import os\ndef rewrite(g, W, P, T):\n    pass\n"


def renamed(src: str) -> str:
    """the same rewrite with other op / key names (same pattern, a different script)"""
    return src.replace('"mrg"', '"mm"').replace('"mrg_act"', '"mm_act"').replace('"mrg_o"', '"mm_o"').replace("d.m.", "d.q.")


class DiscoverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="mb_disc_"))
        cls.ir = MF.branchy_ir(cls.tmp / "fire", MF.FIRE, "fp16")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_d(self, name, scripts, **kw):
        out = self.tmp / name
        s = gd.discover([self.ir], gd.ScriptedToolLoop(scripts), out, **kw)
        return s, json.loads((out / "rewrite_proposals.json").read_text()), json.loads((out / "rewrite_attempts.json").read_text())["attempts"]

    def test_end_to_end_with_the_stub_and_a_fake_evaluator(self):
        scripts = [gd.ScriptProposal("good", fire_script(), "merge the fire expand convs", provenance={"session": "s1"}),
                   gd.ScriptProposal("imports", BAD_IMPORT),
                   gd.ScriptProposal("wrong", fire_script(swap=True), "stacks the filters in the wrong order"),
                   gd.ScriptProposal("good_again", renamed(fire_script()), "same pattern, other names")]
        ev = gd.FakeEvaluator(100.0, 80.0)
        s, doc, att = self.run_d("e2e", scripts, evaluator=ev, max_candidates=4)
        by = {a["script"]: a for a in att}
        self.assertEqual(by["good"]["decision"], "accepted")
        self.assertEqual((by["good"]["stage"], by["good"]["timing"]["original_us"], by["good"]["timing"]["variant_us"]), ("timing", 100.0, 80.0))
        self.assertAlmostEqual(by["good"]["gain"], 0.2)
        self.assertEqual((by["imports"]["decision"], by["imports"]["stage"], by["imports"]["error"]["code"]), ("rejected", "sandbox", "forbidden_construct"))
        self.assertEqual((by["wrong"]["decision"], by["wrong"]["stage"]), ("rejected", "graph_verify"))            # the gate caught it
        self.assertTrue(by["wrong"]["reasons"])
        self.assertEqual(len(ev.calls), 2)                                                                     # only verified candidates are timed
        self.assertEqual((s["attempts"], s["accepted"], s["rejected"]), (4, 2, 2))
        self.assertEqual(s["by_stage"], {"graph_verify": 1, "sandbox": 1, "timing": 2})
        # the proposals file: one pattern, two accepted scripts as evidence
        self.assertEqual(doc["format"], gd.PROPOSALS_FORMAT)
        self.assertEqual(len(doc["proposals"]), 1)
        p = doc["proposals"][0]
        self.assertEqual((p["source"], p["status"], p["suggested_kind"], p["already_implemented"]),
                         ("agent-rewrite", "proposed", "merge_sibling_convs_cat", "merge_sibling_convs"))
        self.assertEqual((p["pattern"]["ops_before"], p["pattern"]["ops_after"]), (5, 2))
        self.assertEqual([o["op"] for o in p["pattern"]["before"]], ["conv2d_f16", "relu_f16", "conv2d_f16", "relu_f16", "cat2_c1_f16"])
        self.assertEqual([o["op"] for o in p["pattern"]["after"]], ["conv2d_f16", "relu_f16"])
        self.assertEqual(p["pattern"]["before"][0]["inputs"], p["pattern"]["before"][2]["inputs"])                  # sibling convs read one input
        self.assertEqual(p["pattern"]["before"][4]["outputs"], ["y0"])
        self.assertEqual(p["pattern"]["after"][1]["outputs"], ["y0"])                                             # the concat's output name is kept
        sk = p["detector_sketch"]
        self.assertEqual(sk["anchor_op"], "cat2_c1_f16")
        self.assertIn("KH", next(c for c in sk["shape_constraints"] if c["op"] == "conv2d_f16")["differing_fields"])
        self.assertIn("IC", next(c for c in sk["shape_constraints"] if c["op"] == "conv2d_f16")["equal_fields"])
        r = p["recipe"]
        self.assertIn("def rewrite", r["script"])
        self.assertEqual(sorted(r["derivations"]), ["d.m.b", "d.m.e1pad", "d.m.w"])
        self.assertTrue(r["exact"])
        self.assertEqual(r["pad_formulas"][0]["formula"], {"axis2": "(KH_max - KH[c0]) // 2", "axis3": "(KW_max - KW[c0]) // 2"})
        self.assertEqual(r["leaf_roles"]["c0.weight"], "weight of the removed conv2d_f16 op 'c0'")
        self.assertEqual(len(p["evidence"]), 2)
        e = p["evidence"][0]
        self.assertEqual((e["problem"], e["original_us"], e["variant_us"], e["decision"], e["timed"]), ("branchy", 100.0, 80.0, "accepted", True))
        self.assertTrue(e["verify"]["ok"] and e["verify"]["checks"]["sampled"])
        self.assertEqual(e["provenance"], {"session": "s1"})
        self.assertIn("note", p)
        # nothing was applied: the IR is untouched, and no hints file is written
        self.assertFalse((self.tmp / "e2e" / "graph_hints.json").exists())
        self.assertEqual(len(json.loads((self.ir / "graph.json").read_text())["ops"]), 9)

    def test_a_rewrite_that_does_not_win_is_rejected_with_evidence(self):
        s, doc, att = self.run_d("lose", [gd.ScriptProposal("good", fire_script())], evaluator=gd.FakeEvaluator(100.0, 99.0))
        self.assertEqual(doc["proposals"], [])
        self.assertEqual((att[0]["decision"], att[0]["stage"]), ("rejected", "timing"))
        self.assertIn("accept_gain", att[0]["why"])
        self.assertTrue(att[0]["verify"]["ok"])

    def test_untimed_candidates_need_the_flag(self):
        s, doc, att = self.run_d("untimed", [gd.ScriptProposal("good", fire_script())])
        self.assertEqual((doc["proposals"], att[0]["decision"], att[0]["stage"]), ([], "not_accepted", "verified_untimed"))
        s, doc, att = self.run_d("untimed2", [gd.ScriptProposal("good", fire_script())], allow_untimed=True)
        self.assertEqual(len(doc["proposals"]), 1)
        self.assertFalse(doc["proposals"][0]["evidence"][0]["timed"])
        self.assertIsNone(doc["proposals"][0]["evidence"][0]["gain"])

    def test_proposals_accumulate_across_runs(self):
        for _ in range(2):
            gd.discover([self.ir], gd.ScriptedToolLoop([gd.ScriptProposal("good", fire_script())]), self.tmp / "acc", evaluator=gd.FakeEvaluator())
        doc = json.loads((self.tmp / "acc" / "rewrite_proposals.json").read_text())
        self.assertEqual((len(doc["proposals"]), len(doc["proposals"][0]["evidence"])), (1, 2))

    def test_a_proposer_that_crashes_loses_only_its_candidates_and_notimplemented_propagates(self):
        class Boom(gd.ToolLoop):
            def run(self, session):
                raise RuntimeError("relay down")

        class Todo(gd.ToolLoop):
            def run(self, session):
                raise NotImplementedError("the LLM loop is not built")
        s = gd.discover([self.ir], Boom(), self.tmp / "boom")
        self.assertEqual(s["attempts"], 1)
        att = json.loads((self.tmp / "boom" / "rewrite_attempts.json").read_text())["attempts"]
        self.assertEqual((att[0]["stage"], att[0]["decision"]), ("proposer", "error"))
        with self.assertRaises(NotImplementedError):
            gd.discover([self.ir], Todo(), self.tmp / "todo")
        with self.assertRaises(TypeError):
            gd.ToolLoop()

    def test_a_tool_loop_repairs_against_try_rewrite(self):
        """the seam an LLM loop uses: try, read the structured error, fix, return the script that works"""
        class Repair(gd.ToolLoop):
            name = "repair"
            seen = []

            def run(self, session):
                ov = session.show()
                self.seen.append(ov["n_ops"])
                bad = session.try_rewrite(BAD_IMPORT)
                self.seen.append(bad["error"]["code"])
                wrong = session.try_rewrite(fire_script(swap=True))      # sandbox ok and structurally valid: only the numerical gate sees it
                self.seen.append(wrong["ok"])
                good = session.try_rewrite(fire_script())
                self.seen.append((good["ok"], good["structural"]["ok"], sorted(good["derivations"])))
                return [gd.ScriptProposal("fixed", fire_script(), "after one repair", provenance={"tries": len(session.tries)})]
        loop = Repair()
        s = gd.discover([self.ir], loop, self.tmp / "repair", evaluator=gd.FakeEvaluator())
        self.assertEqual(loop.seen[:3], [9, "forbidden_construct", True])
        self.assertEqual(loop.seen[3], (True, True, ["d.m.b", "d.m.e1pad", "d.m.w"]))
        self.assertEqual(s["accepted"], 1)
        doc = json.loads((self.tmp / "repair" / "rewrite_proposals.json").read_text())
        self.assertEqual(doc["proposals"][0]["evidence"][0]["provenance"], {"tries": 3})
        self.assertEqual(doc["proposals"][0]["proposers"], ["repair"])

    def test_tool_loop_serves_the_graph_rewrite_proposer_interface(self):
        g = json.loads((self.ir / "graph.json").read_text())
        prop = gd.ToolLoopProposer(gd.ScriptedToolLoop([gd.ScriptProposal("good", fire_script(), "merge", est_gain_pct=9.0), gd.ScriptProposal("bad", BAD_IMPORT)]))
        cands = prop.propose({"name": "branchy", "dir": str(self.ir), "source_ir": str(self.ir), "graph": g}, {"ops": []},
                             {"work": str(self.tmp / "tlp"), "max_candidates": 3})
        self.assertEqual(len(cands), 1)                                                         # the rejected script never becomes a candidate
        c = cands[0]
        self.assertIsInstance(c, grm.RewriteCandidate)
        self.assertEqual((c.edit_type, c.est_gain_pct, c.est_cost["ops_after"]), ("rewire", 9.0, 6))
        v = gv.verify(self.ir, c.edited_ir_dir, tensor_map=c.tensor_map, seeds=(0,))
        self.assertTrue(v.ok, v.summary())
        with self.assertRaises(NotImplementedError):
            gd.ToolLoopProposer().propose({"name": "x", "dir": str(self.ir), "graph": g}, {}, {"work": str(self.tmp / "tlp2")})

    def test_signature_is_stable_across_names_and_differs_across_patterns(self):
        a = self.run_d("sig_a", [gd.ScriptProposal("g", fire_script())], allow_untimed=True)[1]["proposals"][0]["id"]
        b = self.run_d("sig_b", [gd.ScriptProposal("g", renamed(fire_script()))], allow_untimed=True)[1]["proposals"][0]["id"]
        self.assertEqual(a, b)
        other = '''
def rewrite(g, W, P, T):
    # drop the relu after the 1x1 branch: NOT equivalent, so it must not verify; but its signature is another pattern
    r = g.op("a0")
    g.remove_ops(["a0"])
    g.rewire(r["outputs"][0], r["inputs"][0])
'''
        s, doc, att = self.run_d("sig_c", [gd.ScriptProposal("o", other)], allow_untimed=True)
        self.assertEqual(att[0]["stage"], "graph_verify")                                           # relu removed: caught

    def test_cli_is_gated_and_runs_end_to_end(self):
        sd = self.tmp / "scripts"
        sd.mkdir()
        (sd / "fire_merge.py").write_text("# description: merge the expand convs\n" + fire_script())
        (sd / "bad.py").write_text(BAD_IMPORT)
        out = self.tmp / "cli"
        cmd = [sys.executable, "-m", "modelblaster.pipeline.graph_rewrite_discover", str(self.ir), "--out", str(out), "--proposer", f"scripted:{sd}",
               "--evaluator", "fake:100,70"]
        env = {k: v for k, v in os.environ.items() if k != gd.ENV}
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)                                       # off by default
        self.assertIn("--enable-discovery", r.stderr)
        self.assertFalse((out / "rewrite_proposals.json").exists())
        r = subprocess.run(cmd + ["--enable-discovery"], capture_output=True, text=True, timeout=300, env=env)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        s = json.loads(r.stdout)
        self.assertEqual((s["attempts"], s["accepted"]), (2, 1))
        doc = json.loads((out / "rewrite_proposals.json").read_text())
        self.assertEqual(doc["proposals"][0]["pattern"]["description"], "merge the expand convs")
        self.assertAlmostEqual(doc["proposals"][0]["evidence"][0]["gain"], 0.3)
        r = subprocess.run(cmd[:-2] + ["--evaluator", "none"], capture_output=True, text=True, timeout=300, env={**env, gd.ENV: "on"})
        self.assertEqual(r.returncode, 0, r.stderr)                                                  # the env switch works too
        r = subprocess.run(cmd[:-4] + ["--proposer", "llm", "--enable-discovery"], capture_output=True, text=True, timeout=300, env=env)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("tool loop", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
