"""Global knowledge read path of kb_graph_opt (kb_global.py): off changes nothing; read seeds a fresh
knowledge dir once and records it; readwrite against a local store materializes and merges back
(through KernelBlaster's knowledge/v2/merge.py CLI, skipped without its dependencies)."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent))

from modelblaster.pipeline import kb_global, kb_resume  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config  # noqa: E402

KB = Path(os.environ.get("MB_KB_ROOT", ROOT / "third_party" / "KernelBlaster"))


def fake_seed(d: Path, snapshot="v0007", part="cuda:sm_89|fp16") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(d / "knowledge.db")
    c.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    c.execute("INSERT INTO meta VALUES ('gk_seed', ?)", (json.dumps({"snapshot": snapshot, "part": part,
                                                                       "run_uid": "u1", "counts": {"claims": 3}}),))
    c.commit()
    c.close()
    (d / "impl" / "code").mkdir(parents=True)
    (d / "impl" / "index.jsonl").write_text('{"id": "x"}\n')
    return d


def ir(d: Path, dtype="f16") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "graph.json").write_text(json.dumps({"tensors": {"x": {"dtype": dtype}, "w": {"dtype": "f32"}}}))
    return d


class ReadPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = self.tmp / "work"
        self.work.mkdir()

    def test_off_is_a_no_op_and_keeps_the_round_key(self):
        cfg = Config(kb=KB, arch="89")
        state = {"sent": [], "rounds": []}
        self.assertIsNone(kb_global.apply_seed(self.work, cfg, [ir(self.tmp / "ir")], state))
        self.assertNotIn("knowledge_global", state)
        self.assertFalse((self.work / "knowledge").exists())
        self.assertNotIn("knowledge_global", kb_resume.round_key(cfg, [self.tmp / "ir"]))
        self.assertEqual(cfg.knowledge_global, "off")

    def test_read_seeds_a_fresh_dir_once(self):
        seed = fake_seed(self.tmp / "seed")
        cfg = Config(kb=KB, arch="89", knowledge_global="read", knowledge_global_seed=str(seed))
        state = {"sent": [], "rounds": []}
        rec = kb_global.apply_seed(self.work, cfg, [ir(self.tmp / "ir")], state)
        self.assertTrue(rec["applied"])
        self.assertEqual((rec["snapshot"], rec["part"], rec["run_uid"]), ("v0007", "cuda:sm_89|fp16", "u1"))
        self.assertTrue((self.work / "knowledge" / "knowledge.db").exists())
        self.assertEqual(rec["impl_records"], 1)
        self.assertTrue((self.work / "gk_seed" / "knowledge.db").exists())      # the merge base
        # a resumed call (or a newer seed) changes nothing: the work dir keeps what it started from
        cfg2 = Config(kb=KB, arch="89", knowledge_global="read",
                      knowledge_global_seed=str(fake_seed(self.tmp / "seed2", snapshot="v0009")))
        self.assertEqual(kb_global.apply_seed(self.work, cfg2, [ir(self.tmp / "ir")], state)["snapshot"], "v0007")

    def test_existing_knowledge_is_never_overwritten(self):
        (self.work / "knowledge").mkdir()
        (self.work / "knowledge" / "knowledge.db").write_bytes(b"old")
        cfg = Config(kb=KB, arch="89", knowledge_global="read", knowledge_global_seed=str(fake_seed(self.tmp / "s")))
        rec = kb_global.apply_seed(self.work, cfg, [ir(self.tmp / "ir")], {})
        self.assertFalse(rec["applied"])
        self.assertEqual((self.work / "knowledge" / "knowledge.db").read_bytes(), b"old")

    def test_partition_from_arch_and_ir_dtype(self):
        self.assertEqual(kb_global.part_for(Config(kb=KB, arch="89"), [ir(self.tmp / "a")]), "cuda:sm_89|fp16")
        self.assertEqual(kb_global.part_for(Config(kb=KB, arch="100"), [ir(self.tmp / "b", "f32")]), "cuda:sm_100|fp32")
        self.assertEqual(kb_global.part_for(Config(kb=KB, arch="89", knowledge_global_part="x|y"), []), "x|y")

    def test_bad_mode_and_missing_seed(self):
        with self.assertRaises(SystemExit):
            kb_global.apply_seed(self.work, Config(kb=KB, knowledge_global="sometimes"), [], {})
        with self.assertRaises(SystemExit):
            kb_global.apply_seed(self.work, Config(kb=KB, knowledge_global="read"), [], {})


def _kb_ok() -> bool:
    try:
        p = subprocess.run([sys.executable, "-c", "import kernelblaster.knowledge.v2.merge"], capture_output=True,
                           env=dict(os.environ, PYTHONPATH=str(KB / "src")), timeout=60)
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_kb_ok(), "KernelBlaster knowledge v2 not importable")
class ReadWriteStoreTest(unittest.TestCase):
    def test_materialize_from_store_and_merge_back(self):
        tmp = Path(tempfile.mkdtemp())
        work = tmp / "w"
        work.mkdir()
        cfg = Config(kb=KB, arch="89", knowledge_global="readwrite", knowledge_global_store=str(tmp / "store"))
        state = {}
        rec = kb_global.apply_seed(work, cfg, [ir(tmp / "ir")], state)
        self.assertTrue(rec["applied"])
        self.assertIsNone(rec["snapshot"])                     # empty store: the repo seed
        out = kb_global.merge_back(work, cfg, rec)            # nothing learned: still one (empty) merge
        self.assertEqual(out.get("status"), "merged", out)
        self.assertEqual(kb_global.merge_back(work, cfg, rec)["status"], "duplicate")


if __name__ == "__main__":
    unittest.main()
