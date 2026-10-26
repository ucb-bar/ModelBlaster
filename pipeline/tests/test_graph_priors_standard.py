"""--graph-priors standard: class table only (graph_priors_standard.json), no run-derived numbers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_priors as gp  # noqa: E402
from modelblaster.pipeline import graph_search as gsm  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config  # noqa: E402


def fuse(kind):
    return {"id": "fuse:" + kind, "kind": "fuse", "fused_kind": kind}


class StandardPriors(unittest.TestCase):
    def test_table_has_no_counts(self):
        t = json.loads(gp.STANDARD.read_text())
        self.assertNotIn("counts", t)
        self.assertEqual(set(t["classes"]), {"high", "medium", "unknown"})
        self.assertTrue(all(v in t["classes"] for v in t["patterns"].values()))

    def test_classes(self):
        p = gp.Priors.standard("cuda:sm_89|fp16")
        hi = p.prior(fuse("fused__batchnorm2d_f16__relu_f16"))
        self.assertEqual((hi["class"], hi["source"], hi["n"]), ("high", "standard", 0))
        self.assertEqual(p.prior(fuse("fused__conv2d_f16__batchnorm2d_f16__relu_f16"))["class"], "medium")
        self.assertEqual(p.prior(fuse("fused__conv2d_f16__maxpool2d_f16"))["class"], "unknown")
        self.assertEqual(p.prior({"id": "fuse_attention:x", "kind": "fuse_attention"})["class"], "unknown")

    def test_only_high_is_preapplied_and_sweep_table_unused(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), graph_search="interleaved", graph_priors="standard"), Path(t), [])
            self.assertEqual(gs.priors_mode, "preapply")
            self.assertEqual(gs.round_key()["graph_search"]["priors_table"], "standard")
            self.assertTrue(gs.eligible(gs.prior(fuse("fused__batchnorm2d_f16__add_f16__relu_f16"))))
            self.assertFalse(gs.eligible(gs.prior(fuse("fused__conv2d_f16__batchnorm2d_f16"))))
            self.assertEqual(gs.priors.tables, [])
            gs.prior(fuse("fused__conv2d_f16__batchnorm2d_f16"))
            self.assertEqual(gs.priors.sources, {"standard"})

    def test_default_auto_still_uses_the_sweep_table(self):
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), knowledge_global_part="cuda:sm_89|fp16", graph_search="interleaved"), Path(t), [])   # the table is the sm_89 fp16 sweep
            self.assertTrue(gs.priors.tables)


if __name__ == "__main__":
    unittest.main()
