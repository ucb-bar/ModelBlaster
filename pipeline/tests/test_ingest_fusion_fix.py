"""Fused-slot winners are ingested by default (X-137): ingest_specs carries the fused kinds of the round's graphs
without graph search; kill switch MB_INGEST_FUSED=0 restores the old specs."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_search as gsm  # noqa: E402
from modelblaster.pipeline.kb_graph_opt import Config  # noqa: E402
from modelblaster.pipeline.kb_proposals import fuse_pair, spec_from_kind  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS as K  # noqa: E402
from modelblaster.pipeline.tests.test_graph_hints import ewise_graph  # noqa: E402


class IngestFusionFix(unittest.TestCase):
    def _gs(self, t):
        return gsm.GraphSearch(Config(kb=Path(".")), Path(t), [])

    def test_default_adds_fused_kinds(self):
        g, spec = fuse_pair(ewise_graph(), "s", "r", K)
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MB_INGEST_FUSED", None)
            gs = self._gs(t)
            self.assertFalse(gs.active())
            out = gs.ingest_specs(dict(K), {"x": g})
            self.assertIn(spec.op, out)
            self.assertNotIn(spec.op, K)
            same = dict(K)
            self.assertIs(gs.ingest_specs(same, {}), same)                      # nothing to add: identity

    def test_kill_switch_flag(self):
        g, spec = fuse_pair(ewise_graph(), "s", "r", K)
        with tempfile.TemporaryDirectory() as t:
            gs = gsm.GraphSearch(Config(kb=Path("."), ingest_fused="off"), Path(t), [])
            specs = dict(K)
            self.assertIs(gs.ingest_specs(specs, {"x": g}), specs)

    def test_kill_switch(self):
        g, spec = fuse_pair(ewise_graph(), "s", "r", K)
        with tempfile.TemporaryDirectory() as t, mock.patch.dict(os.environ, {"MB_INGEST_FUSED": "0"}):
            specs = dict(K)
            self.assertIs(self._gs(t).ingest_specs(specs, {"x": g}), specs)


if __name__ == "__main__":
    unittest.main()
