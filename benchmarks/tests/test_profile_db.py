"""Unit tests for benchmarks/profile_db.py.

Builds a temporary results tree with synthetic profile_firesim.csv + run.json,
then runs ingest/query/coverage and asserts the expected records appear.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Import the module under test via sys.path so we don't depend on the
# repo being installed.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks import profile_db


PROFILE_CSV_DRONET = """dispatch_id,name,op,shape,cycles
0,conv0,conv2d_s8,N=1;IC=3;IH=8;IW=8;OC=4,12345
1,bn0,batchnorm2d_s8,N=1;C=4;H=4;W=4,2222
2,relu0,relu_s8,n=64,500
"""

PROFILE_CSV_DRONET_REP2 = """dispatch_id,name,op,shape,cycles
0,conv0,conv2d_s8,N=1;IC=3;IH=8;IW=8;OC=4,12500
1,bn0,batchnorm2d_s8,N=1;C=4;H=4;W=4,2200
2,relu0,relu_s8,n=64,510
"""


def _make_run(results_a: Path, cell: str, run_id: str,
              csv_body: str, model: str, target: str, quant: str) -> Path:
    rd = results_a / cell / run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "profile_firesim.csv").write_text(csv_body)
    (rd / "run.json").write_text(json.dumps({
        "schema_version": 1,
        "arm": "A",
        "workload_id": cell,
        "run_id": run_id,
        "git_sha": "deadbeef",
        "started_at": "2026-05-29T00:00:00+00:00",
        "ended_at": "2026-05-29T00:01:00+00:00",
        "wall_clock_s": 60.0,
        "peak_rss_mb": 100.0,
        "exit_status": "ok",
        "model": model,
        "target": target,
        "quant": quant,
        "runner": "firesim",
    }))
    return rd


class IngestQueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results = self.tmp / "results"
        self.db = self.tmp / "profile_db"
        self.results_a = self.results / "A"
        # Two reps of dronet_gemmini_int8 with slightly different cycles
        _make_run(self.results_a, "dronet_gemmini_int8", "2026-05-29T00-00-00Z",
                  PROFILE_CSV_DRONET, "dronet", "gemmini", "int8")
        _make_run(self.results_a, "dronet_gemmini_int8", "2026-05-29T00-30-00Z",
                  PROFILE_CSV_DRONET_REP2, "dronet", "gemmini", "int8")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_ingest_creates_jsonl_file(self):
        n = profile_db.ingest(self.results, self.db)
        self.assertEqual(n, 6)  # 3 dispatches × 2 reps
        jsonl = self.db / "dronet__gemmini__int8.jsonl"
        self.assertTrue(jsonl.exists())
        lines = jsonl.read_text().strip().split("\n")
        self.assertEqual(len(lines), 6)

    def test_ingest_is_idempotent(self):
        first = profile_db.ingest(self.results, self.db)
        second = profile_db.ingest(self.results, self.db)
        self.assertEqual(first, 6)
        self.assertEqual(second, 0)  # nothing new

    def test_query_median_across_reps(self):
        profile_db.ingest(self.results, self.db)
        cycles = profile_db.query("dronet", "gemmini", "int8",
                                  agg="median", db_root=self.db)
        # median of (12345, 12500), (2222, 2200), (500, 510)
        self.assertEqual(cycles[0], 12422)  # median rounds via int(median())
        self.assertEqual(cycles[1], 2211)
        self.assertEqual(cycles[2], 505)

    def test_query_min_max(self):
        profile_db.ingest(self.results, self.db)
        cmin = profile_db.query("dronet", "gemmini", "int8",
                                agg="min", db_root=self.db)
        cmax = profile_db.query("dronet", "gemmini", "int8",
                                agg="max", db_root=self.db)
        self.assertEqual(cmin[0], 12345)
        self.assertEqual(cmax[0], 12500)

    def test_query_op_type_filter(self):
        profile_db.ingest(self.results, self.db)
        only_convs = profile_db.query("dronet", "gemmini", "int8",
                                      op_type="conv2d_s8", db_root=self.db)
        self.assertEqual(set(only_convs.keys()), {0})

    def test_hetero_runs_are_skipped(self):
        # Hetero runs use target="hetero_*" and shouldn't add records to single-backend files.
        _make_run(self.results_a, "dronet_hetero_int8", "2026-05-29T01-00-00Z",
                  PROFILE_CSV_DRONET, "dronet", "hetero_gemmini_opu", "int8")
        profile_db.ingest(self.results, self.db)
        jsonls = list(self.db.glob("*.jsonl"))
        # only one file: dronet__gemmini__int8.jsonl (no hetero file)
        self.assertEqual([p.name for p in jsonls], ["dronet__gemmini__int8.jsonl"])

    def test_coverage_lists_present_and_missing(self):
        profile_db.ingest(self.results, self.db)
        rep = profile_db.coverage_report(self.db)
        present_keys = {(r["network"], r["target"], r["quant"], r["op_type"]) for r in rep["present"]}
        self.assertIn(("dronet", "gemmini", "int8", "conv2d_s8"), present_keys)
        # mlp_control × * × int8 are MISSING in EXPECTED_MATRIX
        missing_keys = {(m["network"], m["target"], m["quant"]) for m in rep["missing"]}
        self.assertIn(("mlp_control", "gemmini", "int8"), missing_keys)


if __name__ == "__main__":
    unittest.main()
