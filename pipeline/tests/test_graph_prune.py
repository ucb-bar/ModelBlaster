"""Dead-op pruning at export (graph_prune, A2 item 1).

Pure-python: liveness on synthetic graphs and on every Level 3 sweep IR kept locally (l3_farm results, skipped when
absent): exactly the known problems have dead ops, all of them pure data movement whose outputs never reach a graph
output. With torch: a tiny bidirectional GRU and a tiny MLP extracted, exported with pruning on and off -- the MLP's
problem files are byte-identical, the GRU's differ only by the dead h_n ops.
"""
from __future__ import annotations

import filecmp
import glob
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from modelblaster.pipeline import graph_prune as gp  # noqa: E402

FARM = Path(__file__).resolve().parents[3].parent / "l3_farm" / "results"
if not FARM.is_dir():
    FARM = Path.home() / "Documents" / "ModelBlaster-clone" / "l3_farm" / "results"

# problems of the Level 3 sweep with dead ops (final IRs), and their dead op names
EXPECTED_DEAD = {
    "35_LSTM": ["lstm#12", "lstm#13"],
    "36_LSTMHn": ["lstm#13"],
    "37_LSTMCn": ["lstm#12"],
    "38_LSTMBidirectional": ["lstm#30", "lstm#31", "lstm#32", "lstm#33", "lstm#34", "lstm#35"],
    "39_GRU": ["gru#8"],
    "40_GRUHidden": ["gru#7"],
    "41_GRUBidirectional": ["gru#20", "gru#21", "gru#22"],
    "42_GRUBidirectionalHidden": ["gru#18", "gru#19"],
    "49_Mamba2ReturnFinalState": ["getitem_3"],
}


def _g(ops, out):
    return {"ops": [{"name": n, "op": k, "inputs": i, "outputs": o} for n, k, i, o in ops],
            "output": {"tensors": out}}


class LivenessTest(unittest.TestCase):
    def test_chain_and_dead_branch(self):
        g = _g([("a", "relu", ["x"], ["t1"]), ("b", "relu", ["t1"], ["t2"]), ("c", "relu", ["t1"], ["dead"]),
                ("d", "catn", ["dead", "t2"], ["dead2"])], ["t2"])
        self.assertEqual(gp.dead_ops(g), [2, 3])
        self.assertEqual(gp.live_tensors(g), {"t2", "t1", "x"})

    def test_multi_output_op_is_live_if_any_output_is(self):
        g = _g([("gru", "gru_seq_f16", ["x", "h0"], ["y", "hn"]), ("cat", "catn_f16", ["hn"], ["hstack"])], ["y"])
        self.assertEqual(gp.dead_ops(g), [1])

    def test_views_and_no_output_ops(self):
        g = _g([("v", "view", ["x"], ["xv"]), ("s", "sink", ["xv"], []), ("r", "relu", ["x"], ["y"])], ["y"])
        self.assertEqual(gp.dead_ops(g), [])            # an op without outputs is kept (side effect)

    def test_serial_kinds_are_live(self):
        from modelblaster.pipeline import kb_concurrency as kc
        g = _g([("w", "write_state", ["x"], ["s"]), ("r", "relu", ["x"], ["y"])], ["y"])
        self.assertEqual(gp.dead_ops(g), [0])
        kc.SERIAL_KINDS.add("write_state")
        try:
            self.assertEqual(gp.dead_ops(g), [])
        finally:
            kc.SERIAL_KINDS.discard("write_state")

    def test_selection_switch_and_all_dead_partition(self):
        g = _g([("a", "relu", ["x"], ["y"]), ("b", "relu", ["x"], ["z"])], ["y"])
        self.assertEqual(gp.prune_selection(g, [0, 1], enable=True), ([0], [1]))
        self.assertEqual(gp.prune_selection(g, [0, 1], enable=False), ([0, 1], []))
        self.assertEqual(gp.prune_selection(g, [1], enable=True), ([1], []))     # never an empty problem
        os.environ[gp.ENV] = "off"
        try:
            self.assertFalse(gp.enabled())
            self.assertEqual(gp.prune_selection(g, [0, 1]), ([0, 1], []))
        finally:
            del os.environ[gp.ENV]
        self.assertTrue(gp.enabled())
        with self.assertRaises(ValueError):
            gp.enabled("maybe")

    def test_no_known_outputs_prunes_nothing(self):
        g = {"ops": [{"name": "a", "op": "relu", "inputs": ["x"], "outputs": ["y"]}], "output": {}}
        self.assertEqual(gp.dead_ops(g), [])


@unittest.skipUnless(FARM.is_dir(), "no local l3_farm results")
class SweepIRsTest(unittest.TestCase):
    """Every kept sweep IR: dead ops only where expected, and only data movement (catn / strided copies)."""

    def test_all_sweep_irs(self):
        files = sorted(glob.glob(str(FARM / "*" / "work" / "final" / "*" / "ir" / "graph.json")))
        self.assertGreaterEqual(len(files), 40)
        seen = {}
        for f in files:
            prob = Path(f).parents[4].name
            g = json.loads(Path(f).read_text())
            r = gp.report(g)
            if r["dead"]:
                seen[prob] = r["dead"]
                self.assertTrue(set(r["dead_kinds"]) <= {"catn_f16", "strided_copy_f16"}, (prob, r))
            # proof obligation: no dead op's output is read by a live op or returned
            live = gp.live_tensors(g)
            for i in gp.dead_ops(g):
                self.assertFalse(set(g["ops"][i]["outputs"]) & live, (prob, g["ops"][i]["name"]))
        for p, names in EXPECTED_DEAD.items():
            if p in seen or any(Path(f).parents[4].name == p for f in files):
                self.assertEqual(seen.get(p), names, p)
        self.assertEqual(set(seen) - set(EXPECTED_DEAD), set())


def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


GRU = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 8, 2, True, False, dropout=0, bidirectional=True)
    def forward(self, x, h0):
        output, h_n = self.gru(x, h0)
        return output
def get_inputs():
    return [torch.rand(4, 2, 8), torch.rand(4, 2, 8)]
def get_init_inputs():
    return []
'''

MLP = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 32)
        self.b = nn.Linear(32, 8)
    def forward(self, x):
        return self.b(torch.relu(self.a(x)))
def get_inputs():
    return [torch.randn(4, 16)]
def get_init_inputs():
    return []
'''


def _extract(src: str, d: Path) -> Path:
    root = Path(__file__).resolve().parents[3]
    (d / "m.py").write_text(src)
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file", str(d / "m.py"),
                         "--bench-target-mb", "0", "--out-dir", str(d / "ir"), "--quant", "fp16"],
                        cwd=root, capture_output=True, text=True, timeout=600,
                        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])})
    if cp.returncode != 0:
        raise unittest.SkipTest("extract failed here: " + cp.stderr[-500:])
    return d / "ir"


def _same_tree(a: Path, b: Path) -> list[str]:
    """Files that differ (the driver embeds its own data.bin path: compared with the directory names swapped)."""
    diffs = []
    for p in sorted(a.rglob("*")):
        if p.is_file():
            q = b / p.relative_to(a)
            if not q.is_file():
                diffs.append(str(p.relative_to(a)))
            elif not filecmp.cmp(p, q, shallow=False):
                if p.name != "driver.cpp" or \
                        p.read_text().replace(str(a.resolve()), "@") != q.read_text().replace(str(b.resolve()), "@"):
                    diffs.append(str(p.relative_to(a)))
    return diffs


@unittest.skipUnless(_have_torch(), "needs torch")
class ExportTest(unittest.TestCase):
    def test_mlp_byte_identical(self):
        from modelblaster.pipeline.kb_export import export_problem
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ir = _extract(MLP, d)
            a = export_problem(ir, d / "on", prune_dead=True)
            b = export_problem(ir, d / "off", prune_dead=False)
            self.assertEqual(_same_tree(a, b), [])

    def test_gru_hn_stack_pruned(self):
        from modelblaster.pipeline.kb_export import export_problem
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ir = _extract(GRU, d)
            from modelblaster.pipeline.graph_hints import unique_names
            g = unique_names(json.loads((ir / "graph.json").read_text()))
            dead = gp.report(g)["dead"]
            self.assertTrue(dead)
            on = json.loads((export_problem(ir, d / "on", prune_dead=True) / "manifest.json").read_text())
            off = json.loads((export_problem(ir, d / "off", prune_dead=False) / "manifest.json").read_text())
            self.assertEqual(on["partition"]["pruned_dead"], dead)
            self.assertNotIn("pruned_dead", off["partition"])
            self.assertEqual(on["name"], off["name"])
            self.assertTrue(on["partition"]["is_whole_graph"])
            self.assertEqual([r["name"] for r in on["ops"]], [r["name"] for r in off["ops"] if r["name"] not in dead])
            # the model's output buffer is checked either way; dead-only tensors are gone
            names_on = {b["name"] for b in on["buffers"]}
            out_on = {on["buffers"][i]["name"] for i in on["outputs"]}
            out_off = {off["buffers"][i]["name"] for i in off["outputs"]}
            self.assertEqual(out_on, out_off & names_on)
            self.assertTrue(set(g["output"]["tensors"]) <= out_on)
            # env switch = the argument
            os.environ[gp.ENV] = "0"
            try:
                env_off = json.loads((export_problem(ir, d / "envoff") / "manifest.json").read_text())
            finally:
                del os.environ[gp.ENV]
            self.assertEqual(_same_tree(d / "off", d / "envoff"), [])
            self.assertEqual(env_off["ops"], off["ops"])


if __name__ == "__main__":
    unittest.main()
