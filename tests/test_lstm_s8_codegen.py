"""An nn.LSTM model lowers to C: extract_graph's lstm_s8 records reach kernel_lstm_s8 (and, after
pipeline/lower_lstm_int.py, kernel_lstm_q_s8) in the generated model.c.

generate_skeleton.emit_model used to carry a second, earlier `lstm_s8` branch written for an IR schema nothing
emits (op["weight_ih"], q["s_x"]); it won, so every nn.LSTM model died with KeyError: 'weight_ih'.
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, cwd):
    env = dict(os.environ)
    return subprocess.run([sys.executable] + args, cwd=cwd, env=env, capture_output=True, text=True)


@pytest.mark.skipif(importlib.util.find_spec("modelblaster") is None,
                    reason="the repository must be importable as `modelblaster` (PYTHONPATH)")
def test_lstm_model_lowers(tmp_path):
    pytest.importorskip("torch")
    gen = tmp_path / "gen"
    r = _run(["-m", "modelblaster.pipeline.extract_graph", "--model", "vitfly_lstm", "--quant", "int8",
              "--out-dir", str(gen)], REPO)
    assert r.returncode == 0, r.stderr[-2000:]
    ops = [o["op"] for o in json.load(open(gen / "graph.json"))["ops"]]
    assert ops.count("lstm_s8") == 2
    for lower in (False,):
        out = tmp_path / ("q" if lower else "f")
        if lower:
            r = _run(["pipeline/lower_lstm_int.py", str(gen / "graph.json"), str(gen / "weights.npz"),
                      str(gen / "io.npz")], REPO)
            assert r.returncode == 0, r.stderr[-2000:]
        r = _run(["-m", "modelblaster.pipeline.generate_skeleton", "--ir", str(gen / "graph.json"),
                  "--weights", str(gen / "weights.npz"), "--io", str(gen / "io.npz"),
                  "--out-dir", str(out), "--backend", "scalar"], REPO)
        assert r.returncode == 0, r.stderr[-2000:]
        model_c = open(out / "model.c").read()
        assert model_c.count("kernel_lstm_q_s8_" if lower else "kernel_lstm_s8_") >= 2
