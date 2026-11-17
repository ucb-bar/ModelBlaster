"""generate_skeleton gives every intermediate buffer MB_N_BUF_SETS copies and
makes each dispatch use copy s->buf_set, so a schedule runner can run
overlapping instances of one network on separate scratch."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from modelblaster.pipeline import generate_skeleton as gs  # noqa: E402


def test_dispatch_references_are_indexed_and_strays_are_errors():
    src = (
        "extern int8_t buf_m_a[MB_N_BUF_SETS][4];\n"
        "static void dispatch_m_0(model_m_state_t *s) {\n"
        "    k(buf_m_a, buf_m_a + 2, sizeof(buf_m_a));\n"
        "}\n"
    )
    out = gs._index_buffers_by_set(src, "m")
    assert "k(buf_m_a[s->buf_set], buf_m_a[s->buf_set] + 2, sizeof(buf_m_a[s->buf_set]));" in out
    assert out.startswith("extern int8_t buf_m_a[MB_N_BUF_SETS][4];")
    with pytest.raises(RuntimeError, match="outside a dispatch function"):
        gs._index_buffers_by_set(src + "void f(void) { g(buf_m_a); }\n", "m")


def test_generated_model_has_buffer_sets(tmp_path):
    pytest.importorskip("torch")
    import os
    import subprocess
    from modelblaster.pipeline.tests.test_kb_export import MODEL
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        filter(None, [str(_ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    (tmp_path / "tiny.py").write_text(MODEL)
    ir = tmp_path / "ir"
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph",
                         "--bench-file", str(tmp_path / "tiny.py"), "--out-dir", str(ir),
                         "--quant", "int8"], cwd=_ROOT, env=env, capture_output=True,
                        text=True, timeout=600)
    if cp.returncode != 0:
        pytest.skip(f"int8 extract of the tiny model failed: {cp.stderr[-300:]}")
    out = tmp_path / "scalar"
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.generate_skeleton",
                         "--ir", str(ir / "graph.json"), "--weights", str(ir / "weights.npz"),
                         "--io", str(ir / "io.npz"), "--out-dir", str(out),
                         "--backend", "scalar"], cwd=_ROOT, env=env, capture_output=True,
                        text=True, timeout=600)
    assert cp.returncode == 0, cp.stderr[-2000:]
    model_c = (out / "model.c").read_text()
    buffers_c = (out / "buffers.c").read_text()
    model_h = (out / "model.h").read_text()
    externs = re.findall(r"^extern \S+ (buf_\w+)\[MB_N_BUF_SETS\]\[\d+\];", model_c, re.M)
    assert externs, "the tiny model has intermediate buffers"
    for b in externs:
        assert re.search(rf"\b{b}\[MB_N_BUF_SETS\]\[\d+\]", buffers_c)
        uses = re.findall(rf"\b{b}\b(\[s->buf_set\])?", model_c.split("#endif", 1)[1])
        assert uses and all(uses[1:]), f"{b}: every use inside a dispatch is indexed"
    assert re.search(r"const int model_\w+_n_buf_sets = MB_N_BUF_SETS;", buffers_c)
    assert "int                         buf_set;" in model_h
