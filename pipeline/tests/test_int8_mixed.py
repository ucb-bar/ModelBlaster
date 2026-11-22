"""int8 extraction with fp32 islands (pipeline/int8_mixed.py) and nn.RMSNorm in the fp32 FX path.

No GPU: the CPU rules of the int8 kinds against their compiled reference C (bit-exact), the pass on a
small attention block that has what made the all-int8 extractor give up on SmolVLA (get_attr
constants: an additive -inf mask and a RoPE-like table; expand; nn.RMSNorm), and the extract_graph
entry points (extract(quant="int8", int8_mixed=...), --int8-mixed, a model's get_int8_mixed_policy).
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "gpu_eval" / "harness"))

from modelblaster.pipeline import int8_mixed as I8  # noqa: E402
from modelblaster.pipeline.extract_graph import extract  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402


@pytest.fixture(scope="module")
def refc(tmp_path_factory):
    cc = os.environ.get("CC", "cc")
    if not shutil.which(cc):
        pytest.skip("no C compiler for the reference kernels")
    d = tmp_path_factory.mktemp("refc")
    src = ["#include <stdint.h>", "#include <stddef.h>", "#include <math.h>"]
    src += [KERNEL_SPECS[k].reference_impl for k in I8.KINDS]
    (d / "ref.c").write_text("\n".join(src))
    subprocess.run([cc, "-O2", "-shared", "-fPIC", str(d / "ref.c"), "-o", str(d / "ref.so"), "-lm"], check=True)
    return ctypes.CDLL(str(d / "ref.so"))


def _p(a):
    return ctypes.c_void_p(a.ctypes.data) if a is not None else None


def test_kinds_registered():
    for k in I8.KINDS:
        assert k in KERNEL_SPECS and KERNEL_SPECS[k].argtypes_factory is not None
        assert k + "_f16" not in KERNEL_SPECS              # no float storage variants of int8 kinds


def test_quantize_rows_rule_is_the_reference_c(refc):
    rng = np.random.default_rng(0)
    M, K = 41, 777
    x = (rng.standard_normal((M, K)) * rng.random((M, 1)) * 20).astype(np.float32)
    x[0] = 0.0
    x[1, :64] = (np.arange(64) - 32 + 0.5).astype(np.float32)        # ties of roundf
    x[1, 64:] = 0.0
    x[1, 70] = 127.0
    x[2, 3], x[3, 4], x[4, 5] = np.nan, np.inf, -np.inf
    x[5, 6] = 1e30
    ci = (rng.random(K) * 3 + 0.1).astype(np.float32)
    ci[:71] = 1.0
    q, s = np.zeros((M, K), np.int8), np.zeros(M, np.float32)
    refc.kernel_quantize_rows_s8(_p(x), _p(ci), _p(q), _p(s), M, K)
    tq, ts = I8.quantize_rows(torch.from_numpy(x), torch.from_numpy(ci))
    assert np.array_equal(tq.numpy(), q) and np.array_equal(ts.numpy(), s)
    assert (q[1, :64] == np.where(np.arange(64) < 32, np.arange(64) - 32, np.arange(64) - 31)).all()  # half away
    assert s[0] == 1.0 and (q[0] == 0).all() and np.abs(q).max() == 127


def test_linear_dq_rule_is_the_reference_c(refc):
    rng = np.random.default_rng(1)
    M, K, N = 13, 2560, 37
    x = rng.integers(-127, 128, (M, K), dtype=np.int8)
    w = rng.integers(-127, 128, (N, K), dtype=np.int8)
    rs = (rng.random(M) * 0.3 + 1e-4).astype(np.float32)
    ws = (rng.random(N) * 1e-2 + 1e-6).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    ir = I8.as_ir({"name": "t", "tensors": {}, "ops": [], "input": {"tensor": "x"}, "output": {"tensor": "y"}},
                  {"w": w, "ws": ws, "b": b})
    for bias in ("b", None):
        y = np.zeros((M, N), np.float32)
        refc.kernel_linear_dq_s8(_p(x), _p(rs), _p(w), _p(b if bias else None), _p(ws), _p(y), M, K, N)
        op = {"op": "linear_dq_s8", "shape": {"M": M, "K": K, "N": N}, "weight": "w", "w_scale": "ws", "bias": bias}
        (t,) = I8._r_linear_dq(ir, op, [torch.from_numpy(x), torch.from_numpy(rs)])
        assert np.array_equal(t.numpy(), y)
        exact = (x.astype(np.float64) @ w.T.astype(np.float64)) * rs[:, None] * ws[None, :] + (b if bias else 0)
        assert np.abs(y - exact).max() <= 1e-5 * np.abs(exact).max()


class _Block(nn.Module):
    """An attention block with SmolVLA's extraction obstacles: an nn.RMSNorm module, buffers
    (get_attr constants: an additive -inf causal mask and a cos table), expand of a time vector."""

    def __init__(self, d=32, s=6, h=2):
        super().__init__()
        self.d, self.s, self.h = d, s, h
        self.norm = nn.RMSNorm(d, eps=1e-5)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.up = nn.Linear(d, 2 * d)
        self.down = nn.Linear(2 * d, d)
        self.head = nn.Linear(d, 4)
        self.register_buffer("mask", torch.triu(torch.full((s, s), float("-inf")), 1), persistent=False)
        self.register_buffer("cosb", torch.cos(torch.arange(s * d, dtype=torch.float32).reshape(s, d) * 0.1),
                             persistent=False)
        with torch.no_grad():
            self.norm.weight.uniform_(0.5, 1.5)

    def forward(self, x, t):
        s, d, h = self.s, self.d, self.h
        x = x + t.reshape(1, 1, d).expand(1, s, d)
        y = self.norm(x)
        q = (self.q(y) * self.cosb).reshape(s, h, d // h).permute(1, 0, 2)
        k = (self.k(y) * self.cosb).reshape(s, h, d // h).permute(1, 2, 0)
        v = self.v(y).reshape(s, h, d // h).permute(1, 0, 2)
        p = F.softmax(torch.bmm(q, k) * (d // h) ** -0.5 + self.mask, dim=-1)
        a = torch.bmm(p, v).permute(1, 0, 2).reshape(1, s, d)
        x = x + self.o(a)
        x = x + self.down(F.silu(self.up(x)))
        return self.head(x)


def _block():
    torch.manual_seed(0)
    m = _Block().eval()
    g = torch.Generator().manual_seed(1)
    sample = [torch.randn(1, 6, 32, generator=g), torch.randn(1, 32, generator=g)]
    cal = [[torch.randn(1, 6, 32, generator=g) * 2, torch.randn(1, 32, generator=g)] for _ in range(4)]
    return m, sample, cal


def test_rmsnorm_module_lowers_in_fp32(tmp_path):
    """nn.RMSNorm used to fall through every call_module branch: no op, a dangling output tensor."""
    import walk_torch
    from mbir import compare, load_ir

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.n, self.n2, self.l = nn.RMSNorm(16, eps=1e-5), nn.RMSNorm(16, elementwise_affine=False), nn.Linear(16, 8)
            with torch.no_grad():
                self.n.weight.uniform_(0.5, 1.5)

        def forward(self, x):
            return self.l(self.n2(self.n(x * 3 + 1)))

    torch.manual_seed(0)
    extract(M().eval(), torch.randn(3, 16), name="rms", out_dir=str(tmp_path), quant="fp32")
    ir = load_ir(tmp_path)
    written = {t for op in ir.ops for t in op["outputs"]}
    assert all(t in written for op in ir.ops for t in op["inputs"]
               if t not in ir.input_tensors("cpu") and not ir.graph["tensors"][t].get("const"))
    assert any(op["op"] == "reduce" for op in ir.ops)
    out, _ = walk_torch.walk(ir, dev="cpu")
    assert compare(out, ir.golden_out, "fp32")["ok"]


def test_int8_mixed_block(tmp_path):
    """The block extracts as an int8_mixed IR: every linear int8, the constants fp32 constant tensors,
    expand a strided copy, the golden reproduced by the CPU walk, close to the fp32 model."""
    from mbir import load_ir
    m, sample, cal = _block()
    rep = extract(m, sample, name="blk", out_dir=str(tmp_path / "ir"), quant="int8",
                  calibration_samples=cal, int8_mixed="linear")
    ir = load_ir(tmp_path / "ir")
    g = ir.graph
    assert g["quant"] == "int8_mixed" and rep["n_int8_linears"] == 7 and rep["n_fp32_linears"] == 0
    kinds = {op["op"] for op in ir.ops}
    assert {"linear_dq_s8", "quantize_rows_s8", "strided_copy", "softmax"} <= kinds and "linear" not in kinds
    # q / k / v read one quantized tensor
    assert sum(op["op"] == "quantize_rows_s8" for op in ir.ops) == 5
    consts = {k: v for k, v in g["tensors"].items() if v.get("const")}
    assert any(np.isinf(ir.weights[v["const"]]).any() for v in consts.values())      # the -inf mask, fp32
    for op in ir.ops:
        if op["op"] == "linear_dq_s8":
            assert ir.weights[op["weight"]].dtype == np.int8 and ir.weights[op["w_scale"]].dtype == np.float32
            assert g["tensors"][op["inputs"][0]]["dtype"] == "i8"
    out = I8.walk(ir, ir.input_tensors("cpu"))
    o = torch.cat([out[t].reshape(-1) for t in ir.output_names()]).numpy()
    assert np.array_equal(o, ir.golden_out.reshape(-1))
    with torch.no_grad():
        ref = m(*sample).reshape(-1).double().numpy()
    cos = o @ ref / (np.linalg.norm(o) * np.linalg.norm(ref))
    assert cos > 0.999 and np.abs(o - ref).max() < 0.05 * np.abs(ref).max(), (cos, np.abs(o - ref).max())


def test_keep_fp32_and_smoothing(tmp_path):
    """keep_fp32 leaves the named linears fp32; SmoothQuant moves an outlier column's range into the
    weight (alpha 0.5 beats alpha 0 on an input with an outlier channel)."""
    from mbir import load_ir
    m, sample, cal = _block()
    extract(m, sample, name="blk", out_dir=str(tmp_path / "fp32"), quant="fp32")
    ir = load_ir(tmp_path / "fp32")
    samples = [I8.inputs_of(ir, s) for s in cal]
    g, w, rep = I8.quantize_graph(ir, samples, {"base": "linear", "keep_fp32": ["head", "o"]})
    assert sorted(rep["fp32_linears"]) == ["head", "o"] and rep["n_int8_linears"] == 5
    assert all(k in w for k in ("head.weight", "o.weight")) and "q.weight" not in w

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(64, 16)

        def forward(self, x):
            return self.l(x)

    torch.manual_seed(2)
    lin = Lin().eval()
    x = torch.randn(8, 64)
    x[:, 3] *= 200.0                                     # an outlier channel
    extract(lin, x, name="lin", out_dir=str(tmp_path / "lin"), quant="fp32")
    li = load_ir(tmp_path / "lin")
    errs = {}
    for a in (0.0, 0.5):
        gg, ww, _ = I8.quantize_graph(li, [I8.inputs_of(li, x)], {"base": "linear", "alpha": a})
        y = I8.walk(I8.as_ir(gg, ww), I8.inputs_of(li, x))[li.output_names()[0]]
        errs[a] = float((y - lin(x).detach()).abs().max())
    assert errs[0.5] < 0.5 * errs[0.0], errs


def test_outlier_columns(tmp_path):
    """outliers: the largest-|x| columns run as an fp32 linear on an index_select of them; with
    an outlier channel the decomposition beats int8 for every column, and the CPU walk is exact."""
    from mbir import load_ir

    class Lin(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(64, 16)

        def forward(self, x):
            return self.l(x)

    torch.manual_seed(3)
    lin = Lin().eval()
    x = torch.randn(8, 64)
    x[:, 5] *= 300.0
    x[:, 40] *= 80.0
    extract(lin, x, name="lin", out_dir=str(tmp_path / "lin"), quant="fp32")
    li = load_ir(tmp_path / "lin")
    errs = {}
    for n in (0, 2):
        g, w, rep = I8.quantize_graph(li, [I8.inputs_of(li, x)], {"base": "linear", "alpha": 0.0, "outliers": {"l": n}})
        kinds = [op["op"] for op in g["ops"]]
        assert ("index_select" in kinds) == (n > 0) and rep["n_fp32_linears"] == 0
        y = I8.walk(I8.as_ir(g, w), I8.inputs_of(li, x))[li.output_names()[0]]
        errs[n] = float((y - lin(x).detach()).abs().max())
        if n:
            assert rep["outlier_columns"] == {"x": 2}
            sel = next(op for op in g["ops"] if op["op"] == "index_select")
            assert sorted(w[g["tensors"][sel["inputs"][1]]["const"]].tolist()) == [5.0, 40.0]
    assert errs[2] < 0.25 * errs[0], errs


def test_policy_resolution():
    assert I8.resolve_policy("linear")["alpha"] == 0.5
    p = I8.resolve_policy('{"base": "linear", "keep_fp32": ["a*"], "alpha": 0.25}')
    assert p["keep_fp32"] == ["a*"] and p["alpha"] == 0.25 and p["int8_kinds"] == ["linear"]
    with pytest.raises(ValueError):
        I8.resolve_policy("bogus")


def test_cli_with_model_policy(tmp_path, monkeypatch):
    """extract_graph --model smolvla_split --quant int8 takes the model's int8_mixed policy (tiny config)."""
    from modelblaster.pipeline import extract_graph as EG
    from modelblaster.models import smolvla_split as S
    monkeypatch.setenv("MB_SMOLVLA_PART", "prefix")
    monkeypatch.setenv("MB_SMOLVLA_IMG", "64")
    monkeypatch.setenv("MB_SMOLVLA_LANG", "4")
    monkeypatch.setenv("MB_SMOLVLA_LAYERS", "2")
    monkeypatch.setenv("MB_SMOLVLA_VLAYERS", "1")
    monkeypatch.delenv("MB_SMOLVLA_REAL", raising=False)
    monkeypatch.delenv("MB_SMOLVLA_CALIB", raising=False)
    S._CACHE.clear()
    try:
        for env, n_int8 in (("all", 1 + 7 + 2 + 2), (None, 0)):
            # all: state_proj + layer 0's 7 + the last layer's k / v (its q feeds nothing the expert
            # reads) + the expert's cross-attention K / V projections of layer 1 (hoist_cross_kv);
            # the default policy keeps the prefix fp32 (models/smolvla_split.INT8_POLICY)
            if env:
                monkeypatch.setenv("MB_SMOLVLA_INT8", env)
            else:
                monkeypatch.delenv("MB_SMOLVLA_INT8", raising=False)
            out = tmp_path / str(env)
            monkeypatch.setattr(sys, "argv", ["extract_graph", "--model", "smolvla_split", "--quant", "int8",
                                              "--num-calibration", "2", "--out-dir", str(out)])
            EG.main()
            g = json.loads((out / "graph.json").read_text())
            assert g["quant"] == "int8_mixed" and g["int8_mixed"]["n_calibration"] == 2
            assert g["int8_mixed"]["n_int8_linears"] == n_int8, env
    finally:
        S._CACHE.clear()
