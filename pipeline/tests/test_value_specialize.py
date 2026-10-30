"""Value specialization of data-dependent ops on fixed integer inputs (concrete_trace.py, extract_graph.py,
gpu_eval/tools/sol_upstream.py): SOL's FlashInfer paged / ragged attention loops over `indptr[i].item()`.

* the tracer evaluates `.item()` / `.tolist()` / `nonzero` / `bincount` at trace time when the operand derives
  only from inputs declared value-specialized (and constants); the same ops on any other tensor are still refused;
* `extract_graph --bench-file` reads the declaration (`MB_VALUE_SPECIALIZED_INPUTS`), keeps the index tensors that
  feed a gather as runtime inputs, records what it baked in graph.json `data_specialized` (input, sha256), and the
  IR walks to the reference output;
* the SOL converter declares the integer safetensors / custom inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MB.parent))
sys.path.insert(0, str(MB / "gpu_eval" / "tools"))
from modelblaster.pipeline import concrete_trace as ct  # noqa: E402
import sol_upstream as su  # noqa: E402


class PagedDecode(torch.nn.Module):
    """A miniature of FlashInfer's paged decode: per sequence, `kv_indptr[b].item()` bounds select the pages of
    `kv_indices`, which gather rows of the cache."""

    def forward(self, q, cache, kv_indptr, kv_indices):
        out = torch.zeros(q.shape[0], q.shape[1])
        for b in range(q.shape[0]):
            s, e = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
            if s >= e:
                continue
            rows = cache[kv_indices[s:e].to(torch.long)]                  # [n, D]
            att = torch.softmax(q[b] @ rows.T, dim=-1)                     # [n]
            out[b] = att @ rows
        return out


def _inputs():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(3, 8, generator=g)
    cache = torch.randn(10, 8, generator=g)
    indptr = torch.tensor([0, 2, 2, 5], dtype=torch.int32)               # sequence 1 is empty
    indices = torch.tensor([7, 1, 3, 9, 0], dtype=torch.int32)
    return [q, cache, indptr, indices]


def test_item_on_declared_input_is_evaluated():
    ins = _inputs()
    gm = ct.concrete_trace(PagedDecode(), ins, specialize=[2])
    assert torch.allclose(gm(*ins), PagedDecode()(*ins), atol=1e-6)
    assert not any(n.op == "call_method" and n.target == "item" for n in gm.graph.nodes)
    vs = gm.meta["value_specialized"]
    # 2 .item() per sequence, 3 sequences; only kv_indptr (placeholder x2) is baked, kv_indices stays runtime
    assert vs["ops"] == {"item": 6} and [i["name"] for i in vs["inputs"]] == ["kv_indptr"]
    assert vs["inputs"][0]["sha256"] == hashlib.sha256(ins[2].numpy().tobytes()).hexdigest()
    names = {n.name for n in gm.graph.nodes if n.op == "placeholder"}
    assert {"kv_indices"} <= names and any("kv_indices" in {i.name for i in n.all_input_nodes}
                                           for n in gm.graph.nodes if n.op != "placeholder")


def test_float_specialized_input_hash_covers_bf16():
    class M(torch.nn.Module):
        def forward(self, x, w):
            return x[: int(w.sum().item()) % 3 + 1]
    w = torch.tensor([1.0, 2.0]).bfloat16()
    gm = ct.concrete_trace(M(), [torch.randn(4), w], specialize=[1])
    h = gm.meta["value_specialized"]["inputs"][0]
    assert h["dtype"] == "bfloat16" and h["sha256"] == hashlib.sha256(w.view(torch.int16).numpy().tobytes()).hexdigest()


def test_item_is_refused_without_the_declaration():
    with pytest.raises(NotImplementedError, match="data-dependent item"):
        ct.concrete_trace(PagedDecode(), _inputs())
    with pytest.raises(NotImplementedError, match="data-dependent item"):
        ct.concrete_trace(PagedDecode(), _inputs(), specialize=[3])          # the wrong input is declared


def test_item_on_a_float_activation_is_still_refused():
    class M(torch.nn.Module):
        def forward(self, x, n):
            k = int(x.sum().item())                      # depends on x, which is not declared
            return x[:k % 3 + 1] * n.float().sum()
    with pytest.raises(NotImplementedError, match="data-dependent item"):
        ct.concrete_trace(M(), [torch.randn(4, 2), torch.tensor([1, 2])], specialize=[1])


def test_mixed_derivation_is_refused():
    """A tensor computed from a declared input AND a free one is not fixed data."""
    class M(torch.nn.Module):
        def forward(self, x, idx):
            return x[: int((idx + x[0].long()).sum().item())]
    with pytest.raises(NotImplementedError, match="data-dependent item"):
        ct.concrete_trace(M(), [torch.ones(8), torch.tensor([1, 2])], specialize=[1])


def test_tolist_nonzero_bincount_are_baked_constants():
    class M(torch.nn.Module):
        def forward(self, x, lens):
            n = lens.tolist()                            # python ints
            nz = torch.nonzero(lens)                     # a constant tensor
            cnt = torch.bincount(lens)                   # a constant tensor
            return x[: n[0]] * nz.sum() + cnt.sum()
    lens = torch.tensor([3, 1, 0, 3])
    x = torch.randn(6)
    gm = ct.concrete_trace(M(), [x, lens], specialize=[1])
    assert torch.allclose(gm(x, lens), M()(x, lens))
    assert gm.meta["value_specialized"]["ops"] == {"tolist": 1, "nonzero": 1, "bincount": 1}


def test_where_single_arg_is_nonzero():
    class M(torch.nn.Module):
        def forward(self, x, m):
            i = torch.where(m > 0)[0]
            return x[i]
    x, m = torch.randn(5), torch.tensor([0, 2, 0, 1, 1])
    gm = ct.concrete_trace(M(), [x, m], specialize=[1])
    assert torch.equal(gm(x, m), x[[1, 3, 4]])
    with pytest.raises(NotImplementedError):
        ct.concrete_trace(M(), [x, m])


BENCH = '''
import torch
import torch.nn as nn

MB_VALUE_SPECIALIZED_INPUTS = [2, 3]


class Model(nn.Module):
    def forward(self, q, cache, kv_indptr, kv_indices, scale):
        out = torch.zeros(q.shape[0], q.shape[1])
        for b in range(q.shape[0]):
            s, e = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
            if s >= e:
                continue
            rows = cache[kv_indices[s:e].to(torch.long)]
            att = torch.softmax((q[b] @ rows.T) * scale, dim=-1)
            out[b] = att @ rows
        return out


def get_inputs():
    g = torch.Generator().manual_seed(1)
    return [torch.randn(3, 8, generator=g), torch.randn(10, 8, generator=g),
            torch.tensor([0, 2, 2, 5], dtype=torch.int32), torch.tensor([7, 1, 3, 9, 0], dtype=torch.int32), 0.5]


def get_init_inputs():
    return []
'''


def _extract(tmp_path, src, quant="native"):
    (tmp_path / "b.py").write_text(src)
    env = {**os.environ, "PYTHONPATH": str(MB.parent)}
    return subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                           str(tmp_path / "b.py"), "--bench-target-mb", "0", "--out-dir", str(tmp_path / "ir"),
                           "--quant", quant], capture_output=True, text=True, timeout=600, env=env)


@pytest.mark.parametrize("quant", ["native", "fp32"])
def test_extract_paged_decode_bench(tmp_path, quant):
    cp = _extract(tmp_path, BENCH, quant)
    assert cp.returncode == 0, cp.stderr[-3000:]
    g = json.loads((tmp_path / "ir" / "graph.json").read_text())
    ds = g["data_specialized"]
    assert ds["ops"] == {"item": 6} and [i["name"] for i in ds["inputs"]] == ["x2"] and ds["inputs"][0]["index"] == 2
    # the page table is a runtime input (a gather's index), only the indptr is baked
    assert "x3" in g["tensors"] and any("x3" in o["inputs"] for o in g["ops"])
    sys.path.insert(0, str(MB / "gpu_eval" / "harness"))
    import mbir
    import walk_torch
    ir = mbir.load_ir(tmp_path / "ir")
    out, _ = walk_torch.walk(ir, "cpu")
    assert mbir.compare(out, ir.golden_out, ir.dtype)["ok"]
    # and the golden is the program on the bench's inputs
    mod = {}
    exec(compile(BENCH, "b", "exec"), mod)
    ref = mod["Model"]()(*mod["get_inputs"]())
    import numpy as np
    assert np.allclose(np.asarray(ir.golden_out, np.float32).reshape(-1), ref.reshape(-1).numpy(), atol=1e-5)


def test_extract_without_declaration_is_refused(tmp_path):
    cp = _extract(tmp_path, BENCH.replace("MB_VALUE_SPECIALIZED_INPUTS = [2, 3]", ""))
    assert cp.returncode != 0 and "data-dependent item" in cp.stderr


def test_converter_declares_integer_data_inputs(tmp_path):
    """Integer inputs a workload carries as data (safetensors / custom) are declared; random ones and
    floats are not."""
    d = {"name": "mini", "op_type": "attn", "axes": {"n": {"type": "var"}, "m": {"type": "var"}},
         "inputs": {"x": {"shape": ["n"], "dtype": "float32"}, "indptr": {"shape": ["m"], "dtype": "int32"},
                    "rnd": {"shape": ["m"], "dtype": "int32"}},
         "outputs": {"y": {"shape": ["n"], "dtype": "float32"}},
         "reference": "import torch\n\ndef run(x, indptr, rnd):\n    k = int(indptr[-1].item())\n    return x[:k]\n"}
    p = tmp_path / "bench" / "FlashInfer-Bench" / "mini"
    p.mkdir(parents=True)
    (p / "definition.json").write_text(json.dumps(d))
    t = torch.tensor([0, 2], dtype=torch.int32)
    raw = t.numpy().tobytes()
    hdr = json.dumps({"ip": {"dtype": "I32", "shape": [2], "data_offsets": [0, len(raw)]}}).encode()
    blob = tmp_path / "blobs" / "x.safetensors"
    blob.parent.mkdir()
    blob.write_bytes(struct.pack("<Q", len(hdr)) + hdr + raw)
    wl = {"uuid": "u0", "axes": {"n": 4, "m": 2},
          "inputs": {"x": {"type": "random"}, "rnd": {"type": "random"},
                     "indptr": {"type": "safetensors", "path": "data/x.safetensors", "tensor_key": "ip"}}}
    (p / "workload.jsonl").write_text(json.dumps(wl) + "\n")
    man = su.convert(tmp_path / "bench", tmp_path / "out", ["FlashInfer-Bench"], "fp32", 1.0, ["smallest"],
                     blob_root=tmp_path / "blobs")
    src = Path(man["workloads"][0]["file"]).read_text()
    assert "MB_VALUE_SPECIALIZED_INPUTS = [1]" in src
    # a small blob is stored in the bench: it runs without the blob directory
    blob.rename(blob.with_suffix(".moved"))
    mod = {}
    exec(compile(src, "b", "exec"), mod)
    assert torch.equal(mod["get_inputs"]()[1], t)


def test_converter_blob_lookup_without_inlining(tmp_path, monkeypatch):
    """A blob over the inline limit is read from its path, else from $SOL_BLOB_ROOT (bench staged elsewhere)."""
    t = torch.tensor([0, 2, 5], dtype=torch.int32)
    raw = t.numpy().tobytes()
    hdr = json.dumps({"ip": {"dtype": "I32", "shape": [3], "data_offsets": [0, len(raw)]}}).encode()
    root = tmp_path / "trace"
    f = root / "blob" / "w" / "x.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(struct.pack("<Q", len(hdr)) + hdr + raw)
    monkeypatch.setattr(su, "INLINE_BLOB_MAX", 0)
    src = su.bench_source.__globals__["_GEN"]
    ns = {"SOL_BLOBS": {"data/flashinfer-trace/blob/w/x.safetensors": "/nonexistent/x"}, "SOL_BLOB_DATA": {},
          "torch": torch}
    exec(compile(src, "gen", "exec"), ns)
    kind = {"path": "data/flashinfer-trace/blob/w/x.safetensors", "tensor_key": "ip"}
    with pytest.raises(FileNotFoundError):
        ns["_load_blob"](kind)
    monkeypatch.setenv("SOL_BLOB_ROOT", str(root))
    assert torch.equal(ns["_load_blob"](kind), t)


def _fp8_problem(tmp_path):
    d = {"name": "mini_fp8", "op_type": "moe", "axes": {"n": {"type": "var"}},
         "inputs": {"a": {"shape": ["n"], "dtype": "float8_e4m3fn"}, "s": {"shape": ["n"], "dtype": "float32"}},
         "outputs": {"y": {"shape": ["n"], "dtype": "bfloat16"}},
         "reference": "import torch\n\ndef run(a, s):\n    return (a.to(torch.float32) * s).to(torch.bfloat16)\n"}
    p = tmp_path / "bench" / "FlashInfer-Bench" / "mini_fp8"
    p.mkdir(parents=True)
    (p / "definition.json").write_text(json.dumps(d))
    (p / "workload.jsonl").write_text(json.dumps({"uuid": "u", "axes": {"n": 16},
                                                  "inputs": {"a": {"type": "random"}, "s": {"type": "random"}}}) + "\n")
    return tmp_path / "bench"


def test_fp8_inputs_refused_unless_widened(tmp_path):
    bench = _fp8_problem(tmp_path)
    man = su.convert(bench, tmp_path / "o1", ["FlashInfer-Bench"], "fp32", 1.0, ["smallest"])
    assert not man["workloads"] and "fp8" in man["problems"][0]["refusal"] + str(man["problems"][0]["skipped"])
    man = su.convert(bench, tmp_path / "o2", ["FlashInfer-Bench"], "fp32", 1.0, ["smallest"], fp8_widen=True)
    w = man["workloads"][0]
    assert w["fp8_widened"] and w["dtype_map"]["float8_e4m3fn"] == "float32"
    mod = {}
    exec(compile(Path(w["file"]).read_text(), "b", "exec"), mod)
    nat, wide = mod["_native_inputs"](), mod["get_native_inputs"]()
    assert nat[0].dtype == torch.float8_e4m3fn and wide[0].dtype == torch.bfloat16
    assert torch.equal(nat[0].to(torch.float32), wide[0].to(torch.float32))           # exact
    assert mod["get_inputs"]()[0].dtype == torch.float32
    # SOL's ground truth runs the original reference on the real fp8 inputs: bit-identical to the widened model
    ref = mod["sol_reference_outputs"]()[0]
    out = mod["Model"]()(*mod["get_inputs"]())
    assert torch.equal(ref.float(), out.to(torch.bfloat16).float())
    assert (nat[0].float().abs() <= 2.0).all()                                         # SOL's clamp(-2, 2)


def test_inplace_write_of_fixed_data_stays_fixed():
    """FlashInfer's MoE routing: zeros_like(...).scatter_(...) of fixed logits, then nonzero of a comparison."""
    class M(torch.nn.Module):
        def forward(self, x, logits):
            _, idx = torch.topk(logits, k=2, dim=1)
            m = torch.zeros_like(logits)
            m.scatter_(1, idx, 1.0)                        # functionalized: the result is still fixed data
            tok = torch.nonzero((m > 0).any(dim=1)).squeeze(1)
            return x.index_select(0, tok) * m.sum()
    x, lg = torch.randn(4, 3), torch.randn(4, 5)
    gm = ct.concrete_trace(M(), [x, lg], specialize=[1])
    assert torch.allclose(gm(x, lg), M()(x, lg))
    assert gm.meta["value_specialized"]["ops"] == {"nonzero": 1}
    with pytest.raises(NotImplementedError):
        ct.concrete_trace(M(), [x, lg], specialize=[0])
