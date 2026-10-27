"""SOL-ExecBench converter (gpu_eval/tools/sol_upstream.py) and its gate checks
(gpu_eval/tools/sol_upstream_gates.py), on two small copied problems (docs/sol_execbench.md)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MB = Path(__file__).resolve().parents[2]
TOOLS = MB / "gpu_eval" / "tools"
FIX = Path(__file__).resolve().parent / "fixtures" / "sol_upstream"
sys.path.insert(0, str(TOOLS))
import sol_upstream as su  # noqa: E402
import sol_upstream_gates as sg  # noqa: E402

ATTN = "001_attention_softmax_dropout_value_matmul_backward"
RMS = "021_rmsnorm_h128"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    out = tmp_path_factory.mktemp("sol")
    man = su.convert(FIX, out, ["L1", "FlashInfer-Bench"], "fp32", cap_mb=4.0,
                     picks=["smallest", "median", "largest"])
    return out, man


def test_resolve_axes_const_var_expr():
    d = json.loads((FIX / "L1" / ATTN / "definition.json").read_text())
    ax = su.resolve_axes(d, {"batch_size": 2, "seq_len_q": 3, "seq_len_kv": 5})
    assert ax["num_key_value_groups"] == 80 // 8 and ax["batch_size"] == 2
    with pytest.raises(KeyError):
        su.resolve_axes(d, {"batch_size": 2})


def test_pick_and_cap(converted):
    out, man = converted
    rms = [w for w in man["workloads"] if w["problem"] == RMS]
    wls = [json.loads(x) for x in (FIX / "FlashInfer-Bench" / RMS / "workload.jsonl").read_text().splitlines()]
    sizes = sorted(w["axes"]["batch_size"] for w in wls)
    got = {w["pick"]: w["axes"]["batch_size"] for w in rms}
    assert got == {"smallest": sizes[0], "median": sizes[(len(sizes) - 1) // 2], "largest": sizes[-1]}
    # every pick is recorded with its uuid; the manifest says why nothing was dropped
    assert all(w["uuid"] for w in rms) and man["cap_mb"] == 4.0
    # the fixture's L1 workloads are tiny: one fits, both are picked (smallest == median dedupes)
    attn = [w for w in man["workloads"] if w["problem"] == ATTN]
    assert {w["uuid"] for w in attn} == {"fixture-0", "fixture-1"}


def test_cap_skips_with_reason(tmp_path):
    man = su.convert(FIX, tmp_path, ["FlashInfer-Bench"], "fp32", cap_mb=0.05, picks=["smallest", "largest"])
    p = man["problems"][0]
    assert p["skipped"] and all(s["reason"].startswith("io ") for s in p["skipped"])
    for w in man["workloads"]:
        assert w["io_mib"] <= 0.05


def test_bench_is_native_and_shrink_proof(converted):
    out, man = converted
    w = next(w for w in man["workloads"] if w["problem"] == ATTN and w["uuid"] == "fixture-0")
    mod = _load(Path(w["file"]), "_b0")
    # no module-level ints: extract_graph's / torch_baseline's shrinkers have nothing to resize
    assert not [k for k, v in vars(mod).items() if isinstance(v, int) and not isinstance(v, bool)
                and not k.startswith("_") and k not in ("SOL_SEED", "SOL_WORKLOAD_INDEX")]
    ins = mod.get_inputs()
    names = list(mod.SOL_INPUTS)
    assert [list(t.shape) for t in ins if torch.is_tensor(t)] == \
        [mod.SOL_INPUTS[n]["shape_v"] for n in names if mod.SOL_INPUTS[n]["shape_v"] is not None]
    by = dict(zip(names, ins))
    assert by["grad_attn_output"].dtype == torch.float32          # bf16 -> fp32 variant
    assert by["dropout_mask"].dtype == torch.bool                 # non-float dtypes kept
    assert by["attention_dropout"] == pytest.approx(0.1)          # custom scalar
    # deterministic: the same values on every call (and for every variant)
    assert torch.equal(mod.get_inputs()[0], ins[0])
    # softmaxed weights from SOL's own generator
    s = by["attn_weights"].sum(-1)
    assert torch.allclose(s, torch.ones_like(s), atol=2e-2)


def test_extract_loader_keeps_shapes(converted):
    """extract_graph's bench loader (with a tiny byte target) leaves the workload's shapes alone."""
    from modelblaster.pipeline import extract_graph as eg
    out, man = converted
    w = next(w for w in man["workloads"] if w["problem"] == RMS and w["pick"] == "largest")
    _, sample, _ = eg._load_kernelbench(w["file"], target_bytes=1024, target_weight_bytes=1024)
    sample = sample if isinstance(sample, list) else [sample]
    assert [list(t.shape) for t in sample] == [[w["axes"]["batch_size"], 128], [128]]


def test_model_matches_sol_reference(converted):
    out, man = converted
    for w in man["workloads"]:
        mod = _load(Path(w["file"]), f"_m_{w['stem']}")
        ins = mod.get_inputs()
        with torch.no_grad():
            got = mod.Model()(*ins)
        got = [got] if torch.is_tensor(got) else list(got)
        refs = mod.sol_reference_outputs()
        flat = torch.cat([g.float().reshape(-1) for g in got])
        r = sg.sol_compare(flat, refs, w["tolerance"])
        assert r["ok"], (w["stem"], r)


def test_bf16_variant_is_the_reference(tmp_path):
    man = su.convert(FIX, tmp_path, ["FlashInfer-Bench"], "bf16", cap_mb=1.0, picks=["smallest"])
    w = man["workloads"][0]
    mod = _load(Path(w["file"]), "_bf")
    assert w["dtype_rewrites"] == 0 and w["dtype_map"]["bfloat16"] == "bfloat16"
    assert all(t.dtype == torch.bfloat16 for t in mod.get_inputs())
    got = mod.Model()(*mod.get_inputs())
    assert torch.equal(got, mod.sol_reference_outputs()[0])


def test_refusals(tmp_path):
    d = json.loads((FIX / "FlashInfer-Bench" / RMS / "definition.json").read_text())
    wl = json.loads((FIX / "FlashInfer-Bench" / RMS / "workload.jsonl").read_text().splitlines()[0])
    d8 = json.loads(json.dumps(d))
    d8["inputs"]["weight"]["dtype"] = "float8_e4m3fn"
    assert su.refusal(d8, wl, None).startswith("unsupported dtype: fp8")
    ws = json.loads(json.dumps(wl))
    ws["inputs"]["weight"] = {"type": "safetensors", "path": "blob/x.safetensors", "tensor_key": "w"}
    assert "safetensors" in su.refusal(d, ws, None)
    assert "blob missing" in su.refusal(d, ws, tmp_path)
    d0 = json.loads(json.dumps(d))
    for v in d0["inputs"].values():
        v["shape"] = None
    assert su.refusal(d0, wl, None).startswith("no tensor inputs")


def test_safetensors_input(tmp_path):
    """A safetensors input is read by the bench's own reader (no safetensors package needed)."""
    t = torch.arange(128, dtype=torch.float32).to(torch.bfloat16)
    raw = t.view(torch.int16).numpy().tobytes()
    hdr = json.dumps({"w": {"dtype": "BF16", "shape": [128], "data_offsets": [0, len(raw)]}}).encode()
    blob = tmp_path / "blobs" / "blob" / "x.safetensors"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(struct.pack("<Q", len(hdr)) + hdr + raw)
    bench = tmp_path / "bench"
    shutil.copytree(FIX / "FlashInfer-Bench", bench / "FlashInfer-Bench")
    p = bench / "FlashInfer-Bench" / RMS / "workload.jsonl"
    wl = json.loads(p.read_text().splitlines()[0])
    wl["inputs"]["weight"] = {"type": "safetensors", "path": "data/blob/x.safetensors", "tensor_key": "w"}
    p.write_text(json.dumps(wl) + "\n")
    man = su.convert(bench, tmp_path / "out", ["FlashInfer-Bench"], "fp32", 1.0, ["smallest"],
                     blob_root=tmp_path / "blobs")
    mod = _load(Path(man["workloads"][0]["file"]), "_st")
    assert torch.equal(mod.get_inputs()[1], t.float())


def test_sol_check_semantics():
    ref = torch.ones(100)
    tol = dict(su.DEFAULT_TOL, max_atol=1e-3, max_rtol=0.0, required_matched_ratio=0.99)
    out = ref.clone()
    out[0] += 1.0                                     # 1 % of the elements off: still matched 0.99
    assert sg.sol_check(out, ref, tol)["ok"]
    out[1] += 1.0
    assert not sg.sol_check(out, ref, tol)["ok"]
    assert not sg.sol_check(ref.clone() + 0.5e-3, ref, dict(tol, max_error_cap=1e-4))["ok"]
    ninf = ref.clone()
    ninf[3] = float("-inf")
    assert not sg.sol_check(ninf, ninf, tol)["ok"]
    assert sg.sol_check(ninf, ninf, dict(tol, allow_negative_inf=True))["ok"]
    assert not sg.sol_check(torch.zeros(100), ref, dict(tol, required_matched_ratio=0.0))["ok"]


def test_l2_tolerance_typo_is_recorded():
    t = su.tolerance({"tolerance": {"max_atol": 1e-5, "max_rtol": 0.05, "required_match_ratio": 0.98}})
    assert t["required_matched_ratio"] == 0.99 and t["ignored_by_sol"] == {"required_match_ratio": 0.98}


OUTER = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, a, b):
        return torch.outer(a, b).cos() + a.outer(b)
def get_inputs():
    return [torch.randn(6), torch.randn(5)]
def get_init_inputs():
    return []
'''


def test_outer_lowers(tmp_path):
    """torch.outer / Tensor.outer (SOL Quant/014's rope frequencies) lower as einsum i,j->ij."""
    import subprocess
    (tmp_path / "outer.py").write_text(OUTER)
    env = {**__import__("os").environ, "PYTHONPATH": str(MB.parent)}
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(tmp_path / "outer.py"), "--out-dir", str(tmp_path / "ir"), "--quant", "fp32"],
                        capture_output=True, text=True, timeout=300, env=env)
    assert cp.returncode == 0, cp.stderr[-2000:]
    sys.path.insert(0, str(MB / "gpu_eval" / "harness"))
    import mbir
    import walk_torch
    ir = mbir.load_ir(tmp_path / "ir")
    out, _ = walk_torch.walk(ir, "cpu")
    assert mbir.compare(out, ir.golden_out, ir.dtype)["ok"]


FP8 = '''import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x.to(torch.float8_e4m3fn).to(torch.float32) * 2.0
def get_inputs():
    return [torch.randn(4, 8)]
def get_init_inputs():
    return []
'''


def test_fp8_cast_is_refused(tmp_path):
    """A cast to fp8 (SOL Quant's in-graph quantisation) is refused, not lowered as a view that
    drops the rounding."""
    import subprocess
    (tmp_path / "fp8.py").write_text(FP8)
    env = {**__import__("os").environ, "PYTHONPATH": str(MB.parent)}
    cp = subprocess.run([sys.executable, "-m", "modelblaster.pipeline.extract_graph", "--bench-file",
                         str(tmp_path / "fp8.py"), "--out-dir", str(tmp_path / "ir"), "--quant", "fp32"],
                        capture_output=True, text=True, timeout=300, env=env)
    assert cp.returncode != 0 and "no fp8/fp4 storage" in cp.stderr
