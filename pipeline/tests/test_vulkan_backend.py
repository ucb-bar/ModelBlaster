"""The Vulkan GPU backend (gpu_eval/harness/vulkan, walk_vulkan.py) and the SmolVLA split graphs.

Always run (no GPU): every Vulkan launcher has exactly its KernelSpec's C signature and a shader;
models/smolvla_split.py extracts (fp32 FX path) into IRs whose PyTorch walk matches their goldens.

GPU (skipped cleanly without one): libmb_vk.so is taken from $MB_VK_LIB, else built into a temp
dir when glslc and g++ exist; a usable Vulkan device must be present (lavapipe counts only with
MB_VK_ALLOW_CPU=1). Then every kernel family is checked against numpy, small IRs that cover the
launchers are walked with walk_vulkan's golden + per-kernel isolation gates, capture / replay is
checked, and the SmolVLA split graphs run end to end against their PyTorch modules.

MB_SMOLVLA_TEST_REAL=1 (needs lerobot and the lerobot/smolvla_base HF cache) also checks the split
graphs against lerobot's own sample_actions.
"""
from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "gpu_eval" / "harness"
VK_DIR = HARNESS / "vulkan"
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HARNESS))

from modelblaster.pipeline.kernel_abi import parse_signature  # noqa: E402
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS  # noqa: E402


def _launchers() -> dict[str, list[str]]:
    """extern "C" kernel_* definitions in mb_vk_kernels.cpp -> normalized param ctypes."""
    src = (VK_DIR / "mb_vk_kernels.cpp").read_text()
    out = {}
    for m in re.finditer(r"^void (kernel_\w+)\(([^)]*)\)\s*\{", src, re.M):
        params = [" ".join(p.split()) for p in m.group(2).split(",") if p.strip()]
        out[m.group(1)] = [re.sub(r"\s*\w+$", "", p).replace(" *", "*") for p in params]
    return out


def _spec_ctypes() -> dict[str, list[str]]:
    out = {}
    for op, spec in KERNEL_SPECS.items():
        try:
            name, params = parse_signature(spec.signature)
        except ValueError:
            continue
        out.setdefault(name, [p.ctype.replace(" *", "*") for p in params])
    return out


# ------------------------------------------------------------------ static (no GPU)
def test_every_launcher_has_its_kernelspec_signature():
    launchers, specs = _launchers(), _spec_ctypes()
    assert len(launchers) >= 60
    for name, ctypes_ in launchers.items():
        assert name in specs, f"{name}: no KernelSpec signature"
        assert ctypes_ == specs[name], f"{name}: {ctypes_} != spec {specs[name]}"


def test_every_pipeline_has_a_shader():
    src = (VK_DIR / "mb_vk_kernels.cpp").read_text()
    names = set(re.findall(r'PIPE\("(\w+)"\)', src)) | set(re.findall(r'mb_vk_pipeline\("(\w+)"\)', src))
    names |= set(re.findall(r'\{"(gemm_\w+)",', src))
    assert {"gemm_16", "gemm_32", "gemm_64", "gemm_big", "splitk_reduce", "softmax", "layer_norm"} <= names
    for n in names:
        assert (VK_DIR / "shaders" / f"{n}.comp").exists(), n


def _tiny_cfg():
    from modelblaster.models import smolvla_split as S
    return S, S.Cfg(img=64, lang=4, v_layers=1, layers=2)


def _extract_split(tmp: Path, parts=None):
    from modelblaster.pipeline.extract_graph import extract
    S, c = _tiny_cfg()
    parts = parts or S.build(None, c)
    g = torch.Generator().manual_seed(3)
    pixels = torch.rand(1, 3, c.img, c.img, generator=g) * 2 - 1
    lang = torch.randn(1, c.lang, c.hidden, generator=g)
    state = torch.randn(1, c.state_dim, generator=g)
    noise = torch.randn(1, c.chunk, c.action_dim, generator=g)
    with torch.no_grad():
        img = parts.vision(pixels)
        kv = parts.prefix(img, lang, state)
    samples = {"vision": pixels, "prefix": [img, lang, state], "step": [kv, noise, S.time_embedding(1.0, c)]}
    for name, x in samples.items():
        extract(getattr(parts, name), x, name=f"smolvla_{name}", out_dir=str(tmp / name), quant="fp32")
    return S, c, parts, dict(pixels=pixels, lang=lang, state=state, noise=noise)


@pytest.fixture(scope="module")
def split_irs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smolvla_split")
    return (tmp,) + _extract_split(tmp)


def test_smolvla_split_irs_walk_on_cpu(split_irs):
    import walk_torch
    from mbir import compare, load_ir
    tmp = split_irs[0]
    for part in ("vision", "prefix", "step"):
        ir = load_ir(tmp / part)
        out, _ = walk_torch.walk(ir, dev="cpu")
        r = compare(out, ir.golden_out, ir.dtype)
        assert r["ok"], (part, r)


def test_smolvla_split_mask_and_rope_buffers():
    S, c = _tiny_cfg()
    p = S.build(None, c)
    P, A = c.prefix_len, c.chunk
    assert P == c.n_img + c.lang + 1
    # prefix: nothing but the state token attends the state token
    assert torch.isinf(p.prefix.bias[:P - 1, P - 1]).all() and (p.prefix.bias[P - 1] == 0).all()
    # expert self-attention rows: all prefix keys, causal over the chunk
    assert (p.step.bias[:, :P] == 0).all() and torch.isinf(p.step.bias[0, P + 1]) and p.step.bias[A - 1, P + A - 1] == 0
    assert p.step.cos_sa.shape == (A, 1, c.head_dim // 2)


@pytest.mark.skipif(not os.environ.get("MB_SMOLVLA_TEST_REAL"), reason="MB_SMOLVLA_TEST_REAL=1: lerobot + HF cache")
def test_smolvla_split_matches_lerobot():
    from modelblaster.models import smolvla_split as S
    policy = S.load_policy(256)
    g = torch.Generator().manual_seed(0)
    images = [torch.rand(1, 3, 256, 256, generator=g) * 2 - 1]
    tokens = torch.randint(0, 1000, (1, 12), generator=g)
    lmask = torch.ones(1, 12, dtype=torch.bool)
    state = torch.randn(1, policy.config.max_state_dim, generator=g)
    noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=g)
    policy.config.num_steps = 3
    with torch.no_grad():
        ref = policy.model.sample_actions(images, [torch.ones(1, dtype=torch.bool)], tokens, lmask, state, noise=noise.clone())
    parts = S.build(policy, S.cfg_from_policy(policy, 256, 12))
    out = parts.sample_actions(images[0], parts.lang_embedding(tokens), state, noise.clone(), 3)
    cos = torch.nn.functional.cosine_similarity(out.flatten().double(), ref.flatten().double(), dim=0)
    assert cos > 0.999999 and (out - ref).abs().max() < 1e-4


# ------------------------------------------------------------------ GPU
@pytest.fixture(scope="module")
def vk(tmp_path_factory):
    from mbvk import VkBackend, VkUnavailable
    lib = os.environ.get("MB_VK_LIB")
    if not lib:
        if not (shutil.which("glslc") and shutil.which(os.environ.get("CXX", "g++"))):
            pytest.skip("no $MB_VK_LIB and no glslc + g++ to build libmb_vk.so")
        import importlib.util
        spec = importlib.util.spec_from_file_location("mb_vk_build", VK_DIR / "build.py")
        vkbuild = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vkbuild)
        out = tmp_path_factory.mktemp("vkbuild")
        if vkbuild.main(["--out", str(out)]) != 0:
            pytest.skip("libmb_vk.so build failed")
        lib = str(out / "libmb_vk.so")
    try:
        return VkBackend(lib)
    except VkUnavailable as e:
        pytest.skip(f"no usable Vulkan device: {e}")


def _p(vk, a):
    return ctypes.c_void_p(vk.ptr(a))


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


@pytest.mark.parametrize("M,K,N", [(8, 96, 200), (31, 960, 300), (31, 2048, 130), (50, 720, 960),
                                   (200, 64, 300), (16, 4096, 960), (37, 100, 30), (5, 7, 3)])
def test_gemm_family(vk, M, K, N):
    rng = np.random.default_rng(M * 7 + K)
    L = vk.lib
    A = vk.upload(rng.standard_normal((M, K), dtype=np.float32))
    W = vk.upload(rng.standard_normal((N, K), dtype=np.float32))
    b = vk.upload(rng.standard_normal(N, dtype=np.float32))
    Bkn = vk.upload(rng.standard_normal((K, N), dtype=np.float32))
    At = vk.upload(np.ascontiguousarray(A.T))
    outs = {k: vk.array((M, N)) for k in ("lin", "mm", "ta", "tb", "tatb")}
    L.kernel_linear(_p(vk, A), _p(vk, W), _p(vk, b), _p(vk, outs["lin"]), M, K, N)
    L.kernel_matmul(_p(vk, A), _p(vk, Bkn), _p(vk, outs["mm"]), M, K, N)
    L.kernel_matmul_ta(_p(vk, At), _p(vk, Bkn), _p(vk, outs["ta"]), M, K, N)
    L.kernel_matmul_tb(_p(vk, A), _p(vk, W), _p(vk, outs["tb"]), M, K, N)
    L.kernel_matmul_tatb(_p(vk, At), _p(vk, W), _p(vk, outs["tatb"]), M, K, N)
    # unaligned (not 16-byte) operands take the scalar-load path
    A1 = vk.array((M * K + 1,))
    A1[1:] = A.reshape(-1)
    o1 = vk.array((M, N))
    L.kernel_linear(ctypes.c_void_p(vk.ptr(A1) + 4), _p(vk, W), None, _p(vk, o1), M, K, N)
    vk.sync()
    A64, W64 = A.astype(np.float64), W.astype(np.float64)
    assert _rel(outs["lin"], A64 @ W64.T + b) < 1e-5
    assert _rel(outs["mm"], A64 @ Bkn) < 1e-5
    assert _rel(outs["ta"], A64 @ Bkn) < 1e-5
    assert _rel(outs["tb"], A64 @ W64.T) < 1e-5
    assert _rel(outs["tatb"], A64 @ W64.T) < 1e-5
    assert _rel(o1, A64 @ W64.T) < 1e-5


def test_pack_weight_layout():
    from walk_vulkan import pack_weight
    w = np.arange(70 * 24, dtype=np.float32).reshape(70, 24)
    p = pack_weight(w).reshape(-1)
    assert p.size == 128 * 24                       # columns padded to 64
    for nb, kq, c, i in [(0, 0, 0, 0), (0, 5, 63, 3), (1, 2, 5, 1), (1, 0, 6, 0)]:
        n, idx = nb * 64 + c, ((nb * 6 + kq) * 64 + c) * 4 + i
        assert p[idx] == (w[n, 4 * kq + i] if n < 70 else 0.0)


@pytest.mark.parametrize("M,K,N", [(1, 32, 960), (7, 720, 130), (31, 960, 300), (50, 720, 2048), (50, 2048, 720),
                                   (64, 1440, 65), (16, 12288, 96)])
@pytest.mark.parametrize("f16", [False, True])
def test_skinny_gemm_packed(vk, M, K, N, f16):
    """A linear of <= 64 rows whose weight has a packed copy (walk_vulkan.Weights does this
    for every such linear) runs the skinny GEMM; fp16 copies hold rounded weights."""
    from walk_vulkan import pack_weight
    rng = np.random.default_rng(M + K + N)
    L = vk.lib
    m = vk.mark()
    A = vk.upload(rng.standard_normal((M, K), dtype=np.float32))
    W = vk.upload(rng.standard_normal((N, K), dtype=np.float32))
    b = vk.upload(rng.standard_normal(N, dtype=np.float32))
    P = vk.upload(pack_weight(W, np.float16 if f16 else np.float32))
    L.mb_vk_set_packed(vk.ptr(W), vk.ptr(P), int(f16))
    out, out_tb = vk.array((M, N)), vk.array((M, N))
    L.kernel_linear(_p(vk, A), _p(vk, W), _p(vk, b), _p(vk, out), M, K, N)
    L.kernel_matmul_tb(_p(vk, A), _p(vk, W), _p(vk, out_tb), M, K, N)
    vk.sync()
    W64 = (W.astype(np.float16) if f16 else W).astype(np.float64)
    ref = A.astype(np.float64) @ W64.T
    assert _rel(out, ref + b) < 1e-5 and _rel(out_tb, ref) < 1e-5
    n0 = L.mb_vk_packed_count()
    vk.release(m)                                   # the registration goes with the buffers
    assert L.mb_vk_packed_count() == n0 - 1


_EW_CASES = [  # (dims, a strides, b strides): SmolVLA's broadcast / layout shapes, plus odd ones (scalar path)
    ((1, 1, 1, 50, 15, 32), (0, 0, 0, 480, 32, 1), (0, 0, 0, 32, 0, 1)),
    ((1, 1, 1, 1, 50, 720), (0, 0, 0, 0, 720, 1), (0, 0, 0, 0, 1, 0)),
    ((1, 1, 1, 1, 15, 961), (0, 0, 0, 0, 961, 1), (0, 0, 0, 0, 0, 1)),
    ((1, 1, 1, 1, 1, 36000), (0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1)),
    ((2, 3, 1, 4, 5, 8), (480, 160, 0, 40, 8, 1), (0, 8, 0, 0, 0, 1)),
    ((2, 3, 4, 5, 6, 7), (2520, 840, 210, 42, 7, 1), (1, 2, 6, 24, 120, 720)),
]


@pytest.mark.parametrize("case", range(len(_EW_CASES)))
def test_ewise2_and_strided_copy_layouts(vk, case):
    """Dim coalescing and the vec4 variants against numpy views (the walker's broadcast / transpose ops)."""
    from numpy.lib.stride_tricks import as_strided
    d, sa, sb = _EW_CASES[case]
    rng = np.random.default_rng(case)
    span = lambda st: 1 + sum((n - 1) * s for n, s in zip(d, st))
    a = vk.upload(rng.standard_normal(span(sa) + 64, dtype=np.float32))
    b = vk.upload(rng.standard_normal(span(sb) + 64, dtype=np.float32) + 3)
    y, yc, yo = (vk.array(d) for _ in range(3))
    L = vk.lib
    L.kernel_ewise2(_p(vk, a), _p(vk, b), _p(vk, y), 2, ctypes.c_float(1.0), *d, *sa, *sb)       # mul
    L.kernel_strided_copy(_p(vk, a), _p(vk, yc), 0, *d, *sa)
    L.kernel_strided_copy(_p(vk, a), _p(vk, yo), 32, *d, *sa)                                    # aligned offset
    yo1 = vk.array(d)
    L.kernel_strided_copy(_p(vk, a), _p(vk, yo1), 1, *d, *sa)                                    # unaligned
    vk.sync()
    view = lambda x, st, off=0: as_strided(np.asarray(x)[off:], d, [4 * v for v in st])
    assert np.array_equal(y, view(a, sa) * view(b, sb))
    assert np.array_equal(yc, view(a, sa)) and np.array_equal(yo, view(a, sa, 32)) and np.array_equal(yo1, view(a, sa, 1))


def test_bmm_and_patch_conv(vk):
    rng = np.random.default_rng(1)
    L = vk.lib
    a = vk.upload(rng.standard_normal((6, 33, 40), dtype=np.float32))
    b = vk.upload(rng.standard_normal((6, 40, 17), dtype=np.float32))
    c = vk.array((6, 33, 17))
    L.kernel_bmm(_p(vk, a), _p(vk, b), _p(vk, c), 6, 33, 40, 17)
    x = vk.upload(rng.standard_normal((1, 3, 64, 48), dtype=np.float32))
    w = vk.upload(rng.standard_normal((20, 3, 16, 16), dtype=np.float32) * 0.1)
    bias = vk.upload(rng.standard_normal(20, dtype=np.float32))
    y = vk.array((1, 20, 4, 3))       # patchify: im2col + GEMM
    L.kernel_conv2d(_p(vk, x), _p(vk, w), _p(vk, bias), _p(vk, y), 1, 3, 64, 48, 20, 16, 16, 16, 16, 0, 0, 1, 1)
    y2 = vk.array((1, 20, 63, 49))    # direct (padded)
    L.kernel_conv2d(_p(vk, x), _p(vk, w), _p(vk, bias), _p(vk, y2), 1, 3, 64, 48, 20, 16, 16, 1, 1, 7, 8, 1, 1)
    vk.sync()
    assert _rel(c, np.einsum("bmk,bkn->bmn", a.astype(np.float64), b)) < 1e-5
    tx, tw, tb = (torch.from_numpy(np.array(t)).double() for t in (x, w, bias))
    assert _rel(y, torch.nn.functional.conv2d(tx, tw, tb, stride=16).numpy()) < 1e-5
    ref2 = torch.nn.functional.conv2d(torch.nn.functional.pad(tx, (8, 8, 7, 7)), tw, tb)
    assert _rel(y2, ref2.numpy()) < 1e-5


def test_row_ops(vk):
    rng = np.random.default_rng(2)
    L = vk.lib
    x = vk.upload(rng.standard_normal((5, 1000, 3), dtype=np.float32) * 4)
    t = torch.from_numpy(np.array(x)).double()
    sm, lsm, rms, l2 = (vk.array(x.shape) for _ in range(4))
    L.kernel_softmax_dim(_p(vk, x), _p(vk, sm), 5, 1000, 3, 0)
    L.kernel_softmax_dim(_p(vk, x), _p(vk, lsm), 5, 1000, 3, 1)
    L.kernel_rms_norm(_p(vk, x), _p(vk, rms), 5, 1000, 3, ctypes.c_float(1e-5))
    L.kernel_l2_norm(_p(vk, x), _p(vk, l2), 5, 1000, 3)
    red = {opc: vk.array((5, 3)) for opc in range(11)}
    for opc, o in red.items():
        L.kernel_reduce(_p(vk, x), _p(vk, o), 5, 1000, 3, opc)
    rows = vk.upload(rng.standard_normal((7, 300), dtype=np.float32) * 3 + 2)
    g, bt = vk.upload(rng.standard_normal(300, dtype=np.float32)), vk.upload(rng.standard_normal(300, dtype=np.float32))
    ln = vk.array((7, 300))
    L.kernel_layer_norm(_p(vk, rows), _p(vk, g), _p(vk, bt), _p(vk, ln), 7, 300, ctypes.c_float(1e-5))
    vk.sync()
    assert _rel(sm, torch.softmax(t, 1).numpy()) < 1e-5
    assert _rel(lsm, torch.log_softmax(t, 1).numpy()) < 1e-5
    assert _rel(rms, (t / torch.sqrt((t * t).mean(1, keepdim=True) + 1e-5)).numpy()) < 1e-5
    assert _rel(l2, (t / t.norm(dim=1, keepdim=True)).numpy()) < 1e-5
    refs = {0: t.sum(1), 1: t.mean(1), 2: t.amax(1), 3: t.amin(1), 5: (t * t).sum(1), 6: t.norm(dim=1), 7: t.argmax(1).double(), 8: t.argmin(1).double(),
            9: t.var(1, unbiased=False), 10: t.var(1)}
    for opc, ref in refs.items():                         # (4, prod of 1000 normals: no finite reference)
        assert _rel(red[opc], ref.numpy()) < 1e-5, opc
    r = torch.from_numpy(np.array(rows)).double()
    assert _rel(ln, torch.nn.functional.layer_norm(r, (300,), torch.from_numpy(np.array(g)).double(),
                                                   torch.from_numpy(np.array(bt)).double(), 1e-5).numpy()) < 1e-5


def test_capture_replay(vk):
    L = vk.lib
    x = vk.upload(np.arange(1000, dtype=np.float32) - 500)
    y = vk.array((1000,))
    z = vk.array((1000,))
    z[:] = 0                                              # (arena memory is not cleared)
    g = L.mb_vk_capture_begin()
    L.kernel_relu(_p(vk, x), _p(vk, y), 1000)
    L.kernel_mul_scalar(_p(vk, y), _p(vk, z), 1000, ctypes.c_float(2.0))
    assert L.mb_vk_capture_end() == g and L.mb_vk_graph_cbs(g) >= 1
    vk.check()
    assert not z.any()                                    # recorded, not run
    vk.replay(g)
    assert np.array_equal(z, 2 * np.maximum(x, 0))
    x[:] = -x
    vk.replay(g)
    assert np.array_equal(z, 2 * np.maximum(x, 0))


_BENCHES = {
    "cnn": '''
import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(s):
        super().__init__(); torch.manual_seed(0)
        s.c1 = nn.Conv2d(3, 8, 3, padding=1); s.bn = nn.BatchNorm2d(8); s.c2 = nn.Conv2d(8, 8, 3, stride=2, padding=1, groups=4)
        s.dw = nn.Conv2d(8, 8, 3, padding=1, groups=8); s.c3 = nn.Conv2d(16, 8, 1); s.fc = nn.Linear(8, 5)
        s.bn.eval(); s.bn.running_mean.uniform_(-0.5, 0.5); s.bn.running_var.uniform_(0.5, 2.0)
    def forward(s, x):
        y = F.max_pool2d(F.relu(s.bn(s.c1(x))), 2)
        z = F.elu(s.c2(y))
        z = F.interpolate(F.hardtanh(s.dw(z), -0.5, 0.5) + F.leaky_relu(z, 0.1), scale_factor=2, mode="nearest")
        u = F.avg_pool2d(s.c3(torch.cat([y, torch.sigmoid(z)], dim=1)), 2)
        return torch.tanh(s.fc(F.adaptive_avg_pool2d(F.softplus(u), 1).flatten(1)))
def get_inputs(): return [torch.randn(2, 3, 16, 16)]
def get_init_inputs(): return []
''',
    "elementwise": '''
import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def forward(s, x):
        a = x.sum(dim=1) + x.mean(dim=1) + x.amax(dim=1) + x.amin(dim=1)
        b = F.softmax(x, dim=1).sum(dim=2) + F.log_softmax(x, dim=-1).mean(dim=-1)
        c = x.var(dim=2) + x.std(dim=2, unbiased=False) + x.norm(dim=2)
        d = F.normalize(x, dim=1).abs().sqrt().exp().log() + x.clamp(-0.3, 0.3) - x.square()
        e = torch.where(x > 0, x, x * 0.5) / (1.0 + x.abs()) + torch.maximum(x, x.flip(1)) + x.pow(2.0)
        f = (F.gelu(x) + F.gelu(x, approximate="tanh") + F.silu(x) + F.mish(x) + F.selu(x) + F.hardswish(x)
             + F.hardsigmoid(x) + torch.erf(x) + torch.rsqrt(x.abs() + 1) + torch.reciprocal(x.abs() + 1)
             + torch.floor(x) + torch.ceil(x) + torch.sign(x) + torch.expm1(x * 0.1) + torch.log1p(x.abs()))
        return torch.cat([a.flatten(), b.flatten(), c.flatten(), d.flatten(), e.flatten(), f.flatten()])
def get_inputs(): return [torch.randn(3, 7, 33)]
def get_init_inputs(): return []
''',
    "matmuls": '''
import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(s):
        super().__init__(); torch.manual_seed(0)
        s.w = nn.Parameter(torch.randn(40, 24) * 0.2)
    def forward(s, x, y):
        a = x @ y
        b = y.t() @ x.t()
        c = torch.bmm(a.reshape(4, 8, 40), a.reshape(4, 8, 40).transpose(1, 2))
        d = torch.triu(a[:, :32]) + torch.tril(a[:, :32], -1) * 2
        e = F.linear(x, s.w) @ s.w
        q = a[:, :16].reshape(1, 2, 32, 8); k = b[:16].t().reshape(1, 2, 32, 8)
        f = F.scaled_dot_product_attention(q, k, q)
        return torch.cat([a.flatten(), b.flatten(), c.flatten(), d.flatten(), e.flatten(), f.flatten()])
def get_inputs(): return [torch.randn(32, 24), torch.randn(24, 40)]
def get_init_inputs(): return []
''',
    "norms": '''
import torch, torch.nn as nn, torch.nn.functional as F
class Model(nn.Module):
    def __init__(s):
        super().__init__(); torch.manual_seed(0)
        s.ln = nn.LayerNorm(300); s.gn = nn.GroupNorm(2, 6)
    def forward(s, x):
        y = s.ln(x) + F.rms_norm(x, (300,), eps=1e-5)
        return torch.cat([y.flatten(), s.gn(x.reshape(1, 6, 10, 30)).flatten(), torch.cumsum(x, dim=1).flatten()])
def get_inputs(): return [torch.randn(6, 300) * 3 + 1]
def get_init_inputs(): return []
''',
}


@pytest.fixture(scope="module")
def specs():
    return {op: {"signature": s.signature} for op, s in KERNEL_SPECS.items()}


@pytest.mark.parametrize("bench", sorted(_BENCHES))
def test_ir_walk_on_vulkan(vk, specs, bench, tmp_path):
    from modelblaster.pipeline.extract_graph import _load_kernelbench, extract
    import walk_vulkan
    f = tmp_path / f"{bench}.py"
    f.write_text(_BENCHES[bench])
    model, sample, name = _load_kernelbench(str(f))
    extract(model, sample, name=name, out_dir=str(tmp_path / "ir"), quant="fp32")
    rec = walk_vulkan.evaluate(str(tmp_path / "ir"), walk_vulkan.VkKernelSet(specs, vk), reps=1)
    assert rec["missing"] == [], rec
    assert rec["ok"], {k: rec.get(k) for k in ("rel_to_range", "first_divergent_op", "intermediate_rel_err_by_op")}


def test_smolvla_split_on_vulkan(vk, specs, split_irs):
    import walk_vulkan
    from smolvla_vulkan import SmolVLAVulkan
    tmp, S, c, parts, x = split_irs
    ks = walk_vulkan.VkKernelSet(specs, vk)
    for part in ("vision", "prefix", "step"):
        rec = walk_vulkan.evaluate(str(tmp / part), ks, reps=1)
        assert rec["ok"] and not rec["missing"], (part, rec)
    m = vk.mark()
    model = SmolVLAVulkan(tmp, vk, specs)
    out = model.sample_actions(x["pixels"].numpy(), x["lang"].numpy(), x["state"].numpy(), x["noise"].numpy(), 3)
    vk.release(m)
    ref = parts.sample_actions(x["pixels"], x["lang"], x["state"], x["noise"].clone(), 3).numpy()
    assert _rel(out, ref) < 1e-4


# ------------------------------------------------------------------ int8 (`_s8`): bit-exact vs the reference C
@pytest.fixture(scope="module")
def refc():
    import walk_vulkan
    if not shutil.which(os.environ.get("CC", "cc")) and not os.environ.get("MB_REF_LIB"):
        pytest.skip("no C compiler for the reference kernels (or $MB_REF_LIB)")
    specs = {op: {"signature": s.signature, "reference_impl": s.reference_impl} for op, s in KERNEL_SPECS.items()}
    ops = ["linear_s8", "matmul_s8", "softmax_s8", "layernorm_s8", "rmsnorm_s8", "gelu_s8", "sin_s8", "cos_s8", "mul_s8"]
    return walk_vulkan.RefC(specs, ops, os.environ.get("MB_REF_LIB"))


def _s8_case(op, rng):
    """(scalar/pointer args in signature order with numpy arrays for pointers, output index) at SmolVLA-like sizes."""
    i8 = lambda *s: rng.integers(-128, 128, size=s, dtype=np.int8)
    f = np.float32
    if op == "linear_s8":
        M, K, N = 50, 960, 320
        return [i8(M, K), i8(N, K), rng.integers(-2**20, 2**20, N, dtype=np.int32), np.zeros((M, N), np.int8),
                M, K, N, 3, -2, 5, 1394847094, 9, -128, 127]
    if op == "matmul_s8":
        M, K, N = 45, 64, 81
        return [i8(M, K), i8(N, K), np.zeros((M, N), np.int8), M, K, N, f(0.011), f(0.0093), f(0.0021), 1, f(8.0), -128, 127]
    if op == "softmax_s8":
        return [i8(50, 300), np.zeros((50, 300), np.int8), 50, 300, f(0.031), f(1 / 127)]
    if op in ("layernorm_s8", "rmsnorm_s8"):
        M, K = 31, 960
        g, b = rng.standard_normal(K).astype(f), rng.standard_normal(K).astype(f)
        args = [i8(M, K), g] + ([b] if op == "layernorm_s8" else []) + [np.zeros((M, K), np.int8), M, K, f(0.05), f(0.03), f(1e-5), -128, 127]
        return args
    if op == "mul_s8":
        return [i8(4096), i8(4096), np.zeros(4096, np.int8), 4096, f(0.0078), f(0.0121), f(0.0049), -128, 127]
    return [i8(4096), np.zeros(4096, np.int8), 4096, f(0.023), f(0.0071), -128, 127]     # gelu / sin / cos


@pytest.mark.parametrize("op", ["linear_s8", "matmul_s8", "softmax_s8", "layernorm_s8", "rmsnorm_s8", "gelu_s8",
                                "sin_s8", "cos_s8", "mul_s8"])
def test_s8_kernels_bit_exact(vk, refc, op):
    rng = np.random.default_rng(abs(hash(op)) % 2**32)
    _run_s8(vk, refc, op, _s8_case(op, rng))


def _s8_dot_cases():
    """linear_s8 / matmul_s8 shapes for the dot4 path (K % 4 == 0) and its scalar fallbacks."""
    rng = np.random.default_rng(7)
    i8 = lambda *s: rng.integers(-128, 128, size=s, dtype=np.int8)
    out = []
    for (M, K, N, io, fo) in [(31, 960, 2560, 128, 0), (50, 720, 320, -3, 7), (5, 36, 9, 0, 0), (3, 37, 5, 11, -4),
                              (1, 32, 960, 128, -128)]:
        out.append(("linear_s8", [i8(M, K), i8(N, K), rng.integers(-2**20, 2**20, N, dtype=np.int32),
                                  np.zeros((M, N), np.int8), M, K, N, io, fo, -6, 1518500250, 10, -128, 127]))
    for (M, K, N, tb) in [(75, 64, 81, 1), (30, 31, 64, 1), (45, 64, 81, 0)]:
        b = i8(N, K) if tb else i8(K, N)
        out.append(("matmul_s8", [i8(M, K), b, np.zeros((M, N), np.int8), M, K, N, np.float32(0.011),
                                  np.float32(0.0093), np.float32(0.0021), tb, np.float32(8.0), -128, 127]))
    return out


@pytest.mark.parametrize("case", range(len(_s8_dot_cases())))
def test_s8_dot4_bit_exact(vk, refc, case):
    op, args = _s8_dot_cases()[case]
    _run_s8(vk, refc, op, args)


def _run_s8(vk, refc, op, args):
    from mbabi import parse_signature as psig
    name, params = psig(KERNEL_SPECS[op].signature)
    gpu_args, cpu_args, outs, keep = [], [], [], []
    for p, a in zip(params, args):
        if isinstance(a, np.ndarray):
            d, c = vk.upload(a), a.copy()
            keep += [d, c]                       # the ctypes args hold bare addresses
            gpu_args.append(ctypes.c_void_p(vk.ptr(d)))
            cpu_args.append(ctypes.c_void_p(c.ctypes.data))
            if not p.is_const_ptr:
                outs.append((d, c))
        else:
            gpu_args.append(a)
            cpu_args.append(a)
    fg = getattr(vk.lib, name)
    fg.argtypes, fg.restype = [p.ctypes_type() for p in params], None
    fc, _ = refc.fn(op)
    m = vk.mark()
    fg(*gpu_args)
    vk.sync()
    fc(*cpu_args)
    for d, c in outs:
        assert np.array_equal(np.array(d), c), f"{op}: {(np.array(d) != c).sum()} of {c.size} differ"
    vk.release(m)


@pytest.mark.parametrize("model", ["attn_block", "norm_block"])
def test_s8_ir_walk_bit_exact(vk, refc, specs, model, tmp_path):
    """MB's int8 SmolVLA proxies (attention + RoPE, norms + GELU) from the FX int8 path."""
    import importlib
    from modelblaster.pipeline.extract_graph import extract
    import walk_vulkan
    mod = importlib.import_module(f"modelblaster.models.{model}")
    extract(mod.get_model(), mod.get_sample_input(), name=model, out_dir=str(tmp_path / "ir"), quant="int8")
    full = {op: {"signature": s.signature, "reference_impl": s.reference_impl} for op, s in KERNEL_SPECS.items()}
    ks = walk_vulkan.VkKernelSet(full, vk)
    rec = walk_vulkan.evaluate(str(tmp_path / "ir"), ks, reps=1, ref_lib=str(refc.lib._name))
    assert rec["ok"] and rec["golden_mismatches"] == 0 and rec["n_ops_not_bit_exact"] == 0, rec
