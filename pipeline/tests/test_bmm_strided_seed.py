"""bmm_strided_f16 seed kernel + fold wiring (X-142): text-level checks that need no GPU or torch; the GPU
correctness matrix lives in l3_farm/analysis/layout_stage1_data/kernel_test.py (run on a node)."""
import copy
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
HAND = ROOT / "gpu_eval" / "harness" / "cuda" / "mb_cuda_kernels_f16_hand.cu"

try:
    from modelblaster.pipeline import kb_proposals as kp, layout_fold as lf
    from modelblaster.pipeline.kb_export import CudaLibrary
except ImportError:  # checkout without the modelblaster alias
    sys.path.insert(0, str(ROOT.parent))
    from ModelBlaster.pipeline import kb_proposals as kp, layout_fold as lf  # noqa: E402
    from ModelBlaster.pipeline.kb_export import CudaLibrary  # noqa: E402
import test_layout_fold as tl  # noqa: E402


def _attn_with_scale(B=2, T=3, nh=2, hs=4):
    """qkv split -> heads -> k^T -> q@k^T -> mul_scalar -> softmax-free tail: bmm followed by an elementwise stage."""
    C = nh * hs
    g = tl.G(); qkv = g.t([B, T, 3 * C], "qkv")
    q, k, v = (g.view(g.slice_last(qkv, i * C, (i + 1) * C), [B, T, nh, hs]) for i in range(3))
    q, k = (g.permute(t, [0, 2, 1, 3]) for t in (q, k))
    kT = g.permute(k, [0, 1, 3, 2])
    att = g.bmm(q, kT, B * nh, T, hs, T)
    sc = g.t([B * nh, T, T], "sc")
    g.ops.append({"name": "scale", "op": "mul_scalar_f16", "inputs": [att], "outputs": [sc], "scalar": 0.5,
                  "shape": {"n": B * nh * T * T}})
    return g.graph("qkv", sc)


def test_slot_source_is_self_contained():
    """The slot of kernel_bmm_strided_f16 = the entry + its kernels + helper; no macro / include of its own, static smem only."""
    lib = CudaLibrary(HAND.read_text())
    src = lib.slot_source("kernel_bmm_strided_f16")
    for name in ("k_bmm_strided", "k_bmm_strided_scalar", "mbs_load8", "kernel_bmm_strided_f16"):
        assert name in lib.items and lib.items[name].text in src, name      # (an attribute before the name hides an item)
    assert "#define" not in src and "#include" not in src
    assert "extern __shared__" not in src and "cudaStream" not in src and "cudaDeviceSynchronize" not in src
    assert "mma.sync.aligned.m16n8k16" in src and "ldmatrix" in src


def test_find_chains_matches_strided_head_after_fold():
    """layout_packing.md 5.2 to-do 2: fusion proposed AFTER a fold finds bmm_strided_f16 heads (generic find_chains)."""
    specs = kp._all_specs()
    g2, rep = lf.fold(_attn_with_scale())
    assert [o["op"] for o in g2["ops"] if "bmm" in o["op"]] == ["bmm_strided_f16"]
    kind = "fused__bmm_strided_f16__mul_scalar_f16"
    chains = kp.find_chains(g2, kind, specs)
    assert len(chains) == 1 and len(chains[0]) == 2
    g3, spec = kp.fuse_chain(g2, chains[0], specs)
    assert spec.op == kind and g3["ops"][-1]["sub_ops"][0]["op"] == "bmm_strided_f16"
    # the reverse order (fuse, then fold) yields the same fused kind and stage-0 shape
    g0 = _attn_with_scale()
    g4, _ = kp.fuse_chain(g0, [next(o["name"] for o in g0["ops"] if o["op"] == "bmm_f16"), "scale"], specs)
    g5, rep5 = lf.fold(g4)
    f = next(o for o in g5["ops"] if "sub_ops" in o)
    assert f["op"] == kind
    assert f["sub_ops"][0]["shape"] == g3["ops"][-1]["sub_ops"][0]["shape"]


def test_fused_head_shape_tracks_folded_strides():
    """A folded fused head's op["shape"]["a"] is the strided shape (it is the instance's shape key for guards / reuse)."""
    specs = kp._all_specs()
    g = _attn_with_scale()
    bm = next(o["name"] for o in g["ops"] if o["op"] == "bmm_f16")
    g4, _ = kp.fuse_chain(g, [bm, "scale"], specs)
    g5, _ = lf.fold(g4)
    f = next(o for o in g5["ops"] if "sub_ops" in o)
    assert f["shape"]["a"] == f["sub_ops"][0]["shape"] and "a_sm" in f["shape"]["a"] and "batch" not in f["shape"]["a"]


if __name__ == "__main__":
    for fn in (test_slot_source_is_self_contained, test_find_chains_matches_strided_head_after_fold,
               test_fused_head_shape_tracks_folded_strides):
        fn(); print("ok", fn.__name__)
