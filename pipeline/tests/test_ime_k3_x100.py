"""K3 X100 IME backend (ime_k3_x100) and the build-time conv weight tiles.

Pinned here:
  * ime_k3_x100 is rvv_k3 plus kernels/ime_k3_x100/: same zvl256b flags (legal on
    the X100), lineage ime_k3_x100 -> rvv_k3 -> rvv, IHWOC conv weights;
  * the curated probe finds our conv / linear / maxpool for it, and the conv entry
    is registered with the IHWOC layout (so the layout contract holds);
  * the tiles are byte-for-byte what the X100 conv packs per call from IHWOC
    (port of its packing loop), behind the 64-byte "MBX100TW" header;
  * which weights are tiled (ime_k3_x100 conv2d_s8 only; never split, sharded,
    schedule-sharded, shared, asymmetric or tiny), MODELBLASTER_IME2_TILES=0 turns
    it off, and every other backend's weights.c is unchanged;
  * a tiled weight is never OC-sliced by the IHWOC pool wrapper;
  * generate_kernels refuses a kernel that does not read the tiles.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from modelblaster.pipeline import generate_skeleton as gs
from modelblaster.pipeline import k3_weight_tiles as kt
from modelblaster.pipeline.backends import BACKENDS, backend_lineage

RNG = np.random.default_rng(7)


def test_backend_is_rvv_k3_plus_x100_kernels():
    x, k3 = BACKENDS["ime_k3_x100"], BACKENDS["rvv_k3"]
    assert x.kernel_cflags == k3.kernel_cflags
    assert any(f == "-march=rv64gcv_zvl256b_zfh_zvfh" for f in x.kernel_cflags)
    assert not any("zvl1024b" in f for f in x.kernel_cflags)   # must be legal on the X100
    assert x.curated_aliases == ("rvv_k3", "rvv")
    assert backend_lineage("ime_k3_x100") == ("ime_k3_x100", "rvv_k3", "rvv")
    assert x.verify_method == k3.verify_method


def test_conv_weights_ihwoc_like_rvv_k3():
    assert gs._conv_weight_layout_for_op("conv2d_s8", "ime_k3_x100") == "ihwoc"
    assert gs._conv_weight_layout_for_op("conv2d_s8", "rvv_k3") == "ihwoc"


@pytest.mark.parametrize("op,algo", [
    ("conv2d_s8", "x100_vmadot_4x4x8"),
    ("maxpool2d_s8", "direct"),
])
def test_curated_probe_finds_our_kernels(op, algo):
    path = kt.selected_kernel_path("ime_k3_x100", op)
    assert path is not None and path.endswith(f"ime_k3_x100_{op}_{algo}.c")
    src = open(path).read()
    assert f"/* algorithm: {algo} */" in src


def test_no_x100_linear():
    # Measured slower than the rvv linear on every (batch-1) linear in mlp_control and
    # dronet (k3-work TASK-015), so linear_s8 falls through to rvv_k3 / rvv.
    assert kt.selected_kernel_path("ime_k3_x100", "linear_s8") is None


def test_conv_entry_declares_ihwoc():
    from modelblaster.pipeline.reference_kernels import KERNEL_SPECS
    algo = next(a for a in KERNEL_SPECS["conv2d_s8"].algorithms if a.name == "x100_vmadot_4x4x8")
    assert algo.target_affinity == ("ime_k3_x100",)
    assert algo.weight_layout == "ihwoc"


def kernel_x100_tiles(w_oihw: np.ndarray) -> np.ndarray:
    """Port of the X100 conv's per-call B packing from IHWOC weight[k*OC + oc]:
    packed_b[ks*(n_panels*32) + oc*8 + q], k = 8ks + q, zero padded."""
    oc_n, ic_n, kh_n, kw_n = w_oihw.shape
    src = np.transpose(w_oihw, (1, 2, 3, 0)).reshape(-1)   # [k][oc], what rvv_k3 emits
    k = ic_n * kh_n * kw_n
    ks_n, panels = (k + 7) // 8, (oc_n + 3) // 4
    out = np.zeros(ks_n * panels * 32, np.int8)
    for ks in range(ks_n):
        for oc in range(oc_n):
            for q in range(8):
                kk = ks * 8 + q
                if kk < k:
                    out[ks * panels * 32 + oc * 8 + q] = src[kk * oc_n + oc]
    return out


def header(rows: int, k: int, magic: bytes) -> bytes:
    return magic + np.array([rows, k], "<u4").tobytes() + bytes(48)


@pytest.mark.parametrize("oc,ic,kh,kw", [(32, 3, 3, 3), (64, 32, 1, 1), (80, 80, 3, 3),
                                         (6, 5, 3, 3), (2, 2, 1, 1)])
def test_tiles_are_what_the_kernel_packs(oc, ic, kh, kw):
    w = RNG.integers(-128, 128, (oc, ic, kh, kw), dtype=np.int8)
    got = kt.tile_pack(w, "x100_conv", oc, ic * kh * kw)
    assert got[:64].tobytes() == header(oc, ic * kh * kw, b"MBX100TW")
    np.testing.assert_array_equal(got[64:], kernel_x100_tiles(w))


def _conv(did, w, oc=32, ic=16, **extra):
    return {"op": "conv2d_s8", "dispatch_id": did, "name": f"c{did}", "weight": w,
            "shape": {"N": 1, "IC": ic, "IH": 8, "IW": 8, "OC": oc, "KH": 3, "KW": 3,
                      "SH": 1, "SW": 1, "PH": 1, "PW": 1}, **extra}


def _ir():
    return {"name": "m", "ops": [
        _conv(0, "c.w"),                                                  # tiled
        _conv(1, "split.w", split_from={"axis": "OC", "tile": 0}),        # split: plain
        _conv(2, "sched.w", _xpurt_schedule_shard_factor=4),              # composite: plain
        _conv(3, "shared.w"), _conv(4, "shared.w"),                       # two readers: plain
        _conv(5, "asym.w", quant={"input_offset": -3, "filter_offset": 0}),   # asymmetric: plain
        _conv(6, "small.w", oc=1, ic=1),                                  # 9 bytes: plain
        {"op": "linear_s8", "dispatch_id": 7, "name": "l7", "weight": "fc.w",
         "shape": {"M": 64, "K": 64, "N": 64}},                           # X100: convs only
    ]}


def test_plan():
    assert kt.tile_plan(_ir(), "ime_k3_x100") == {"c.w": ("conv2d_s8", "x100_conv", 32, 144)}


def test_plan_other_backends_and_off_switch(monkeypatch):
    for be in ("rvv_k3", "rvv_k3_a100", "rvv_x60", "ime_x60", None):
        assert kt.tile_plan(_ir(), be) == {}
    # Sam's A100 conv packs IHWOC itself: no tiles until a tile-reading kernel is selected.
    assert kt.tile_plan(_ir(), "ime_k3_a100") == {}
    monkeypatch.setenv("MODELBLASTER_IME2_TILES", "0")
    assert kt.tile_plan(_ir(), "ime_k3_x100") == {}


def _weights():
    rng = np.random.default_rng(3)
    names = ["c.w", "split.w", "sched.w", "shared.w", "asym.w", "small.w"]
    ws = {n: rng.integers(-128, 128, (32, 16, 3, 3), dtype=np.int8) for n in names}
    ws["small.w"] = rng.integers(-128, 128, (1, 1, 3, 3), dtype=np.int8)
    ws["fc.w"] = rng.integers(-128, 128, (64, 64), dtype=np.int8)
    return ws


def test_emit_weights_tiles_and_manifest(tmp_path):
    gs.emit_weights("m", _weights(), str(tmp_path), "ime_k3_x100", _ir())
    c = (tmp_path / "weights.c").read_text()
    h = (tmp_path / "weights.h").read_text()
    n_tiled = 64 + ((144 + 7) // 8) * (32 // 4) * 32
    assert f"[{n_tiled}];" in h
    assert "__attribute__((aligned(64))) const int8_t" in c
    assert c.count("MBX100TW header") == 1
    man = json.loads((tmp_path / kt.WEIGHT_LAYOUTS_JSON).read_text())["k3_tiles"]
    assert man["op_magic"] == {"conv2d_s8": ["MBX100TW"]}
    assert list(man["weights"]) == ["c.w"]


def test_rvv_k3_weights_unchanged_and_off_switch(tmp_path, monkeypatch):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()
    gs.emit_weights("m", _weights(), str(a), "rvv_k3", _ir())
    monkeypatch.setenv("MODELBLASTER_IME2_TILES", "0")
    gs.emit_weights("m", _weights(), str(b), "ime_k3_x100", _ir())
    gs.emit_weights("m", _weights(), str(c), "rvv_k3", _ir())
    assert not (a / kt.WEIGHT_LAYOUTS_JSON).exists()
    assert not (b / kt.WEIGHT_LAYOUTS_JSON).exists()
    # Untiled ime_k3_x100 weights are rvv_k3's IHWOC weights (names carry the backend).
    strip = lambda p, be: (p / "weights.c").read_text().replace(be, "BE")
    assert strip(b, "ime_k3_x100") == strip(c, "rvv_k3") == strip(a, "rvv_k3")


def test_stale_manifest_removed(tmp_path, monkeypatch):
    gs.emit_weights("m", _weights(), str(tmp_path), "ime_k3_x100", _ir())
    assert (tmp_path / kt.WEIGHT_LAYOUTS_JSON).exists()
    monkeypatch.setenv("MODELBLASTER_IME2_TILES", "0")
    gs.emit_weights("m", _weights(), str(tmp_path), "ime_k3_x100", _ir())
    assert not (tmp_path / kt.WEIGHT_LAYOUTS_JSON).exists()


def test_kernel_check_refuses_kernel_without_header(tmp_path):
    gs.emit_weights("m", _weights(), str(tmp_path), "ime_k3_x100", _ir())
    ours = open(kt.selected_kernel_path("ime_k3_x100", "conv2d_s8")).read()
    kt.check_kernels_read_tiles({"conv2d_s8": ours}, str(tmp_path))
    with pytest.raises(SystemExit, match="MBX100TW"):
        kt.check_kernels_read_tiles({"conv2d_s8": "void kernel_conv2d_s8(void) {}"}, str(tmp_path))


def test_ihwoc_pool_wrapper_never_slices_tiles():
    src = gs._conv2d_s8_ihwoc_wrapper()
    assert 'memcmp(w, "MBX100TW", 8) == 0' in src
    assert 'memcmp(w, "MBIME2TW", 8) == 0' in src
