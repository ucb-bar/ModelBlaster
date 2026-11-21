"""K3 IME per-channel kernels (conv2d_s8_pc, linear_s8_pc; k3-work TASK-019).

Host checks only: registration, weight layout, kernel selection, weight tiles, and that a
per-channel conv is never planned for per-shard weights (it has no pool entry point). The
kernels themselves are checked on the board (k3-work experiments/vint-pc-kernels-20261117).
"""
from __future__ import annotations

import pytest

from modelblaster.pipeline import generate_skeleton as gs
from modelblaster.pipeline import k3_weight_tiles as kt
from modelblaster.pipeline.reference_kernels import KERNEL_SPECS

PC_KERNELS = [
    ("ime_k3_x100", "conv2d_s8_pc", "x100_vmadot_4x4x8", "ihwoc"),
    ("ime_k3_a100", "conv2d_s8_pc", "ime_vmadot_8x16x8", "ihwoc"),
    ("ime_k3_a100", "linear_s8_pc", "ime_vmadot_8x16x8", "oihw"),
]


@pytest.mark.parametrize("backend,op,algo,layout", PC_KERNELS)
def test_registered_for_its_backend_only(backend, op, algo, layout):
    cands = [a for a in KERNEL_SPECS[op].algorithms if a.target_affinity == (backend,)]
    assert [a.name for a in cands] == [algo]
    assert cands[0].weight_layout == layout


@pytest.mark.parametrize("backend,op,algo,layout", PC_KERNELS)
def test_selected_and_signature(backend, op, algo, layout):
    path = kt.selected_kernel_path(backend, op)
    assert path is not None and path.endswith(f"{backend}_{op}_{algo}.c")
    src = open(path).read()
    assert f"/* algorithm: {algo} */" in src
    assert f"void kernel_{op}(" in src


def test_conv_weights_packed_ihwoc_on_k3_ime_only():
    assert gs._conv_weight_layout_for_op("conv2d_s8_pc", "ime_k3_x100") == "ihwoc"
    assert gs._conv_weight_layout_for_op("conv2d_s8_pc", "ime_k3_a100") == "ihwoc"
    for be in ("rvv_k3", "rvv_k3_a100", "rvv_x60", "ime_x60"):
        assert gs._conv_weight_layout_for_op("conv2d_s8_pc", be) is None   # OIHW, as before


def test_no_x100_linear_pc():
    # On the X100 linear_s8_pc runs on the generic rvv kernel, as linear_s8 does.
    assert kt.selected_kernel_path("ime_k3_x100", "linear_s8_pc") is None


def test_x100_pc_conv_reads_the_tiles():
    assert kt.TILE_OPS["ime_k3_x100"]["conv2d_s8_pc"] == "x100_conv"
    assert kt._kernel_reads("ime_k3_x100", "conv2d_s8_pc", kt.TILE_MAGIC["x100_conv"])
    assert "conv2d_s8_pc" not in kt.TILE_OPS.get("ime_k3_a100", {})


def _pc_conv_ir(shard_factor):
    return {"ops": [{
        "name": "c0", "op": "conv2d_s8_pc", "dispatch_id": 0, "weight": "c0.w", "bias": "c0.b",
        "shard_factor": shard_factor,
        "shape": {"N": 1, "IC": 16, "IH": 8, "IW": 8, "OC": 32, "OH": 8, "OW": 8,
                  "KH": 3, "KW": 3, "SH": 1, "SW": 1, "PH": 1, "PW": 1},
        "quant": {"input_offset": 0, "filter_offset": 0, "output_offset": 0,
                  "output_multiplier_per_oc_key": "c0.m", "output_shift_per_oc_key": "c0.s",
                  "activation_min": -128, "activation_max": 127}}]}


@pytest.mark.parametrize("backend", ["ime_k3_x100", "ime_k3_a100", "rvv_k3", "rvv_x60"])
def test_pc_conv_never_planned_for_shard_weights(backend):
    # Its dispatch is one plain kernel call that names the full weight; per-shard arrays
    # would drop that array (the build broke at MB_SHARD_FACTOR > 1 before TASK-019).
    assert gs.shard_conv_weights(_pc_conv_ir(4), backend, 4) == {}
    assert "conv2d_s8_pc" not in gs.shareable_ops()
