"""Host bit-exact end-to-end check of the split axes (OC, C, E, OH).

Builds the unsplit and the split K3 IR with the scalar reference kernels on the
BUILD HOST (native cc, `--platform host`), runs both on the model's test input
and requires the split op's output tensor to be element-for-element identical
and the golden error to be unchanged. Driven through
`scripts/verify_ir_rewrite_host.py`, the same gate XPU-RT's rewrite_arm uses.

Skips when the K3 build IRs (build/k3/<m>/int8) or a host `cc`/`make` are absent.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
VERIFY = REPO / "scripts" / "verify_ir_rewrite_host.py"


def _ir_dir(model: str) -> Path:
    d = REPO / "build" / "k3" / model / "int8"
    for f in ("graph.json", "weights.npz", "io.npz"):
        if not (d / f).exists():
            pytest.skip(f"{d / f} missing (K3 build not present)")
    if not (shutil.which("cc") and shutil.which("make")):
        pytest.skip("host cc/make unavailable")
    return d


def _run(tmp_path, model, split_ops, tensor, allow_dirty_baseline=False):
    d = _ir_dir(model)
    hint = tmp_path / "hint.json"
    hint.write_text(json.dumps({
        "contract": "modelblaster.split_hints/v1",
        "networks": [{"network": model, "split_ops": split_ops}]}))
    split_ir = tmp_path / "graph.json"
    env = {"PYTHONPATH": f"{REPO / 'src'}:{REPO}", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run([sys.executable, "-m", "modelblaster.pipeline.apply_split_hint",
                    "--hint", str(hint), "--model", model,
                    "--ir", str(d / "graph.json"), "--out", str(split_ir)],
                   check=True, env=env, capture_output=True)
    r = subprocess.run(
        [sys.executable, str(VERIFY), "--baseline-ir", str(d / "graph.json"),
         "--rewritten-ir", str(split_ir), "--weights", str(d / "weights.npz"),
         "--io", str(d / "io.npz"), "--tensor", tensor,
         "--work", str(tmp_path / "work"), "--json", str(tmp_path / "res.json")],
        env=env, capture_output=True, text=True)
    res = json.loads((tmp_path / "res.json").read_text())
    # exit 3 == rewrite faithful but the BASELINE already disagrees with its
    # golden (yolov8_nano on the host scalar path: a property of the build).
    assert r.returncode in ((0, 3) if allow_dirty_baseline else (0,)), r.stdout[-800:]
    assert res["rewrite_faithful"], res
    assert res["max_abs_diff"] == 0 and res["n_differing"] == 0
    return res


def test_dronet_all_axes(tmp_path):
    """OC (d0), C (maxpool d1), OH (conv d4, uneven d5), E (add d6, relu d17)."""
    res = _run(tmp_path, "dronet", [
        {"op": 0, "n_splits": 2},
        {"op": 1, "axis": "C", "n_splits": 2},
        {"op": 4, "axis": "OH", "n_splits": 2},
        {"op": 5, "axis": "OH", "tile_sizes": [6, 8]},
        {"op": 6, "n_splits": 2},
        {"op": 15, "n_splits": 2},
        {"op": 17, "n_splits": 2}], "relu_modules_6")
    assert res["bit_exact"]


@pytest.mark.parametrize("axis_ops,tensor", [
    ([{"op": 0, "axis": "OH", "n_splits": 2}], "conv_modules_0"),     # OH, stem
    ([{"op": 0, "tile_sizes": [16, 8, 8]}], "conv_modules_0"),        # uneven OC
    ([{"op": 1, "axis": "C", "tile_sizes": [16, 10, 6]}], "maxpool1"),  # uneven C
    ([{"op": 6, "tile_sizes": [1000, 2272, 3000]}], "add"),           # uneven E
])
def test_dronet_single_axis(tmp_path, axis_ops, tensor):
    assert _run(tmp_path, "dronet", axis_ops, tensor)["bit_exact"]


def test_yolov8_nano_oc_and_e(tmp_path):
    """Fused conv OC split with tile_oc a multiple of 16 (d9: OC=64), plus E
    on an add_s8. OH is unsupported on fused convs (the stem is one)."""
    _run(tmp_path, "yolov8_nano",
         [{"op": 9, "n_splits": 4}, {"op": 6, "n_splits": 2}], "l3_act",
         allow_dirty_baseline=True)
