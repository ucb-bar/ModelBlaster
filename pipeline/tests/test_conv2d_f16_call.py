"""The skeleton's conv2d_f16 call must match the conv2d_f16 kernel signature.

The spec, the reference and the curated rvv_f16 kernels take dilation (DH, DW);
a skeleton that stops at PW leaves every fp16-conv model (ViNT's goal encoder)
unbuildable: "too few arguments to function 'kernel_conv2d_f16_<model>'".
Generates a three-op model (cast, dilated conv2d_f16, cast) and compiles the
generated model.c and kernels.c with the host compiler (-fsyntax-only: the
timer reads are RISC-V assembly, and the type check is what catches arity).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CC = shutil.which("gcc") or shutil.which("cc")


def _host_has_float16() -> bool:
    if CC is None:
        return False
    r = subprocess.run([CC, "-x", "c", "-c", "-o", os.devnull, "-"],
                       input="_Float16 h = 1;\n", capture_output=True, text=True)
    return r.returncode == 0


def _write_model(d: Path) -> None:
    q = {"scale": 0.05, "zero_point": 0}
    graph = {
        "name": "f16c", "version": 1, "quant": "int8",
        "input": {"tensor": "x"}, "output": {"tensors": ["y"], "tensor": "y"},
        "tensors": {
            "x": {"shape": [1, 2, 6, 6], "dtype": "i8", "quant": q},
            "xh": {"shape": [1, 2, 6, 6], "dtype": "f16"},
            "c": {"shape": [1, 3, 2, 2], "dtype": "f16"},
            "y": {"shape": [1, 3, 2, 2], "dtype": "i8", "quant": q},
        },
        "ops": [
            {"name": "xh", "op": "cast_i8_to_f16", "precision": "fp16",
             "inputs": ["x"], "outputs": ["xh"], "shape": {"n": 72}, "quant": q,
             "dispatch_id": 0, "depends_on": []},
            {"name": "c", "op": "conv2d_f16", "precision": "fp16",
             "inputs": ["xh"], "outputs": ["c"], "weight": "c.weights", "bias": None,
             "shape": {"N": 1, "IC": 2, "IH": 6, "IW": 6, "OC": 3, "KH": 3, "KW": 3,
                       "SH": 1, "SW": 1, "PH": 0, "PW": 0, "DH": 2, "DW": 2,
                       "groups": 1},
             "dispatch_id": 1, "depends_on": [0]},
            {"name": "y", "op": "cast_f16_to_i8", "precision": "int8",
             "inputs": ["c"], "outputs": ["y"], "shape": {"n": 12}, "quant": q,
             "dispatch_id": 2, "depends_on": [1]},
        ],
        "dispatches": [0, 1, 2],
    }
    (d / "graph.json").write_text(json.dumps(graph))
    w = (np.arange(54).reshape(3, 2, 3, 3) % 5 - 2).astype(np.float16)
    np.savez(d / "weights.npz", **{"c.weights": w})
    np.savez(d / "io.npz", input=(np.arange(72) % 7 - 3).astype(np.int8),
             output=np.zeros(12, np.int8))


@pytest.mark.skipif(not _host_has_float16(), reason="host C compiler lacks _Float16")
def test_conv2d_f16_call_passes_dilation_and_compiles(tmp_path):
    _write_model(tmp_path)
    gen = tmp_path / "gen"
    env = dict(os.environ, PYTHONPATH=f"{ROOT / 'src'}{os.pathsep}{ROOT}")
    for args in (
        ["modelblaster.pipeline.generate_skeleton", "--ir", "graph.json",
         "--weights", "weights.npz", "--io", "io.npz", "--out-dir", str(gen),
         "--backend", "rvv_f16", "--platform", "linux"],
        ["modelblaster.pipeline.generate_kernels", "--ir", "graph.json",
         "--out-dir", str(gen), "--target", "scalar", "--backend", "reference",
         "--quant", "int8"],
    ):
        r = subprocess.run([sys.executable, "-m", *args], cwd=tmp_path, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout}\n{r.stderr}"

    call = next(ln for ln in (gen / "model.c").read_text().splitlines()
                if "kernel_conv2d_f16_f16c(" in ln)
    assert call.rstrip().endswith("0, 0, 2, 2);"), call

    r = subprocess.run([CC, "-fsyntax-only", "-DMODELBLASTER_PLATFORM_LINUX", "-I.",
                        f"-I{ROOT / 'harness_linux' / 'src'}", "model.c", "kernels.c"],
                       cwd=gen, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
