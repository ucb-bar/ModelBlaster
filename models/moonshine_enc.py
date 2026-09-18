"""Moonshine Tiny's speech encoder, 4.0 s window -- see fpga/pynq-z2/modelblaster/moonshine/.

Installed by patches/0100-modelblaster-moonshine-ops.patch.  Everything real (the port,
the pinned checkpoint loader and the speech split) lives in the iiswc-tutorial tree.
"""
import os
import pathlib
import sys

_ROOT = pathlib.Path(os.environ.get("IISWC_ROOT", "/nonexistent"))
sys.path.insert(0, str(_ROOT / "fpga" / "pynq-z2" / "modelblaster" / "moonshine"))
import moonshine_enc as _m  # noqa: E402


def get_model(seed: int = 0):
    return _m.build_encoder()


def get_sample_input(seed: int = 1):
    return _m.calibration_inputs(1)[0]


def get_calibration_samples(n: int):
    return _m.calibration_inputs(n)
