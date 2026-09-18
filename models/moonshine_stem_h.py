"""Moonshine Tiny's convolution stem alone, time on H -- the layout lever, measured on
Moonshine's own stem.  See fpga/pynq-z2/modelblaster/moonshine/.

Installed by patches/0100-modelblaster-moonshine-ops.patch.
"""
import os
import pathlib
import sys

_ROOT = pathlib.Path(os.environ.get("IISWC_ROOT", "/nonexistent"))
sys.path.insert(0, str(_ROOT / "fpga" / "pynq-z2" / "modelblaster" / "moonshine"))
import moonshine_enc as _m  # noqa: E402

_LAYOUT = "h"


def get_model(seed: int = 0):
    return _m.build_stem(_LAYOUT)


def get_sample_input(seed: int = 1):
    return _m.calibration_inputs(1, _LAYOUT)[0]


def get_calibration_samples(n: int):
    return _m.calibration_inputs(n, _LAYOUT)
