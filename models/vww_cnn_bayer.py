"""Visual Wake Words 'cnn_bayer' -- see fpga/pynq-z2/modelblaster/vision/ in the iiswc-tutorial repo.

Installed by patches/0070-modelblaster-vww-models.patch.  The architecture, the trained
weights and the calibration frames live outside this submodule on purpose: the submodule
is a pinned read-only input and a weight blob cannot be carried in a patch.
"""
import os
import pathlib
import sys

_ROOT = pathlib.Path(os.environ.get("IISWC_ROOT", "/nonexistent"))
sys.path.insert(0, str(_ROOT / "fpga" / "pynq-z2" / "modelblaster" / "vision"))
import mb_shim  # noqa: E402

_ARCH = "cnn_bayer"


def get_model(seed: int = 0):
    return mb_shim.load(_ARCH)[0]


def get_sample_input(seed: int = 1):
    return mb_shim.calib(_ARCH, 1)[0]


def get_calibration_samples(n: int):
    return mb_shim.calib(_ARCH, n)
