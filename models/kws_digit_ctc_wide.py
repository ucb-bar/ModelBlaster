"""Connected-digit CTC transcriber 'digit_ctc_wide' -- see fpga/pynq-z2/modelblaster/kws/.

Installed by patches/0020-modelblaster-kws-models.patch.
"""
import os
import pathlib
import sys

_ROOT = pathlib.Path(os.environ.get("IISWC_ROOT", "/nonexistent"))
sys.path.insert(0, str(_ROOT / "fpga" / "pynq-z2" / "modelblaster" / "kws"))
import mb_shim  # noqa: E402

_ARCH = "digit_ctc_wide"


def get_model(seed: int = 0):
    return mb_shim.load(_ARCH)[0]


def get_sample_input(seed: int = 1):
    return mb_shim.calib(_ARCH, 1)[0]


def get_calibration_samples(n: int):
    return mb_shim.calib(_ARCH, n)
