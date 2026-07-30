"""yolov8_nano variant with 64x64 input baked in.

Same network architecture as models/yolov8_nano.py but pinned to a 64x64
input resolution (vs the default 160x160). At 1/6.25× the spatial area
the per-conv compute scales down proportionally — useful for fitting a
multi-network bundle (1 yolo + 2 dronet + 4 mlp_control) into a tighter
period budget on the Gemmini+RVV/OPU hetero bitstream.

The qrb5165 reference (notes/figures/qrb5165_reference image) shows the
same bundle at 75.71 ms; yolov8_nano@160 alone has ~420 ms of compute
which cannot fit. The @64 variant has ~67 ms of compute — comfortably
under the 75 ms budget.

The wrapper sets MODELBLASTER_YOLOV8N_INPUT=64 BEFORE importing the
backing model module, so the model construction sees the right input
size. Output dir convention: examples/yolov8_nano_64/<quant>/generated/.
"""

import os
# Must set BEFORE importing the yolov8_nano module — the latter reads
# the env var at import / model-build time.
os.environ["MODELBLASTER_YOLOV8N_INPUT"] = "64"

from . import yolov8_nano as _base  # noqa: E402


def get_model():
    return _base.get_model()


def get_sample_input():
    return _base.get_sample_input()


# Re-export anything else the pipeline expects from the base module.
__all__ = ["get_model", "get_sample_input"]
