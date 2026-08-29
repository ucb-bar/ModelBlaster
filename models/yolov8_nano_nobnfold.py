"""yolov8_nano, BN-folding A/B arm B (folding OFF).

Architecturally identical to models/yolov8_nano.py; this exists only so the
A/B experiment gets its own examples/ tree, IR, kernel cache and build dir.
examples/yolov8_nano/ is shared with other work and _run_lib.sh skips extract
when graph.json is already present, so re-extracting there would either be a
no-op (wrong IR) or clobber another agent's build.

Pair: models/yolov8_nano_bnfold.py (arm A, folding on).
"""

from . import yolov8_nano as _base


def get_model(seed: int = 0):
    return _base.get_model(seed)


def get_sample_input(seed: int = 1):
    return _base.get_sample_input(seed)


__all__ = ["get_model", "get_sample_input"]
