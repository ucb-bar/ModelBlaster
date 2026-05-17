"""Calibration data loaders for the modelblaster pipeline.

Each model declares a ``calibration_spec`` dict describing where each of
its inputs comes from. The walker (extract_graph_export) resolves that
spec at extract time, materializing real-image calibration samples for
PTQ scale computation. The spec is serialized into
``<example>/<quant>/generated/calibration_spec.json`` alongside the rest
of the IR artifacts so any extracted model is reproducible from disk
alone: anyone with the same dataset and the same spec gets the same
calibration scales.

See ``modelblaster/datasets/base.py`` for the dispatcher + protocol; each
concrete loader lives in its own file (``image_dir.py``,
``isaaclab_forest_render.py``, etc).
"""

from modelblaster.datasets.base import (  # noqa: F401
    DatasetItem,
    load_dataset,
    materialize_calibration_samples,
)
