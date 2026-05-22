"""Extractors for the pipeline's passes_applied.json artifact.

The IR extractors (``pipeline/extract_graph.py`` and
``pipeline/extract_graph_export.py``) each emit
``passes_applied.json`` next to ``graph.json`` listing every fusion /
fold pass that fired during the build, with the names of the IR
sites where it matched. The runner copies that file into the cell's
run dir; these extractors surface a few aggregate counts as
dashboard columns so "did my new fusion pattern actually run?" is
answerable without grep.

Schema (both extractors):

  {
    "schema_version": 1,
    "extractor": "extract_graph" | "extract_graph_export",
    "n_fx_nodes": int (extract_graph) | n_aten_nodes (export),
    "n_ir_ops": int,
    "passes": {
      "<pass_name>": {"fired": int, "sites": list[str]},
      ...
    }
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def passes_fired_total(path: Path) -> int:
    """Sum of `fired` across every pass entry. The dashboard column
    answers "how many opportunities did the pipeline collapse this
    run?" Each fired pass corresponds to one eliminated / folded op."""
    data = _load(path)
    total = 0
    for entry in (data.get("passes") or {}).values():
        total += int(entry.get("fired", 0) or 0)
    return total


def ir_op_count(path: Path) -> Optional[int]:
    """Number of compute ops in the IR after all passes fired. Pair
    with `n_fx_nodes` (or `n_aten_nodes`) to see how aggressive the
    front-end was at lowering."""
    data = _load(path)
    v = data.get("n_ir_ops")
    return int(v) if v is not None else None


def linear_relu_fuse_fired(path: Path) -> Optional[int]:
    """Count of Linear+ReLU fusions in the FX extractor. None when
    the file is from the export extractor (different pass set)."""
    p = (_load(path).get("passes") or {}).get("linear_relu_fuse")
    return int(p["fired"]) if isinstance(p, dict) and "fired" in p else None


def conv2d_relu_fuse_fired(path: Path) -> Optional[int]:
    """Count of Conv2d+ReLU fusions in the FX extractor."""
    p = (_load(path).get("passes") or {}).get("conv2d_relu_fuse")
    return int(p["fired"]) if isinstance(p, dict) and "fired" in p else None


def bn_fold_fired(path: Path) -> Optional[int]:
    """Count of BatchNorm-into-Conv2d folds (export extractor only).
    None on the FX path where BN is emitted as a runtime
    batchnorm2d_s8 record rather than folded into preceding conv."""
    p = (_load(path).get("passes") or {}).get("bn_fold_into_conv2d")
    return int(p["fired"]) if isinstance(p, dict) and "fired" in p else None


def pad_fold_fired(path: Path) -> Optional[int]:
    """Count of (symmetric or all-zero) pad folds into conv2d
    padding (export extractor only)."""
    p = (_load(path).get("passes") or {}).get("pad_fold_into_conv2d")
    return int(p["fired"]) if isinstance(p, dict) and "fired" in p else None
