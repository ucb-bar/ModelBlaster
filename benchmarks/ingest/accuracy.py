"""Extractors for the per-run accuracy summary.

`accuracy.json` is emitted by the runner after parsing the harness's
MODELBLASTER_VERIFY marker (or, for host runs, the direct ctypes
compare against the PyTorch golden). Fields:

    linf        max abs error vs reference
    rmse        sqrt(mean((out - ref)**2))
    cosine      1 - cos_distance
    n_samples   number of comparison elements
    bit_exact   linf == 0
    verify_pass passed the backend's tolerance gate
    atol_used   tolerance the runner applied (after per-backend overrides)
    rtol_used   relative tolerance the runner applied

Per-backend tolerance overrides come from
`pipeline.backends.Backend.atol_override` / `.rtol_override`; the
runner records the effective values into `accuracy.json` so the
aggregator can flag a "pass" that hides drift from a loose backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def linf(path: Path) -> Optional[float]:
    v = _load(path).get("linf")
    return float(v) if v is not None else None


def rmse(path: Path) -> Optional[float]:
    v = _load(path).get("rmse")
    return float(v) if v is not None else None


def cosine(path: Path) -> Optional[float]:
    v = _load(path).get("cosine")
    return float(v) if v is not None else None


def verify_pass(path: Path) -> Optional[bool]:
    v = _load(path).get("verify_pass")
    return bool(v) if v is not None else None


def bit_exact(path: Path) -> Optional[bool]:
    """True when the harness output matched the reference at every
    element (linf == 0). Surfaces as a separate column from verify_pass
    because verify_pass tolerates the backend's atol/rtol envelope --
    a kernel that's "verify_pass=true but bit_exact=false" is correct
    by spec but has measurable drift from the reference, which is the
    relevant signal when tuning an int8 quant scheme."""
    v = _load(path).get("bit_exact")
    return bool(v) if v is not None else None
