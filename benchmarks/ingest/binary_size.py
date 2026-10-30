"""Extractors for the per-run binary / artifact size summary.

``binary_size.json`` is emitted by the arm driver after run.sh
finishes. It captures the "did this kernel set bloat the build"
question that complements cycles + LLM cost: a kernel that's faster
but doubles the binary may be a regression for embedded targets.

Schema:

  {
    "schema_version": 1,
    "zephyr_elf_bytes": int | null,    # output of stat zephyr.elf
    "kernels_c_bytes":  int | null,    # generated kernels.c size
    "kernels_c_loc":    int | null,    # line count
    "weights_npz_bytes": int | null,   # per-model packed weights
  }

Any field may be null when the corresponding artifact isn't where
the arm driver expected (hetero builds, missing-extras layouts, etc).
The extractors below return None on null fields so the dashboard
flags the gap rather than silently zero-imputing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def zephyr_elf_bytes(path: Path) -> Optional[int]:
    v = _load(path).get("zephyr_elf_bytes")
    return int(v) if v is not None else None


def kernels_c_bytes(path: Path) -> Optional[int]:
    v = _load(path).get("kernels_c_bytes")
    return int(v) if v is not None else None


def kernels_c_loc(path: Path) -> Optional[int]:
    """Line count of generated kernels.c. Pair with cycles_firesim to
    see "did the LLM trade lines of code for cycles?" -- a 3x kernel
    that runs 1.5x faster is plausibly worth it; a 10x kernel that
    runs 1.05x faster usually isn't."""
    v = _load(path).get("kernels_c_loc")
    return int(v) if v is not None else None


def weights_npz_bytes(path: Path) -> Optional[int]:
    v = _load(path).get("weights_npz_bytes")
    return int(v) if v is not None else None
