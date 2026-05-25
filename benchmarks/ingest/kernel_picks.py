"""Extractors for the per-op kernel source breakdown.

``kernel_picks.json`` is emitted by ``pipeline/generate_kernels.py``
and copied into each run dir by the arm driver. It records where each
op's kernel implementation came from:

  reference   scalar reference_impl (no curated, no LLM, no cache)
  curated     hand-written file from kernels/<target>/<target>_<op>_<algo>.c
  cached      previously-PASSed kernel reused from <model>/cache/<target>/
  llm         LLM-synthesized this run

Plus the algorithm name chosen (when applicable) and the source file
path. Schema:

  {
    "schema_version": 1,
    "target": "rvv_opu",
    "picks": {
      "conv2d_s8": {
        "source": "curated",
        "algorithm": "indir_gemm",
        "path": "/.../kernels/rvv_opu/rvv_opu_conv2d_s8_indir_gemm.c"
      },
      ...
    }
  }

For Arm A, this answers "did curated kernels exist for every hot op or
did we silently fall back to scalar?" For Arm B, "which algorithm did
the LLM end up picking per op?"

Today the LLM path lumps cached + LLM-fresh into source=="llm" because
that distinction lives inside `generate_one_llm` and isn't yet plumbed
out. A follow-up commit there will let this surface `cached` vs `llm`
separately; the extractors below already key on the right field so
no schema change is needed at the dashboard side.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _picks(path: Path) -> dict[str, dict[str, Any]]:
    return _load(path).get("picks") or {}


def _count_source(path: Path, source: str) -> Optional[int]:
    picks = _picks(path)
    if not picks:
        return None
    return sum(1 for v in picks.values()
               if v.get("source") == source)


def n_kernels_curated(path: Path) -> Optional[int]:
    """Ops whose kernel was swapped in from kernels/<target>/. High on
    cells where the curated kernel set covers all hot ops; low on
    cells that fall back to scalar reference for accelerator ISAs."""
    return _count_source(path, "curated")


def n_kernels_reference(path: Path) -> Optional[int]:
    """Ops that ran the inline reference_impl (no curated, no LLM).
    On Arm A, this is "scalar fallback" -- the ops where curated
    kernels are missing for this target."""
    return _count_source(path, "reference")


def n_kernels_cached(path: Path) -> Optional[int]:
    """Ops served by the cache directory (previously-PASSed LLM
    kernel reused without spending tokens this run). For Arm B
    today this is folded into n_kernels_llm; future plumbing will
    separate it."""
    return _count_source(path, "cached")


def n_kernels_llm(path: Path) -> Optional[int]:
    """Ops where the LLM produced the kernel this run. Today this
    counter includes cache hits served by the LLM path (see module
    docstring); split lands when generate_one_llm propagates its
    internal source up."""
    return _count_source(path, "llm")


def n_kernels_total(path: Path) -> Optional[int]:
    """Total ops with a recorded pick. Should equal n_ops in the
    graph except for ops the pipeline classified as alias/noop."""
    picks = _picks(path)
    return len(picks) if picks else None


def algorithms_distinct_count(path: Path) -> Optional[int]:
    """How many distinct algorithm names showed up across all op
    picks. Useful Arm-B signal: "did the LLM converge on one style
    everywhere, or pick a different algorithm per op?" High values
    suggest shape-specialized picks; low values suggest a default
    algorithm dominates. Arm A inherits the diversity of curated
    files present in `kernels/<target>/`."""
    picks = _picks(path)
    algos = {v.get("algorithm") for v in picks.values()
             if v.get("algorithm")}
    return len(algos) if algos else None


def algorithm_mode(path: Path) -> Optional[str]:
    """Most-common algorithm name across all op picks. The "default"
    style the kernel set leaned on. None when no pick has an
    algorithm name (e.g. an all-reference Arm A cell)."""
    picks = _picks(path)
    algos = [v.get("algorithm") for v in picks.values()
             if v.get("algorithm")]
    if not algos:
        return None
    return Counter(algos).most_common(1)[0][0]
