"""Extractor for the end-to-end wall-cycle count.

Source file: ``wall_cycles.txt`` -- a single integer written by the
spike / firesim runner that records the cycle count between
the harness's ``MODELBLASTER_WALL_CYCLES_BEGIN`` marker (just before
the model forward pass starts) and ``MODELBLASTER_WALL_CYCLES_END``
(just after it finishes). This is what end users actually feel as
"inference latency on the target".

Complements ``profile_spike.csv``-derived ``cycles_spike``, which is
the SUM of per-dispatch cycles. The two diverge when dispatches run
concurrently (multi-tile hetero), when the harness has non-zero
between-dispatch overhead, or when ``extra_shapes`` verify probes
inflate the dispatch-cycle sum past the actual e2e cost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def wall_cycles(path: Path) -> Optional[int]:
    """Read the single integer from ``wall_cycles.txt``. The aggregator
    passes the resolved path directly (run_dir / source). Returns
    ``None`` when the file is missing (e.g. spike-hetero runs that
    fail verify before wall_cycles emission, or runs pre-dating the
    marker)."""
    try:
        txt = Path(path).read_text().strip()
    except OSError:
        return None
    if not txt:
        return None
    try:
        return int(txt)
    except ValueError:
        return None
