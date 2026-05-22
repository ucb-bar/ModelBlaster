"""Extractors for the beam-search optimize trajectory.

``pipeline/generate_kernels.py``'s optimize loop writes
``beam_search_trajectory.jsonl`` next to ``optimize_summary.json``
when the Arm B-* drivers fire (``BACKEND=llm`` + ``OPTIMIZE=1``).
Each line records one LLM-produced candidate:

  {
    "spec": "conv2d_s8",
    "baseline_cycles": 14000,
    "iter": 1, "parent_idx": 0, "exp_idx": 1,
    "parent_cycles": 14000,
    "result": "ok" | "duplicate" | "build_fail" | "verify_fail",
    "tokens_in": 1234, "tokens_out": 567,
    "cycles": 12500,      // only when result == "ok"
    "diag": "..."         // only on failures
  }

The extractors below surface a handful of aggregates as dashboard
columns; the full JSONL stays available for the offline questions
(per-iter best-cycles trajectory, best-of-K curve, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _iter_records(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _by_result(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in _iter_records(path):
        r = str(rec.get("result", ""))
        counts[r] = counts.get(r, 0) + 1
    return counts


def n_candidates_total(path: Path) -> int:
    """Count of every candidate the LLM produced (across all iters,
    parents, expansions, op kinds). A direct knob on cost when paired
    with `tokens_per_candidate_mean`."""
    return sum(1 for _ in _iter_records(path))


def n_candidates_viable(path: Path) -> int:
    """Candidates that built + verified + produced a cycle count.
    The ratio against `n_candidates_total` is the LLM's effective
    yield rate -- a key knob when tuning beam/expansions."""
    return _by_result(path).get("ok", 0)


def n_candidates_build_fail(path: Path) -> int:
    return _by_result(path).get("build_fail", 0)


def n_candidates_verify_fail(path: Path) -> int:
    return _by_result(path).get("verify_fail", 0)


def n_candidates_duplicate(path: Path) -> int:
    """Suggestions the LLM repeated; cost paid (tokens) but no new
    code to test. High duplicate rates suggest the prompt is too
    constrained or the temperature is too low."""
    return _by_result(path).get("duplicate", 0)


def tokens_per_candidate_mean(path: Path) -> Optional[float]:
    """Mean (input + output) token usage per candidate across the
    whole optimize call. Multiply by `n_candidates_total` for the
    total LLM cost of one (workload, arm) cell's optimize loop."""
    n = 0
    total = 0
    for rec in _iter_records(path):
        n += 1
        total += int(rec.get("tokens_in", 0) or 0)
        total += int(rec.get("tokens_out", 0) or 0)
    return float(total) / n if n else None


def best_improvement_pct(path: Path) -> Optional[float]:
    """Across every op the optimize loop touched, the largest
    baseline-vs-best percent improvement any single spec achieved.
    A useful "did the beam help at all on this cell?" signal --
    near-zero across a long run is the regression to watch."""
    best_by_spec: dict[str, tuple[Optional[int], Optional[int]]] = {}
    for rec in _iter_records(path):
        spec = str(rec.get("spec", ""))
        if not spec:
            continue
        cur = best_by_spec.get(spec, (None, None))
        baseline, best = cur
        if baseline is None:
            baseline = int(rec.get("baseline_cycles", 0) or 0) or None
        if rec.get("result") == "ok":
            cycles = int(rec.get("cycles", 0) or 0)
            if best is None or cycles < best:
                best = cycles
        best_by_spec[spec] = (baseline, best)

    pcts = []
    for baseline, best in best_by_spec.values():
        if baseline and best and baseline > 0:
            pcts.append((baseline - best) / baseline * 100.0)
    return max(pcts) if pcts else None


def candidates_to_best_iter(path: Path) -> Optional[int]:
    """For the op whose best improvement is largest, which iteration
    did the best candidate land in? Tells you whether more iterations
    would have helped or you plateaued early."""
    by_spec: dict[str, list[dict[str, Any]]] = {}
    for rec in _iter_records(path):
        if rec.get("result") != "ok":
            continue
        spec = str(rec.get("spec", ""))
        by_spec.setdefault(spec, []).append(rec)
    if not by_spec:
        return None
    # Pick the spec with the largest improvement.
    best_spec, best_iter = None, None
    best_delta = -1.0
    for spec, recs in by_spec.items():
        baseline = int(recs[0].get("baseline_cycles", 0) or 0)
        if baseline <= 0:
            continue
        # Best record per spec.
        winner = min(recs, key=lambda r: int(r.get("cycles", 0) or 0))
        delta = (baseline - int(winner.get("cycles", 0) or 0)) / baseline
        if delta > best_delta:
            best_delta = delta
            best_spec = spec
            best_iter = int(winner.get("iter", 0) or 0)
    return best_iter
