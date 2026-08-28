"""Provenance-carrying per-op cost database.

Every cost the scheduler consumes must be traceable to the exact artifact
that produced it. Two audited failure modes motivate that rule, and both
produced confident, stable, WRONG numbers that were invisible in summary
tables:

  (a) An IR rewrite removed ops from the scheduler's view with no fused
      kernel existing to do the merged work (see
      ``artifacts/agentic_fuse_split/WARNING.md`` on
      ``origin/feat/agentic-fusion-loop``). "The schedule counted the work
      as gone; the hardware would still need to do it." The corrective
      patch then edited a ``results.csv`` the scheduler does not read, so
      the "honest" number matched the fiction.

  (b) A sharding benchmark where the hardware did real work under a
      dispatch table that did not describe it -- timings real, labels
      fiction.

Storage layout (append-only JSONL, one file per (network, target, quant)):

    benchmarks/profile_db/<network>__<target>__<quant>.jsonl

Two record generations coexist in that store:

  * **v1** (``schema_version: 1``) -- what the original FireSim ingest
    wrote. Keyed on ``(run_id, dispatch_id)``, no unit, no impl hash, no
    core identity. Still readable; normalized on load; permanently
    labelled ``provenance: "partial"`` and unit ``"unknown_cycles"``, which
    is deliberately NOT convertible to seconds.

  * **v2** (``schema_version: 2``) -- written by :func:`ingest_run`. Keyed
    on ``(run_id, stable_key, core_kind, hart)`` and carrying the full
    provenance chain. :func:`ingest_run` REFUSES to write a record without
    an implementation hash: a missing hash is an error, never a default.

Why each v2 field exists
------------------------
``stable_key``
    ``op_name | op_type | signature``. The rewriters
    (:mod:`pipeline.apply_fusion_hint`, :mod:`pipeline.apply_split_hint`)
    renumber ``dispatch_id`` contiguously and do NOT emit their internal
    ``id_remap`` into the output graph. A positional id therefore does not
    survive a rewrite; the name+signature pair does.
``dispatch_id``
    Kept, but explicitly as *the positional id at measurement time*.
``split_from`` / ``fused_from``
    Shard/tile and fusion lineage copied verbatim from the IR, so a tile's
    cost is attributable to the op it came from.
``impl_hash``
    Content hash of the kernel source(s) that actually ran. ``git_sha``
    alone is insufficient: a regenerated kernel at the same SHA is
    indistinguishable from the original.
``target`` + ``core_kind`` + ``hart``
    The old DB dropped hetero runs outright (``if
    target.startswith("hetero"): continue``). That is exactly where a
    multi-model real-time objective lives, so hetero and multi-core
    records are first-class here.
``unit``
    Load-bearing. The old code stored rdcycle core cycles in one place and
    mtime/rdtime ticks in another, both spelled ``cycles``, and a consumer
    divided by ``clock_mhz`` as if they were cycles. On the K1 ``rdtime``
    ticks at 24 MHz and ``rdcycle`` SIGILLs -- the Linux codegen even
    ``#define``\\s a function literally named ``rdcycle()`` that reads
    ``rdtime``. Mixing those two silently is a 66x error. A query spanning
    more than one unit raises :class:`UnitMismatchError`; it never
    converts.
``source_artifact`` (+ its own hash)
    The stdout/log the numbers were parsed from.
``dispatch_table_hash``
    So a trace's labels can be proven to come from the artifact the binary
    executed. Required whenever an XPU-RT trace block is consumed.

Public API
----------
    stable_key(op_name, op_type, signature) -> str
    content_hash(paths) -> "sha256:..."
    impl_hash_from_kernel_picks(picks_path, extra=...) -> "sha256:..."

    ingest_run(stdout, ...) -> IngestResult   # parse a harness stdout
    ingest(results_root, db_root) -> int      # legacy FireSim results tree
    query(network, target, quant, ...) -> dict[key, Measurement]
    query_records(...) -> list[dict]          # JSON-safe aggregates
    coverage_report(db_root) -> dict

CLI:
    python -m benchmarks.profile_db ingest
    python -m benchmarks.profile_db ingest-run --stdout run.log ...
    python -m benchmarks.profile_db coverage
    python -m benchmarks.profile_db query --network dronet --target gemmini \
        --quant int8 --agg p90 --key stable
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import math
import os
import pathlib
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "benchmarks" / "results"
DEFAULT_DB_ROOT = REPO_ROOT / "benchmarks" / "profile_db"

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProfileDBError(Exception):
    """Base class for every refusal this module raises."""


class MissingImplHashError(ProfileDBError):
    """No implementation hash for a measurement.

    Deliberately not defaulted: without it, a regenerated kernel at the
    same git SHA is indistinguishable from the one that was measured.
    """


class MissingProvenanceError(ProfileDBError):
    """A required provenance field (source artifact, unit, ...) is absent."""


class MissingDispatchTableError(ProfileDBError):
    """An XPU-RT trace was consumed without the dispatch table that labels it.

    Failure mode (b): the hardware does real work under a dispatch table
    that does not describe it. Without the table's hash the trace's
    labels cannot be tied to the artifact the binary executed.
    """


class UnknownUnitError(ProfileDBError):
    """A unit was used that this module cannot convert or does not know."""


class UnitMismatchError(ProfileDBError):
    """A query spans records measured in different units.

    Never resolved by conversion -- the caller must narrow the query.
    """


class StableKeyCollisionError(ProfileDBError):
    """One dispatch_id maps to more than one stable op identity.

    Symptom of an IR rewrite that renumbered ops between runs. Re-query
    with ``key="stable"``.
    """


class IRMismatchError(ProfileDBError):
    """The graph.json handed to ingest does not describe the measured run."""


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

UNIT_CORE_CYCLES = "core_cycles"
UNIT_RDTIME_24MHZ = "rdtime_ticks_24mhz"
UNIT_MTIME_TICKS = "mtime_ticks"
UNIT_NS = "ns"
UNIT_US = "us"
UNIT_MS = "ms"
UNIT_UNKNOWN = "unknown_cycles"

#: unit -> (ticks per second, human description). ``None`` for the rate
#: means "not convertible without an explicitly supplied clock".
UNITS: dict[str, tuple[Optional[float], str]] = {
    UNIT_CORE_CYCLES: (
        None,
        "rdcycle deltas. Convert with the CORE clock, which the caller "
        "must supply -- it is not a property of the unit.",
    ),
    UNIT_RDTIME_24MHZ: (
        24_000_000.0,
        "SpaceMiT K1 rdtime ticks. Fixed 24.000 MHz timebase (device-tree "
        "timebase-frequency), NOT the 1.6 GHz core clock. rdcycle SIGILLs "
        "from userspace on this board, so the Linux codegen defines a "
        "function named rdcycle() that actually reads rdtime.",
    ),
    UNIT_MTIME_TICKS: (
        None,
        "Zephyr k_cycle_get_64 / mtime ticks. Cross-hart correct, coarser "
        "than rdcycle; the ratio to core cycles is platform-specific "
        "(1000x on the FireSim configs that produced the v1 records).",
    ),
    UNIT_NS: (1e9, "nanoseconds"),
    UNIT_US: (1e6, "microseconds"),
    UNIT_MS: (1e3, "milliseconds"),
    UNIT_UNKNOWN: (
        None,
        "Legacy v1 record: the unit was never recorded. Cannot be "
        "converted to time. Re-measure through ingest_run() to fix.",
    ),
}


def check_unit(unit: str) -> str:
    if unit not in UNITS:
        raise UnknownUnitError(
            f"unknown unit {unit!r}; known units: {sorted(UNITS)}")
    return unit


def to_seconds(value: float, unit: str,
               clock_hz: Optional[float] = None) -> float:
    """Convert `value` in `unit` to seconds.

    Raises :class:`UnknownUnitError` for any unit whose rate is not
    intrinsic unless the caller supplies `clock_hz` explicitly. This is the
    only sanctioned conversion path: nothing in this module divides a raw
    count by a clock on the caller's behalf.
    """
    rate, _desc = UNITS[check_unit(unit)]
    if rate is None:
        if clock_hz is None:
            raise UnknownUnitError(
                f"unit {unit!r} has no intrinsic rate ({UNITS[unit][1]}); "
                f"pass clock_hz= explicitly to convert")
        rate = float(clock_hz)
    if rate <= 0:
        raise UnknownUnitError(f"non-positive rate for unit {unit!r}")
    return float(value) / rate


# ---------------------------------------------------------------------------
# Identity + hashing
# ---------------------------------------------------------------------------

_STABLE_SEP = "|"


def stable_key(op_name: str, op_type: str, signature: str) -> str:
    """Op identity that survives an IR rewrite.

    ``dispatch_id`` does not: :mod:`pipeline.apply_fusion_hint` and
    :mod:`pipeline.apply_split_hint` both re-assign ids contiguously and
    keep their ``id_remap`` local -- it never reaches the emitted
    graph.json. Name plus signature does survive, and split tiles stay
    distinct because the splitters rename to ``<name>.tile_<t>`` and
    shrink the shape.
    """
    return f"{op_name}{_STABLE_SEP}{op_type}{_STABLE_SEP}{signature}"


def signature_from_shape(shape: Any) -> str:
    """Render an IR shape (dict or already-a-string) as ``K=V;K=V``."""
    if shape is None:
        return ""
    if isinstance(shape, str):
        return shape
    if isinstance(shape, Mapping):
        return ";".join(f"{k}={v}" for k, v in shape.items())
    return str(shape)


def content_hash(paths: Iterable[os.PathLike | str],
                 *, extra: Iterable[bytes] = ()) -> str:
    """SHA-256 over the *content* of `paths`, path-name and length framed.

    Same recipe as ``pipeline/ingest_xpurt_schedule.py``'s ``pdb_hash`` so
    the two agree. Paths are sorted, so the hash is order-independent; the
    name and byte length are folded in so a rename or a truncation cannot
    collide. A missing path is an error, not a skip -- silently hashing
    fewer files is how a stale hash starts matching.
    """
    h = hashlib.sha256()
    ps = sorted({str(p) for p in paths})
    if not ps and not list(extra):
        raise MissingProvenanceError(
            "content_hash() called with no paths and no extra bytes")
    for p in ps:
        try:
            data = pathlib.Path(p).read_bytes()
        except OSError as exc:
            raise MissingProvenanceError(
                f"cannot hash {p!r}: {exc}. Refusing to emit a hash over a "
                f"subset of the sources that ran.") from exc
        h.update(os.path.basename(p).encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "little"))
        h.update(data)
    for blob in extra:
        h.update(len(blob).to_bytes(8, "little"))
        h.update(blob)
    return "sha256:" + h.hexdigest()


def file_hash(path: os.PathLike | str) -> str:
    """SHA-256 of one file's bytes, prefixed ``sha256:``."""
    return content_hash([path])


def impl_hash_from_kernel_picks(picks_path: os.PathLike | str,
                                *, extra: Iterable[os.PathLike | str] = (),
                                ) -> str:
    """Implementation hash for a build described by ``kernel_picks.json``.

    ``kernel_picks.json`` (written by ``pipeline/generate_kernels.py``)
    names the curated kernel source for every op kind that got one.
    Reference-kernel picks have ``path: null``; their code lives in the
    generated ``kernels.c``, so pass that via `extra`.

    The picks file itself is hashed too: it records which algorithm was
    selected, and a different selection is a different implementation even
    when the source files are unchanged.
    """
    picks_path = pathlib.Path(picks_path)
    try:
        picks = json.loads(picks_path.read_text())
    except OSError as exc:
        raise MissingProvenanceError(
            f"cannot read kernel picks {picks_path}: {exc}") from exc
    srcs: list[str] = [str(picks_path)]
    for _op, pick in sorted((picks.get("picks") or {}).items()):
        p = (pick or {}).get("path")
        if p:
            srcs.append(str(p))
    srcs.extend(str(p) for p in extra)
    return content_hash(srcs)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_SAFE = re.compile(r"[^A-Za-z0-9._+-]")


def _slug(s: str) -> str:
    return _SAFE.sub("_", str(s))


def _db_path(db_root: pathlib.Path, network: str, target: str,
             quant: str) -> pathlib.Path:
    return db_root / f"{_slug(network)}__{_slug(target)}__{_slug(quant)}.jsonl"


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100].

    Defined for any non-empty sequence, so a 1-sample record still has a
    p90 (equal to its only sample -- which is why ``n_samples`` travels
    beside every aggregate).
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    s = sorted(float(v) for v in values)
    if len(s) == 1:
        return s[0]
    rank = (q / 100.0) * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


_AGGS: dict[str, Any] = {
    "median": statistics.median,
    "mean": statistics.mean,
    "min": min,
    "max": max,
    "p90": lambda vs: _percentile(vs, 90.0),
}


def _summarize(samples: Sequence[float]) -> dict:
    return {
        "n_samples": len(samples),
        "median": float(statistics.median(samples)),
        "mean": float(statistics.mean(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "p90": _percentile(samples, 90.0),
    }


def record_key(rec: Mapping[str, Any]) -> tuple:
    """Idempotency key for one stored measurement.

    ``(run_id, stable_key)`` is the identity the brief asks for; core
    identity is appended so the same op measured on two harts inside one
    hetero run stays two records instead of clobbering itself. For a
    single-core run the extra components are constant and the key
    degenerates to ``(run_id, stable_key)``.
    """
    return (
        rec.get("run_id"),
        rec.get("stable_key"),
        rec.get("core_kind") or "",
        -1 if rec.get("hart") is None else int(rec["hart"]),
        -1 if rec.get("instance") is None else int(rec["instance"]),
        rec.get("record_kind", "op_measurement"),
    )


def _normalize(rec: dict) -> dict:
    """Upgrade a stored record to the v2 field set, in memory only.

    v1 records keep their meaning: unit becomes the explicitly
    non-convertible ``unknown_cycles`` and provenance is ``partial``, so a
    consumer that needs a real unit or an impl hash can filter them out
    rather than being handed a plausible-looking default.
    """
    if int(rec.get("schema_version", 1)) >= 2:
        rec.setdefault("record_kind", "op_measurement")
        return rec
    out = dict(rec)
    out["schema_version"] = 1
    out["record_kind"] = rec.get("record_kind", "op_measurement")
    cyc = rec.get("cycles")
    samples = rec.get("samples")
    if samples is None:
        samples = [] if cyc is None else [cyc]
    out["samples"] = list(samples)
    out.setdefault("stable_key", stable_key(
        rec.get("op_name", ""), rec.get("op_type", ""),
        rec.get("signature", "")))
    out.setdefault("unit", UNIT_UNKNOWN)
    out.setdefault("provenance", "partial")
    out.setdefault("impl_hash", None)
    out.setdefault("impl_sources", [])
    out.setdefault("core_kind", None)
    out.setdefault("hart", None)
    out.setdefault("instance", None)
    out.setdefault("source_artifact", None)
    out.setdefault("dispatch_table_hash", None)
    if out["samples"]:
        out.update({k: v for k, v in _summarize(out["samples"]).items()
                    if k not in out})
    return out


def load_records(db_root: pathlib.Path, network: str, target: str,
                 quant: str) -> list[dict]:
    """All normalized records for one (network, target, quant) shard."""
    return [_normalize(r)
            for r in _load_jsonl(_db_path(db_root, network, target, quant))]


def _append(path: pathlib.Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Harness stdout parsing
# ---------------------------------------------------------------------------

_PROF_BLOCK_RE = re.compile(
    r"=== MODELBLASTER_PROFILE_BEGIN(?: \[([^\]]+)\])? ===\n"
    r"(.*?)"
    r"=== MODELBLASTER_PROFILE_END(?: \[([^\]]+)\])? ===",
    re.S,
)
_WALL_RE = re.compile(
    r"=== MODELBLASTER_WALL_CYCLES(?: \[([^\]]+)\])? === (\d+)")
_XPURT_TRACE_RE = re.compile(
    r"=== MODELBLASTER_XPURT_TRACE_BEGIN ===\n(.*?)"
    r"=== MODELBLASTER_XPURT_TRACE_END ===",
    re.S,
)

#: Column sets the harnesses emit. ``harness/`` and ``harness_linux/``
#: print the 5-column form; ``pipeline/generate_xpurt_main.py`` prefixes a
#: ``backend`` column. Both are accepted; anything else is refused rather
#: than guessed at.
_PROFILE_COLUMNS_5 = ["dispatch_id", "name", "op", "shape", "cycles"]
_PROFILE_COLUMNS_6 = ["backend"] + _PROFILE_COLUMNS_5


def parse_profile_blocks(text: str) -> list[dict]:
    """Every ``MODELBLASTER_PROFILE_BEGIN..END`` block in `text`.

    Returns ``[{"tag": str|None, "rows": [ {...}, ... ]}, ...]`` in stdout
    order. ``tag`` is the ``[<model>]`` label the multi-model / XPU-RT
    harnesses print, or ``None`` for the untagged single-model harness.
    """
    blocks: list[dict] = []
    for m in _PROF_BLOCK_RE.finditer(text):
        begin_tag, body, end_tag = m.group(1), m.group(2), m.group(3)
        if begin_tag != end_tag:
            raise ProfileDBError(
                f"mismatched PROFILE markers: BEGIN [{begin_tag}] vs "
                f"END [{end_tag}] -- refusing to guess which model this "
                f"block belongs to")
        lines = [ln for ln in body.splitlines() if ln.strip()]
        rows: list[dict] = []
        if lines:
            header = [c.strip() for c in lines[0].split(",")]
            if header not in (_PROFILE_COLUMNS_5, _PROFILE_COLUMNS_6):
                raise ProfileDBError(
                    f"unrecognized profile header {header!r}; expected "
                    f"{_PROFILE_COLUMNS_5} or {_PROFILE_COLUMNS_6}")
            for ln in lines[1:]:
                cells = [c.strip() for c in ln.split(",")]
                if len(cells) != len(header):
                    raise ProfileDBError(
                        f"profile row has {len(cells)} fields, header has "
                        f"{len(header)}: {ln!r}")
                rec = dict(zip(header, cells))
                rec["dispatch_id"] = int(rec["dispatch_id"])
                rec["cycles"] = int(rec["cycles"])
                rows.append(rec)
        blocks.append({"tag": begin_tag, "rows": rows})
    return blocks


def parse_wall_cycles(text: str) -> list[tuple[Optional[str], int]]:
    """Every ``MODELBLASTER_WALL_CYCLES`` marker as ``(tag, value)``."""
    return [(m.group(1), int(m.group(2))) for m in _WALL_RE.finditer(text)]


def parse_xpurt_trace(text: str) -> list[dict]:
    """Rows of the ``MODELBLASTER_XPURT_TRACE`` block, or ``[]``.

    The block is only emitted when the binary was built with
    ``-DMODELBLASTER_XPURT_TRACE``. Its columns are
    ``entry_id,network,instance,dispatch_id,op,name,core_kind,hart,
    predicted_start_ms,predicted_duration_ms,worker_kind_idx,
    actual_start_cycles,actual_end_cycles``.
    """
    m = _XPURT_TRACE_RE.search(text)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    return list(csv.DictReader(io.StringIO(body)))


def _split_tag(tag: Optional[str], networks: Optional[Sequence[str]],
               fallback: Optional[str]) -> tuple[str, int]:
    """Resolve a block tag into ``(network, instance)``.

    The XPU-RT harness tags per *job* (``mlp_control3``), not per network.
    Splitting the trailing digits off blindly is wrong -- ``yolov8_nano_64``
    ends in digits and is a network name -- so a known-network list is
    required to disambiguate. Without one the tag is taken verbatim as the
    network with instance 0.
    """
    if tag is None:
        if fallback is None:
            raise MissingProvenanceError(
                "untagged profile block and no network= given")
        return fallback, 0
    if networks:
        for cand in sorted(networks, key=len, reverse=True):
            if tag == cand:
                return cand, 0
            if tag.startswith(cand):
                rest = tag[len(cand):]
                if rest.isdigit():
                    return cand, int(rest)
    return tag, 0


# ---------------------------------------------------------------------------
# ingest_run
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Outcome of one :func:`ingest_run` call."""

    added: int = 0
    skipped: int = 0
    records: list[dict] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"IngestResult(added={self.added}, skipped={self.skipped}, "
                f"paths={self.paths})")


def _load_graph(graph: Any) -> Optional[dict]:
    if graph is None:
        return None
    if isinstance(graph, Mapping):
        return dict(graph)
    return json.loads(pathlib.Path(graph).read_text())


def _graph_index(graph: Optional[dict]) -> dict[int, dict]:
    if not graph:
        return {}
    out: dict[int, dict] = {}
    for op in graph.get("ops", []):
        did = op.get("dispatch_id")
        if did is not None:
            out[int(did)] = op
    return out


def ingest_run(
    stdout: str | os.PathLike,
    *,
    run_id: str,
    unit: str,
    target: str,
    quant: str,
    network: Optional[str] = None,
    networks: Optional[Sequence[str]] = None,
    impl_hash: Optional[str] = None,
    impl_sources: Optional[Sequence[os.PathLike | str]] = None,
    source_artifact: Optional[os.PathLike | str] = None,
    core_kind: Optional[str] = None,
    hart: Optional[int] = None,
    cpu_id: Optional[int] = None,
    dispatch_table: Optional[os.PathLike | str] = None,
    dispatch_table_hash: Optional[str] = None,
    graph: Any = None,
    strict_ir: bool = True,
    git_sha: str = "",
    captured_at: Optional[str] = None,
    workload_id: Optional[str] = None,
    runner: str = "k1",
    extra: Optional[Mapping[str, Any]] = None,
    db_root: pathlib.Path = DEFAULT_DB_ROOT,
    write: bool = True,
) -> IngestResult:
    """Parse a K1 harness stdout into provenance-carrying records.

    `stdout` is either the text itself or a path to the captured log
    (``validation/k1_runner.py --save-output``). When a path is given it
    becomes the default `source_artifact`.

    Refusals, all deliberate:

    * no `impl_hash` and no `impl_sources` -> :class:`MissingImplHashError`.
      A build with no recorded implementation identity cannot be told apart
      from a different build at the same git SHA.
    * no `source_artifact` -> :class:`MissingProvenanceError`.
    * an XPU-RT trace block present but no dispatch table ->
      :class:`MissingDispatchTableError`. The trace's ``core_kind`` / ``hart``
      labels come from that table; unhashed, they are unfalsifiable.
    * a `graph` whose op at `dispatch_id` does not match the measured row
      -> :class:`IRMismatchError` (unless ``strict_ir=False``, which records
      ``ir_mismatch: true`` on the record instead).

    Idempotent on :func:`record_key`, so re-ingesting the same log adds
    nothing.
    """
    check_unit(unit)
    if unit == UNIT_UNKNOWN:
        raise MissingProvenanceError(
            f"{UNIT_UNKNOWN!r} is the marker for legacy records with no "
            f"recorded unit; a new measurement must name its real unit "
            f"(one of {sorted(u for u in UNITS if u != UNIT_UNKNOWN)})")

    text: str
    if isinstance(stdout, (str, bytes)) and not _looks_like_path(stdout):
        text = stdout if isinstance(stdout, str) else stdout.decode()
    else:
        p = pathlib.Path(os.fspath(stdout))
        text = p.read_text()
        if source_artifact is None:
            source_artifact = p

    if source_artifact is None:
        raise MissingProvenanceError(
            "source_artifact is required: every cost must name the "
            "stdout/log it was parsed from. Pass the captured log path "
            "(k1_runner --save-output) or a Path as `stdout`.")
    source_artifact = pathlib.Path(os.fspath(source_artifact))
    try:
        source_artifact_hash = file_hash(source_artifact)
    except MissingProvenanceError:
        raise MissingProvenanceError(
            f"source_artifact {source_artifact} does not exist; a path that "
            f"cannot be read is not provenance") from None

    if impl_hash is None:
        if not impl_sources:
            raise MissingImplHashError(
                "impl_hash (or impl_sources to compute one) is required. "
                "git_sha alone is insufficient: a regenerated kernel at the "
                "same SHA is indistinguishable from the one measured. See "
                "impl_hash_from_kernel_picks().")
        impl_hash = content_hash(impl_sources)
    impl_sources_out = [str(p) for p in (impl_sources or [])]

    if dispatch_table_hash is None and dispatch_table is not None:
        dispatch_table_hash = file_hash(dispatch_table)

    trace_rows = parse_xpurt_trace(text)
    if trace_rows and dispatch_table_hash is None:
        raise MissingDispatchTableError(
            "stdout carries an XPU-RT trace block but no dispatch table was "
            "given. The trace's network/instance/core_kind/hart labels come "
            "from that table; without its hash they cannot be tied to the "
            "artifact the binary executed. Pass dispatch_table= (the "
            "*_dispatch_table.c emitted by ingest_xpurt_schedule) or "
            "dispatch_table_hash=.")

    # (network, instance, dispatch_id) -> trace row, for core attribution.
    trace_index: dict[tuple[str, int, int], dict] = {}
    for tr in trace_rows:
        try:
            k = (tr["network"], int(tr["instance"]), int(tr["dispatch_id"]))
        except (KeyError, ValueError):
            continue
        trace_index[k] = tr

    graph_doc = _load_graph(graph)
    ops_by_id = _graph_index(graph_doc)
    graph_hash = None
    if graph is not None and not isinstance(graph, Mapping):
        graph_hash = file_hash(graph)

    captured_at = captured_at or _dt.datetime.now(
        _dt.timezone.utc).isoformat()

    blocks = parse_profile_blocks(text)
    if not blocks:
        raise ProfileDBError(
            "no MODELBLASTER_PROFILE_BEGIN/END block found in stdout")

    new_by_file: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    all_records: list[dict] = []

    for block in blocks:
        net, instance = _split_tag(block["tag"], networks, network)
        # Multiple rows can share a stable key inside one block (an op that
        # runs more than once). Pool them into one record's samples rather
        # than dropping all but the first.
        pooled: dict[tuple, dict] = {}
        for row in block["rows"]:
            op_name = row["name"]
            op_type = row["op"]
            signature = row["shape"]
            did = int(row["dispatch_id"])
            skey = stable_key(op_name, op_type, signature)

            row_core = core_kind
            row_hart = hart
            row_backend = row.get("backend")
            tr = trace_index.get((net, instance, did))
            if tr is not None:
                row_core = tr.get("core_kind") or row_core
                try:
                    row_hart = int(tr["hart"])
                except (KeyError, TypeError, ValueError):
                    pass
            if row_core is None and row_backend:
                row_core = row_backend

            split_from = None
            fused_from = None
            ir_mismatch = False
            op = ops_by_id.get(did)
            if op is not None:
                ir_name = op.get("name")
                ir_op = op.get("op")
                ir_sig = signature_from_shape(op.get("shape"))
                if (ir_name, ir_op) != (op_name, op_type) or (
                        ir_sig and ir_sig != signature):
                    msg = (
                        f"IR/measurement disagreement at dispatch_id={did} "
                        f"for {net}: graph says "
                        f"{ir_name!r}/{ir_op!r}/{ir_sig!r}, harness reported "
                        f"{op_name!r}/{op_type!r}/{signature!r}. The graph "
                        f"handed to ingest is not the IR this binary ran.")
                    if strict_ir:
                        raise IRMismatchError(msg)
                    ir_mismatch = True
                split_from = op.get("split_from")
                fused_from = op.get("fused_from")
            elif ops_by_id:
                msg = (f"dispatch_id={did} measured for {net} but absent from "
                       f"the supplied graph ({len(ops_by_id)} dispatches)")
                if strict_ir:
                    raise IRMismatchError(msg)
                ir_mismatch = True

            pkey = (skey, row_core, row_hart)
            rec = pooled.get(pkey)
            if rec is None:
                rec = {
                    "schema_version": SCHEMA_VERSION,
                    "record_kind": "op_measurement",
                    "provenance": "full",

                    # --- what was measured -------------------------------
                    "network": net,
                    "instance": instance,
                    "target": target,
                    "quant": quant,
                    "workload_id": workload_id or f"{net}_{target}_{quant}",

                    # --- stable op identity ------------------------------
                    "stable_key": skey,
                    "op_name": op_name,
                    "op_type": op_type,
                    "signature": signature,

                    # --- positional identity at measurement time ---------
                    "dispatch_id": did,
                    "dispatch_id_is_positional": True,

                    # --- shard / fusion lineage --------------------------
                    "split_from": split_from,
                    "fused_from": fused_from,

                    # --- where it ran ------------------------------------
                    "backend": row_backend,
                    "core_kind": row_core,
                    "hart": row_hart,
                    "cpu_id": cpu_id if cpu_id is not None else row_hart,

                    # --- how it was built --------------------------------
                    "impl_hash": impl_hash,
                    "impl_sources": impl_sources_out,
                    "git_sha": git_sha,

                    # --- the measurement ---------------------------------
                    "run_id": run_id,
                    "runner": runner,
                    "unit": unit,
                    "samples": [],

                    # --- provenance of the numbers themselves ------------
                    "source_artifact": str(source_artifact),
                    "source_artifact_hash": source_artifact_hash,
                    "dispatch_table_hash": dispatch_table_hash,
                    "dispatch_table_path": (str(dispatch_table)
                                            if dispatch_table else None),
                    "graph_hash": graph_hash,
                    "ir_mismatch": ir_mismatch,
                    "captured_at": captured_at,
                }
                if extra:
                    rec["extra"] = dict(extra)
                pooled[pkey] = rec
            rec["samples"].append(int(row["cycles"]))

        # Wall-clock total for this block, stored beside the ops so a
        # consumer can check sum(ops) against the wall it was drawn from.
        wall = None
        for tag, val in parse_wall_cycles(text):
            if tag == block["tag"]:
                wall = val
                break
        if wall is not None:
            wkey = ("__wall__", core_kind, hart)
            pooled[wkey] = {
                "schema_version": SCHEMA_VERSION,
                "record_kind": "wall",
                "provenance": "full",
                "network": net,
                "instance": instance,
                "target": target,
                "quant": quant,
                "workload_id": workload_id or f"{net}_{target}_{quant}",
                "stable_key": "__wall__",
                "op_name": "__wall__",
                "op_type": "__wall__",
                "signature": "",
                "dispatch_id": -1,
                "dispatch_id_is_positional": True,
                "split_from": None,
                "fused_from": None,
                "backend": None,
                "core_kind": core_kind,
                "hart": hart,
                "cpu_id": cpu_id if cpu_id is not None else hart,
                "impl_hash": impl_hash,
                "impl_sources": impl_sources_out,
                "git_sha": git_sha,
                "run_id": run_id,
                "runner": runner,
                # The wall marker is a k_cycle_get_64/mtime delta on Zephyr
                # and an rdtime delta on Linux; it is NOT necessarily the
                # same unit as the per-op counts, so it is labelled
                # separately and never pooled with them.
                "unit": unit,
                "samples": [int(wall)],
                "source_artifact": str(source_artifact),
                "source_artifact_hash": source_artifact_hash,
                "dispatch_table_hash": dispatch_table_hash,
                "dispatch_table_path": (str(dispatch_table)
                                        if dispatch_table else None),
                "graph_hash": graph_hash,
                "ir_mismatch": False,
                "captured_at": captured_at,
            }

        for rec in pooled.values():
            rec.update(_summarize(rec["samples"]))
            # `cycles` mirrors the median so v1-era readers (coverage_report,
            # older consumers) keep working. It is NOT the source of truth --
            # `samples` + `unit` are.
            rec["cycles"] = int(rec["median"])
            all_records.append(rec)
            new_by_file[(net, target, quant)].append(rec)

    result = IngestResult()
    for key, recs in new_by_file.items():
        path = _db_path(db_root, *key)
        existing = {record_key(_normalize(r)) for r in _load_jsonl(path)}
        fresh: list[dict] = []
        seen_this_call: set[tuple] = set()
        for r in recs:
            k = record_key(r)
            if k in existing or k in seen_this_call:
                result.skipped += 1
                continue
            seen_this_call.add(k)
            fresh.append(r)
        if fresh and write:
            _append(path, fresh)
        result.added += len(fresh)
        result.records.extend(fresh)
        result.paths.append(str(path))
    return result


def _looks_like_path(s: Any) -> bool:
    if not isinstance(s, str):
        return True
    if "\n" in s:
        return False
    return len(s) < 4096 and os.path.exists(s)


# ---------------------------------------------------------------------------
# Legacy ingest (FireSim results tree)
# ---------------------------------------------------------------------------

# (network, target, quant) tuples we expect to see for a "complete" matrix.
# Anything in this set without records becomes a MISSING row in coverage.
EXPECTED_MATRIX: list[tuple[str, str, str]] = [
    ("dronet", "gemmini", "int8"),
    ("dronet", "gemmini_q31", "int8"),
    ("dronet", "rvv_opu", "int8"),
    ("yolov8_nano", "gemmini", "int8"),
    ("yolov8_nano", "gemmini_q31", "int8"),
    ("yolov8_nano", "rvv_opu", "int8"),
    ("mlp_control", "gemmini", "int8"),
    ("mlp_control", "gemmini_q31", "int8"),
    ("mlp_control", "rvv_opu", "int8"),
]


def _read_run_meta(run_dir: pathlib.Path) -> Optional[dict]:
    rj = run_dir / "run.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text())
    except json.JSONDecodeError:
        return None


def _read_profile_csv(csv_path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "dispatch_id": int(r["dispatch_id"]),
                "op_name": r["name"],
                "op_type": r["op"],
                "signature": r["shape"],
                "cycles": int(r["cycles"]),
                "backend": r.get("backend"),
            })
    return rows


def _read_trace_index(run_dir: pathlib.Path) -> dict[int, dict]:
    """dispatch_id -> {core_kind, hart} from a run's ``xpurt_trace.csv``."""
    p = run_dir / "xpurt_trace.csv"
    if not p.exists():
        return {}
    out: dict[int, dict] = {}
    with p.open() as f:
        for r in csv.DictReader(f):
            try:
                did = int(r["dispatch_id"])
            except (KeyError, ValueError):
                continue
            hart: Optional[int]
            try:
                hart = int(r.get("hart", ""))
            except (TypeError, ValueError):
                hart = None
            out.setdefault(did, {"core_kind": r.get("core_kind"),
                                 "hart": hart})
    return out


def ingest(
    results_root: pathlib.Path = DEFAULT_RESULTS_ROOT,
    db_root: pathlib.Path = DEFAULT_DB_ROOT,
    verbose: bool = False,
) -> int:
    """Walk all ``<cell>/<run-id>/profile_firesim.csv`` and append new records.

    Idempotent on ``(run_id, dispatch_id)`` as before. Returns the number of
    NEW records added.

    Two behaviour changes versus the original:

    * **Hetero runs are no longer skipped.** The old code did
      ``if target.startswith("hetero"): continue``, which dropped exactly the
      multi-model / multi-core measurements a real-time objective is about.
      They land in their own ``<network>__<hetero-target>__<quant>.jsonl``
      shard, so no single-backend query sees them by accident, and per-op
      ``core_kind`` / ``hart`` are joined in from ``xpurt_trace.csv`` when
      that file is present.
    * Records carry ``unit`` (from ``run.json``'s ``profile_unit`` if the
      runner recorded one, else the explicitly non-convertible
      ``unknown_cycles``) and ``source_artifact``. They stay
      ``provenance: "partial"`` -- this tree has no implementation hash, and
      inventing one would defeat the point.
    """
    db_root.mkdir(parents=True, exist_ok=True)
    arm_a = results_root / "A"
    if not arm_a.exists():
        return 0

    new_by_file: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for cell_dir in sorted(arm_a.iterdir()):
        if not cell_dir.is_dir():
            continue
        for run_dir in sorted(cell_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "latest":
                continue
            csv_path = run_dir / "profile_firesim.csv"
            if not csv_path.exists():
                continue
            meta = _read_run_meta(run_dir)
            if meta is None:
                if verbose:
                    print(f"  skip (no run.json): {run_dir}", file=sys.stderr)
                continue
            network = meta.get("model")
            target = meta.get("target")
            quant = meta.get("quant")
            if not (network and target and quant):
                continue

            key = (network, target, quant)
            file_path = _db_path(db_root, *key)
            existing = {(r.get("run_id"), r.get("dispatch_id"))
                        for r in _load_jsonl(file_path)}

            run_id = meta.get("run_id") or run_dir.name
            git_sha = meta.get("git_sha", "")
            captured_at = meta.get("started_at", "")
            workload_id = meta.get("workload_id", cell_dir.name)
            unit = meta.get("profile_unit") or UNIT_UNKNOWN
            check_unit(unit)
            trace_index = _read_trace_index(run_dir)

            for row in _read_profile_csv(csv_path):
                if (run_id, row["dispatch_id"]) in existing:
                    continue
                tr = trace_index.get(row["dispatch_id"], {})
                new_by_file[key].append({
                    "schema_version": 1,
                    "provenance": "partial",
                    "network": network,
                    "target": target,
                    "quant": quant,
                    "workload_id": workload_id,
                    "run_id": run_id,
                    "dispatch_id": row["dispatch_id"],
                    "op_type": row["op_type"],
                    "op_name": row["op_name"],
                    "signature": row["signature"],
                    "stable_key": stable_key(row["op_name"], row["op_type"],
                                             row["signature"]),
                    "cycles": row["cycles"],
                    "unit": unit,
                    "core_kind": tr.get("core_kind") or row.get("backend"),
                    "hart": tr.get("hart"),
                    "source_artifact": str(csv_path),
                    "impl_hash": None,
                    "git_sha": git_sha,
                    "captured_at": captured_at,
                })

    added = 0
    for key, recs in new_by_file.items():
        file_path = _db_path(db_root, *key)
        _append(file_path, recs)
        added += len(recs)
        if verbose:
            print(f"  +{len(recs):>5} -> {file_path.name}")
    return added


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class Measurement(int):
    """One aggregated cost, with its provenance attached.

    Subclasses :class:`int` so the pre-existing consumers
    (``scripts/run_xpurt_scheduler.py``, ``scripts/periodic_partition_schedule.py``,
    ``scripts/run_xpurt_scheduler_multi.py``) that treat ``query()``'s values
    as plain integers keep working unchanged, while a provenance-aware
    caller reads the attributes:

        m = query(...)[0]
        m + 1            # 12346 -- still an int
        m.n_samples      # 1 vs 10 is now visible
        m.unit           # never guessed
        m.impl_hashes    # which build(s) produced this
    """

    #: The attached fields, in a fixed order (`as_dict` and `__new__` both
    #: iterate it). NOT `__slots__`: CPython refuses `nonempty __slots__` on a
    #: subclass of a variable-length built-in, so `class Measurement(int)` with
    #: `__slots__` set raises TypeError at class-creation time and the whole
    #: module fails to import. An int subclass therefore carries a `__dict__`
    #: whatever we do -- there is no layout for `__slots__` to save -- so the
    #: tuple keeps its only remaining job, which is naming the fields once.
    _FIELDS = ("value", "agg", "unit", "n_samples", "n_runs", "stable_key",
               "op_name", "op_type", "signature", "dispatch_ids",
               "impl_hashes", "run_ids", "core_kinds", "harts",
               "source_artifacts", "dispatch_table_hashes", "provenance",
               "samples", "median", "p90", "mean", "min", "max")

    def __new__(cls, value: float, **kw: Any) -> "Measurement":
        # int() truncation, matching the original query()'s int(median).
        obj = super().__new__(cls, int(value))
        obj.value = float(value)
        for k in cls._FIELDS:
            if k == "value":
                continue
            setattr(obj, k, kw.get(k))
        return obj

    def as_dict(self) -> dict:
        """JSON-safe view of this measurement and everything behind it."""
        return {k: getattr(self, k) for k in self._FIELDS}

    def to_seconds(self, clock_hz: Optional[float] = None) -> float:
        """Convert to seconds, refusing when the unit does not permit it."""
        return to_seconds(self.value, self.unit, clock_hz=clock_hz)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Measurement({self.value:g} {self.unit}, agg={self.agg}, "
                f"n={self.n_samples}, key={self.stable_key!r})")


_KEY_FIELDS = {
    "dispatch_id": "dispatch_id",
    "stable": "stable_key",
    "stable_key": "stable_key",
    "op_name": "op_name",
}


def query(
    network: str,
    target: str,
    quant: str,
    op_type: Optional[str] = None,
    agg: str = "median",
    db_root: pathlib.Path = DEFAULT_DB_ROOT,
    *,
    key: str = "dispatch_id",
    core_kind: Optional[str] = None,
    hart: Optional[int] = None,
    instance: Optional[int] = None,
    impl_hash: Optional[str] = None,
    run_id: Optional[str] = None,
    unit: Optional[str] = None,
    require_impl_hash: bool = False,
    require_unit: bool = False,
    record_kind: Optional[str] = "op_measurement",
) -> dict[Any, Measurement]:
    """Aggregated cost per op for one (network, target, quant).

    Returns ``{key -> Measurement}``. ``Measurement`` is an ``int``, so the
    legacy call ``query(net, tgt, q, agg="median")`` still yields something
    a caller can do arithmetic on and ``json.dumps``; the sample count,
    unit, impl hashes and source artifacts ride along as attributes.

    `agg` is one of ``median``, ``mean``, ``min``, ``max``, ``p90``,
    computed over the pooled samples of every matching record.

    `key` selects the grouping:

    * ``"dispatch_id"`` (default, legacy shape). If one dispatch_id maps to
      more than one stable identity -- what an IR rewrite produces -- this
      raises :class:`StableKeyCollisionError` instead of averaging a conv
      together with the elu that inherited its number.
    * ``"stable"`` groups by ``op_name|op_type|signature``, which survives a
      rewrite.
    * ``"op_name"`` groups by op name alone.

    Units are never converted. If the matching records disagree on `unit`,
    :class:`UnitMismatchError` is raised; narrow with ``unit=``.
    """
    if agg not in _AGGS:
        raise ValueError(f"unknown agg: {agg!r}; choose from {sorted(_AGGS)}")
    if key not in _KEY_FIELDS:
        raise ValueError(f"unknown key: {key!r}; choose from "
                         f"{sorted(set(_KEY_FIELDS))}")
    field_name = _KEY_FIELDS[key]

    records = load_records(pathlib.Path(db_root), network, target, quant)
    if not records:
        return {}

    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in records:
        if record_kind is not None and r.get("record_kind",
                                             "op_measurement") != record_kind:
            continue
        if op_type is not None and r.get("op_type") != op_type:
            continue
        if core_kind is not None and r.get("core_kind") != core_kind:
            continue
        if hart is not None and r.get("hart") != hart:
            continue
        if instance is not None and r.get("instance") != instance:
            continue
        if impl_hash is not None and r.get("impl_hash") != impl_hash:
            continue
        if run_id is not None and r.get("run_id") != run_id:
            continue
        if unit is not None and r.get("unit") != unit:
            continue
        if require_impl_hash and not r.get("impl_hash"):
            continue
        if require_unit and r.get("unit") in (None, UNIT_UNKNOWN):
            continue
        if not r.get("samples"):
            continue
        groups[r.get(field_name)].append(r)

    out: dict[Any, Measurement] = {}
    for gkey, recs in groups.items():
        units = {r.get("unit") or UNIT_UNKNOWN for r in recs}
        if len(units) > 1:
            raise UnitMismatchError(
                f"{network}/{target}/{quant} {key}={gkey!r} mixes units "
                f"{sorted(units)}. These are different quantities "
                f"(e.g. rdcycle core cycles vs 24 MHz rdtime ticks); this "
                f"module will not convert between them silently. Narrow the "
                f"query with unit=<one of {sorted(units)}>.")
        skeys = {r.get("stable_key") for r in recs}
        if key == "dispatch_id" and len(skeys) > 1:
            raise StableKeyCollisionError(
                f"{network}/{target}/{quant}: dispatch_id={gkey} maps to "
                f"{len(skeys)} different ops across runs "
                f"({sorted(s for s in skeys if s)}). dispatch_ids are "
                f"positional and get renumbered by the IR rewriters, so this "
                f"lookup would average unrelated ops. Re-query with "
                f"key='stable'.")

        samples: list[float] = []
        for r in recs:
            samples.extend(float(v) for v in r["samples"])
        value = _AGGS[agg](samples)
        first = recs[0]
        out[gkey] = Measurement(
            value,
            agg=agg,
            unit=units.pop(),
            n_samples=len(samples),
            n_runs=len({r.get("run_id") for r in recs}),
            stable_key=first.get("stable_key"),
            op_name=first.get("op_name"),
            op_type=first.get("op_type"),
            signature=first.get("signature"),
            dispatch_ids=sorted({r.get("dispatch_id") for r in recs
                                 if r.get("dispatch_id") is not None}),
            impl_hashes=sorted({r.get("impl_hash") for r in recs
                                if r.get("impl_hash")}),
            run_ids=sorted({r.get("run_id") for r in recs if r.get("run_id")}),
            core_kinds=sorted({r.get("core_kind") for r in recs
                               if r.get("core_kind")}),
            harts=sorted({r.get("hart") for r in recs
                          if r.get("hart") is not None}),
            source_artifacts=sorted({r.get("source_artifact") for r in recs
                                     if r.get("source_artifact")}),
            dispatch_table_hashes=sorted({r.get("dispatch_table_hash")
                                          for r in recs
                                          if r.get("dispatch_table_hash")}),
            provenance=("full" if all(r.get("provenance") == "full"
                                      for r in recs) else "partial"),
            samples=samples,
            median=float(statistics.median(samples)),
            p90=_percentile(samples, 90.0),
            mean=float(statistics.mean(samples)),
            min=float(min(samples)),
            max=float(max(samples)),
        )
    return out


def query_records(*args: Any, **kw: Any) -> list[dict]:
    """:func:`query` as a list of plain JSON-serializable dicts.

    Same arguments. Use this when the ``int``-subclass ergonomics of
    :class:`Measurement` get in the way (e.g. writing a report).
    """
    return [{"key": k, **m.as_dict()} for k, m in
            sorted(query(*args, **kw).items(), key=lambda kv: str(kv[0]))]


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _all_records(db_root: pathlib.Path = DEFAULT_DB_ROOT) -> Iterable[dict]:
    db_root = pathlib.Path(db_root)
    if not db_root.exists():
        return
    for path in sorted(db_root.glob("*.jsonl")):
        for r in _load_jsonl(path):
            yield _normalize(r)


def coverage_report(db_root: pathlib.Path = DEFAULT_DB_ROOT) -> dict:
    """What (network, target, quant, op_type) tuples are present.

    Adds two provenance columns the original did not have: ``n_full``
    (records carrying an impl hash) and ``units`` (every distinct unit seen
    for the tuple -- more than one is a red flag).
    """
    by_key: dict[tuple, dict] = defaultdict(
        lambda: {"runs": set(), "cycles": [], "full": 0, "units": set(),
                 "cores": set()})
    seen_combos: set[tuple[str, str, str]] = set()
    for r in _all_records(db_root):
        if r.get("record_kind", "op_measurement") != "op_measurement":
            continue
        combo = (r["network"], r["target"], r["quant"])
        seen_combos.add(combo)
        k = (*combo, r["op_type"])
        agg = by_key[k]
        agg["runs"].add(r["run_id"])
        agg["cycles"].extend(r.get("samples") or [r.get("cycles", 0)])
        agg["units"].add(r.get("unit") or UNIT_UNKNOWN)
        if r.get("impl_hash"):
            agg["full"] += 1
        if r.get("core_kind"):
            agg["cores"].add(r["core_kind"])

    present = []
    for (network, target, quant, op_type), agg in sorted(by_key.items()):
        cycles = agg["cycles"]
        present.append({
            "network": network,
            "target": target,
            "quant": quant,
            "op_type": op_type,
            "n_runs": len(agg["runs"]),
            "n_dispatches": len(cycles),
            "n_full_provenance": agg["full"],
            "units": sorted(agg["units"]),
            "core_kinds": sorted(agg["cores"]),
            "median": int(statistics.median(cycles)),
            "min": int(min(cycles)),
            "max": int(max(cycles)),
        })

    missing = []
    for combo in EXPECTED_MATRIX:
        if combo not in seen_combos:
            missing.append({"network": combo[0], "target": combo[1],
                            "quant": combo[2]})

    return {"present": present, "missing": missing}


def _print_coverage(report: dict) -> None:
    print(f"\n=== Profile DB Coverage ({len(report['present'])} present rows, "
          f"{len(report['missing'])} MISSING combos) ===\n")
    if report["present"]:
        print(f"{'network':<14}{'target':<20}{'quant':<6}{'op_type':<24}"
              f"{'runs':>6}{'disp':>6}{'full':>6}{'median':>14}{'min':>14}"
              f"{'max':>14}  units")
        print("-" * 140)
        for r in report["present"]:
            print(f"{r['network']:<14}{r['target']:<20}{r['quant']:<6}"
                  f"{r['op_type']:<24}{r['n_runs']:>6}{r['n_dispatches']:>6}"
                  f"{r['n_full_provenance']:>6}"
                  f"{r['median']:>14,}{r['min']:>14,}{r['max']:>14,}  "
                  f"{','.join(r['units'])}")
    if report["missing"]:
        print("\nMISSING combos (in EXPECTED_MATRIX but no records):")
        for m in report["missing"]:
            print(f"  - {m['network']:<14}{m['target']:<14}{m['quant']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Scan results/ and append new records.")
    p_ing.add_argument("--results-root", type=pathlib.Path,
                       default=DEFAULT_RESULTS_ROOT)
    p_ing.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)
    p_ing.add_argument("-v", "--verbose", action="store_true")

    p_run = sub.add_parser(
        "ingest-run", help="Ingest one K1 harness stdout with full provenance.")
    p_run.add_argument("--stdout", required=True,
                       help="captured harness stdout (k1_runner --save-output)")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--network", default=None)
    p_run.add_argument("--networks", default=None,
                       help="comma-separated known network names, to split "
                            "instance suffixes off tagged blocks")
    p_run.add_argument("--target", required=True)
    p_run.add_argument("--quant", required=True)
    p_run.add_argument("--unit", required=True, choices=sorted(
        u for u in UNITS if u != UNIT_UNKNOWN),
        help="rdtime_ticks_24mhz for the K1 Linux harness")
    p_run.add_argument("--impl-hash", default=None)
    p_run.add_argument("--kernel-picks", default=None,
                       help="generated/kernel_picks.json; hashed together "
                            "with --impl-source to form the impl hash")
    p_run.add_argument("--impl-source", action="append", default=[],
                       help="repeatable kernel/source path")
    p_run.add_argument("--core-kind", default=None)
    p_run.add_argument("--hart", type=int, default=None)
    p_run.add_argument("--cpu-id", type=int, default=None)
    p_run.add_argument("--graph", default=None,
                       help="graph.json the binary was generated from")
    p_run.add_argument("--no-strict-ir", action="store_true")
    p_run.add_argument("--dispatch-table", default=None)
    p_run.add_argument("--git-sha", default="")
    p_run.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)

    p_cov = sub.add_parser("coverage", help="Print coverage matrix.")
    p_cov.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)
    p_cov.add_argument("--json", action="store_true",
                       help="emit JSON instead of table")

    p_q = sub.add_parser("query", help="Return per-op aggregated cost.")
    p_q.add_argument("--network", required=True)
    p_q.add_argument("--target", required=True)
    p_q.add_argument("--quant", required=True)
    p_q.add_argument("--op-type", default=None)
    p_q.add_argument("--agg", default="median", choices=sorted(_AGGS))
    p_q.add_argument("--key", default="dispatch_id",
                     choices=sorted(set(_KEY_FIELDS)))
    p_q.add_argument("--core-kind", default=None)
    p_q.add_argument("--hart", type=int, default=None)
    p_q.add_argument("--unit", default=None)
    p_q.add_argument("--require-impl-hash", action="store_true")
    p_q.add_argument("--full", action="store_true",
                     help="emit full provenance per key, not just the value")
    p_q.add_argument("--db-root", type=pathlib.Path, default=DEFAULT_DB_ROOT)

    args = ap.parse_args()

    if args.cmd == "ingest":
        added = ingest(args.results_root, args.db_root, verbose=args.verbose)
        print(f"added {added} record(s)")
        return 0

    if args.cmd == "ingest-run":
        impl_hash = args.impl_hash
        sources = list(args.impl_source)
        if impl_hash is None and args.kernel_picks:
            impl_hash = impl_hash_from_kernel_picks(args.kernel_picks,
                                                    extra=sources)
            sources = [args.kernel_picks] + sources
        res = ingest_run(
            args.stdout,
            run_id=args.run_id,
            unit=args.unit,
            target=args.target,
            quant=args.quant,
            network=args.network,
            networks=(args.networks.split(",") if args.networks else None),
            impl_hash=impl_hash,
            impl_sources=sources or None,
            core_kind=args.core_kind,
            hart=args.hart,
            cpu_id=args.cpu_id,
            graph=args.graph,
            strict_ir=not args.no_strict_ir,
            dispatch_table=args.dispatch_table,
            git_sha=args.git_sha,
            db_root=args.db_root,
        )
        print(f"added {res.added} record(s), skipped {res.skipped} "
              f"already-present -> {', '.join(res.paths)}")
        return 0

    if args.cmd == "coverage":
        report = coverage_report(args.db_root)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            _print_coverage(report)
        return 0

    if args.cmd == "query":
        out = query(args.network, args.target, args.quant,
                    op_type=args.op_type, agg=args.agg, db_root=args.db_root,
                    key=args.key, core_kind=args.core_kind, hart=args.hart,
                    unit=args.unit,
                    require_impl_hash=args.require_impl_hash)
        if args.full:
            print(json.dumps([{"key": k, **m.as_dict()}
                              for k, m in sorted(out.items(),
                                                 key=lambda kv: str(kv[0]))],
                             indent=2))
        else:
            print(json.dumps(
                {str(k): {"value": m.value, "unit": m.unit,
                          "n_samples": m.n_samples, "agg": m.agg}
                 for k, m in sorted(out.items(), key=lambda kv: str(kv[0]))},
                indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
