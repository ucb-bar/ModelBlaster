"""Trace store for ModelBlaster's side of the graph optimiser (``kb_graph_opt``).

Like :mod:`kb_telemetry`, ModelBlaster loads KernelBlaster's stdlib-only
``kernelblaster.trace`` package from the checkout it drives (``Config.kb``),
under a private module name; a checkout without it, or ``--trace off`` /
``KB_TRACE=off``, gives the no-op :class:`_Off`.

``<work>/trace`` (same format as a KernelBlaster run's ``<out>/trace``, see
KernelBlaster ``docs/trace.md``) gets one record per ModelBlaster step:

* ``mb.profile`` -- an IR's whole-graph profile (``profile.json``: per-op times,
  CUPTI attribution, device), with the build and driver output of its runs;
* ``mb.plan`` / ``mb.export`` -- the plan's inputs (profiles, excluded items) and
  output (``plan.json``), the exported problems' manifests;
* ``mb.kb_run`` -- one KernelBlaster run: command (secrets scrubbed), exit code,
  its log, each problem's result summary, and the run's own trace store
  (``kb_trace``: the prompts, candidates, builds and decisions inside it);
* ``mb.ingest`` -- what went into the kernel cache;
* ``mb.hint`` -- a hint decision: the original / variant IR graphs and manifests,
  KernelBlaster's two results, the re-measured times (raw driver output of each
  re-timing run in ``driver`` records under ``mb_stage=remeasure``), gain, decision;
* ``mb.kb_time`` -- a run of KernelBlaster's ``time_problems.py`` (jobs, result, log);
* ``mb.verify`` / ``mb.model_run`` -- the final verification and the model's result.

Records carry ``mb_round`` (from the telemetry ``mb.round`` span), ``mb_stage``
and ``ir`` / ``hint`` where known. ``scripts/trace_view.py WORK --mb`` (in the
KernelBlaster checkout) lists them; ``--list`` lists every KernelBlaster run's steps.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_MOD = "_mb_kb_trace"


class _Off:
    """No-op stand-in (tracing off or not available in the checkout)."""

    available = False

    def configure(self, *a, **kw):
        return None

    def close(self):
        return None

    def enabled(self):
        return False

    def scope(self, **kw):
        return contextlib.nullcontext({})

    def set_attrs(self, **kw):
        pass

    def record(self, *a, **kw):
        return None

    def put(self, *a, **kw):
        return None

    def put_json(self, *a, **kw):
        return None

    def tool(self, *a, **kw):
        pass

    def output(self, *a, **kw):
        pass

    def key(self, *a, **kw):
        return ""

    def store(self):
        return None


_OFF = _Off()


def _level(level: str | None) -> str:
    v = (level or os.getenv("KB_TRACE") or "full").strip().lower()
    return {"0": "off", "false": "off", "no": "off", "1": "full", "true": "full", "on": "full"}.get(v, v)


def load(kb_root, level: str | None = None) -> object:
    """KernelBlaster's trace package from ``kb_root``, or :class:`_Off`."""
    if _level(level) == "off":
        return _OFF
    if _MOD in sys.modules:
        return sys.modules[_MOD]
    pkg = Path(kb_root) / "src" / "kernelblaster" / "trace"
    if not (pkg / "__init__.py").exists():
        return _OFF
    try:
        spec = importlib.util.spec_from_file_location(_MOD, pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_MOD] = mod
        spec.loader.exec_module(mod)
        mod.available = True
        return mod
    except Exception:  # noqa: BLE001 -- tracing must never break the flow
        sys.modules.pop(_MOD, None)
        return _OFF


def get() -> object:
    """The loaded trace package (after :func:`load`) when a store is open, else the no-op."""
    m = sys.modules.get(_MOD)
    return m if m is not None and m.enabled() else _OFF


def configure(kb_root, work: Path, level: str | None = None):
    """Open ``work/trace`` at ``level`` (``KB_TRACE``; default full) -> the store or None."""
    lv = _level(level)
    t = load(kb_root, lv)
    if t is _OFF:
        return None
    return t.configure(Path(work) / "trace", level=lv, run_id=f"mb-{Path(work).name}")


def close():
    m = sys.modules.get(_MOD)
    return m.close() if m is not None else None


def kb_supports_trace(kb_root) -> bool:
    """The checkout's run_multislot.py takes ``--trace`` (older pins do not)."""
    try:
        return '"--trace"' in (Path(kb_root) / "scripts" / "run_multislot.py").read_text()
    except OSError:
        return False


def files(paths: dict) -> dict:
    """{label: path} -> {label: blob ref} for the files that exist (text, capped by the store)."""
    t = get()
    out = {}
    for k, p in paths.items():
        try:
            if p is not None and Path(p).is_file():
                out[k] = t.put(Path(p).read_text(errors="replace"))
        except OSError:
            pass
    return out


def record(kind: str, *, rid_parts=(), **fields) -> None:
    """One ModelBlaster record (``kind`` gets the ``mb.`` prefix); never raises."""
    t = get()
    if t is _OFF:
        return
    try:
        rid = t.key("mb", kind.removeprefix("mb."), *rid_parts) if rid_parts else None
        t.record(kind if kind.startswith("mb.") else f"mb.{kind}", rid=rid, **fields)
    except Exception:  # noqa: BLE001
        pass


def put_json(obj):
    t = get()
    try:
        return t.put(json.dumps(obj, indent=1, default=str)) if t is not _OFF else None
    except Exception:  # noqa: BLE001
        return None
