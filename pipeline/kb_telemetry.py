"""Telemetry for ModelBlaster's graph optimiser (``kb_graph_opt``).

ModelBlaster does not re-implement the span machinery: it loads
KernelBlaster's stdlib-only ``kernelblaster.telemetry`` package from the
checkout it already drives (``Config.kb``), under a private module name so
nothing else of KernelBlaster is imported into this process. A checkout
without it (an older pin) or ``KB_TELEMETRY=0`` gives the no-op
:class:`_Off`, and the flow runs unchanged.

ModelBlaster's own spans (``mb.*``: profile, propose, plan, export,
kb_wait, ingest, decide, prepare, remeasure, finish, verify) go to
``<work>/telemetry.jsonl``; every KernelBlaster run writes its own
``telemetry.jsonl`` and ``summary.json`` telemetry, which
:func:`kb_run_summary` reads back so ``round.json`` and ``summary.json``
carry both sides: where ModelBlaster's wall clock went and, summed over the
KernelBlaster runs, tokens by role / model, truncations and the agent-loop
breakdown. ``scripts/telemetry_report.py WORK`` in the KernelBlaster
checkout prints the tables for all of them.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

_MOD = "_mb_kb_telemetry"


class _Off:
    """No-op stand-in (telemetry disabled or not available)."""

    available = False

    def span(self, *a, **kw):
        return contextlib.nullcontext(None)

    def within(self, sp):
        return contextlib.nullcontext(sp)

    def current(self):
        return None

    def count(self, *a, **kw):
        pass

    def configure(self, *a, **kw):
        return None

    def close(self):
        pass

    def full_summary(self, sp):
        return None

    def lap(self, *a, **kw):
        pass

    def end_laps(self, sp):
        pass

    def rollup(self, summaries):
        return None


_OFF = _Off()


def load(kb_root) -> object:
    """KernelBlaster's telemetry package from ``kb_root``, or :class:`_Off`."""
    if os.getenv("KB_TELEMETRY", "1").strip().lower() in ("0", "false", "no", "off"):
        return _OFF
    if _MOD in sys.modules:
        return sys.modules[_MOD]
    pkg = Path(kb_root) / "src" / "kernelblaster" / "telemetry"
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
    except Exception:  # noqa: BLE001  -- telemetry must never break the flow
        sys.modules.pop(_MOD, None)
        return _OFF


def get() -> object:
    """The loaded telemetry module (after :func:`load`), else the no-op."""
    return sys.modules.get(_MOD) or _OFF


def run_within(parent, name: str, fn, *a, **kw):
    """``fn(*a, **kw)`` in span ``name`` under ``parent`` -- for worker
    threads, which do not inherit the submitting thread's span."""
    t = get()
    with t.within(parent):
        with t.span(name):
            return fn(*a, **kw)


class section:
    """A span opened and closed by calls (``s = section("x"); ...; s.end()``),
    for a region of a long function that should not be re-indented."""

    def __init__(self, name: str, **attrs):
        self._cm = get().span(name, **attrs)
        self.span = self._cm.__enter__()

    def end(self):
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None


def kb_run_summary(out: Path, results: list[dict] | None = None) -> dict | None:
    """One KernelBlaster run's run-level telemetry (its ``summary.json``
    entry: every problem plus the final reflection passes), or None."""
    try:
        for entry in json.loads((Path(out) / "summary.json").read_text()):
            if isinstance(entry, dict) and set(entry) == {"telemetry"} and isinstance(entry["telemetry"], dict):
                return entry["telemetry"]
    except (OSError, ValueError, TypeError):
        pass
    return None


def _loaded():
    return sys.modules.get(_MOD)


def kb_rollup(summaries: list) -> dict | None:
    tel = _loaded()
    if tel is None or not any(summaries):
        return None
    return tel.rollup(summaries)


def compact(s: dict | None) -> dict | None:
    """The parts of a collector summary worth keeping in round.json."""
    if not s:
        return None
    keep = ("wall_s", "time_s", "time_pct", "busy_s", "counters", "gpu_busy_s", "derived")
    return {k: s[k] for k in keep if k in s}
