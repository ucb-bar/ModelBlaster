"""Dead-op pruning at export (A2 capabilities, item 1).

An op is *dead* when none of its outputs can reach a graph output: no live op reads any of them and none of
them is a graph output. Liveness is computed backwards over the whole IR graph (views and aliases are ops whose
output is the reader's input, so they are covered by the same rule). Ops without outputs, and ops of kinds with
side effects beyond their outputs (``kb_concurrency.SERIAL_KINDS``), are always live.

``kb_export.export_problem`` drops dead ops from the partition it exports (and does not count their reads as
"consumed outside", so a live op's output that only a dead op read becomes an intermediate instead of a checked
partition output). The IR itself is never rewritten: profiles, plans and hint matching still see every op;
dead ops simply have no call in the walker and no time in the profile.

Switch: ``$MB_PRUNE_DEAD`` (``1``/``on`` default, ``0``/``off`` = legacy export, byte-identical), or the
``prune_dead`` argument of ``export_problem``. Example: 41_GRUBidirectional's h_n stack (3 catn ops, ~22 us)
whose result the model never returns.
"""
from __future__ import annotations

import os

ENV = "MB_PRUNE_DEAD"


def enabled(value: bool | str | None = None) -> bool:
    """Resolve the switch: an explicit value wins, else ``$MB_PRUNE_DEAD``, else ON."""
    if value is None:
        value = os.environ.get(ENV, "1")
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("1", "on", "true", "yes", ""):
        return True
    if v in ("0", "off", "false", "no"):
        return False
    raise ValueError(f"{ENV}={value!r}: expected on|off")


def _side_effect_kinds() -> set[str]:
    try:
        from .kb_concurrency import SERIAL_KINDS
        return set(SERIAL_KINDS)
    except Exception:                                   # pragma: no cover - import guard
        return set()


def live_tensors(graph: dict, outputs: list[str] | None = None) -> set[str]:
    """Every tensor whose value can reach a graph output."""
    return _analyse(graph, outputs)[1]


def dead_ops(graph: dict, outputs: list[str] | None = None) -> list[int]:
    """Indices (into ``graph["ops"]``) of ops whose outputs never reach a graph output, in IR order."""
    return sorted(_analyse(graph, outputs)[0])


def _graph_outputs(graph: dict) -> list[str]:
    o = graph.get("output") or {}
    return list(o.get("tensors") or ([o["tensor"]] if o.get("tensor") else []))


def _analyse(graph: dict, outputs: list[str] | None):
    ops = graph["ops"]
    outs = list(outputs) if outputs is not None else _graph_outputs(graph)
    if not outs:                                         # nothing known to be returned: prune nothing
        return set(), set()
    serial = _side_effect_kinds()
    live = set(outs)
    dead: set[int] = set()
    for i in range(len(ops) - 1, -1, -1):
        o = ops[i]
        produced = list(o.get("outputs", []))
        keep = (not produced) or o["op"] in serial or any(t in live for t in produced)
        if keep:
            live.update(o.get("inputs", []))
        else:
            dead.add(i)
    return dead, live


def prune_selection(graph: dict, sel: list[int], outputs: list[str] | None = None,
                    enable: bool | str | None = None) -> tuple[list[int], list[int]]:
    """(live part of ``sel``, dead op indices of the whole graph). With pruning off, or when every selected op
    is dead (a partition holding only dead ops is exported as is, so callers never get an empty problem), ``sel``
    is returned unchanged; the second element is then empty."""
    if not enabled(enable):
        return list(sel), []
    dead = dead_ops(graph, outputs)
    if not dead:
        return list(sel), []
    ds = set(dead)
    keep = [i for i in sel if i not in ds]
    if not keep:
        return list(sel), []
    return keep, dead


def report(graph: dict, outputs: list[str] | None = None) -> dict:
    """Summary for logs / tests: dead op names and kinds."""
    dead = dead_ops(graph, outputs)
    ops = graph["ops"]
    return {"n_ops": len(ops), "dead": [ops[i]["name"] for i in dead],
            "dead_kinds": sorted({ops[i]["op"] for i in dead})}
