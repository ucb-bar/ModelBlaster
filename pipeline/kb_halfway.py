"""Halfway sync of global knowledge inside a round (node side; ``--knowledge-global-halfway on``).

With ``refresh: round`` / ``merge_on: round`` knowledge is written when a round completes and read at the START of
the next, so a problem that lasts about one round only ever sees its seed snapshot. The halfway sync does the same
exchange (same mechanisms, same package format) once more, at a clean step boundary in the MIDDLE of the round's
KernelBlaster run, without restarting the search:

1. the plan's KernelBlaster run is started with ``--pause-spec`` (run_multislot.py): every plan problem parks once all
   its rollouts completed ``ceil(steps / 2)`` steps (``pause_at``), flushes its journal (each rollout's current kernel,
   the composite, the counters, the LLM journal) and the run exits 77 (``EXIT_PAUSED``) with ``paused.json``;
2. this module then writes ``<round dir>/halfway.json`` ``{round, stage: "paused", steps_done, ts}`` and waits (bounded,
   ``--knowledge-global-halfway-wait-s``, default 180 s; heartbeat file ``halfway.hb``) for the controller (farm.py
   ``halfway_sync``) to pull the run's knowledge.db, merge it (``why="round<n>h"``), build a refresh package of what
   OTHER runs merged since this run's sync point, upload it to ``--knowledge-global-refresh`` and write
   ``halfway-r<round>.ack.json`` next to it;
3. the package is imported with the existing refresh import (``kb_global.refresh_at_round``: one transaction, no
   KernelBlaster process has the db open: the run has exited), and the SAME run_multislot command is started again
   without the pause: each rollout continues from ITS OWN state (its next step and current kernel), the composite
   is the composite, no journaled LLM response is paid twice. Only the knowledge store and the implementation
   store changed under it.

Never blocks or fails the run: no ack within the timeout -> resume without the package (``halfway_timeout``); any
error -> ``halfway_error`` and the run continues. ``halfway.json`` ``stage`` makes it resume-safe: absent = the first
half runs (a killed first half resumes from its journals and parks at the same step); ``paused`` = a crash during the
pause: no KernelBlaster run is started, the wait / import is redone; ``done`` = the second half runs, never paused
again. Everything is off (no flag, no marker in the round key, KernelBlaster started exactly as before) unless the
flag is on, a seed was applied and a refresh inbox is configured.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXIT_PAUSED = 77
FORMAT = "mb-halfway/1"


def kb_supports_pause(kb) -> bool:
    """The KernelBlaster checkout has the pause hook (``run_multislot.py --pause-spec``)."""
    try:
        return "--pause-spec" in (Path(kb) / "scripts" / "run_multislot.py").read_text()
    except OSError:
        return False


def enabled(cfg, gk) -> bool:
    """On only when asked for, a seed was applied (the run can import: its sync point is known) and the refresh
    inbox is set; KernelBlaster must have the hook (checked by the caller via :func:`kb_supports_pause`)."""
    return (str(getattr(cfg, "knowledge_global_halfway", "off") or "off").lower() in ("on", "1", "true", "yes")
            and gk is not None and bool(gk.get("applied")) and bool(getattr(cfg, "knowledge_global_refresh", ""))
            and kb_supports_pause(getattr(cfg, "kb", ".")))


def key_fields(cfg, gk) -> dict:
    """Round key: nothing when off (unchanged key); a constant marker when on (never changes across the pause)."""
    return {"knowledge_global_halfway": True} if enabled(cfg, gk) else {}


def _read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _write(p: Path, obj) -> None:
    from .kb_resume import atomic_write_json
    atomic_write_json(Path(p), obj)


def pause_at(steps: int, frac: float = 0.5) -> int | None:
    """Step at which a problem with ``steps`` steps per rollout parks (same rule as run_multislot ``_pause_at``)."""
    import math
    n = math.ceil(int(steps) * frac - 1e-9)
    return n if 0 < n < int(steps) else None


class _PausedProc:
    """Stands for a KernelBlaster run that already exited paused (a crash during the pause): ``wait()`` -> 77."""
    returncode = EXIT_PAUSED
    pid = 0

    def poll(self):
        return EXIT_PAUSED

    def wait(self, timeout=None):
        return EXIT_PAUSED

    def send_signal(self, sig):
        pass

    def kill(self):
        pass


class Halfway:
    """The halfway sync of one flow run (``kb_graph_opt._run``): ``arm`` before the round's plan run, then
    :class:`~modelblaster.pipeline.kb_graph_opt.KBRunner` asks :meth:`flags` / :meth:`mode` at start and calls
    :meth:`sync` when the run exits paused."""

    def __init__(self, cfg, work: Path, state: dict, state_path: Path, gk: dict, *, sleep=time.sleep, now=time.time):
        self.cfg, self.work, self.state, self.state_path, self.gk = cfg, Path(work), state, state_path, gk
        self.sleep, self.now = sleep, now
        self.wait_s = float(getattr(cfg, "knowledge_global_halfway_wait_s", 180.0) or 0.0)
        self.poll_s = float(os.environ.get("MB_HALFWAY_POLL_S", "1.0"))
        self.armed: dict[str, dict] = {}                      # str(out dir) -> {"round", "problems"}

    # -- arming
    def arm(self, rnd: int, out: Path, problems: list[str]) -> None:
        """The run in ``out`` (the round's plan run) pauses the named plan problems halfway."""
        if problems:
            self.armed[str(out)] = {"round": int(rnd), "problems": sorted(problems)}

    def disarm(self, out: Path) -> None:
        self.armed.pop(str(out), None)

    def active(self, out: Path) -> bool:
        return str(out) in self.armed

    def _file(self, out: Path) -> Path:
        return Path(out).parent / "halfway.json"

    def mode(self, out: Path) -> str:
        """normal (not armed / second half) | pause (first half) | paused (crash during the pause)."""
        if not self.active(out):
            return "normal"
        st = (_read(self._file(out)) or {}).get("stage")
        return {"done": "normal", "paused": "paused"}.get(st, "pause")

    def flags(self, out: Path) -> list[str]:
        """Extra run_multislot arguments (first half only)."""
        if self.mode(out) != "pause":
            return []
        spec = Path(out) / "pause_spec.json"
        Path(out).mkdir(parents=True, exist_ok=True)
        _write(spec, {"frac": 0.5, "problems": self.armed[str(out)]["problems"], "round": self.armed[str(out)]["round"]})
        return ["--pause-spec", str(spec)]

    # -- the pause
    def sync(self, out: Path) -> dict:
        """The run in ``out`` exited paused: wait for the controller's package, import it, record. Never raises."""
        info = self.armed[str(out)]
        r = info["round"]
        hf = self._file(out)
        t0 = self.now()
        rec = {"round": r, "ts": t0, "outcome": None}
        try:
            paused = _read(Path(out) / "paused.json") or {}
            prev = _read(hf) or {}
            _write(hf, {"format": FORMAT, "round": r, "stage": "paused", "ts": prev.get("ts") or t0,
                        "steps_done": {k: v.get("steps_done") for k, v in (paused.get("problems") or {}).items()},
                        "pause_at": {k: v.get("pause_at") for k, v in (paused.get("problems") or {}).items()},
                        "wait_s": self.wait_s, "pid": os.getpid()})
            res = self._wait_and_import(out, r, rec)
        except Exception as e:  # noqa: BLE001 -- knowledge never fails a run
            res = None
            rec.update(outcome="error", error=repr(e)[:300])
        rec["wait_s"] = round(self.now() - t0, 2)
        try:
            h = self.state.get("knowledge_global")
            if h is not None:
                t = h.setdefault("halfway", {"n": 0, "imported_rows": 0, "wait_s": 0.0, "timeouts": 0, "errors": 0,
                                             "events": []})
                t["n"] += 1
                t["wait_s"] = round(t["wait_s"] + rec["wait_s"], 2)
                t["timeouts"] += rec["outcome"] == "timeout"
                t["errors"] += rec["outcome"] == "error"
                t["imported_rows"] += int(rec.get("imported_rows") or 0)
                t["events"] = (t["events"] + [rec])[-20:]
                if rec["outcome"] == "timeout":
                    t["halfway_timeout"] = True
                if rec["outcome"] == "error":
                    t["halfway_error"] = rec.get("error")
                from .kb_resume import atomic_write_json
                atomic_write_json(self.state_path, self.state)
        except Exception as e:  # noqa: BLE001
            rec.setdefault("error", repr(e)[:300])
        try:
            _write(hf, {"format": FORMAT, "round": r, "stage": "done", "ts": (_read(hf) or {}).get("ts") or t0,
                        "done_ts": self.now(), **{k: v for k, v in rec.items() if k not in ("ts", "round")}})
        except Exception:  # noqa: BLE001
            pass
        print(json.dumps({"round": r, "knowledge_global_halfway": {k: rec.get(k) for k in
                                                                    ("outcome", "wait_s", "imported_rows", "error")}}),
              flush=True)
        return rec

    def _wait_and_import(self, out: Path, r: int, rec: dict):
        inbox = Path(self.cfg.knowledge_global_refresh)
        ack_f = inbox / f"halfway-r{r:02d}.ack.json"
        hb = Path(out).parent / "halfway.hb"
        deadline = self.now() + self.wait_s
        ack = None
        while True:
            ack = _read(ack_f) if ack_f.exists() else None
            if ack is not None:
                break
            if self.now() >= deadline:
                break
            try:
                hb.write_text(json.dumps({"ts": self.now(), "pid": os.getpid()}))
            except OSError:
                pass
            self.sleep(min(self.poll_s, max(0.0, deadline - self.now())) or 0.01)
        if ack is None:
            rec.update(outcome="timeout")
            return None
        rec["ack"] = {k: ack.get(k) for k in ("package", "status", "merge", "error") if k in ack}
        from . import kb_global
        rf = kb_global.refresh_at_round(self.work, self.cfg, self.state, r, why="halfway")
        try:
            done = inbox / "done"
            done.mkdir(exist_ok=True)
            os.replace(ack_f, done / ack_f.name)
        except OSError:
            pass
        if rf is None:
            rec.update(outcome="none", imported_rows=0)             # nothing was delivered (nothing new for this run)
            return None
        n = sum(v for v in (rf.get("counts") or {}).values() if isinstance(v, int))
        rec.update(outcome="imported" if rf.get("status") in ("applied", "already_applied", "up_to_date") else
                   "error" if rf.get("status") in ("error", "refused") else rf.get("status"),
                   import_status=rf.get("status"), counts=rf.get("counts"), imported_rows=n,
                   **({"error": str(rf.get("error") or rf.get("reason"))[:300]} if rf.get("error") or
                      rf.get("status") in ("error", "refused") else {}))
        return rf


def stub_ack(inbox: Path, r: int, package: str | None = None, **kw) -> Path:
    """What the controller writes after uploading (tests / scripted controllers)."""
    f = Path(inbox) / f"halfway-r{r:02d}.ack.json"
    _write(f, {"round": r, "package": package, "ts": time.time(), **kw})
    return f
