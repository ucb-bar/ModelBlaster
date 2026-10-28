"""Resume for the graph optimiser (``kb_graph_opt``) inside a round, and its signal handling.

A round of ``kb_graph_opt`` runs these phases:

1. profile;
2. propose;
3. plan;
4. export;
5. the KernelBlaster run;
6. ingest;
7. KernelBlaster's proposals;
8. the hint decisions, in waves.

A round that is killed (SIGKILL, OOM, reboot, ssh drop, Ctrl-C) continues
in the interrupted phase when ``kb_graph_opt`` is rerun on the work dir.
``rNN/progress.json`` (written atomically) holds:

* ``profile``: the profiles, so the proposals and the plan are the ones the
  killed attempt made;
* ``export``: the exported problem dirs; they are never re-exported, so
  KernelBlaster's journals still match them;
* ``ingest``: the winners ingest report;
* ``decide``: the candidate hints, their waves, and the decisions of each
  finished wave.

The KernelBlaster runs resume by themselves: their out dirs keep
KernelBlaster's journals (``run_multislot.py --resume``). Finished problems are
skipped and in-flight steps reuse their journaled LLM responses. A hint's
exported original / variant problems are kept in ``<hint dir>/job.json``
(:func:`prepared`), so a wave that is redone runs the same problems.

A round started under another configuration or another IR set is moved aside
(``rNN.stale-<time>``) and redone from the start. The work dir itself
(hints, cache, knowledge) is shared across runs by design.

Signals (:class:`Stop`):

* **First SIGTERM / SIGINT / SIGHUP.** It is forwarded to the running
  KernelBlaster processes. Each stops starting steps, finishes its in-flight
  ones, flushes its journals and exits 75. Then ``kb_graph_opt`` exits
  ``EXIT_INTERRUPTED`` (75).
* **Second signal.** It is forwarded too; KernelBlaster exits at once, and so
  does ``kb_graph_opt`` (``EXIT_ABORTED``, 76).

KernelBlaster children run in their own session, so a terminal's Ctrl-C reaches
them once, through ``kb_graph_opt``. They also stop by themselves when
``kb_graph_opt`` dies (``KB_STOP_WITH_PARENT``).
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Callable

FORMAT = "mb-round-progress/1"
EXIT_INTERRUPTED = 75
EXIT_ABORTED = 76

# Config fields that decide what a round does (operational ones -- GPUs, hosts, parallelism,
# evaluator -- may change between a killed attempt and its resume)
ROUND_FIELDS = ("model", "reflect_model", "arch", "rollouts", "steps", "variant_steps", "max_fix", "coverage",
                "top_k", "min_share", "min_gain_pct", "max_variants", "remeasure", "accept_gain", "profile_runs",
                "seed")


class Interrupted(Exception):
    """The run stopped on a signal with its progress saved: rerun to resume."""


# ---------------------------------------------------------------- files

def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, default=str))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def round_key(cfg, irs) -> dict:
    key = {k: getattr(cfg, k, None) for k in ROUND_FIELDS}
    key["irs"] = sorted(str(Path(d).resolve()) for d in irs)
    if getattr(cfg, "target", "cuda") != "cuda":      # (absent for CUDA: its keys are unchanged)
        key["target"] = cfg.target
    return key


def key_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- rounds

class RoundProgress:
    """``rNN/progress.json``: the phases a round has finished and what they produced."""

    def __init__(self, R: Path, key: dict):
        self.R = Path(R)
        self.path = self.R / "progress.json"
        self.key, self.key_hash = key, key_hash(key)
        self.moved: str | None = None
        d = read_json(self.path)
        if self.R.exists() and (d is None or d.get("key_hash") != self.key_hash):
            # an attempt under another configuration / IR set, or from before progress.json existed
            stale = self.R.with_name(f"{self.R.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
            self.R.rename(stale)
            self.moved, d = str(stale), None
        self.resumed = d is not None
        self.data = d or {"format": FORMAT, "key": key, "key_hash": self.key_hash, "phases": {},
                          "attempts": []}
        self.data["attempts"].append({"pid": os.getpid(), "start": time.time(),
                                      "resumed_after": sorted(self.data["phases"])})
        self._save()

    def _save(self) -> None:
        atomic_write_json(self.path, self.data)

    def done(self, phase: str) -> bool:
        return phase in self.data["phases"]

    def get(self, phase: str, default=None):
        return self.data["phases"].get(phase, {}).get("value", default)

    def mark(self, phase: str, value=None) -> None:
        self.data["phases"][phase] = {"value": value, "ts": time.time()}
        self._save()

    def once(self, phase: str, fn: Callable[[], Any]):
        """``fn()`` the first time; its saved value on a resumed attempt."""
        if self.done(phase):
            return self.get(phase)
        v = fn()
        self.mark(phase, v)
        return v


def prepared(prepare: Callable, h: dict, prof: dict, out: Path, **kw) -> dict:
    """``prepare_hint`` once per hint and round: the exported original / variant problems are
    kept (``out/job.json``), so a redone wave runs exactly the problems KernelBlaster journaled
    (re-exporting could seed them differently: an earlier attempt of the wave may have
    ingested the variant's slots into the cache already)."""
    from .kb_proposals import with_fused_specs

    f = Path(out) / "job.json"
    d = read_json(f)
    if d and d.get("hint_id") == h["id"] and all((Path(d["job"][k]) / "manifest.json").exists()
                                                  for k in ("orig", "variant")):
        graph = json.loads((Path(prof["ir_dir"]) / "graph.json").read_text())
        job = dict(d["job"], hint=h, specs=with_fused_specs(graph, kw["specs"]), resumed=True)
        job["orig"], job["variant"] = Path(job["orig"]), Path(job["variant"])
        return job
    job = prepare(h, prof, out, **kw)
    if "early" not in job:
        atomic_write_json(f, {"hint_id": h["id"], "job": {k: (str(v) if isinstance(v, Path) else v)
                                                          for k, v in job.items() if k not in ("hint", "specs")}})
    return job


def decide_hints(prog: RoundProgress, cands: list, graphs: dict, out: Path, width: int, *, decide_wave,
                 schedule, footprint, specs: dict, hints_path: Path, save_hints) -> tuple[list, list]:
    """``kb_graph_opt.decide_hints`` with its waves and the finished waves' decisions in
    ``prog`` (phase ``decide``): a resumed round skips the finished waves and redoes the
    interrupted one (its KernelBlaster run resumes from its journals). ``decide_wave(k, items)``
    decides one wave and saves the hints file."""
    st = prog.get("decide") or {}
    ids = [h["id"] for h, _ in cands]
    if st.get("cands") != ids:
        fps = [footprint(h, p["ir"], graphs[p["ir"]], specs) for h, p in cands]
        st = {"cands": ids, "waves": schedule(fps, width), "decided": {}}
        prog.mark("decide", st)
    decided: dict[int, dict] = {int(i): d for i, d in st["decided"].items()}
    for k, w in enumerate(st["waves"]):
        if all(i in decided for i in w):
            continue
        items = [(cands[i][0], cands[i][1], out / _safe(cands[i][0]["id"])) for i in w]
        for i, d in zip(w, decide_wave(k, items)):
            decided[i] = d
        save_hints()
        st["decided"] = {str(i): d for i, d in decided.items()}
        prog.mark("decide", st)                  # after the hints file: a decision is never recorded twice
        STOP.check()
    return [[ids[i] for i in w] for w in st["waves"]], [decided[i] for i in sorted(decided)]


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def kb_supports_resume(kb: Path) -> bool:
    """The KernelBlaster checkout has resumable runs (``run_multislot.py --resume``)."""
    try:
        return "--force-restart" in (Path(kb) / "scripts" / "run_multislot.py").read_text()
    except OSError:
        return False


# ---------------------------------------------------------------- signals

class Stop:
    """First SIGTERM / SIGINT / SIGHUP: forwarded to the KernelBlaster children (they stop
    cleanly); the flow raises :class:`Interrupted` at its next check. Second: forwarded again,
    and the process exits ``EXIT_ABORTED`` at once."""

    def __init__(self):
        self.requested: int | None = None
        self._procs: set = set()
        self._lock = threading.Lock()
        self.installed = False

    def install(self) -> None:
        if self.installed or threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._on)
        os.environ["KB_STOP_WITH_PARENT"] = "1"      # KernelBlaster children stop when this process dies
        self.installed = True

    def _on(self, sig, _frame) -> None:
        first = self.requested is None
        self.requested = sig
        self._forward()
        if not first:
            os._exit(EXIT_ABORTED)
        print(f"[kb_graph_opt] {signal.Signals(sig).name}: stopping after the running KernelBlaster steps "
              f"(resumable: rerun the same command); a second signal exits at once", flush=True)

    def _forward(self) -> None:
        with self._lock:
            procs = list(self._procs)
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass

    def register(self, proc) -> None:
        with self._lock:
            self._procs.add(proc)
        if self.requested is not None:
            self._forward()

    def unregister(self, proc) -> None:
        with self._lock:
            self._procs.discard(proc)

    def check(self) -> None:
        if self.requested is not None:
            raise Interrupted(f"stopped on {signal.Signals(self.requested).name}")


STOP = Stop()


def kb_exit(rc: int, log: Path) -> None:
    """A KernelBlaster run's exit status: 75 = stopped on a signal (resumable)."""
    if rc == EXIT_INTERRUPTED or (STOP.requested is not None and rc != 0):
        raise Interrupted(f"KernelBlaster stopped on a signal (exit {rc}); see {log}")
