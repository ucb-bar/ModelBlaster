"""Park after the characterization pass (barrier scheduling, node side; ``--park-after-char on``; l3_farm/analysis/barrier_scheduling.md).

Barrier scheduling (X-151) runs the sweep in two phases. Phase 1: every problem extracts its IR, profiles the whole model and
runs A4's characterization pass (round 0: a cheap KernelBlaster pass on the largest ops; ``--graph-characterize N``), then PARKS:
the flow writes ``<work>/parked.json``, flushes its state and exits **78** (``EXIT_PARKED``) without starting the re-profile, the
pre-apply or the plan. The farm controller pulls the parked run's knowledge, merges it, frees the node / driver slot and, once
(nearly) every problem has parked (a global soft barrier), resumes the parked problems for phase 2.

Resume is the flow's own: the SAME command without the flag continues the round at its next phase (``rNN/progress.json``
phases ``gs_char`` is done, ``gs_char_profile`` is not), so the park flag is not part of the round key (a parked, then resumed
problem keeps its key). At resume start the newest refresh package (the controller built it for the post-barrier store) is
imported once (``kb_global.refresh_at_round(..., why="barrier")``: atomic, never double counted, never blocks or fails the run).

Flag off (the default): nothing here runs, ``state.json`` / ``progress.json`` / the round key / the command are unchanged.
Unparkable runs just continue (``state.json`` ``barrier.park_skipped`` records why): characterization off, nothing to
characterize. A failure before the characterization (e.g. an fp32 extraction failure) is an ordinary failure.

The module imports only the stdlib at load time (``worker5.sh`` loads ``drop_partial_rounds`` by file path, before the node's
venv is active).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

EXIT_PARKED = 78
FORMAT = "mb-parked/1"
PARKED_FILE = "parked.json"
_ON = ("on", "1", "true", "yes")


class Parked(Exception):
    """The run parked after its characterization pass with its state saved: the process exits ``EXIT_PARKED``."""


def enabled(cfg) -> bool:
    return str(getattr(cfg, "park_after_char", "off") or "off").lower() in _ON


def char_sync_enabled(cfg, gk) -> bool:
    """The characterization sync (wait mode): asked for, a seed was applied and a refresh inbox is configured."""
    return (str(getattr(cfg, "knowledge_global_char_sync", "off") or "off").lower() in _ON and gk is not None
            and bool(gk.get("applied")) and bool(getattr(cfg, "knowledge_global_refresh", "")))


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _write(p, obj) -> None:
    from .kb_resume import atomic_write_json
    atomic_write_json(Path(p), obj)


class Barrier:
    """The node side of one flow run: the park hook (``GraphSearch.on_char_done``) and the resume-time import."""

    def __init__(self, cfg, work: Path, state: dict, state_path: Path, gk: dict | None, irs=(), now=time.time):
        self.cfg, self.work, self.state, self.state_path, self.gk, self.now = cfg, Path(work), state, state_path, gk, now
        self.irs = [Path(str(d)).name for d in irs]
        self.file = self.work / PARKED_FILE
        self.key_hash: str | None = None            # set by the flow once the round key is known (recorded only)

    # -- bookkeeping in state.json (only with the flag on: flag off never touches state)
    def _rec(self) -> dict:
        return self.state.setdefault("barrier", {})

    def _save(self) -> None:
        _write(self.state_path, self.state)

    def skipped(self, reason: str) -> None:
        """Parking was asked for but there is nothing to park after: the run just continues (recorded)."""
        rec = self._rec()
        if rec.get("park_skipped") != reason:
            rec["park_skipped"] = reason
            self._save()
        print(json.dumps({"barrier": {"park_skipped": reason}}), flush=True)

    # -- the park hook: called by GraphSearch.characterize right after the characterization KernelBlaster run (and
    #    its ingest, and its sync marker / ack when the characterization sync is active) and before the re-profile
    def char_sync(self, R: Path, prog, rep: dict | None) -> dict | None:
        """Wait mode: ``<R>/char_sync.json`` stage ``paused`` (the controller's probe sees it), wait up to
        ``knowledge_global_char_sync_wait_s`` for ``<inbox>/char-r<NN>.ack.json``, import the package (``why="char"``), stage
        ``done``. Resume-safe (done: never again; paused: wait / import redone). Never blocks or fails the run."""
        r = int(Path(R).name[1:])
        f = Path(R) / "char_sync.json"
        prev = _read(f) or {}
        if prev.get("stage") == "done" or prog.done("gs_char_profile") or (self.work / PARKED_FILE).exists():
            return None
        wait = float(getattr(self.cfg, "knowledge_global_char_sync_wait_s", 180.0) or 0.0)
        inbox = Path(self.cfg.knowledge_global_refresh)
        ack_f = inbox / f"char-r{r:02d}.ack.json"
        t0 = self.now()
        rec = {"round": r, "ts": t0, "outcome": None}
        try:
            _write(f, {"format": FORMAT, "tag": "char", "round": r, "stage": "paused", "ts": prev.get("ts") or t0,
                       "char": {k: (rep or {}).get(k) for k in ("items", "llm_calls", "usd")}, "wait_s": wait, "pid": os.getpid()})
            hb = Path(R) / "char_sync.hb"
            poll = float(os.environ.get("MB_HALFWAY_POLL_S", "1.0"))
            deadline = t0 + wait
            ack = None
            while True:
                ack = _read(ack_f) if ack_f.exists() else None
                if ack is not None or self.now() >= deadline:
                    break
                try:
                    hb.write_text(json.dumps({"ts": self.now(), "pid": os.getpid()}))
                except OSError:
                    pass
                time.sleep(min(poll, max(0.0, deadline - self.now())) or 0.01)
            if ack is None:
                rec["outcome"] = "timeout"
            else:
                from . import kb_global
                rf = kb_global.refresh_at_round(self.work, self.cfg, self.state, r, why="char")
                try:
                    (inbox / "done").mkdir(exist_ok=True)
                    os.replace(ack_f, inbox / "done" / ack_f.name)
                except OSError:
                    pass
                if rf is None:
                    rec.update(outcome="none", imported_rows=0)
                else:
                    n = sum(v for v in (rf.get("counts") or {}).values() if isinstance(v, int))
                    rec.update(outcome="imported" if rf.get("status") in ("applied", "already_applied", "up_to_date") else
                               "error" if rf.get("status") in ("error", "refused") else rf.get("status"),
                               import_status=rf.get("status"), counts=rf.get("counts"), imported_rows=n)
        except Exception as e:  # noqa: BLE001 -- knowledge never fails a run
            rec.update(outcome="error", error=repr(e)[:300])
        rec["wait_s"] = round(self.now() - t0, 2)
        try:
            b = self._rec().setdefault("char_sync", {"n": 0, "imported_rows": 0, "timeouts": 0, "errors": 0, "events": []})
            b["n"] += 1
            b["imported_rows"] += int(rec.get("imported_rows") or 0)
            b["timeouts"] += rec["outcome"] == "timeout"
            b["errors"] += rec["outcome"] == "error"
            b["events"] = (b["events"] + [rec])[-10:]
            self._save()
            _write(f, {"format": FORMAT, "tag": "char", "round": r, "stage": "done", "ts": prev.get("ts") or t0,
                       "done_ts": self.now(), **{k: v for k, v in rec.items() if k not in ("ts", "round")}})
        except Exception:  # noqa: BLE001
            pass
        print(json.dumps({"round": r, "knowledge_global_char_sync": {k: rec.get(k) for k in
                                                                      ("outcome", "wait_s", "imported_rows", "error")}}), flush=True)
        return rec

    def after_char(self, R: Path, prog, rep: dict | None) -> None:
        if not enabled(self.cfg):
            if char_sync_enabled(self.cfg, self.gk):
                self.char_sync(R, prog, rep)
            return
        if prog.done("gs_char_profile"):
            return self.skipped("the characterization re-profile was already done")
        if not rep or not (rep.get("items") or 0):
            return self.skipped("the characterization pass had no items (nothing learned to share)")
        prev = _read(self.file) or {}
        doc = {"format": FORMAT, "phase": "char", "round": int(Path(R).name[1:]), "stage": "parked",
               "ts": self.now(), "first_ts": prev.get("first_ts") or prev.get("ts") or self.now(), "parks": int(prev.get("parks") or 0) + 1,
               "pid": os.getpid(), "key_hash": self.key_hash, "problems": self.irs,
               "char": {k: rep.get(k) for k in ("items", "llm_calls", "usd", "ingested")}}
        rec = self._rec()
        rec.update(parked=True, parks=doc["parks"], parked_ts=doc["ts"], phase="char", round=doc["round"])
        _write(self.file, doc)
        self._save()
        print(json.dumps({"barrier": {"parked": "char", "round": doc["round"], "parks": doc["parks"]}}), flush=True)
        raise Parked(f"parked after the characterization pass of round {doc['round']} ({self.file})")

    # -- resume: import the post-barrier package once, mark the run resumed
    def resume_import(self) -> dict | None:
        """At the start of a resumed run: when ``parked.json`` says ``parked``, import the newest refresh package once
        (``why="barrier"``), then mark it ``resumed``. Never raises. -> the import record or None."""
        try:
            doc = _read(self.file)
            if not doc or doc.get("stage") != "parked":
                return None
            rec: dict = {"round": doc.get("round", 0), "ts": self.now()}
            rf = None
            try:
                from . import kb_global
                if self.gk is not None and getattr(self.cfg, "knowledge_global_refresh", ""):
                    rf = kb_global.refresh_at_round(self.work, self.cfg, self.state, int(doc.get("round") or 0), why="barrier")
            except Exception as e:  # noqa: BLE001 -- knowledge never fails a run
                rec["error"] = repr(e)[:300]
            if rf is not None:
                rec.update(status=rf.get("status"), counts=rf.get("counts"),
                           imported_rows=sum(v for v in (rf.get("counts") or {}).values() if isinstance(v, int)))
                if rf.get("error") or rf.get("reason"):
                    rec["error"] = str(rf.get("error") or rf.get("reason"))[:300]
            else:
                rec.setdefault("status", "none")                 # no package delivered / nothing new for this run
            rec["parked_s"] = round(self.now() - float(doc.get("ts") or self.now()), 1)
            b = self._rec()
            b.setdefault("resumes", []).append(rec)
            b["parked"] = False
            h = self.state.get("knowledge_global")
            if h is not None and rf is not None:
                h.setdefault("barrier", {"n": 0, "imported_rows": 0})
                h["barrier"]["n"] += 1
                h["barrier"]["imported_rows"] += int(rec.get("imported_rows") or 0)
            self._save()
            _write(self.file, {**doc, "stage": "resumed", "resumed_ts": self.now(), "import": rec})
            print(json.dumps({"barrier": {"resumed": True, **{k: rec.get(k) for k in
                                                              ("status", "imported_rows", "parked_s", "error")}}}), flush=True)
            return rec
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"barrier": {"resume_import_error": repr(e)[:300]}}), flush=True)
            return None


# ---------------------------------------------------------------- worker5.sh: what a resume keeps of a parked round

def drop_partial_rounds(work) -> list:
    """What ``worker5.sh`` does before it restarts a run from a resume pack (the pack carries no KernelBlaster journals, so a
    round that was in progress cannot continue mid-way): every ``rNN`` with NN >= the number of rounds in ``state.json`` is
    dropped -- EXCEPT the round a barrier parked after its characterization pass. That round is rolled back to its park
    point instead: ``progress.json`` keeps only the phases ``profile`` and ``gs_char`` (finished, their products are in the pack:
    the winners cache, the knowledge db, the profile jsons), the other products of the round are removed, so the
    characterization is never run twice. Without ``parked.json`` this is exactly the old behaviour. -> list of messages."""
    w = Path(work)
    st = _read(w / "state.json") or {}
    n = len(st.get("rounds", []))
    pk = _read(w / PARKED_FILE) or {}
    out = []
    for p in sorted(w.glob("r[0-9][0-9]")):
        k = int(p.name[1:])
        if k < n:
            continue
        prog = _read(p / "progress.json")
        if pk.get("phase") == "char" and pk.get("round") == k and prog and "gs_char" in (prog.get("phases") or {}):
            keep = {"profile", "gs_char"}
            prog["phases"] = {a: b for a, b in prog["phases"].items() if a in keep}
            for c in p.iterdir():
                if c.name in ("progress.json", "profile", "char"):
                    continue
                shutil.rmtree(c, ignore_errors=True) if c.is_dir() else c.unlink()
            tmp = p / ".progress.json.tmp"
            tmp.write_text(json.dumps(prog, indent=1, default=str))
            os.replace(tmp, p / "progress.json")
            out.append(f"kept parked round {p.name} (rolled back to its park point)")
        else:
            shutil.rmtree(p)
            out.append(f"dropped partial round {p.name}")
    return out
