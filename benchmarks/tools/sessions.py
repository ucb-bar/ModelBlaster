"""Named time-window sessions for the LLM cost tracker.

A session is a labeled window of wall-clock time. Cost incurred while
a session is active is attributed to it. Single-active model:
starting a new session auto-ends the previous one, so a user can't
accidentally double-count by forgetting to close one.

Why time-window instead of tagging each call? The LLM clients already
write per-call records with a ``ts`` field. Tagging would require a
schema change AND propagating a session id env var into every
subprocess. Time-window sessions need zero changes to the clients --
the cost monitor filters records by timestamp at query time.

State lives at ``benchmarks/results/.sessions.json`` (gitignored).
Single-file, easy to inspect, easy to wipe.

Public surface:
    SessionLedger.load()            # idempotent; creates empty ledger
    ledger.start(name, label=None)  # ends any active session, opens new
    ledger.end()                    # closes the active session
    ledger.active                   # property -> Session | None
    ledger.list_all()               # all sessions, newest first
    ledger.get(session_id)          # specific session
    ledger.save()                   # atomic write back

Each Session exposes ``contains(ts)`` so the cost monitor can decide
whether a JSONL record falls within the session's time window.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_LEDGER_PATH = (
    Path(__file__).resolve().parents[1] / "results" / ".sessions.json"
)


def _now_iso() -> str:
    """UTC ISO 8601 with explicit +00:00 suffix so JSONL ts fields
    and session boundaries are directly comparable as strings AND
    via datetime.fromisoformat."""
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        d = datetime.fromisoformat(ts)
    except ValueError:
        return None
    # Records without explicit tz get treated as UTC -- matches
    # bedrock_client's _append_call_log convention.
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d


@dataclass
class Session:
    id: str
    label: Optional[str]
    started_at: str           # ISO 8601 UTC
    ended_at: Optional[str]   # ISO 8601 UTC, None while active

    @property
    def is_active(self) -> bool:
        return self.ended_at is None

    @property
    def started_dt(self) -> Optional[datetime]:
        return _parse_ts(self.started_at)

    @property
    def ended_dt(self) -> Optional[datetime]:
        return _parse_ts(self.ended_at) if self.ended_at else None

    def contains(self, ts: str) -> bool:
        """True when the JSONL record at ``ts`` falls within this
        session's window. Open sessions extend to "now."""
        rec_dt = _parse_ts(ts)
        if rec_dt is None:
            return False
        start = self.started_dt
        if start is None or rec_dt < start:
            return False
        end = self.ended_dt
        if end is not None and rec_dt > end:
            return False
        return True


class SessionLedger:
    """Persistent ledger of sessions stored at ``path``. Use
    ``SessionLedger.load(path=None)`` to construct -- it tolerates
    missing files and corrupted JSON (treated as "no sessions yet").
    """

    def __init__(self, path: Path,
                 active_id: Optional[str],
                 sessions: list[Session]):
        self.path = path
        self.active_id = active_id
        self.sessions = sessions

    # ───────────────────── constructors / IO ─────────────────────

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SessionLedger":
        path = path or DEFAULT_LEDGER_PATH
        if not path.exists():
            return cls(path=path, active_id=None, sessions=[])
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable ledger -- treat as empty. The
            # alternative (raising) would lock users out of mb-cost.
            return cls(path=path, active_id=None, sessions=[])
        sessions = [Session(**s) for s in data.get("sessions", [])]
        active_id = data.get("active")
        return cls(path=path, active_id=active_id, sessions=sessions)

    def save(self) -> None:
        """Atomic write: stage to a sibling .tmp file, then rename.
        Safe under concurrent reads (`mb-cost live` watches without a
        lock)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "active": self.active_id,
            "sessions": [asdict(s) for s in self.sessions],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    # ───────────────────── queries ─────────────────────

    @property
    def active(self) -> Optional[Session]:
        if self.active_id is None:
            return None
        return self.get(self.active_id)

    def get(self, session_id: str) -> Optional[Session]:
        for s in self.sessions:
            if s.id == session_id:
                return s
        return None

    def list_all(self) -> list[Session]:
        """Newest first."""
        return sorted(self.sessions,
                      key=lambda s: s.started_at, reverse=True)

    # ───────────────────── mutations ─────────────────────

    def start(self, name: str,
              label: Optional[str] = None) -> Session:
        """Open a new session named ``name``. If another session is
        active, end it first so only ever one is active. ``name``
        must be unique across the ledger -- enforced to avoid
        accidentally re-opening an already-closed session."""
        name = name.strip()
        if not name:
            raise ValueError("session name must be non-empty")
        if self.get(name) is not None:
            raise ValueError(
                f"session {name!r} already exists; pick a unique name"
            )
        if self.active is not None:
            self.end()
        session = Session(
            id=name, label=label,
            started_at=_now_iso(), ended_at=None,
        )
        self.sessions.append(session)
        self.active_id = session.id
        self.save()
        return session

    def end(self) -> Optional[Session]:
        """Close the active session. No-op when none is active."""
        active = self.active
        if active is None:
            return None
        active.ended_at = _now_iso()
        self.active_id = None
        self.save()
        return active

    def delete(self, session_id: str) -> bool:
        """Remove a session permanently. Useful for cleaning up test
        sessions. Returns True if anything was removed."""
        before = len(self.sessions)
        self.sessions = [s for s in self.sessions if s.id != session_id]
        if self.active_id == session_id:
            self.active_id = None
        if before != len(self.sessions):
            self.save()
            return True
        return False


# ───────────────────── helpers for the monitor ─────────────────────


def month_start_iso(now: Optional[datetime] = None) -> str:
    """First day of the current UTC month at 00:00:00, ISO 8601."""
    d = now or datetime.now(timezone.utc)
    first = d.replace(day=1, hour=0, minute=0, second=0,
                      microsecond=0)
    return first.isoformat()


def is_within_current_month(ts: str,
                            now: Optional[datetime] = None) -> bool:
    """True when the JSONL record at ``ts`` falls in the current
    UTC month."""
    rec_dt = _parse_ts(ts)
    if rec_dt is None:
        return False
    n = now or datetime.now(timezone.utc)
    return rec_dt.year == n.year and rec_dt.month == n.month
