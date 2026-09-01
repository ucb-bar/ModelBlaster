# firesim-queue → /opt + systemd unit (closure for #93)

## Task

#93 — Tier 1: promote firesim-queue to /opt/ with a systemd unit.

## Current state

`firesim-queue` lives at `/scratch2/agustin/firesim_queue/bin/firesim-queue`
with a SQLite backing store in the same directory. All ModelBlaster
FireSim runs go through this queue via `FIRESIM_QUEUE=1` (per
`reference_firesim_queue.md` memory). Test battery passes (#92,
24/24).

## Why this was parked

The original "parked" status (in-task notes) was: "only resurface if
multi-user demand grows." The reason for promoting it would be:
- multi-user fairness (currently one shared sqlite owned by `agustin`)
- automatic restart on host reboot
- log rotation under journalctl

None of those are forcing functions today — the queue is single-user
(everyone shares `agustin`'s account on this host) and host reboots
are rare. The current /scratch2 location works fine; promoting to
/opt would require sudo and would couple us more tightly to the host
OS than to the merlin-dev environment.

## Decision

**Keep #93 parked.** The current /scratch2 location + on-demand
invocation is the right choice while:
- the queue is single-user (no contention with another lab account)
- the host is stable enough that uptime-driven systemd doesn't pay
  for itself
- the merlin-dev environment lives on /scratch2 (so collocating the
  queue keeps everything on one filesystem)

If the lab grows another user who needs FIRESIM_QUEUE=1, revisit.
Until then, this is a deliberate non-action.

## Cleanup

Closing #93 as deliberately-parked, with this memo as the
durable rationale (so a future agent doesn't pick the task up
without re-reading the context).
