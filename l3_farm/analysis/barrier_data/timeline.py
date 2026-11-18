#!/usr/bin/env python3
"""Timeline model of barrier scheduling vs the plain pooled sweep for the Level 3 set (X-151). A model, not a measurement:

  W_i        problem i's wall time (min) when it runs alone to the end: the sweep's measured wall (results/<n>/final.tar.gz mtime
             - farm.json started; old farm, 3 rounds, no char pass, ~12-18 concurrent) x SCALE
  phase 1    d1_i = P1_BASE + P1_FRAC x W_i  (extract, whole-model profile, characterization pass on the 8 largest ops, pack + merge)
  park cost  PARK_S seconds (pack pull + knowledge merge happen on the controller; the slot is released when the process exits)
  resume     RESUME_S seconds of start-up overhead on top of the remaining work (process start, tar unpack, verify the restored state)
  phase 2    W_i - d1_i + RESUME_S

Slots: ``--slots`` concurrent problems (19). Policies: (A) no barrier: cheap-first list scheduling on the slots; (A') no barrier, longest
first; (B) barrier: phase 1 cheap-first on the slots, the barrier opens when MIN_FRAC of the problems have parked (or STALL min without
a new park), phase 2 longest-first (a problem still in phase 1 parks late and goes straight to the head of the phase-2 queue).
Prints makespan, slot idle time, the barrier-open time and the idle slot-minutes lost at the barrier. Usage: timeline.py [--slots 19]
[--min-frac 0.9] [--p1-frac 0.12] [--p1-base 8] [--park-s 60] [--resume-s 120]"""
import heapq, json, math, os, sys
from pathlib import Path

L3 = Path(__file__).resolve().parents[2]


def walls(scale=1.0):
    s = json.load(open(L3 / "state" / "farm.json"))
    out = {}
    for n, p in s["problems"].items():
        f = L3 / "results" / n / "final.tar.gz"
        if p.get("status") == "done" and f.exists() and p.get("started"):
            out[n] = (os.path.getmtime(f) - p["started"]) / 60 * scale
    return out


def listsched(jobs, slots, t0=0.0):
    """jobs: [(name, duration)] in start order -> (finish times {name: t}, idle slot-minutes inside [t0, makespan])."""
    heap = [t0] * slots
    fin = {}
    for n, d in jobs:
        t = heapq.heappop(heap)
        fin[n] = t + d
        heapq.heappush(heap, t + d)
    mk = max(fin.values())
    return fin, sum(mk - t for t in heap) , mk


def barrier(W, slots, min_frac, p1_frac, p1_base, park_s, resume_s, stall=90, head=0):
    d1 = {n: p1_base + p1_frac * w for n, w in W.items()}
    names = sorted(W, key=lambda n: (W[n], n))                    # phase 1: cheap first
    if head:                                                      # ... the `head` most expensive admitted first (phase1_head)
        top = sorted(W, key=lambda n: (-W[n], n))[:head]
        names = top + [n for n in names if n not in top]
    heap = [(0.0, i) for i in range(slots)]
    heapq.heapify(heap)
    park = {}
    for n in names:
        t, i = heapq.heappop(heap)
        park[n] = t + d1[n] + park_s / 60
        heapq.heappush(heap, (park[n], i))
    order = sorted(park.values())
    need = math.ceil(min_frac * len(W))
    t_open = order[need - 1]
    # stall rule (no new park for `stall` min while >= 1 parked): look for a gap in the park times
    for a, b in zip(order, order[1:] + [float("inf")]):
        if b - a >= stall and order.index(a) + 1 >= 1:
            t_open = min(t_open, a + stall)
            break
    # slots: a slot that parks a problem is free at park time but (phase 1 queue empty) idles until the barrier opens
    free_at = sorted(t for t, _ in heap)                         # when each slot last became free (phase 1 over)
    # problems that had not parked at t_open continue running in phase 1 on their slot; they then park and resume at once
    slotfree = []
    late = [n for n in names if park[n] > t_open]
    idle = 0.0
    pool = []                                                    # (time slot becomes available for phase 2)
    # slots that were not holding a late problem become available at t_open (idle from their last park)
    nlate = len(late)
    # each late problem occupies a slot until its park time, and is itself resumed there (same slot)
    avail = [t_open] * (slots - nlate)
    for n in late:
        avail.append(park[n])
    idle += sum(max(0.0, t_open - t) for t in sorted(t for t, _ in heap)[: slots - nlate])
    rem = {n: W[n] - d1[n] + resume_s / 60 for n in names}
    # queue: late problems first (their slot), then the rest longest first
    q = sorted([n for n in names if n not in late], key=lambda n: (-rem[n], n))
    heap2 = list(avail)
    heapq.heapify(heap2)
    fin = {}
    for n in late:                                               # resumes on the slot it parked on
        t = park[n]
        heap2.remove(t)
        fin[n] = t + rem[n]
        heapq.heappush(heap2, fin[n])
    for n in q:
        t = heapq.heappop(heap2)
        fin[n] = t + rem[n]
        heapq.heappush(heap2, fin[n])
    mk = max(fin.values())
    return {"makespan_min": mk, "barrier_open_min": t_open, "late": late, "idle_at_barrier_slot_min": idle,
            "phase2_start_order": q[:8]}


def main():
    a = sys.argv[1:]
    o = {"--slots": 19, "--min-frac": 0.9, "--p1-frac": 0.12, "--p1-base": 8.0, "--park-s": 60, "--resume-s": 120, "--scale": 1.0}
    while a:
        k = a.pop(0)
        if k in o:
            o[k] = type(o[k])(a.pop(0))
    W = walls(o["--scale"])
    names = sorted(W, key=lambda n: (W[n], n))
    print(f"{len(W)} problems, total work {sum(W.values()) / 60:.1f} problem-hours, median {sorted(W.values())[len(W) // 2]:.0f} min, "
          f"max {max(W.values()):.0f} min ({max(W, key=W.get)}); slots {o['--slots']}")
    fin, idle, mk = listsched([(n, W[n]) for n in names], o["--slots"])
    print(f"(A)  no barrier, cheap first : makespan {mk / 60:5.2f} h, idle slot-time {idle / 60:5.1f} h (tail)")
    fin, idle, mk = listsched([(n, W[n]) for n in reversed(names)], o["--slots"])
    print(f"(A') no barrier, longest first: makespan {mk / 60:5.2f} h, idle slot-time {idle / 60:5.1f} h (tail)")
    for mf, hd in ((o["--min-frac"], 0), (o["--min-frac"], 3), (1.0, 0)):
        r = barrier(W, o["--slots"], mf, o["--p1-frac"], o["--p1-base"], o["--park-s"], o["--resume-s"], head=hd)
        print(f"(B)  barrier min_frac {mf}{' + phase1_head ' + str(hd) if hd else ''}: makespan {r['makespan_min'] / 60:5.2f} h, barrier opens at {r['barrier_open_min']:.0f} min, "
              f"late (in phase 1 at open) {len(r['late'])}, idle slot-time at the barrier {r['idle_at_barrier_slot_min'] / 60:.1f} h; "
              f"phase 2 starts with {r['phase2_start_order'][:4]}")


if __name__ == "__main__":
    main()
