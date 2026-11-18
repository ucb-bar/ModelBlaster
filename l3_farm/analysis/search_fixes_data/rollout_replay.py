#!/usr/bin/env python3
"""A3 item 4 (rollout acceptance): replay of the Level 3 sweep's logged KernelBlaster trajectories (zero LLM).

For every slot problem journal (results/*/work/**/journal/state.json; steps in commit order) the composite best is
tracked through the merges, and every step is placed by the LAG of the state it started from (rollout state's deployed
time / composite best at commit). Outputs (next to this script, rollout_replay.json) and a printed table:

1. how often the sweep's acceptance rule (composite: g = min(own gain, confirmed gain vs composite)) DISCARDED a verified
   gain over the rollout's own state (gain >= accept, not merged, not kept), and how far behind those rollouts were;
2. the empirical merge rate P(step merges into the composite | lag bucket of its start state);
3. an estimate of what rollout-local acceptance (keep own gains) would have changed: for every discarded gain, the same
   rollout's later steps would have started at the improved lag instead of the old one; expected extra merges =
   sum over those steps of P(merge | new lag) - P(merge | old lag) (rate model; NOT a counterfactual run);
4. the same for the 'lineage' resync variant (A3): a stalled lineage lagging > R restarts from the composite (lag 1).

Usage: python3 rollout_replay.py [--accept 0.01] [--resync 1.5]
"""
import argparse
import bisect
import collections
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
BUCKETS = [1.0, 1.02, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf")]


def bucket(lag):
    i = bisect.bisect_right(BUCKETS, lag) - 1
    i = max(0, min(i, len(BUCKETS) - 2))
    return f"{BUCKETS[i]:g}-{BUCKETS[i + 1]:g}"


def load_runs():
    for f in sorted(glob.glob(os.path.join(RES, "*", "work", "**", "journal", "state.json"), recursive=True)):
        try:
            s = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        steps = s.get("steps_log") or []
        if steps and any((x.get("context") or {}).get("e2e") for x in steps):
            yield os.path.relpath(f, RES), s, steps


def replay(acc, resync):
    rows = []
    for path, s, steps in load_runs():
        # composite start: the first step-0 state's deployed time
        c = next((x["e2e_before"] for x in steps if x.get("e2e_before")), None)
        if not c:
            continue
        c0 = c
        start_c = {}                      # rollout -> composite when its previous step committed (= this step's start)
        for i, x in enumerate(steps):
            e = (x.get("context") or {}).get("e2e") or {}
            before = x.get("e2e_before") or e.get("before_us")
            cs = start_c.get(x["rollout"], c0)
            lag = (before / cs) if before and cs else None
            gain, gc = e.get("gain"), e.get("confirm_gain")
            merged = bool(e.get("accept"))
            g = gain if gain is not None else 0.0
            if gc is not None:
                g = min(g, gc)
            kept = gain is not None and g > -acc / 2
            local_after = before * (1 - gain) if (before and gain is not None) else None
            rows.append({"run": path, "i": i, "rollout": x["rollout"], "step": x["step"], "status": x["status"],
                         "lag": lag, "gain": gain, "confirm_gain": gc, "merged": merged, "kept": kept,
                         "discarded_local_gain": bool(gain is not None and gain >= acc and not merged and not kept),
                         "local_lag": (local_after / cs) if local_after and cs else None, "comp": cs})
            if merged:
                c = e.get("confirm_us") or x.get("e2e_after") or c
            start_c[x["rollout"]] = c
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept", type=float, default=0.01)
    ap.add_argument("--resync", type=float, default=1.5, help="lineage: restart a stalled lineage lagging > this")
    a = ap.parse_args()
    rows = replay(a.accept, a.resync)
    n = len(rows)
    lagged = [r for r in rows if r["lag"] is not None]
    rate = collections.defaultdict(lambda: [0, 0, 0.0])
    for r in lagged:
        if r["step"] == 0:                  # step 0 starts from the problem's seed state (easy wins): rates from later steps
            continue
        b = bucket(r["lag"])
        rate[b][0] += r["merged"]
        rate[b][1] += 1
        rate[b][2] += max(r["confirm_gain"] or 0.0, 0.0) if r["merged"] else 0.0
    p = {b: (m / t if t else 0.0) for b, (m, t, _) in rate.items()}
    v = {b: (g / t if t else 0.0) for b, (_, t, g) in rate.items()}     # expected composite gain per step
    disc = [r for r in rows if r["discarded_local_gain"]]
    by_roll = collections.defaultdict(list)
    for r in rows:
        by_roll[(r["run"], r["rollout"])].append(r)
    est = {k: [0.0, 0.0, 0] for k in ("local", "resync", "lineage")}    # [extra merges, extra gain, steps shifted]

    def shift(k, x, new_lag):
        if new_lag is None or new_lag >= x["lag"]:
            return
        est[k][0] += p.get(bucket(new_lag), 0.0) - p.get(bucket(x["lag"]), 0.0)
        est[k][1] += v.get(bucket(new_lag), 0.0) - v.get(bucket(x["lag"]), 0.0)
        est[k][2] += 1

    events = []
    for key, traj in by_roll.items():
        traj = sorted(traj, key=lambda x: x["step"])
        local_state = None                 # absolute deployed time of the local lineage (None = same as real)
        for j, x in enumerate(traj):
            if x["lag"] is None:
                continue
            cs = x["comp"]
            prev = traj[j - 1] if j else None
            stalled = prev is not None and not (prev["kept"] and (prev["gain"] or 0) >= a.accept)
            # local: the rollout continues from its best own state since the last discard
            l_local = min(x["lag"], local_state / cs) if local_state else x["lag"]
            shift("local", x, l_local)
            # resync (composite acceptance): a stalled rollout lagging > R restarts from the composite
            shift("resync", x, 1.0 if (stalled and x["lag"] > a.resync) else x["lag"])
            # lineage: local, and a stalled lineage still lagging > R restarts from the composite
            shift("lineage", x, 1.0 if (stalled and l_local > a.resync) else l_local)
            if x["discarded_local_gain"] and x["local_lag"]:
                local_state = x["local_lag"] * cs
                events.append({"run": x["run"], "rollout": x["rollout"], "step": x["step"],
                               "lag_before": round(x["lag"], 3), "lag_after_local": round(x["local_lag"], 3),
                               "gain": x["gain"], "confirm_gain": x["confirm_gain"]})
            elif x["merged"]:
                local_state = None
    affected = est["local"][2]
    extra_local, extra_lineage = est["local"][0], est["lineage"][0]
    # lagging-start steps in reality: how often did a step that started behind the composite merge?
    behind = [r for r in lagged if r["lag"] > 1.1]
    merges = sum(r["merged"] for r in rows)
    out = {"accept": a.accept, "steps": n, "steps_with_lag": len(lagged), "merges": merges,
           "steps_started_behind_1.1x": len(behind), "merges_from_behind_1.1x": sum(r["merged"] for r in behind),
           "discarded_local_gains": len(disc), "discarded_share_of_steps": len(disc) / n if n else 0,
           "discard_lag_before_median": sorted(r["lag"] for r in disc)[len(disc) // 2] if disc else None,
           "discard_lag_after_local_median": sorted(r["local_lag"] for r in disc if r["local_lag"])[len(disc) // 2]
           if disc else None,
           "later_steps_affected": affected,
           "est_extra_merges_local": round(extra_local, 2), "est_extra_merges_lineage": round(extra_lineage, 2),
           "estimate": {k: {"extra_merges": round(e[0], 1), "extra_merges_pct": round(100 * e[0] / max(merges, 1), 2),
                            "extra_gain_sum": round(e[1], 3), "steps_shifted": e[2]} for k, e in est.items()},
           "resync_lag": a.resync,
           "merge_rate_by_lag": {b: {"merges": rate[b][0], "steps": rate[b][1], "p": round(p[b], 4),
                                     "gain_per_step": round(v[b], 5)}
                                 for b in sorted(rate, key=lambda k: float(k.split("-")[0]))},
           "events": events}
    json.dump(out, open(os.path.join(HERE, "rollout_replay.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "events"}, indent=1))


if __name__ == "__main__":
    main()
