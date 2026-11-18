#!/usr/bin/env python3
"""Extract per-application state/technique counts from a global knowledge snapshot (SQLite, read-only).
Usage: python3 extract_counts.py <path/to/vNNNN.db>   -> counts_states.csv, counts_techniques.csv
Source used: knowledge_global_headline/snapshots/v0278.db (X-152 headline L3 sweep, L40S, 48 problems)."""
import csv, json, sqlite3, sys, collections
from families import family
db = sys.argv[1]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
based = {}
for hid, data in c.execute("select id, data from gk_hypotheses"):
    try: based[hid] = json.loads(data).get("based_on") or []
    except Exception: pass
fam_pair = collections.Counter()
leaf, first, tech, pair = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for rec, v2 in c.execute("select rec, v2 from gk_outcomes"):
    r, v = json.loads(rec), (json.loads(v2) if v2 else {})
    ch = v.get("chain") or []
    if ch:
        leaf[ch[-1]] += 1; first[ch[0]] += 1
    tech[r.get("technique")] += 1
    if ch:
        pair[(ch[-1], r.get("technique"))] += 1
        t = r.get("technique") or ""
        if t.startswith("h_"):   # agent-proposed hypothesis: attribute to the family of its parent techniques
            fams = [f for f in (family(x) for x in based.get(t, [])) if f != "Other"]
            f = collections.Counter(fams).most_common(1)[0][0] if fams else "Agent-proposed (unresolved)"
        else:
            f = family(t)
        fam_pair[(ch[-1], f)] += 1
for name, cnt in (("states", leaf), ("coarse_states", first), ("techniques", tech)):
    with open(f"counts_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow([name[:-1] if name != "coarse_states" else "coarse_state", "applications"])
        for k, n in cnt.most_common(): w.writerow([k, n])
with open("counts_state_technique.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["state", "technique", "applications"])
    for (st, tq), n in pair.most_common(): w.writerow([st, tq, n])
with open("counts_state_family.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["state", "family", "applications"])
    for (st, fm), n in fam_pair.most_common(): w.writerow([st, fm, n])
print(sum(leaf.values()), "applications;", len(leaf), "states;", len(tech), "techniques")
