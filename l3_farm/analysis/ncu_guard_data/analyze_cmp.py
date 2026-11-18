import json, statistics as st, sys
sys.path.insert(0, "/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src")
from kernelblaster.knowledge import ncu as N
d = json.load(open("ncu_cmp.json"))
KEYS = ["duration_us", "compute_throughput", "memory_throughput", "dram_throughput", "l1_throughput", "l2_throughput",
        "achieved_occupancy", "l1_hit_rate", "l2_hit_rate"]
tot = {"slots": 0, "state_same": 0, "rule1_same": 0}
out = {}
for pn, pd in d["problems"].items():
    rows = pd["rows"]
    print("##", pn)
    print("| slot | op | dur base us (3 reps) | dur none us (3 reps) | none/base | max |dSOL| pct-pts (comp,mem,dram) | state base/none | top rule base/none |")
    print("|---|---|---|---|---|---|---|---|")
    for sid in sorted(rows["base"][0]["by"]):
        b = [r["by"].get(sid) for r in rows["base"]]; n = [r["by"].get(sid) for r in rows["none"]]
        if not all(b) or not all(n): continue
        db = [x["duration_us"] for x in b]; dn = [x["duration_us"] for x in n]
        mb = lambda k: st.median(x[k] for x in b if k in x); mn = lambda k: st.median(x[k] for x in n if k in x)
        dsol = max(abs(mb(k) - mn(k)) for k in ("compute_throughput", "memory_throughput", "dram_throughput") if k in b[0])
        # noise within base across reps
        sb = {x["state"] for x in b}; sn = {x["state"] for x in n}
        r1b = (b[0]["rules"][0]["rule"] if b[0]["rules"] else "-"); r1n = (n[0]["rules"][0]["rule"] if n[0]["rules"] else "-")
        tot["slots"] += 1; tot["state_same"] += (st.mode(x["state"] for x in b) == st.mode(x["state"] for x in n)); tot["rule1_same"] += r1b == r1n
        print(f"| {sid} | {pd['slots'][sid]['op']} | {', '.join(f'{x:.2f}' for x in db)} | {', '.join(f'{x:.2f}' for x in dn)} | {st.median(dn)/st.median(db):.3f} | {dsol:.2f} | {'/'.join(sorted(sb))} {'/'.join(sorted(sn))} | {r1b} / {r1n} |")
print(tot)
