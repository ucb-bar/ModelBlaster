import os, glob, json, re
out = []
for d in sorted(glob.glob("/tmp/krisd/farm_headline/*/")):
    n = os.path.basename(d.rstrip("/"))
    r = {"p": n, "refresh": None, "rows": None, "parked_s": None, "char_reused": None, "reuse_later": 0, "rounds": 0}
    try:
        for l in open(d + "run.log", errors="ignore"):
            if '"barrier"' in l and "imported_rows" in l:
                b = json.loads(l)["barrier"]; r.update(refresh=b.get("status"), rows=b.get("imported_rows"), parked_s=b.get("parked_s"))
    except Exception: pass
    try:
        m = re.findall(r'"reused":\s*(\d+)', open(d + "work/r00/char/run.log", errors="ignore").read())
        r["char_reused"] = sum(map(int, m)) if m else None
    except Exception: pass
    for f in glob.glob(d + "work/r0[1-9]/**/run.log", recursive=True):
        r["rounds"] = max(r["rounds"], int(re.search(r"/r0(\d)/", f).group(1)))
        for x in re.findall(r'"reused":\s*(\d+)', open(f, errors="ignore").read()): r["reuse_later"] += int(x)
    out.append(r)
print(json.dumps(out))
