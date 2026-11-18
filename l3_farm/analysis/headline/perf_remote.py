import json, glob, os, collections
out = []
for d in sorted(glob.glob("/tmp/krisd/farm_headline/*/")):
    n = os.path.basename(d.rstrip("/")); r = {"p": n, "char_before": None, "char_after": None, "dec": collections.Counter(), "gains": [], "rounds": 0, "last_ts": None}
    try:
        for l in open(d + "work/graph_decisions.jsonl"):
            try: j = json.loads(l)
            except Exception: continue
            r["rounds"] = max(r["rounds"], j.get("round", 0)); r["last_ts"] = j.get("ts")
            if j.get("event") == "characterize":
                b = j.get("partition_us_before") or {}; a = j.get("partition_us_after") or {}
                r["char_before"] = sum(b.values()) if b else None; r["char_after"] = sum(a.values()) if a else None
            if j.get("event") == "decision":
                r["dec"][j.get("status")] += 1
                if j.get("measured") and j.get("gain") is not None: r["gains"].append([j.get("hint"), j.get("gain"), j.get("status")])
    except Exception: pass
    try:
        import time
        fs = [os.path.getmtime(f) for f in glob.glob(d + "work/**/*", recursive=True)[:20000] if os.path.isfile(f)]
        r["active_ts"] = max(fs) if fs else None
    except Exception: r["active_ts"] = None
    r["dec"] = dict(r["dec"]); out.append(r)
print(json.dumps(out))
