#!/usr/bin/env python3
"""Also plant the failing kernel in the run's implementation store as the FASTEST record of the shape set (what the
sweep had: the rejected DenseNet winner stayed the store's fastest 960x7x7 record, so every winners export re-exported
it and a re-optimized winner never reached ModelBlaster). usage: inject_store.py WORK"""
import hashlib
import json
import sys
from pathlib import Path

work = Path(sys.argv[1])
cache = json.loads((work / "cache" / "index.json").read_text())
bad = next(e for e in cache["entries"] if e["impl_id"] == "inject:bad")
code = (work / "cache" / bad["file"]).read_text()
impl = work / "knowledge" / "impl"
recs = [json.loads(l) for l in (impl / "index.jsonl").read_text().splitlines() if l.strip()]
guard = sorted(bad["shape_guard"])
same = [r for r in recs if r["op"] == bad["op"] and sorted({json.dumps(s, sort_keys=True) for s in r["shapes"]}) == guard]
base = min(same, key=lambda r: r["slot_us"])
sha = hashlib.sha1(code.encode()).hexdigest()
(impl / "code" / f"{sha}.cu").write_text(code)
rec = {**base, "code_sha": sha, "id": "injectbad01", "slot_us": base["slot_us"] / 10, "speedup_vs_seed":
       (base.get("speedup_vs_seed") or 1) * 10, "speedup": None, "parent_id": None, "techniques": ["inject"]}
with open(impl / "index.jsonl", "a") as f:
    f.write(json.dumps(rec, sort_keys=True) + "\n")
print(json.dumps({"planted": rec["id"], "slot_us": rec["slot_us"], "was_fastest": base["slot_us"], "records_same_shape": len(same)}))
