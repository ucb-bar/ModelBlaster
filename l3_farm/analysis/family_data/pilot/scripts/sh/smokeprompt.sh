WK=/tmp/krisd/family/pilot/smoke/r00/run/trace
python3 - <<PY
import json,os,glob,gzip
recs=[json.loads(l) for l in open("$WK/index.jsonl")]
print(sorted({r.get("kind") or r.get("t") or "?" for r in recs}))
r=[r for r in recs if "llm" in json.dumps(r)[:120]][:2]
print(json.dumps(r)[:1500])
PY
