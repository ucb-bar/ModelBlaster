import json, sys, re
from pathlib import Path
def find(o, key, out):
    if isinstance(o, dict):
        if key in o and isinstance(o[key], dict): out.append(o[key])
        for v in o.values(): find(v, key, out)
    elif isinstance(o, list):
        for v in o: find(v, key, out)
KEYS = ("remote", "local", "infra", "full", "compile_errors", "remote_cached", "spilled")
for d in sorted(Path("/tmp/krisd/cpool/e3").glob("*_*")):
    groups = {}; bcg = {}
    for f in (d / "work").rglob("*.json"):
        if f.stat().st_size > 5_000_000: continue
        try: r = json.loads(f.read_text())
        except Exception: continue
        rel = f.relative_to(d / "work").parts
        g = "/".join(rel[:3]) if rel[0].startswith("r") and len(rel) > 3 else "/".join(rel[:-1])
        l = []; find(r, "remote_build", l)
        for x in l:
            cur = groups.setdefault(g, {k: 0 for k in KEYS})
            for k in KEYS: cur[k] = max(cur[k], x.get(k, 0))
        l = []; find(r, "build_cache", l)
        for x in l:
            cur = bcg.setdefault(g, {})
            for k, v in x.items():
                if isinstance(v, (int, float)): cur[k] = max(cur.get(k, 0), v)
    rb = {k: sum(g[k] for g in groups.values()) for k in KEYS}
    bc = {}
    for g in bcg.values():
        for k, v in g.items(): bc[k] = bc.get(k, 0) + v
    res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    print(d.name, "wall", res.get("wall"), "passed", res.get("passed"), "status", {k[:34]: v for k, v in (res.get("status") or {}).items()}, "fused", res.get("fused"), "speedup", res.get("speedup"))
    print("   remote_build", rb, "build_cache", bc, "groups", len(groups))
