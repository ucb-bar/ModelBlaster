"""Summarise walk_torch.py JSON lines: PASS / FAIL / unsupported / zero-op IRs."""
import json, sys, collections
rs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
c = collections.Counter()
for r in rs:
    k = ("ERR" if "n_ops" not in r else "zero_ops" if r["n_ops"] == 0 else
         "unsupported:" + ",".join(r["unsupported"]) if r["unsupported"] else "PASS" if r["ok"] else "FAIL")
    c[k] += 1
    if k not in ("PASS",) and "-q" not in sys.argv:
        print(k, r["ir"].rstrip("/").split("/")[-1], r.get("rel_to_range", r.get("error", ""))[:120] if isinstance(r.get("rel_to_range", r.get("error", "")), str) else r.get("rel_to_range"))
print(len(rs), dict(c))
