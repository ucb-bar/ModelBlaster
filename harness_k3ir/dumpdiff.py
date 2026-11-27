"""Byte comparison of two k3ir feed dumps of the same plan (host reference C, Spike, board).

    python dumpdiff.py NAMES_JSON DUMP_A DUMP_B [--exact-kinds k1,k2,...] [--json OUT]
Per producing op kind: outputs compared, outputs with any differing byte, largest difference in units in the
last place (fp16 / fp32 outputs, bit-pattern distance). ok: no output of an --exact-kinds kind differs."""
import json
import sys

import numpy as np

from ref_walk import read_dump

names = json.load(open(sys.argv[1]))
a, b = read_dump(sys.argv[2]), read_dump(sys.argv[3])
exact = set(sys.argv[sys.argv.index("--exact-kinds") + 1].split(",")) if "--exact-kinds" in sys.argv else set()
U = {"f16": (np.float16, np.uint16), "f32": (np.float32, np.uint32)}
res, ok = {}, True
for i in sorted(set(a) & set(b)):
    t = names["tensors"][i]
    k = names["producer"].get(t, "?") + ("_f16" if names["dtypes"][t] == "f16" else "")
    r = res.setdefault(k, {"outputs": 0, "differ": 0, "max_ulps": 0})
    r["outputs"] += 1
    if a[i] != b[i]:
        r["differ"] += 1
        dt = names["dtypes"][t]
        if dt in U:
            xi = np.frombuffer(a[i], U[dt][1]).astype(np.int64)
            yi = np.frombuffer(b[i], U[dt][1]).astype(np.int64)
            # sign-magnitude -> ordered integers
            sb = 15 if dt == "f16" else 31
            xi = np.where(xi >> sb, -(xi & ((1 << sb) - 1)), xi)
            yi = np.where(yi >> sb, -(yi & ((1 << sb) - 1)), yi)
            r["max_ulps"] = max(r["max_ulps"], int(np.abs(xi - yi).max()))
        else:
            r["max_ulps"] = -1
        if names["producer"].get(t) in exact:
            ok = False
out = {"ok": ok, "by_kind": res, "n_compared": len(set(a) & set(b))}
print(json.dumps(out))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
