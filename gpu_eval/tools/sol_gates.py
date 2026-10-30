"""Per-problem gate table for the SOL-ExecBench run.
Usage: sol_gates.py IR_ROOT WALK_CPU_JSON HOST_ROOT H3_JSON [--md]

Gates: extract (IR_ROOT/status_all.txt), CPU IR walk vs io.npz golden, host C
(MB codegen on x86; err <= 1e-3*max|golden| + 1e-5, the Level2/3 bound), and
the hand-CUDA walk (golden + every kernel checked on the reference's inputs).
"""
import json, re, sys
from pathlib import Path
import numpy as np

irr, walk_j, host_root, h3_j = (Path(a) for a in sys.argv[1:5])
md = "--md" in sys.argv

status = {}
for line in (irr / "status_all.txt").read_text().splitlines():
    stem, st, *err = line.split("\t")
    status[stem] = (st, err[0] if err else "")


# walk/h3 records name the IR by its graph name (or its path, on error);
# graph names aren't unique across levels (and drop "."), so take each
# record's first unused dir with that name, in the order the IRs were walked
names = {}
for d in sorted(irr.glob("*/io.npz")):
    names.setdefault(json.loads((d.parent / "graph.json").read_text())["name"], []).append(d.parent.name)


def by_name(recs):
    left = {k: list(v) for k, v in names.items()}
    out = {}
    for r in recs:
        n = r["ir"].rstrip("/")
        if "/" in n:
            out[n.split("/")[-1]] = r
        else:
            out[left[n].pop(0)] = r
    return out


walk = by_name(json.load(open(walk_j)))
gpu = by_name(json.load(open(h3_j)))


def host(stem):
    f = host_root / stem / "run.out"
    if not f.exists():
        return None, "no run"
    m = re.search(r"max_abs_err=(\S+)", f.read_text())
    if not m:
        return False, "no verify line"
    err = float(m.group(1))
    g = np.load(irr / stem / "io.npz")["output"].astype(np.float32)
    fin = g[np.isfinite(g)]
    bound = 1e-3 * (float(np.abs(fin).max()) if fin.size else 0.0) + 1e-5
    return bool(err <= bound), f"{err / max(bound - 1e-5, 1e-12) * 1e-3:.2g}"


rows, tot = [], {"extract": 0, "walk": 0, "host": 0, "gpu": 0}
for stem in sorted(status, key=lambda s: (s.removeprefix("l2_")[:3], s)):
    st, err = status[stem]
    r = {"problem": stem, "extract": st == "OK"}
    if st == "OK":
        w = walk.get(stem, {})
        r["walk"] = bool(w.get("ok"))
        r["walk_rel"] = w.get("rel_to_range")
        r["host"], r["host_rel"] = host(stem)
        h = gpu.get(stem, {})
        r["gpu"] = bool(h.get("ok") and not h.get("first_divergent_op"))
        r["gpu_rel"] = h.get("rel_to_range")
        r["gpu_kernel_rel"] = h.get("worst_intermediate_rel_err")
        r["gpu_note"] = h.get("first_divergent_op") or h.get("error") or (h.get("missing") or None)
    else:
        r["why"] = err
    for g in tot:
        tot[g] += bool(r.get(g))
    rows.append(r)

n = len(rows)
print(f"problems {n}: extract {tot['extract']}, CPU walk {tot['walk']}, host C {tot['host']}, "
      f"hand-CUDA walk {tot['gpu']}")
for r in rows:
    if r["extract"] and not (r["walk"] and r["host"] and r["gpu"]):
        print("  gate miss:", r["problem"], {k: r.get(k) for k in
              ("walk", "walk_rel", "host", "host_rel", "gpu", "gpu_rel", "gpu_note")})
if md:
    f = lambda v: "—" if v is None else f"{v:.1e}" if isinstance(v, float) else str(v)
    print("\n| problem | CPU walk err/range | host C err/range | CUDA err/range | worst CUDA kernel (isolated) |")
    print("|---|---|---|---|---|")
    for r in rows:
        if r["extract"]:
            print(f"| {r['problem']} | {f(r['walk_rel'])}{'' if r['walk'] else ' ✗'} | "
                  f"{r['host_rel']}{'' if r['host'] else ' ✗'} | {f(r['gpu_rel'])}{'' if r['gpu'] else ' ✗'} | "
                  f"{f(r['gpu_kernel_rel'])} |")
    print("\n| problem | not extracted: reason |")
    print("|---|---|")
    for r in rows:
        if not r["extract"]:
            print(f"| {r['problem']} | {r['why'][:120]} |")
