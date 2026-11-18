import os, sys, json, filecmp, difflib, re
from pathlib import Path
A, B, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
SKIP_BIN = {"data.bin"}
def norm(t):   # run-dir paths / cache ids are not part of the comparison
    return re.sub(r"/tmp/krisd/mc/ip/dcout/[^/\"\s]+", "<D>", t)
res = {}
for p in sorted(x.name for x in A.iterdir() if x.is_dir()):
    r = {"same": 0, "diff": [], "only_a": [], "only_b": [], "data_bin": "same"}
    fa = {str(f.relative_to(A / p)) for f in (A / p).rglob("*") if f.is_file()}
    fb = {str(f.relative_to(B / p)) for f in (B / p).rglob("*") if f.is_file()}
    r["only_a"], r["only_b"] = sorted(fa - fb), sorted(fb - fa)
    for f in sorted(fa & fb):
        a, b = A / p / f, B / p / f
        if os.path.basename(f) in SKIP_BIN:
            if a.stat().st_size != b.stat().st_size or not filecmp.cmp(a, b, shallow=False): r["data_bin"] = "DIFF"
            continue
        if filecmp.cmp(a, b, shallow=False): r["same"] += 1; continue
        try:
            ta, tb = norm(a.read_text()), norm(b.read_text())
        except UnicodeDecodeError:
            r["diff"].append(f + " (binary)"); continue
        if ta == tb: r["same"] += 1; continue
        r["diff"].append(f)
        d = "".join(list(difflib.unified_diff(ta.splitlines(True), tb.splitlines(True), f"a/{f}", f"b/{f}", n=1))[:400])
        (OUT / (p + "__" + f.replace("/", "__") + ".diff")).write_text(d)
    res[p] = r
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
for p, r in res.items(): print(p, "same", r["same"], "DIFF", len(r["diff"]), "only_a", len(r["only_a"]), "only_b", len(r["only_b"]), "data.bin", r["data_bin"])
