"""python3 aggregate.py -> summary.csv + REPORT.md (in this directory)."""
import json, csv, math, re, collections, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
stems = [l.strip() for l in open(H / "problems.txt") if l.strip()]
lvl = lambda s: "L1" if s.startswith("61") and len(s.split("_")[0]) == 5 else "L2" if s.startswith("62") and len(s.split("_")[0]) == 5 else "L3"
rows = []
for s in stems:
    f = H / "results" / s / "torch_posthoc.json"
    r = dict(level=lvl(s), name=s, eager_us="", graph_us="", compile_us="", compile_s="", compile_error="", node="", gpu="")
    if f.exists():
        d = json.loads(f.read_text())
        r.update(eager_us=d.get("eager_us") or "", graph_us=d.get("graph_us") or "", compile_us=d.get("compile_us") or "",
                 compile_s=d.get("compile_s", ""), compile_error=d.get("compile_error") or (d.get("graph_error") and "graph:" + d["graph_error"]) or "",
                 node=d.get("host", ""), gpu=d.get("gpu", ""))
    else:
        r["compile_error"] = "NO_RESULT (bench crashed/timeout/srun failure)"
    rows.append(r)
with open(H / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
gm = lambda v: math.exp(sum(map(math.log, v)) / len(v)) if v else float("nan")
def reason(e):
    e = e or ""
    if e.startswith("NO_RESULT"): return "no result (crash/timeout)"
    if "OutOfMemory" in e or "out of memory" in e: return "OOM"
    if "Timeout" in e: return "timeout"
    return "compile error" if e else "ok"
out = ["# torch.compile sweep (fp16, native sizes, L40S)\n",
       "Method: eager / CUDA graph / torch.compile (default mode) in the same process, median of 50 cudaEvent pairs after 10 warmups. "
       "The compiled callable is timed like eager (includes Python/guard dispatch), so small problems show dispatch overhead.\n",
       "## Counts\n", "| level | n | compile ok | failed | failure reasons |", "|---|---|---|---|---|"]
by = collections.defaultdict(list)
for r in rows: by[r["level"]].append(r)
for L in ("L1", "L2", "L3"):
    rs = by[L]; ok = [r for r in rs if r["compile_us"] != ""]
    rc = collections.Counter(reason(r["compile_error"]) for r in rs if r["compile_us"] == "")
    out.append(f"| {L} | {len(rs)} | {len(ok)} | {len(rs)-len(ok)} | {dict(rc)} |")
out += ["\n## Geomean speedups (n = problems with all needed numbers)\n", "| level | n | eager/compile | graph/compile | compile/eager (=1/first) |", "|---|---|---|---|---|"]
for L in ("L1", "L2", "L3", "ALL"):
    rs = [r for r in rows if (L == "ALL" or r["level"] == L) and r["compile_us"] != "" and r["eager_us"] != ""]
    g = [r for r in rs if r["graph_us"] != ""]
    e = gm([r["eager_us"] / r["compile_us"] for r in rs]); gg = gm([r["graph_us"] / r["compile_us"] for r in g])
    out.append(f"| {L} | {len(rs)} (graph {len(g)}) | {e:.3f} | {gg:.3f} | {1/e if rs else float('nan'):.3f} |")
out.append("\n(compile/eager speedup as asked = eager_us/compile_us; >1 means compile faster.)\n")
slow = sorted([r for r in rows if r["compile_us"] != "" and r["eager_us"] != "" and r["compile_us"] > 1.10 * r["eager_us"]], key=lambda r: -r["compile_us"] / r["eager_us"])
out += [f"## Compile >10% slower than eager: {len(slow)} problems\n", "| level | problem | eager us | compile us | ratio |", "|---|---|---|---|---|"]
out += [f"| {r['level']} | {r['name']} | {r['eager_us']} | {r['compile_us']} | {r['compile_us']/r['eager_us']:.2f} |" for r in slow]
cs = sorted(r["compile_s"] for r in rows if isinstance(r["compile_s"], (int, float)))
if cs:
    q = lambda p: cs[min(len(cs) - 1, int(p * len(cs)))]
    out += ["\n## Compile time (s, 3 warm calls incl. compile)\n", f"n={len(cs)} min {cs[0]} p25 {q(.25)} median {q(.5)} p75 {q(.75)} p90 {q(.9)} max {cs[-1]}; total {sum(cs)/3600:.2f} h\n"]
out += ["## Failures\n"] + [f"- {r['level']} {r['name']}: {r['compile_error'][:160]}" for r in rows if r["compile_us"] == ""]
out += ["\n## Nodes / GPU\n", "; ".join(f"{k or '?'}: {v}" for k, v in collections.Counter((r["node"], r["gpu"]) for r in rows).items()).replace("(", "").replace(")", "")]
(H / "REPORT.md").write_text("\n".join(out) + "\n")
print("\n".join(out[:20]))
