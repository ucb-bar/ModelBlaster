import csv, json, glob, math, statistics as st
from pathlib import Path
H = Path(__file__).resolve().parent
tasks_dir = H / "robust-kbench/tasks/kernelbench"
def name(lv, k):
    p = list((tasks_dir / lv / f"task_{k}").glob("*.py"))
    p = [x for x in p if x.name not in ("func_forward.py",)]
    return p[0].stem if p else ""
cyc, old = {}, {}
for lv, f, g in (("level_1", "baseline_sakana_l40s_clean.csv", "results_hf_level_1.csv"), ("level_2", "baseline_sakana_l40s_level2_clean.csv", "results_hf_level_2.csv")):
    for n, v in list(csv.reader(open(H / "old_results" / f, encoding="utf-8-sig")))[1:]:
        cyc[(lv, int(n.split("_")[0]))] = v.strip()
    for r in csv.DictReader(open(H / "old_results" / g, encoding="utf-8-sig")):
        if r["Task_ID"].strip().isdigit(): old[(lv, int(r["Task_ID"]))] = (r["Speedup_native"], r["Speedup_compile"])
res = {}
for f in glob.glob(str(H / "results/level_*/task_*/result.json")):
    r = json.load(open(f)); res[(r["level"], int(r["task"].split("/")[1].split("_")[1]))] = r
import re
def err(lv, k):
    t = (H / f'results/{lv}/task_{k}/run.log').read_text()
    m = re.search(r'"error": "([^"]{3,240})"', t) or re.search(r'Error[^\n]{0,200}', t)
    return (m.group(1) if m and m.groups() else (m.group(0) if m else '')).replace('\\n', ' ')[:200]
rows = []
for (lv, k), r in sorted(res.items()):
    c = cyc.get((lv, k), "")
    rows.append({"level": lv, "task": k, "name": name(lv, k), "old_csv_status": "dirty" if c.lower() == "d" else ("N/A" if c.upper() == "N/A" else "cycles"),
                 "old_ncu_cycles": c if c.lower() not in ("d", "n/a") else "", "correct": r["correct"], "torch_native_ms": r["torch_native_ms"], "torch_compile_ms": r["torch_compile_ms"],
                 "cuda_ms": r["cuda_ms"], "speedup_native": r["speedup_native"], "speedup_compile": r["speedup_compile"],
                 "old_hf_speedup_native": old.get((lv, k), ("", ""))[0], "error": err(lv, k) if not r["correct"] else ""})
with open(H / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def gm(x): return math.exp(sum(map(math.log, x)) / len(x)) if x else float("nan")
out = []
for lv in ("level_1", "level_2", "all"):
    rs = [r for r in rows if (lv == "all" or r["level"] == lv) and r["old_csv_status"] != "dirty"]   # dirty tasks excluded from the statistics
    ok = [r for r in rs if r["correct"] and r["speedup_native"]]
    sn = [r["speedup_native"] for r in ok]; sc = [r["speedup_compile"] for r in ok if r["speedup_compile"]]
    out.append(f"| {lv} | {len(rs)} | {len(ok)} | {gm(sn):.2f}x | {st.median(sn):.2f}x | {sum(s > 1 for s in sn)} | {gm(sc):.2f}x | {sum(s > 1 for s in sc)} |")
print("\n".join(out))
