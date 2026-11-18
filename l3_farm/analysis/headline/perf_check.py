"""Performance check for the headline sweep. Run: FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/headline/perf_check.py
 - finished problems: final model_e2e_us vs first sweep (results/validated.md) -> flags regressions > 10%
 - in-progress problems: characterization partition_us before->after, decision outcomes, measured gains."""
import sys, os, re, json, glob, time
here = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(here, "../..")
sys.path.insert(0, R)
import farm
farm.load()
old = {}
for l in open(R + "/results/validated.md"):
    c = [x.strip() for x in l.split("|")]
    if len(c) > 6 and re.match(r"\d+_", c[1]):
        m = lambda s: (float(re.match(r"[\d.]+", s).group(0)) if re.match(r"[\d.]+", s) else None)
        old[c[1]] = dict(seed=m(c[2]), final=m(c[3]), torch=m(c[5]))
print("== finished problems (this sweep vs first sweep)")
fin = []
for f in glob.glob(R + "/results_headline*/*/work/model_runtime.json"):
    n = f.split("/")[-3]; j = json.load(open(f)); j = j.get(n, j)
    if not j.get("passed"): print(f"  {n}: NOT PASSED"); continue
    new = j["model_e2e_us"]; o = old.get(n, {})
    ratio = (o.get("final") / new) if o.get("final") else None
    t = o.get("torch")
    fin.append((n, new, o.get("final"), ratio, t))
    flag = "  <-- REGRESSION >10% vs first sweep" if ratio and ratio < 0.9 else ""
    print(f"  {n}: {new:.0f} us (first sweep {o.get('final')}; {ratio:.2f}x better)" if ratio else f"  {n}: {new:.0f} us", f"| vs torch {t/new:.2f}x" if t else "", flag)
if not fin: print("  none finished yet")
job = "5321587"
farm.put(job, here + "/perf_remote.py", "/tmp/krisd/perf_remote.py")
rc, o, e = farm.ssh(f"srun --jobid={job} --overlap python3 /tmp/krisd/perf_remote.py")
rows = json.loads(o.decode().strip().splitlines()[-1])
print("== in progress (char partition us before->after; decisions; measured gains; rounds past char)")
now = time.time()
for r in rows:
    cb, ca = r["char_before"], r["char_after"]
    g = sorted(r["gains"], key=lambda x: x[1])
    best = g[0] if g else None
    print(f"  {r['p']:34s} char {cb and round(cb)}->{ca and round(ca)} ({cb/ca:.1f}x)" if cb and ca else f"  {r['p']:34s} char n/a", f"| dec {r['dec']} | measured {len(g)} best {best} | round {r['rounds']} | last decision {int((now - r['last_ts'])/60) if r['last_ts'] else '?'} min ago | files active {int((now - r['active_ts'])/60) if r.get('active_ts') else '?'} min ago")
