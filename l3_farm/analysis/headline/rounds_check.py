"""Average rounds progressed (fractional). Run: FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/headline/rounds_check.py"""
import sys, os, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here + "/../..")
import farm
farm.load()
farm.put("5321587", here + "/rounds_remote.py", "/tmp/krisd/rounds_remote.py")
rc, o, e = farm.ssh("srun --jobid=5321587 --overlap python3 /tmp/krisd/rounds_remote.py")
rows = json.loads(o.decode().strip().splitlines()[-1])
heavy = {"44_MiniGPTBlock": 3.0}
tot = []
for n, done, cur, frac, ph in rows:
    if n in heavy: continue
    if done is None: continue
    v = done + (frac if cur >= done else 0)
    tot.append((v, n, done, cur, frac, ph))
tot.sort(reverse=True)
for v, n, done, cur, frac, ph in tot: print(f"  {n:34s} {v:5.2f}  (completed {done}, in round {cur}, phase fraction {frac}, phases {','.join(ph[-3:])})")
vals = [v for v, *_ in tot] + [3.0]
print(f"\nproblems counted: {len(vals)} (incl. 44 done=3.0; 18_SqueezeNet on heavy node excluded); mean rounds progressed = {sum(vals)/len(vals):.2f} of 3 | completed rounds only = {sum(t[2] for t in tot)+3}/{len(vals)} = {(sum(t[2] for t in tot)+3)/len(vals):.2f}")
