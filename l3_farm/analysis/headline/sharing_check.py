"""Sharing evidence for the headline sweep. Run: FARM_CONFIG=analysis/headline/config_headline.json python3 analysis/headline/sharing_check.py"""
import sys, os, json, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import farm
farm.load()
here = os.path.dirname(os.path.abspath(__file__))
job = "5321587"
farm.put(job, here + "/sharing_remote.py", "/tmp/krisd/sharing_remote.py")
rc, o, e = farm.ssh(f"srun --jobid={job} --overlap python3 /tmp/krisd/sharing_remote.py")
rows = json.loads(o.decode().strip().splitlines()[-1])
st = "knowledge_global_headline"
root = os.path.join(here, "../../..", st)
hist = [json.loads(l) for l in open(root + "/history.jsonl")]
print(f"store: current={open(root + '/current').read().strip()} merges={len(hist)} problems_merged={len({(h.get('run') or {}).get('problem') for h in hist})} impl_code={len(os.listdir(root + '/impl_code'))}")
ap = [r for r in rows if r["refresh"] == "applied"]
print(f"barrier refresh applied: {len(ap)}/{len(rows)}; imported_rows min/median/max:",
      (min(r['rows'] for r in ap), sorted(r['rows'] for r in ap)[len(ap)//2], max(r['rows'] for r in ap)) if ap else None)
print("refresh not applied / missing:", [r["p"] for r in rows if r["refresh"] != "applied"][:40])
print(f"kernel reuse: char-pass reused total={sum(r['char_reused'] or 0 for r in rows)}; later rounds reused total={sum(r['reuse_later'] for r in rows)}; problems past r00: {[r['p'] for r in rows if r['rounds']]}")
