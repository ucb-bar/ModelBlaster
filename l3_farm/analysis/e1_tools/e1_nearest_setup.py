"""Build + push the cross-problem pool for the nearest-K arm.  e1_nearest_setup.py build | push JOB [JOB..]
Pool = every sweep problem's verified impl records (speedup_vs_seed>=1.02, cuda): per (problem, op, shape set) the 2 best distinct
codes, shapes de-duplicated. -> e1_data/stage/npool.tar.gz (index.jsonl + code/<sha>.cu, _from_problem set)."""
import sys, io, json, tarfile, collections
sys.path.insert(0, '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/analysis/e1_tools')
import e1_prep as p
OUT = p.OUT / "npool.tar.gz"
def build():
    g = collections.defaultdict(list)
    for r in p.all_records():
        sh = [s for s in (r.get('shapes') or []) if isinstance(s, dict)]
        if not sh or r['_src'] in p.EXCLUDE and False:
            continue
        uniq = list({json.dumps(s, sort_keys=True): s for s in sh}.values())
        r = dict(r, shapes=uniq)
        g[(r['_src'], r['op'], json.dumps(sorted(json.dumps(s, sort_keys=True) for s in uniq)))].append(r)
    rows, codes = [], {}
    for v in g.values():
        seen = set()
        for r in sorted(v, key=lambda r: -(r.get('speedup_vs_seed') or 0)):
            if r['code_sha'] in seen: continue
            seen.add(r['code_sha'])
            if len(seen) > 2: break
            c = p.RES / r['_src'] / 'work' / 'knowledge' / 'impl' / 'code' / f"{r['code_sha']}.cu"
            if not c.exists(): continue
            rr = {k: x for k, x in r.items() if k != '_src'}
            rr['_from_problem'] = r['_src']
            rows.append(rr); codes[r['code_sha']] = c
    with tarfile.open(OUT, 'w:gz') as tw:
        p.add(tw, 'npool/index.jsonl', ("\n".join(json.dumps(r) for r in rows) + "\n").encode())
        for sha, c in codes.items():
            p.add(tw, f'npool/code/{sha}.cu', c.read_bytes())
    print(len(rows), 'records', len(codes), 'codes', OUT.stat().st_size / 1e6, 'MB')
def push(jobs):
    import e1_run
    for j in jobs:
        e1_run.put(j, OUT, f"{e1_run.R}/npool.tar.gz")
        rc, o, e = e1_run.run(j, f"cd {e1_run.R} && rm -rf npool && tar xzf npool.tar.gz && ls npool/code | wc -l", 300)
        print(j, o.strip(), e[-200:])
if __name__ == '__main__':
    build() if sys.argv[1] == 'build' else push(sys.argv[2:])
