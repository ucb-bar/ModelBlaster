"""Old L40S results (firesim2 /nscratch/krisd/tachyon-rl-fresh/results: ncu Elapsed Cycles, GPT-5 agent, older KernelBench sizes) vs this sweep (whole-model CUDA-graph us vs eager us)."""
import csv,glob,json,math,os,re,sys
D='analysis/l12_sweep/old_firesim2/'
gm=lambda a: math.exp(sum(map(math.log,a))/len(a)) if a else float('nan')
def fl(x):
    try: return float(x)
    except: return None
pt={r['problem']:fl(r['avg_Elapsed_Cycles']) for r in csv.DictReader(open(D+'combined_l40s_baseline_pytorch.csv'))}
old={}
for lvl,f in ((1,'combined_l40s_level1_cycles.csv'),(2,'combined_l40s_level2_cycles.csv')):
    for r in csv.DictReader(open(D+f)):
        k=r['Kernel ID']; vals=[fl(r[c]) for c in r if c.startswith('Best_')]; vals=[v for v in vals if v]
        base=fl(r['Baseline'])
        if not vals or not base: continue
        old[(lvl,k)]=dict(seed=base,final=vals[-1],best=min(vals),torch=pt.get(k))
out=[]
for lvl in (1,2):
    rows=[v for (l,k),v in old.items() if l==lvl and v['torch']]
    for nm,key in (('final column','final'),('best-ever','best')):
        sp=sorted(v['torch']/v[key] for v in rows)
        out.append(f"OLD L{lvl} {nm}: n={len(sp)} geomean vs pytorch {gm(sp):.2f} median {sp[len(sp)//2]:.2f} faster {sum(x>1 for x in sp)} ({100*sum(x>1 for x in sp)//len(sp)}%) >=2x {sum(x>=2 for x in sp)}")
def key(n):
    m=re.match(r'(6[12])(\d{3})_(.*)',n); return (1 if m.group(1)=='61' else 2), f"{int(m.group(2)):03d}_"+m.group(3).rstrip('_')
new={}
for f in glob.glob('results_l12/6*/work/summary.json'):
    n=f.split('/')[1]
    try: r=list(json.load(open(f))['irs'].values())[0]
    except Exception: continue
    t=f"analysis/l12_sweep/torch_posthoc/{n}/torch_posthoc.json"
    if r.get('model_passed') and r.get('model_e2e_us') and os.path.exists(t):
        e=json.load(open(t)).get('eager_us')
        if e: new[key(n)]=(r['model_e2e_us'],e)
oldk={(l,re.sub(r'_+$','',k)):v for (l,k),v in old.items()}
paired=sorted((k,v,oldk[k]) for k,v in new.items() if k in oldk and oldk[k]['torch'])
out.append(f"NEW collected with eager: {len(new)} {{L1: {sum(1 for k in new if k[0]==1)}, L2: {sum(1 for k in new if k[0]==2)}}}; paired with old: {len(paired)}")
ns=[];of=[];ob=[]
out.append(f"{'problem':46s} new x eager | old x pytorch (final / best-ever)")
for k,v,o in paired:
    nx=v[1]/v[0]; ns.append(nx); of.append(o['torch']/o['final']); ob.append(o['torch']/o['best'])
    out.append(f"L{k[0]} {k[1][:42]:43s} {nx:7.2f} | {of[-1]:8.2f} {ob[-1]:8.2f}")
if ns: out.append(f"PAIRED geomean: new {gm(ns):.2f} | old final {gm(of):.2f} | old best-ever {gm(ob):.2f}  (n={len(ns)})")
print("\n".join(out))
open('analysis/l12_sweep/compare_old_output.txt','w').write("\n".join(out)+"\n")
