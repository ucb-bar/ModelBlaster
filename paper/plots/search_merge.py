import sys, itertools; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, pandas as pd
import reproduce_from_raw as R
from parse_cycles_per_iter import compute_cumulative_best_cycles

paper={('L40S',1):(1.080,1.037,59.72),('H100',1):(1.497,1.137,70.37),('L40S',2):(2.214,2.074,72.60),('H100',2):(2.592,2.291,84.85)}
def run_best(d: Path):
    try:
        h,rows=compute_cumulative_best_cycles(d,max_iter=10,require_checker=True)
    except Exception: return None
    if not rows: return None
    df=pd.DataFrame(rows,columns=h)
    v=np.vstack([pd.to_numeric(df[f'Best_{i}'],errors='coerce').to_numpy(float) for i in range(1,11)])
    v=np.where(v>0,v,np.nan)
    with np.errstate(all='ignore'): b=np.nanmin(v,axis=0)
    return pd.Series(b,index=df['Kernel ID'].astype(str))
# candidate runs per cell: (path)
cands={}
for base in (R.RAW, Path('raw2')):
    for run in sorted(base.iterdir()):
        for model in sorted(run.iterdir()):
            for lvl in (1,2):
                d=model/f'level{lvl}'
                if d.is_dir():
                    s=run_best(d)
                    if s is not None and len(s)>10: cands[(run.name,model.name,lvl)]=s
print(len(cands),'candidate (run,model,level)')
for k,s in cands.items(): print(' ',k,len(s))
def stats(sp):
    return (np.exp(np.log(sp).mean()),np.median(sp),100*(sp>1).mean(),sp.size)
gpu_key={'H100':['H100'],'L40S':['l40s','l40']}
for (label,level),(tg,tm,tp) in paper.items():
    gp,_=None,None
    run,gpu=R.RUNS[(label,level)]
    comp=R.load_base(R.base_csv(gpu,level,'compile')); eag=R.load_base(R.base_csv(gpu,level,'eager'))
    m=pd.merge(eag,comp,on='problem',how='outer',suffixes=('_e','_c'))
    pt=pd.DataFrame({'problem':m['problem'].astype(str),'cycles':np.fmin(m['cycles_e'],m['cycles_c'])}).set_index('problem')['cycles']
    key=label.lower() if label!='H100' else 'h100'
    pool=[(k,s) for k,s in cands.items() if k[2]==level and (key in k[0].lower()) and ('gpt-5' in k[1] or 'gpt-4' in k[1]) and (label!='L40S' or 'l40' in k[0].lower()) and (label!='H100' or 'H100' in k[0])]
    if label=='H100' and level==1: pool+= [(k,s) for k,s in cands.items() if k[0].endswith('H100_header') ]
    print(f"\n=== {label} L{level}: candidates={[p[0][0].replace('rl_baseline_debug_computelab_','')+'/'+p[0][1][:9] for p in pool]} | table geo={tg} med={tm} >1={tp}")
    res=[]
    for r in range(1,min(len(pool),5)+1):
        for sub in itertools.combinations(range(len(pool)),r):
            ser=pd.concat([pool[i][1] for i in sub],axis=1).min(axis=1)
            j=pd.concat([pt,ser],axis=1,join='inner').dropna(); j.columns=['b','o']
            j=j[(j.b>0)&(j.o>0)]
            if j.empty: continue
            sp=(j.b/j.o).to_numpy()
            g,md,p,n=stats(sp)
            res.append((abs(np.log(g/tg))+abs(np.log(md/tm))+abs(p-tp)/50,sub,g,md,p,n))
    res.sort(key=lambda x:x[0])
    for e in res[:3]:
        names=[pool[i][0][0].replace('rl_baseline_debug_computelab_','')+'/'+pool[i][0][1][:9] for i in e[1]]
        print(f"   dist={e[0]:.3f} n={e[5]} geo={e[2]:.3f} med={e[3]:.3f} >1={e[4]:.1f}%  <- {names}")
