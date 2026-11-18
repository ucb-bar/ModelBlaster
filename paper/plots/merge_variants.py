"""Test whether the table = ours merged with cuDNN/cuBLAS baselines (plot_fastp_compare_sakana.py logic)."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import reproduce_from_raw as R

paper={('L40S',1):(93,4.502,1.080,1.037,59.72),('H100',1):(86,2.817,1.497,1.137,70.37),
       ('L40S',2):(95,9.419,2.214,2.074,72.60),('H100',2):(81,10.223,2.592,2.291,84.85)}
B=R.BASE
def cd_path(gpu,level): return B/(f"baseline_cudnn{'_level2' if level==2 else ''}_{gpu}.csv")
def sak_clean(gpu,level): return B/(f"baseline_sakana_{gpu}{'_level2' if level==2 else ''}_clean.csv")
def dirty_set(p):
    df=pd.read_csv(p); df.columns=[c.strip() for c in df.columns]
    pc,cc=df.columns[0],df.columns[1]
    return set(df.loc[df[cc].astype(str).str.strip().str.lower().eq('d'),pc].astype(str))
def load_cycles(p):
    df=pd.read_csv(p); df.columns=[c.strip() for c in df.columns]
    return {str(r.iloc[0]):float(pd.to_numeric(r.iloc[1],errors='coerce')) for _,r in df.iterrows()}
def stats(sp):
    sp=sp[np.isfinite(sp)&(sp>0)]
    return f"n={sp.size} mean={sp.mean():.3f} geo={np.exp(np.log(sp).mean()):.3f} med={np.median(sp):.3f} >1={100*(sp>1).mean():.2f}% <1={100*(sp<1).mean():.2f}%"
for (label,level),(run,gpu) in R.RUNS.items():
    if gpu=='a6000': continue
    ours=R.ours_table(run,level)
    comp=R.load_base(R.base_csv(gpu,level,'compile')); eag=R.load_base(R.base_csv(gpu,level,'eager'))
    m=pd.merge(eag,comp,on='problem',how='outer',suffixes=('_e','_c'))
    pt=pd.DataFrame({'problem':m['problem'].astype(str),'cycles':np.fmin(m['cycles_e'],m['cycles_c'])})
    mm=pd.merge(pt,ours,on='problem',how='left')
    b=mm['cycles'].to_numpy(float); o=mm['ours'].to_numpy(float)
    base_sp=np.where((b>0)&(o>0),b/o,np.nan)
    cd=load_cycles(cd_path(gpu,level)); dirty=dirty_set(sak_clean(gpu,level))
    probs=mm['problem'].tolist()
    def merge(use_dirty):
        out=base_sp.copy()
        if use_dirty:
            for i,p in enumerate(probs):
                if p in dirty:
                    sp=1.0; c=cd.get(p)
                    if b[i]>0 and c and c>0 and c<b[i]: sp=b[i]/c
                    out[i]=sp
        for i,p in enumerate(probs):
            c=cd.get(p)
            if b[i]>0 and c and c>0:
                s=b[i]/c
                if not np.isfinite(out[i]) or s>out[i]: out[i]=s
        return out
    print(f"{label} L{level} pool={len(probs)} dirty={len(dirty)}  paper(valid%,mean,geo,med,>1)={paper.get((label,level))}")
    print("   ours only      :",stats(base_sp))
    print("   +cuDNN subst   :",stats(merge(False)))
    print("   +dirty+cuDNN   :",stats(merge(True)))
