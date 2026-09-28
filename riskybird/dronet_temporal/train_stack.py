#!/usr/bin/env python3
"""Train a FRAME-STACKED (temporal) DroNet steering head on the real HM01B0
PULP-DroNet v3 dataset. N consecutive grayscale frames -> N input channels;
only conv0 input depth changes, downstream identical -> stays in int8-Gemmini
envelope with zero new operators.

Adapted from himax/train_himax.py. Correctness rules:
  - frames time-ordered by int(filename) within an acquisition
  - stack for target i = [i-(N-1) .. i] (channel order oldest->newest), label=target yaw
  - NEVER stack across acquisition boundaries
  - no temporal leakage: all N frames must share the target's partition
  - hflip flips all N channels together and negates the label
  - drop anomalous huge-int filenames (>1e12; the ~31 tii uint64-overflow frames)
"""
import os, sys, time, csv, glob, argparse
X="/scratch2/dima/misc_sw/XPU-RT"; sys.path.insert(0, X)
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES=True
from torchvision import transforms
import torchvision.transforms.functional as TF
from qnn_models.dronet import DronetTorch

ap=argparse.ArgumentParser()
ap.add_argument("--N",type=int,default=int(os.environ.get("N","3")))
ap.add_argument("--epochs",type=int,default=int(os.environ.get("EPOCHS","45")))
ap.add_argument("--bs",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3)
ap.add_argument("--root",default=f"{X}/datasets/pulp_dronet_himax/raw/Dataset_PULP_Dronet_v3")
ap.add_argument("--out",default=None)
ap.add_argument("--norm",type=float,default=90.0)
a=ap.parse_args()
N=a.N
OUT=a.out or f"{X}/logs/dronet_himax_stack{N}"
dev="cuda" if torch.cuda.is_available() else "cpu"
HUGE=1e12

# ---- gather rows grouped by acquisition, build leakage-free N-stacks ----
def build(partition):
    """Return list of (tuple_of_N_paths_oldest_to_newest, yaw_norm)."""
    stacks=[]; n_target=0; drop_bnd=0; drop_part=0; drop_huge=0
    for c in sorted(glob.glob(os.path.join(a.root,"**","labels_partitioned.csv"),recursive=True)):
        d=os.path.dirname(c); imgd=os.path.join(d,"images")
        keyed=[]
        for r in csv.DictReader(open(c)):
            try: k=int(r["filename"].rsplit(".",1)[0])
            except: continue
            if k>HUGE: drop_huge+=1; continue
            try: y=max(-1.0,min(1.0,float(r["label_yaw_rate"])/a.norm))
            except: continue
            p=os.path.join(imgd,r["filename"])
            if not os.path.exists(p): continue
            keyed.append((k, r["partition"], p, y))
        keyed.sort(key=lambda t:t[0])
        parts=[t[1] for t in keyed]
        for i in range(len(keyed)):
            if parts[i]!=partition: continue
            n_target+=1
            if i-(N-1)<0: drop_bnd+=1; continue
            win=keyed[i-(N-1):i+1]
            if any(w[1]!=partition for w in win): drop_part+=1; continue
            stacks.append((tuple(w[2] for w in win), keyed[i][3]))
    print(f"[build {partition}] N={N} target_frames={n_target} usable_stacks={len(stacks)} "
          f"drop_boundary={drop_bnd} drop_partition_mismatch={drop_part} drop_huge_int={drop_huge}",flush=True)
    return stacks

tr=build("train"); va=build("valid")
ytr=np.array([y for _,y in tr]); yva=np.array([y for _,y in va])
print(f"[data] train_stacks={len(tr)} valid_stacks={len(va)}",flush=True)
print(f"[yaw] train std={ytr.std():.3f} frac|y|>0.02={np.mean(np.abs(ytr)>0.02):.3f}  "
      f"valid std={yva.std():.3f} frac_nz={np.mean(np.abs(yva)>0.02):.3f}",flush=True)

class Stack(Dataset):
    def __init__(self,items,train): self.items=items; self.train=train
    def __len__(self): return len(self.items)
    def _load(self,paths,size):
        chans=[]
        for p in paths:
            with Image.open(p) as im: im=im.convert("L").resize((size,size),Image.BILINEAR)
            chans.append(TF.to_tensor(im))          # (1,size,size)
        return torch.cat(chans,dim=0)               # (N,size,size), oldest->newest
    def __getitem__(self,i):
        paths,y=self.items[i]
        if self.train:
            x=self._load(paths,128)                 # (N,128,128)
            t,l,h,w=transforms.RandomCrop.get_params(x,(112,112))
            x=TF.crop(x,t,l,h,w)                    # SAME crop across all N channels
            if np.random.rand()<0.5:
                x=torch.flip(x,dims=[2]); y=-y      # flip all channels + negate label
        else:
            x=self._load(paths,112)                 # (N,112,112)
        return x, torch.tensor([y],dtype=torch.float32)

tl=DataLoader(Stack(tr,True),batch_size=a.bs,shuffle=True,num_workers=8,pin_memory=True,drop_last=True)
vl=DataLoader(Stack(va,False),batch_size=256,shuffle=False,num_workers=8,pin_memory=True)

def eva(pred,gt):
    v=gt.var(); return float(1-((pred-gt).var()/v)) if v>0 else float('nan')
def evaluate(m):
    m.eval(); P=[];G=[]
    with torch.no_grad():
        for x,y in vl:
            s,_=m(x.to(dev)); P.append(s.squeeze(1).cpu().numpy()); G.append(y.squeeze(1).numpy())
    P=np.concatenate(P);G=np.concatenate(G); nz=np.abs(G)>0.02
    return float(((P-G)**2).mean()), eva(P,G), (eva(P[nz],G[nz]) if nz.sum()>2 else float('nan')), float(np.abs(P-G).mean())

model=DronetTorch(img_dims=(112,112),img_channels=N,output_dim=1,small=True).to(dev)
opt=optim.Adam(model.parameters(),lr=a.lr,weight_decay=1e-4)
sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
outd=__import__("pathlib").Path(OUT); outd.mkdir(parents=True,exist_ok=True); best=-1e9; best_nz=float('nan')
print(f"[model] N={N} img_channels={N} params={sum(p.numel() for p in model.parameters())} conv0={list(model.conv_modules[0].weight.shape)}",flush=True)
for ep in range(1,a.epochs+1):
    model.train(); t0=time.time(); ls=0;nc=0
    for x,y in tl:
        x=x.to(dev);y=y.to(dev); opt.zero_grad()
        s,_=model(x); loss=nn.functional.mse_loss(s,y); loss.backward(); opt.step()
        ls+=loss.item()*len(x); nc+=len(x)
    sch.step(); vmse,veva,veva_nz,vmae=evaluate(model)
    print(f"[ep{ep:02d}] train_mse={ls/nc:.4f} val_mse={vmse:.4f} val_EVA={veva:.3f} val_EVA_nz={veva_nz:.3f} val_MAE={vmae:.4f} ({time.time()-t0:.1f}s)",flush=True)
    torch.save(model.state_dict(),outd/"last.pt")
    if veva>best: best=veva; best_nz=veva_nz; torch.save(model.state_dict(),outd/"best.pt"); print(f"  ^best EVA={best:.3f} (nz={best_nz:.3f})",flush=True)
print(f"[done] N={N} best_val_EVA={best:.3f} best_val_EVA_nz={best_nz:.3f} ckpt={outd/'best.pt'}",flush=True)
print(f"STACK_TRAIN_DONE N={N} best_EVA={best:.4f} best_EVA_nz={best_nz:.4f}",flush=True)
