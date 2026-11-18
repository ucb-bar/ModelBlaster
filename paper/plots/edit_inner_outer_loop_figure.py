from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS=None
im=Image.open('orig.png').convert('RGB')
a=np.asarray(im).astype(int)
dk=(a.sum(axis=2)<380)          # glyph / line pixels
BG=(255,253,209); WHITE=(255,255,255); BLACK=(0,0,0)

def bbox(mask, x0,y0,x1,y1):
    sub=mask[y0:y1,x0:x1]
    ys=np.where(sub.any(axis=1))[0]; xs=np.where(sub.any(axis=0))[0]
    return x0+xs[0], y0+ys[0], x0+xs[-1]+1, y0+ys[-1]+1
def words(mask,x0,y0,x1,y1,gap):
    cols=mask[y0:y1,x0:x1].any(axis=0); out=[]; s=None; last=None
    for i,c in enumerate(cols):
        if c:
            if s is None: s=i
            elif last is not None and i-last>gap: out.append((x0+s,x0+last+1)); s=i
            last=i
    if s is not None: out.append((x0+s,x0+last+1))
    return out

# ---------- glyph sources ----------
# Fix Agent label (single line, same font size)
fb=bbox(dk,9230,3880,9990,4110); print('Fix Agent text bbox',fb)
fw=words(dk,fb[0],fb[1],fb[2],fb[3],30); print('Fix Agent words',fw)
ag=(fw[-1][0],fb[1],fw[-1][1],fb[3])            # "Agent"
# baseline of Agent = bottom of the leading 'A'
A=bbox(dk,ag[0],ag[1],ag[0]+110,ag[3]); print('A glyph',A)
agent_img=im.crop(ag); agent_base=A[3]-ag[1]; agent_capH=A[3]-A[1]
# Build + Evaluate title
tb=bbox(dk,8420,2740,10150,2960); print('title bbox',tb)
tw=words(dk,tb[0],tb[1],tb[2],tb[3],30); print('title words',tw)
B=bbox(dk,tb[0],tb[1],tb[0]+100,tb[3]); print('B glyph',B)
title_img=im.crop(tb); title_base=B[3]-tb[1]; title_capH=B[3]-B[1]
print('cap heights: B',title_capH,' A',agent_capH)
sp=int(np.median([tw[i+1][0]-tw[i][1] for i in range(len(tw)-1)])); print('space',sp)

# ---------- 1. Build + Evaluate Agent ----------
L,R,T,Bt=8380,10192,2715,3593; d=160
Lp,Rp=L-d,R+d
ib=bbox(dk,L+60,3000,R-60,3440); print('inner boxes bbox',ib); interior=im.crop((ib[0]-6,ib[1]-6,ib[2]+6,ib[3]+6)); ipos=(ib[0]-6,ib[1]-6)
# arrowhead (Actor -> Build) at the right edge
yh=(T+Bt)//2
head=im.crop((R-2,yh-80,R+140,yh+80))
out=im.copy()
from PIL import ImageDraw
dr=ImageDraw.Draw(out)
dr.rectangle((Rp-12,yh-90,Rp+150,yh+90),fill=BG)       # clear the line stub where the head will move
dr.rectangle((Lp,T,Rp,Bt),fill=WHITE)
for rect in ((Lp,T,Rp,T+11),(Lp,Bt-11,Rp,Bt),(Lp,T,Lp+11,Bt),(Rp-11,T,Rp,Bt)): dr.rectangle(rect,fill=BLACK)
out.paste(interior,ipos)
dr.rectangle((L+14,tb[1]-12,R-13,tb[3]+30),fill=WHITE)   # erase old title
head_x=Rp-2
out.paste(head,(head_x,yh-80))
total=(tb[2]-tb[0])+sp+(ag[2]-ag[0]); cx=(L+R)//2; x0=cx-total//2
base=tb[1]+title_base
out.paste(title_img,(x0,base-title_base))
out.paste(agent_img,(x0+(tb[2]-tb[0])+sp, base-agent_base))

# ---------- 2. bigger Critic Reflection Agent box ----------
cL,cR,cT,cB=7150,8004,3672,4345
txt=bbox(dk,cL+16,cT+16,cR-16,cB-16); print('critic text bbox',txt)
t_img=im.crop(txt); s=1.12
t_img=t_img.resize((int(t_img.width*s),int(t_img.height*s)),Image.LANCZOS)
nL,nR,nT,nB=cL-95,cR+95,cT,cB+95
dr.rectangle((nL,nT,nR,nB),fill=WHITE)
for rect in ((nL,nT,nR,nT+11),(nL,nB-11,nR,nB),(nL,nT,nL+11,nB),(nR-11,nT,nR,nB)): dr.rectangle(rect,fill=BLACK)
out.paste(t_img,((nL+nR)//2-t_img.width//2,(nT+nB)//2-t_img.height//2))
out.save('new.png',optimize=True)
print('saved', out.size)
f=6.7355
out.crop((int(960*f),int(380*f),int(1620*f),int(680*f))).resize((1500,int(1500*(300/660)))).save('preview.png')
