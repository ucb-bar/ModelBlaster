from PIL import Image, ImageDraw, ImageFont
import numpy as np
Image.MAX_IMAGE_PIXELS=None
FONT='/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'
im=Image.open('cur.png').convert('RGB')
def ink(text,size):
    f=ImageFont.truetype(FONT,size)
    W=int(size*len(text)*0.8)+200; H=int(size*2)
    c=Image.new('L',(W,H),255); ImageDraw.Draw(c).text((50,int(size*0.4)),text,font=f,fill=0)
    a=np.asarray(c)<128; ys=np.where(a.any(axis=1))[0]; xs=np.where(a.any(axis=0))[0]
    return c.crop((xs[0],ys[0],xs[-1]+1,ys[-1]+1))
# calibrate the font size on the old title (ink width 2817 px, height 207 px)
lo,hi=100,600
for _ in range(25):
    mid=(lo+hi)/2
    w=ink('(a) Outer Loop: Model Level',int(round(mid))).width
    if w<2817: lo=mid
    else: hi=mid
size=int(round(lo)); t=ink('(a) Outer Loop: Model Level',size)
print('size',size,'old-title render',t.size,'(target 2817x207)')
old_a=(1507,4765,4324,4972); old_b=(8625,4755,11510,4963)
dr=ImageDraw.Draw(im)
for b in (old_a,old_b): dr.rectangle((b[0]-12,b[1]-12,b[2]+12,b[3]+12),fill=(255,255,255))
def put(text,center_x,top):
    g=ink(text,size).convert('L')
    img=Image.new('RGB',g.size,(0,0,0)); mask=Image.eval(g,lambda v:255-v)
    im.paste(img,(int(center_x-g.width/2),top),mask)
    print(text,g.size,'x from',int(center_x-g.width/2),'to',int(center_x+g.width/2))
put('(a) Outer Loop: Model Level Optimization',(old_a[0]+old_a[2])//2,old_a[1])
put('(b) Inner Loop: Kernel Level Optimization',(old_b[0]+old_b[2])//2+400,old_b[1])
im.save('new_titles.png',optimize=True)
f=6.7355
im.crop((0,int(690*f),int(2000*f),int(745*f))).resize((2000,int(55*f*2000/(2000*f)))).save('titles_preview.png')
