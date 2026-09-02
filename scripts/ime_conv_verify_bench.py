#!/usr/bin/env python3
"""Board verify + IME-vs-RVV cycle bench for the conv2d_s8 IME kernel.

Compiles three implementations into ONE riscv binary — the new IME kernel
(kernels/ime/ime_conv2d_s8_ime_vmadot_4x4x8.c), the board-proven RVV kernel
(kernels/rvv/rvv_conv2d_s8_rvv_oc_blocked.c), and an independent scalar oracle
in the harness — over the REAL (deduped) conv shapes of dronet + yolov8_nano.
On the K1 (pinned to cluster 0, where smt.vmadot is legal) it checks
max_abs_err(IME, oracle)==0 AND max_abs_err(RVV, oracle)==0 per shape, and times
IME vs RVV with rdcycle (min of N reps). Emits artifacts/ime_conv/*.
"""
import json, os, subprocess, sys, textwrap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS = os.environ.get("CROSS",
    "/scratch2/agustin/chipyard/.conda-env/riscv-tools/bin/riscv64-unknown-linux-gnu-")
HOST = os.environ.get("MODELBLASTER_K1_HOST", "k1")
REMOTE = os.environ.get("MODELBLASTER_K1_REMOTE_ROOT", "/root/mb_k1") + "/ime_conv"
OUT = os.path.join(REPO, "artifacts", "ime_conv")
MARCH = ["-march=rv64gcv_zvl256b", "-mabi=lp64d", "-O3"]


def load_shapes():
    seen, shapes = set(), []
    for net in ("dronet", "yolov8_nano"):
        g = os.path.join(REPO, "build", "k1_xpurt", net, "int8", "graph.json")
        if not os.path.exists(g):
            g = os.path.join(REPO, "build", "k1", net, "int8", "graph.json")
        ir = json.load(open(g))
        for op in ir["ops"]:
            conv = None
            if op["op"] == "conv2d_s8":
                conv = op
            elif op.get("sub_ops"):
                conv = {s["op"]: s for s in op["sub_ops"]}.get("conv2d_s8")
            if not conv:
                continue
            s, q = conv["shape"], conv.get("quant", {})
            if q.get("input_offset", 0) or q.get("filter_offset", 0):
                continue  # asymmetric -> RVV only
            key = (s["IC"], s["IH"], s["IW"], s["OC"], s["KH"], s["KW"],
                   s["SH"], s["SW"], s["PH"], s["PW"])
            if key in seen:
                continue
            seen.add(key)
            OH = (s["IH"] + 2*s["PH"] - s["KH"])//s["SH"] + 1
            OW = (s["IW"] + 2*s["PW"] - s["KW"])//s["SW"] + 1
            shapes.append(dict(net=net, name=op["name"], IC=s["IC"], IH=s["IH"],
                IW=s["IW"], OC=s["OC"], KH=s["KH"], KW=s["KW"], SH=s["SH"],
                SW=s["SW"], PH=s["PH"], PW=s["PW"], OH=OH, OW=OW, M=OH*OW,
                out_mult=q.get("output_multiplier", 1<<30),
                out_shift=q.get("output_shift", 0),
                amin=q.get("activation_min", -128), amax=q.get("activation_max", 127)))
    return shapes


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void ime_conv(const int8_t*,const int8_t*,const int32_t*,int8_t*,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int);
void rvv_conv(const int8_t*,const int8_t*,const int32_t*,int8_t*,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int);

static int32_t q31(int32_t x,int32_t mult,int32_t shift){
    int64_t p=(int64_t)x*(int64_t)mult; p=(p+(1LL<<30))>>31; int32_t s=(int32_t)p;
    if(shift>0){int32_t r=(1<<(shift-1)); return (s+r)>>shift;} return s<<(-shift);
}
/* Independent scalar oracle. IHWOC weights, NCHW act, symmetric (offsets 0). */
static void ref_conv(const int8_t*in,const int8_t*w,const int32_t*bias,int8_t*out,
  int IC,int IH,int IW,int OC,int KH,int KW,int SH,int SW,int PH,int PW,
  int mult,int shift,int amin,int amax){
    int OH=(IH+2*PH-KH)/SH+1, OW=(IW+2*PW-KW)/SW+1;
    for(int oc=0;oc<OC;oc++) for(int oh=0;oh<OH;oh++) for(int ow=0;ow<OW;ow++){
        int32_t acc=bias?bias[oc]:0;
        for(int ic=0;ic<IC;ic++) for(int kh=0;kh<KH;kh++) for(int kw=0;kw<KW;kw++){
            int ih=oh*SH-PH+kh, iw=ow*SW-PW+kw;
            if(ih<0||ih>=IH||iw<0||iw>=IW) continue;
            int k=(ic*KH+kh)*KW+kw;
            acc += (int32_t)in[(ic*IH+ih)*IW+iw]*(int32_t)w[k*OC+oc];
        }
        int32_t s=q31(acc,mult,shift); if(s<amin)s=amin; if(s>amax)s=amax;
        out[(oc*OH+oh)*OW+ow]=(int8_t)s;
    }
}
#include <time.h>
static uint64_t rdcyc(void){struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts); return (uint64_t)ts.tv_sec*1000000000ull+ts.tv_nsec;}
static uint32_t rng=2463534242u;
static int8_t r8(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return (int8_t)((rng&0xff)-128);}

typedef struct{const char*net;const char*name;int IC,IH,IW,OC,KH,KW,SH,SW,PH,PW,OH,OW,M,mult,shift,amin,amax;} Shape;
#include "shapes.inc"
#define NREP 5

int main(void){
    printf("net,name,IC,IH,IW,OC,KH,KW,M,rvv_cyc,ime_cyc,speedup,verify\n");
    for(int t=0;t<(int)(sizeof(SHAPES)/sizeof(SHAPES[0]));t++){
        Shape S=SHAPES[t];
        int K=S.IC*S.KH*S.KW, insz=S.IC*S.IH*S.IW, wsz=K*S.OC, osz=S.OC*S.OH*S.OW;
        int8_t*in=malloc(insz),*w=malloc(wsz),*o_r=malloc(osz),*o_i=malloc(osz),*o_ref=malloc(osz);
        int32_t*bias=malloc(S.OC*sizeof(int32_t));
        for(int i=0;i<insz;i++)in[i]=r8(); for(int i=0;i<wsz;i++)w[i]=r8();
        for(int i=0;i<S.OC;i++)bias[i]=(int32_t)r8()*137;
        ref_conv(in,w,bias,o_ref,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,S.mult,S.shift,S.amin,S.amax);
        rvv_conv(in,w,bias,o_r,1,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,0,0,0,S.mult,S.shift,S.amin,S.amax);
        ime_conv(in,w,bias,o_i,1,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,0,0,0,S.mult,S.shift,S.amin,S.amax);
        int err_r=0,err_i=0;
        for(int i=0;i<osz;i++){int e;e=abs(o_r[i]-o_ref[i]);if(e>err_r)err_r=e;e=abs(o_i[i]-o_ref[i]);if(e>err_i)err_i=e;}
        uint64_t cr=~0ull,ci=~0ull;
        for(int r=0;r<NREP;r++){uint64_t a=rdcyc();rvv_conv(in,w,bias,o_r,1,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,0,0,0,S.mult,S.shift,S.amin,S.amax);uint64_t b=rdcyc()-a;if(b<cr)cr=b;}
        for(int r=0;r<NREP;r++){uint64_t a=rdcyc();ime_conv(in,w,bias,o_i,1,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,0,0,0,S.mult,S.shift,S.amin,S.amax);uint64_t b=rdcyc()-a;if(b<ci)ci=b;}
        printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%.3f,%s\n",S.net,S.name,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.M,
            (unsigned long long)cr,(unsigned long long)ci,(double)cr/(double)(ci?ci:1),
            (err_r==0&&err_i==0)?"OK":(err_i?"IME_MISMATCH":"RVV_MISMATCH"));
        free(in);free(w);free(o_r);free(o_i);free(o_ref);free(bias);
    }
    return 0;
}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    shapes = load_shapes()
    print(f"[shapes] {len(shapes)} unique symmetric conv shapes")
    inc = "static Shape SHAPES[] = {\n"
    for s in shapes:
        inc += ('  {"%s","%s",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d},\n' % (
            s["net"], s["name"], s["IC"], s["IH"], s["IW"], s["OC"], s["KH"], s["KW"],
            s["SH"], s["SW"], s["PH"], s["PW"], s["OH"], s["OW"], s["M"],
            s["out_mult"], s["out_shift"], s["amin"], s["amax"]))
    inc += "};\n"
    bd = os.path.join(OUT, "build")
    os.makedirs(bd, exist_ok=True)
    open(os.path.join(bd, "shapes.inc"), "w").write(inc)
    open(os.path.join(bd, "harness.c"), "w").write(HARNESS)

    cc = CROSS + "gcc"
    objs = []
    for src, rename, extra in [
        (os.path.join(REPO, "kernels/ime/ime_conv2d_s8_ime_vmadot_4x4x8.c"), "ime_conv", []),
        (os.path.join(REPO, "kernels/rvv/rvv_conv2d_s8_rvv_oc_blocked.c"), "rvv_conv",
         ["-DMODELBLASTER_RVV_IHWOC_WEIGHTS"]),
    ]:
        obj = os.path.join(bd, rename + ".o")
        cmd = [cc, *MARCH, *extra, f"-Dkernel_conv2d_s8={rename}", "-c", src, "-o", obj]
        print("[cc]", " ".join(cmd[-4:]))
        subprocess.run(cmd, check=True)
        objs.append(obj)
    hobj = os.path.join(bd, "harness.o")
    subprocess.run([cc, *MARCH, f"-I{bd}", "-c", os.path.join(bd, "harness.c"), "-o", hobj], check=True)
    binp = os.path.join(bd, "ime_conv_bench")
    subprocess.run([cc, *MARCH, "-static", hobj, *objs, "-o", binp], check=True)
    print("[link] ->", binp)

    subprocess.run(["ssh", HOST, f"mkdir -p {REMOTE}"], check=True)
    subprocess.run(["scp", "-q", binp, f"{HOST}:{REMOTE}/bench"], check=True)
    # cluster 0 only (smt.vmadot legal on harts 0-3)
    r = subprocess.run(["ssh", HOST, f"taskset -c 0 {REMOTE}/bench"],
                       capture_output=True, text=True)
    print("[run] rc", r.returncode)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr); sys.exit(1)
    csv = r.stdout
    open(os.path.join(OUT, "ime_vs_rvv_conv.csv"), "w").write(csv)
    print(csv)

    # summarize + plot
    rows = [l.split(",") for l in csv.strip().splitlines()[1:] if l.strip()]
    bad = [r for r in rows if r[-1] != "OK"]
    wins = [r for r in rows if float(r[11]) > 1.0 and r[-1] == "OK"]
    summ = f"verify: {len(rows)-len(bad)}/{len(rows)} bit-exact (max_abs_err=0)"
    if bad:
        summ += f"  !! {len(bad)} MISMATCH: " + ", ".join(f"{r[1]}({r[-1]})" for r in bad[:6])
    # net cycle reduction: picker takes IME where it wins, RVV else
    for net in ("dronet", "yolov8_nano"):
        nr = [r for r in rows if r[0] == net and r[-1] == "OK"]
        if not nr:
            continue
        all_rvv = sum(int(r[9]) for r in nr)
        best = sum(min(int(r[9]), int(r[10])) for r in nr)
        summ += (f"\n{net}: {len(nr)} shapes, IME wins {sum(1 for r in nr if int(r[10])<int(r[9]))}; "
                 f"picker(best-of) {best} vs all-RVV {all_rvv} = {100*(1-best/all_rvv):.1f}% conv-cycle reduction")
    open(os.path.join(OUT, "ime_vs_rvv_conv.txt"), "w").write(summ + "\n")
    print("\n" + summ)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ok = [r for r in rows if r[-1] == "OK"]
        ok.sort(key=lambda r: int(r[8]))  # by M
        M = [int(r[8]) for r in ok]; sp = [float(r[11]) for r in ok]
        col = ["#228833" if s > 1 else "#cc3311" for s in sp]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(range(len(ok)), sp, color=col)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(ok)))
        ax.set_xticklabels([f"{r[1]}\nM={r[8]}" for r in ok], rotation=90, fontsize=5)
        ax.set_ylabel("IME speedup vs RVV (>1 = IME wins)")
        ax.set_title("conv2d-on-IME vs RVV per layer (green=IME wins) — bit-exact", weight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "ime_vs_rvv_conv.png"), dpi=140)
        fig.savefig(os.path.join(OUT, "ime_vs_rvv_conv.pdf"))
        print("wrote", os.path.join(OUT, "ime_vs_rvv_conv.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
