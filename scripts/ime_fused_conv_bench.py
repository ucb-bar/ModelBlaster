#!/usr/bin/env python3
"""Board bench for the FUSED conv+BN+SiLU kernel: IME vs the DEPLOYED RVV kernel.

scripts/ime_conv_verify_bench.py compares the IME conv against the *standalone*
RVV conv, which is not what the build runs. The deployed op is
`conv2d_batchnorm2d_silu_s8` and its RVV implementation is
kernels/rvv/rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c.
This harness compiles that kernel and the IME one into a single binary over the
real fused-conv shapes (and real quant parameters) of the deployed graph, and on
the board it (a) requires the IME output to be BYTE-IDENTICAL to the RVV output
-- the deployment contract, `max_abs_err=0` -- and (b) times both back to back.

It exists so a kernel change can be measured per shape in ~30 s of board time
instead of a full model build + harness run.
"""
import json, os, subprocess, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROSS = os.environ.get("CROSS", "riscv64-unknown-linux-gnu-")   # toolchain prefix; on PATH unless set
HOST = os.environ.get("MODELBLASTER_K1_HOST", "k1")
REMOTE = os.environ.get("MODELBLASTER_K1_REMOTE_ROOT", "/root/mb_k1") + "/ime_fused_conv"
OUT = os.environ.get("MB_IME_FUSED_BENCH_OUT") or os.path.join(REPO, "artifacts", "ime_fused_conv")
IME_SRC = os.environ.get("MB_IME_FUSED_SRC",
    os.path.join(REPO, "kernels/ime/ime_conv2d_batchnorm2d_silu_s8_ime_vmadot_4x4x8.c"))
MARCH = ["-march=rv64gcv_zvl256b", "-mabi=lp64d", "-O3"]


def load_shapes():
    out = []
    for net in os.environ.get("MB_IME_FUSED_NETS", "yolov8_nano_64x96").split(","):
        g = os.path.join(REPO, "build", "k1_xpurt", net, "int8", "graph.json")
        ir = json.load(open(g))
        for op in ir["ops"]:
            if op["op"] != "conv2d_batchnorm2d_silu_s8":
                continue
            subs = {s["op"]: s for s in op["sub_ops"]}
            conv, bn, silu = subs["conv2d_s8"], subs["batchnorm2d_s8"], subs["silu_s8"]
            s, q = conv["shape"], conv["quant"]
            if q.get("input_offset", 0) or q.get("filter_offset", 0):
                continue
            out.append(dict(
                did=op["dispatch_id"], name=op["name"],
                IC=s["IC"], IH=s["IH"], IW=s["IW"], OC=s["OC"], KH=s["KH"], KW=s["KW"],
                SH=s["SH"], SW=s["SW"], PH=s["PH"], PW=s["PW"], OH=s["OH"], OW=s["OW"],
                mult=q["output_multiplier"], shift=q["output_shift"],
                amin=q["activation_min"], amax=q["activation_max"],
                bn_si=bn["quant"]["scale_in"], bn_so=bn["quant"]["scale_out"],
                bn_amin=bn["quant"]["activation_min"], bn_amax=bn["quant"]["activation_max"],
                si_si=silu["quant"]["scale_in"], si_so=silu["quant"]["scale_out"],
                si_amin=silu["quant"]["activation_min"], si_amax=silu["quant"]["activation_max"]))
    return out


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef void (*fused_fn)(const int8_t*,const int8_t*,const int32_t*,const float*,const float*,int8_t*,
    int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,
    float,float,int,int,float,float,int,int);
void ime_fused(const int8_t*,const int8_t*,const int32_t*,const float*,const float*,int8_t*,
    int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,
    float,float,int,int,float,float,int,int);
void rvv_fused(const int8_t*,const int8_t*,const int32_t*,const float*,const float*,int8_t*,
    int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,int,
    float,float,int,int,float,float,int,int);

static uint64_t ns(void){struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
    return (uint64_t)ts.tv_sec*1000000000ull+ts.tv_nsec;}
static uint32_t rng=2463534242u;
static uint32_t nr(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static int8_t r8(void){return (int8_t)((nr()&0xff)-128);}
static float rf(float lo,float hi){return lo+(hi-lo)*((float)(nr()&0xffff)/65535.0f);}

typedef struct{int did;const char*name;int IC,IH,IW,OC,KH,KW,SH,SW,PH,PW,OH,OW;
    int mult,shift,amin,amax; float bn_si,bn_so; int bn_amin,bn_amax;
    float si_si,si_so; int si_amin,si_amax;} Shape;
#include "shapes.inc"
#define NREP 5

static void run(fused_fn f,Shape S,const int8_t*in,const int8_t*w,const int32_t*b,
                const float*bs,const float*bb,int8_t*o){
    f(in,w,b,bs,bb,o,1,S.IC,S.IH,S.IW,S.OC,S.KH,S.KW,S.SH,S.SW,S.PH,S.PW,
      0,0,0,S.mult,S.shift,S.amin,S.amax,S.bn_si,S.bn_so,S.bn_amin,S.bn_amax,
      S.si_si,S.si_so,S.si_amin,S.si_amax);
}

int main(int argc,char**argv){
    int only=(argc>1)?atoi(argv[1]):-1;
    printf("did,name,IC,IH,IW,OC,KH,KW,M,rvv_ns,ime_ns,speedup,verify\n");
    for(int t=0;t<(int)(sizeof(SHAPES)/sizeof(SHAPES[0]));t++){
        Shape S=SHAPES[t];
        if(only>=0&&S.did!=only) continue;
        int K=S.IC*S.KH*S.KW, insz=S.IC*S.IH*S.IW, wsz=K*S.OC, osz=S.OC*S.OH*S.OW;
        int8_t*in=malloc(insz),*w=malloc(wsz),*o_r=calloc(osz,1),*o_i=calloc(osz,1);
        int32_t*bias=malloc(S.OC*sizeof(int32_t));
        float*bs=malloc(S.OC*sizeof(float)),*bb=malloc(S.OC*sizeof(float));
        for(int i=0;i<insz;i++)in[i]=r8();
        for(int i=0;i<wsz;i++)w[i]=r8();
        for(int i=0;i<S.OC;i++){bias[i]=(int32_t)r8()*137; bs[i]=rf(0.4f,2.0f); bb[i]=rf(-1.5f,1.5f);}
        run(rvv_fused,S,in,w,bias,bs,bb,o_r);
        run(ime_fused,S,in,w,bias,bs,bb,o_i);
        int err=0; for(int i=0;i<osz;i++){int e=abs(o_i[i]-o_r[i]); if(e>err)err=e;}
        uint64_t cr=~0ull,ci=~0ull;
        for(int r=0;r<NREP;r++){uint64_t a=ns();run(rvv_fused,S,in,w,bias,bs,bb,o_r);uint64_t d=ns()-a;if(d<cr)cr=d;}
        for(int r=0;r<NREP;r++){uint64_t a=ns();run(ime_fused,S,in,w,bias,bs,bb,o_i);uint64_t d=ns()-a;if(d<ci)ci=d;}
        printf("%d,%s,%d,%d,%d,%d,%d,%d,%d,%llu,%llu,%.3f,%s\n",S.did,S.name,S.IC,S.IH,S.IW,S.OC,
            S.KH,S.KW,S.OH*S.OW,(unsigned long long)cr,(unsigned long long)ci,
            (double)cr/(double)(ci?ci:1), err==0?"OK":"IME_MISMATCH");
        fflush(stdout);
        free(in);free(w);free(o_r);free(o_i);free(bias);free(bs);free(bb);
    }
    return 0;
}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    shapes = load_shapes()
    print(f"[shapes] {len(shapes)} fused conv dispatches")
    inc = "static Shape SHAPES[] = {\n"
    for s in shapes:
        inc += ('  {%d,"%s",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.17gf,%.17gf,%d,%d,%.17gf,%.17gf,%d,%d},\n' % (
            s["did"], s["name"], s["IC"], s["IH"], s["IW"], s["OC"], s["KH"], s["KW"],
            s["SH"], s["SW"], s["PH"], s["PW"], s["OH"], s["OW"],
            s["mult"], s["shift"], s["amin"], s["amax"],
            s["bn_si"], s["bn_so"], s["bn_amin"], s["bn_amax"],
            s["si_si"], s["si_so"], s["si_amin"], s["si_amax"]))
    inc += "};\n"
    bd = os.path.join(OUT, "build")
    os.makedirs(bd, exist_ok=True)
    open(os.path.join(bd, "shapes.inc"), "w").write(inc)
    open(os.path.join(bd, "harness.c"), "w").write(HARNESS)

    cc = CROSS + "gcc"
    objs = []
    for src, rename in [
        (IME_SRC, "ime_fused"),
        (os.path.join(REPO, "kernels/rvv/rvv_conv2d_batchnorm2d_silu_s8_rvv_oc_blocked_bn_silu_epilogue.c"),
         "rvv_fused"),
    ]:
        obj = os.path.join(bd, rename + ".o")
        subprocess.run([cc, *MARCH, f"-Dkernel_conv2d_batchnorm2d_silu_s8={rename}",
                        "-c", src, "-o", obj], check=True)
        objs.append(obj)
    hobj = os.path.join(bd, "harness.o")
    subprocess.run([cc, *MARCH, f"-I{bd}", "-c", os.path.join(bd, "harness.c"), "-o", hobj], check=True)
    binp = os.path.join(bd, "ime_fused_bench")
    subprocess.run([cc, *MARCH, "-static", hobj, *objs, "-lm", "-o", binp], check=True)
    print("[link] ->", binp)

    lock = os.path.join(REPO, "..", "results", "codesign_feedback", "board.lock")
    subprocess.run(["ssh", HOST, f"mkdir -p {REMOTE}"], check=True)
    subprocess.run(["scp", "-q", binp, f"{HOST}:{REMOTE}/bench"], check=True)
    args = sys.argv[1] if len(sys.argv) > 1 else ""
    r = subprocess.run(["ssh", HOST, f"taskset -c 0 {REMOTE}/bench {args}"],
                       capture_output=True, text=True)
    print("[run] rc", r.returncode)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr, file=sys.stderr); sys.exit(1)
    csv = r.stdout
    open(os.path.join(OUT, "ime_vs_rvv_fused_conv.csv"), "w").write(csv)
    print(csv)
    rows = [l.split(",") for l in csv.strip().splitlines()[1:] if l.strip()]
    bad = [r for r in rows if r[-1] != "OK"]
    wins = [r for r in rows if float(r[11]) > 1.0 and r[-1] == "OK"]
    tot_r = sum(int(r[9]) for r in rows)
    tot_i = sum(int(r[10]) for r in rows)
    best = sum(min(int(r[9]), int(r[10])) for r in rows if r[-1] == "OK")
    print(f"\nverify: {len(rows)-len(bad)}/{len(rows)} byte-identical to RVV"
          + ("  !! MISMATCH: " + ", ".join(r[1] for r in bad[:8]) if bad else ""))
    print(f"IME wins {len(wins)}/{len(rows)};  all-RVV {tot_r/1e6:.2f} ms  "
          f"all-IME {tot_i/1e6:.2f} ms  best-of {best/1e6:.2f} ms")


if __name__ == "__main__":
    main()
