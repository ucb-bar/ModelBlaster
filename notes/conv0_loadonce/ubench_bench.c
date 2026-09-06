/*
 * Load-once conv0 validation microbench (riskybird-35).
 *
 * Exercises the REAL DroNet grayscale conv0 + fused maxpool1:
 *   conv : N=1, IC=1, 112x112 -> 56x56, OC=32, 3x3 stride-2 pad-1
 *   pool : 3x3 stride-2 pad-0  (overlapping) -> 27x27x32
 *   requant: Q0.31 folded scale, mult=1118622017 shift=7, clamp [-128,127]
 * (params lifted verbatim from generated_gray_fusedpool/hetero_tiled model.c).
 *
 * Three ways, all NHWC in / NHWC pooled out, checked against a scalar golden:
 *   A) tiled_conv_auto with the HW pool tail  == the CURRENT fused conv0 path
 *   E) mb_conv2d_pool_loadonce_s8             == the load-once kernel under test
 * Prints per-path wall cycles + Gemmini EX/LD/ST/ALL3 perf-counter deltas, and
 *   errA  = A vs scalar golden
 *   errE  = E vs scalar golden
 *   errAE = E vs A   (the bit-exact gate: must be 0)
 */
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <zephyr/sys/printk.h>
#include <gemmini.h>
#include <gemmini_params.h>
#include <gemmini_counter.h>
#include "/home/cobble/Tools/mb_wt_conv0loadonce/kernels/gemmini_q31_rvv/conv2d_pool_loadonce.h"

/* ---- conv0 shape + requant (the real DroNet first layer) ---- */
enum {
    IC = 1, IH = 112, IW = 112, OC = 32,
    K = 3, S = 2, P = 1,
    POOL_K = 3, POOL_S = 2, POOL_P = 0,
    R_MULT = 1118622017, R_SHIFT = 7, R_MIN = -128, R_MAX = 127,
};

static inline uint64_t rdc(void){ uint64_t c; asm volatile("rdcycle %0":"=r"(c)); return c; }

static elem_t in_nhwc [IH*IW*IC]        __attribute__((aligned(64)));
static elem_t w_hwio  [K*K*IC*OC]       __attribute__((aligned(64)));
static acc_t  bias_buf[OC]              __attribute__((aligned(64)));
/* generous pooled-output buffers (OHp=OWp=27) */
static elem_t out_ref [64*64*OC]        __attribute__((aligned(64)));
static elem_t out_lo  [64*64*OC]        __attribute__((aligned(64)));
static elem_t out_gold[64*64*OC]        __attribute__((aligned(64)));
static elem_t conv_tmp[IH*IW*OC]        __attribute__((aligned(64)));  /* scalar conv (int8) */

static uint32_t rng;
static inline int8_t rnd8(int lo,int span){ rng=rng*1664525u+1013904223u; return (int8_t)(lo+(int)((rng>>17)%(unsigned)span)); }

static int max_abs_err(const elem_t*a,const elem_t*b,size_t n){
    int m=0; for(size_t i=0;i<n;i++){ int d=(int)a[i]-(int)b[i]; if(d<0)d=-d; if(d>m)m=d; } return m;
}

/* folded Q0.31 requant (ACC_SCALE), clamp int8 -- matches HW config_st scale */
static inline int8_t requant(int32_t acc, int32_t scale_q31){
    int64_t y = ((int64_t)acc*(int64_t)scale_q31 + (1LL<<30)) >> 31;
    if (y > R_MAX) y = R_MAX; if (y < R_MIN) y = R_MIN; return (int8_t)y;
}

extern void bench_print(const char*);
static char line[256];
static void emit(const char*fmt,...){ va_list ap; va_start(ap,fmt); vsnprintk(line,sizeof line,fmt,ap); va_end(ap); bench_print(line); }

static void gemmini_on(void){ asm volatile("csrs mstatus, %0"::"r"(0x18000):"memory"); gemmini_flush(0); }

void bench_all(void)
{
    const int OH = (IH+2*P-K)/S + 1;               /* 56 */
    const int OW = (IW+2*P-K)/S + 1;               /* 56 */
    const int OHp= (OH-POOL_K)/POOL_S + 1;         /* 27 */
    const int OWp= (OW-POOL_K)/POOL_S + 1;         /* 27 */
    const size_t pooled = (size_t)OHp*OWp*OC;
    const int32_t scale_q31 = (R_SHIFT==0) ? R_MULT
        : (int32_t)(((int64_t)R_MULT + (1LL<<(R_SHIFT-1))) >> R_SHIFT);

    bench_print("=== GEMMINI_UBENCH_BEGIN ===\n");
    emit("cfg,IC=%d,IH=%d,OC=%d,K=%d,S=%d,P=%d,poolK=%d,poolS=%d,OH=%d,OHp=%d,scaleq31=%d\n",
         IC,IH,OC,K,S,P,POOL_K,POOL_S,OH,OHp,scale_q31);

    /* deterministic data */
    rng = 0x1234567u;
    for (size_t i=0;i<sizeof in_nhwc;i++) in_nhwc[i]=rnd8(-4,8);
    for (size_t i=0;i<sizeof w_hwio;i++)  w_hwio[i]=rnd8(-4,8);
    for (int oc=0;oc<OC;oc++) bias_buf[oc]=rnd8(-16,32);

    /* ---- scalar golden: conv -> requant -> maxpool (NHWC) ---- */
    for (int oh=0;oh<OH;oh++) for(int ow=0;ow<OW;ow++) for(int oc=0;oc<OC;oc++){
        int32_t acc = bias_buf[oc];
        for (int kh=0;kh<K;kh++){ int ih=oh*S-P+kh;
          for (int kw=0;kw<K;kw++){ int iw=ow*S-P+kw;
            int oob=(ih<0||ih>=IH||iw<0||iw>=IW);
            for (int ic=0;ic<IC;ic++){
              int32_t v = oob?0:in_nhwc[((size_t)ih*IW+iw)*IC+ic];
              acc += v * (int32_t)w_hwio[(((size_t)kh*K+kw)*IC+ic)*OC+oc];
            } } }
        conv_tmp[((size_t)oh*OW+ow)*OC+oc] = requant(acc, scale_q31);
    }
    for (int ph=0;ph<OHp;ph++) for(int pw=0;pw<OWp;pw++) for(int oc=0;oc<OC;oc++){
        int8_t m = INT8_MIN;
        for (int kh=0;kh<POOL_K;kh++){ int oh=ph*POOL_S-POOL_P+kh; if(oh<0||oh>=OH)continue;
          for (int kw=0;kw<POOL_K;kw++){ int ow=pw*POOL_S-POOL_P+kw; if(ow<0||ow>=OW)continue;
            int8_t v = conv_tmp[((size_t)oh*OW+ow)*OC+oc]; if(v>m)m=v;
          } }
        out_gold[((size_t)ph*OWp+pw)*OC+oc] = m;
    }

    uint64_t t0,t1;
    uint32_t exA,ldA,stA,a3A, exE,ldE,stE,a3E;

    /* ---- Path A: tiled_conv_auto + HW pool tail (the current fused conv0) ---- */
    gemmini_on();
    counter_reset();
    counter_configure(0,MAIN_EX_CYCLES); counter_configure(1,MAIN_LD_CYCLES);
    counter_configure(2,MAIN_ST_CYCLES); counter_configure(3,MAIN_LD_ST_EX_CYCLES);
    memset(out_ref,0x55,pooled);
    asm volatile("fence":::"memory");
    uint32_t x=counter_read(0),l=counter_read(1),st=counter_read(2),a3=counter_read(3);
    t0=rdc();
    tiled_conv_auto(1, IH, IW, IC, OC, OH, OW, S, 1, 1, P, K,
        false,false,false,false,false,
        in_nhwc, w_hwio, bias_buf, out_ref,
        NO_ACTIVATION, (acc_scale_t)scale_q31,
        POOL_K, POOL_S, POOL_P, WS);
    gemmini_fence(); gemmini_flush(0);
    t1=rdc();
    exA=counter_read(0)-x; ldA=counter_read(1)-l; stA=counter_read(2)-st; a3A=counter_read(3)-a3;
    uint64_t wallA=t1-t0;

    /* ---- Path E: load-once kernel ---- */
    gemmini_on();
    counter_reset();
    counter_configure(0,MAIN_EX_CYCLES); counter_configure(1,MAIN_LD_CYCLES);
    counter_configure(2,MAIN_ST_CYCLES); counter_configure(3,MAIN_LD_ST_EX_CYCLES);
    memset(out_lo,0x55,pooled);
    asm volatile("fence":::"memory");
    x=counter_read(0);l=counter_read(1);st=counter_read(2);a3=counter_read(3);
    t0=rdc();
    int rc = mb_conv2d_pool_loadonce_s8(in_nhwc, w_hwio, bias_buf, out_lo,
        IC, IH, IW, OC, K, S, P, POOL_K, POOL_S,
        NO_ACTIVATION, (acc_scale_t)scale_q31);
    gemmini_fence(); gemmini_flush(0);
    t1=rdc();
    exE=counter_read(0)-x; ldE=counter_read(1)-l; stE=counter_read(2)-st; a3E=counter_read(3)-a3;
    uint64_t wallE=t1-t0;

    int errA = max_abs_err(out_ref, out_gold, pooled);
    int errE = max_abs_err(out_lo,  out_gold, pooled);
    int errAE= max_abs_err(out_lo,  out_ref,  pooled);

    emit("UBP,c0gA,wall=%llu,EX=%u,LD=%u,ST=%u,ALL3=%u\n",
         (unsigned long long)wallA,exA,ldA,stA,a3A);
    emit("UBP,c0gE,wall=%llu,EX=%u,LD=%u,ST=%u,ALL3=%u,rc=%d\n",
         (unsigned long long)wallE,exE,ldE,stE,a3E,rc);
    emit("UBV,errA=%d,errE=%d,errAE=%d,pooled=%u\n",
         errA,errE,errAE,(unsigned)pooled);
    bench_print("=== GEMMINI_UBENCH_END ===\n");
}
