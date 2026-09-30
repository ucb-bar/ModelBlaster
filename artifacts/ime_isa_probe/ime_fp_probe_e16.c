#define _GNU_SOURCE
/* Fair test of the fp16 matrix family: the spec selects the MAC unit from vl*SEW and raises
 * illegal-instruction for an unsupported CONFIG as well as for an absent instruction, so a
 * probe of vfmadot under e8 cannot distinguish the two. Here each fp candidate runs under an
 * e16 config, with an e16 vmadot as the control for that same config. */
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <sched.h>
static sigjmp_buf jb;
static void ill(int s){ (void)s; siglongjmp(jb,1); }
static int8_t A[128] __attribute__((aligned(64)));
static int8_t B[128] __attribute__((aligned(64)));
#define PROBE(name, body) do {                                             \
    struct sigaction sa,old; memset(&sa,0,sizeof sa);                      \
    sa.sa_handler=ill; sigaction(SIGILL,&sa,&old);                         \
    const int8_t *pa=A,*pb=B; size_t n16=16,n32=32; (void)n32;             \
    if (sigsetjmp(jb,1)==0){ body; printf("  %-34s EXECUTES\n",name);}     \
    else                    printf("  %-34s SIGILL\n",name);               \
    sigaction(SIGILL,&old,NULL);                                           \
} while(0)
#define S16 "vsetvli t0, %[n16], e16, m1, ta, ma\n\t" "vmv.v.i v8,0\n\t" "vmv.v.i v9,0\n\t" \
            "vle16.v v0, (%[pa])\n\t" "vle16.v v4, (%[pb])\n\t"
#define O : [pa]"+r"(pa),[pb]"+r"(pb) : [n16]"r"(n16) : "t0","v0","v1","v4","v5","v8","v9","memory"
int main(int argc,char**argv){
    int hart=argc>1?atoi(argv[1]):0; cpu_set_t m; CPU_ZERO(&m); CPU_SET(hart,&m);
    if(sched_setaffinity(0,sizeof m,&m)){perror("aff");return 2;}
    for(int i=0;i<128;i++){A[i]=(int8_t)(i%7-3);B[i]=(int8_t)(i%5-2);}
    printf("hart %d, SEW=16 configs\n",hart);
    PROBE("vmadot ss  e16 (f7=0x71,f3=3)", __asm__ volatile(S16 ".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t" O));
    for (int f7=0x74; f7<=0x75; f7++) {
      if(f7==0x74){
        PROBE("vfmadot e16 (f7=0x74,f3=0)", __asm__ volatile(S16 ".insn r 0x2b, 0, 0x74, x8, x0, x4\n\t" O));
        PROBE("vfmadot e16 (f7=0x74,f3=1)", __asm__ volatile(S16 ".insn r 0x2b, 1, 0x74, x8, x0, x4\n\t" O));
      } else {
        PROBE("vfmadot e16 (f7=0x75,f3=0)", __asm__ volatile(S16 ".insn r 0x2b, 0, 0x75, x8, x0, x4\n\t" O));
        PROBE("vfmadot e16 (f7=0x75,f3=1)", __asm__ volatile(S16 ".insn r 0x2b, 1, 0x75, x8, x0, x4\n\t" O));
      }
    }
    PROBE("vfmadot e16 (f7=0x76,f3=0)", __asm__ volatile(S16 ".insn r 0x2b, 0, 0x76, x8, x0, x4\n\t" O));
    PROBE("vfmadot e16 (f7=0x77,f3=0)", __asm__ volatile(S16 ".insn r 0x2b, 0, 0x77, x8, x0, x4\n\t" O));
    return 0;
}
