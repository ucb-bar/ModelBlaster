/* K3 A100 layout-op micro-costs for the IME conv kernels. Usage: k3_ime_layout_probe <cpu>
 * Times segment loads/stores, strided e64, gathers, scalar copies, ... Each result is ns per
 * single operation (median of REPS runs of an inner loop of ITERS ops). */
#define _GNU_SOURCE
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_placement.h"
#define REPS 31
#define TICK_NS (1000.0 / 24.0)
#define SINK(v) __asm__ volatile("" ::"vr"(v) : "memory")
static inline uint64_t rdt(void) { uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); return t; }
static int cmpu(const void *a, const void *b) { uint64_t x = *(uint64_t *)a, y = *(uint64_t *)b; return x < y ? -1 : x > y; }
static double med_ns(uint64_t *t) { qsort(t, REPS, sizeof(*t), cmpu); return t[REPS / 2] * TICK_NS; }
static signed char src[16384] __attribute__((aligned(64)));
static signed char dst[16384] __attribute__((aligned(64)));
static volatile long gsink;
#define N 64
#define BENCH(NAME, ...) \
	static double NAME(void) { uint64_t t[REPS]; \
		for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt(); for (int i = 0; i < N; i++) { __VA_ARGS__; } uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; } \
		return med_ns(t) / N; }
#define P(NAME) printf("%-34s %9.1f ns\n", #NAME, NAME())

BENCH(vle8_128, { vint8m1_t v = __riscv_vle8_v_i8m1(src + (i & 7) * 128, 128); SINK(v); })
BENCH(vle8_128_unal, { vint8m1_t v = __riscv_vle8_v_i8m1(src + (i & 7) * 128 + 16, 128); SINK(v); })
BENCH(vse8_128, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 128); __riscv_vse8_v_i8m1(dst + (i & 7) * 128, v, 128); })
BENCH(vle8_16, { vint8m1_t v = __riscv_vle8_v_i8m1(src + (i & 7) * 128, 16); SINK(v); })
BENCH(vsseg8e8_vl64, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 64); __riscv_vsseg8e8_v_i8m1x8(dst, __riscv_vcreate_v_i8m1x8(v,v,v,v,v,v,v,v), 64); })
BENCH(vsseg8e8_vl8, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 8); __riscv_vsseg8e8_v_i8m1x8(dst, __riscv_vcreate_v_i8m1x8(v,v,v,v,v,v,v,v), 8); })
BENCH(vlseg8e8_vl64, { vint8m1x8_t u = __riscv_vlseg8e8_v_i8m1x8(src, 64); __riscv_vse8_v_i8m1(dst, __riscv_vget_v_i8m1x8_i8m1(u, 3), 64); })
BENCH(vlseg8e8_vl8, { vint8m1x8_t u = __riscv_vlseg8e8_v_i8m1x8(src, 8); __riscv_vse8_v_i8m1(dst, __riscv_vget_v_i8m1x8_i8m1(u, 3), 8); })
BENCH(vlseg2e8_vl64, { vint8m1x2_t u = __riscv_vlseg2e8_v_i8m1x2(src, 64); __riscv_vse8_v_i8m1(dst, __riscv_vget_v_i8m1x2_i8m1(u, 1), 64); })
BENCH(vsseg2e64_m4_vl16, { vint64m4_t x = __riscv_vle64_v_i64m4((int64_t*)src, 16); __riscv_vsseg2e64_v_i64m4x2((int64_t*)dst, __riscv_vcreate_v_i64m4x2(x,x), 16); })
BENCH(vsseg2e64_m1_vl8, { vint64m1_t x = __riscv_vle64_v_i64m1((int64_t*)src, 8); __riscv_vsseg2e64_v_i64m1x2((int64_t*)dst, __riscv_vcreate_v_i64m1x2(x,x), 8); })
BENCH(vlseg4e64_m2_vl16, { vint64m2x4_t z = __riscv_vlseg4e64_v_i64m2x4((int64_t*)src, 16); __riscv_vse64_v_i64m2((int64_t*)dst, __riscv_vget_v_i64m2x4_i64m2(z, 2), 16); })
BENCH(vsse64_vl8, { vint64m1_t x = __riscv_vle64_v_i64m1((int64_t*)src, 8); __riscv_vsse64_v_i64m1((int64_t*)dst, 160, x, 8); })
BENCH(vlse64_vl8, { vint64m1_t x = __riscv_vlse64_v_i64m1((int64_t*)src, 32, 8); SINK(x); })
BENCH(vlse64_vl8_s16, { vint64m1_t x = __riscv_vlse64_v_i64m1((int64_t*)src, 16, 8); SINK(x); })
BENCH(vsse8_vl8, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 8); __riscv_vsse8_v_i8m1(dst, 160, v, 8); })
BENCH(scalar_ld_sd_x8, { long *s = (long*)(src + (i&7)*64), *d = (long*)(dst + (i&7)*64); long a0=s[0],a1=s[1],a2=s[2],a3=s[3],a4=s[4],a5=s[5],a6=s[6],a7=s[7]; d[0]=a0;d[1]=a1;d[2]=a2;d[3]=a3;d[4]=a4;d[5]=a5;d[6]=a6;d[7]=a7; })
BENCH(scalar_lb_sb_x64, { volatile signed char *s = src + (i&7)*64, *d = dst + (i&7)*64; for (int k = 0; k < 64; k++) d[k] = s[k]; })
BENCH(vrgather_e8m1_128, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 128); vuint8m1_t ix = __riscv_vid_v_u8m1(128); ix = __riscv_vxor_vx_u8m1(ix, (uint8_t)i, 128); vint8m1_t g = __riscv_vrgather_vv_i8m1(v, ix, 128); SINK(g); })
BENCH(vrgather_e8m2_64, { vuint8m2_t t = __riscv_vle8_v_u8m2((uint8_t*)src, 256); vuint8m2_t ix = __riscv_vid_v_u8m2(64); ix = __riscv_vxor_vx_u8m2(ix, (uint8_t)i, 64); vuint8m2_t g = __riscv_vrgather_vv_u8m2(t, ix, 64); SINK(g); })
BENCH(vluxei16_64, { vuint16m1_t ix = __riscv_vid_v_u16m1(64); ix = __riscv_vsll_vx_u16m1(ix, 2, 64); vint8mf2_t g = __riscv_vluxei16_v_i8mf2(src, ix, 64); SINK(g); })
BENCH(vwmul_e8, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 128); vint16m2_t w = __riscv_vwadd_vx_i16m2(v, 3, 128); SINK(w); })
BENCH(vfdiv_f32m2_64, { vfloat32m2_t v = __riscv_vle32_v_f32m2((float*)src, 64); vfloat32m2_t w = __riscv_vfdiv_vf_f32m2(v, 1.7f + i, 64); SINK(w); })
BENCH(vfmul_f32m2_64, { vfloat32m2_t v = __riscv_vle32_v_f32m2((float*)src, 64); vfloat32m2_t w = __riscv_vfmul_vf_f32m2(v, 1.7f + i, 64); SINK(w); })
BENCH(vsmul_vnclip_m2_64, { vint32m2_t v = __riscv_vle32_v_i32m2((int*)src, 64); vint32m2_t s = __riscv_vsmul_vx_i32m2(v, 1207959552, __RISCV_VXRM_RNU, 64); vint16m1_t o = __riscv_vnclip_wx_i16m1(s, 7, __RISCV_VXRM_RNU, 64); SINK(o); })
BENCH(vcvt_rmm_m2_64, { vfloat32m2_t v = __riscv_vle32_v_f32m2((float*)src, 64); vint32m2_t w = __riscv_vfcvt_x_f_v_i32m2_rm(v, __RISCV_FRM_RMM, 64); SINK(w); })
BENCH(vslideup_e8m1, { vint8m1_t v = __riscv_vle8_v_i8m1(src, 128); vint8m1_t w = __riscv_vslideup_vx_i8m1(v, v, 8, 128); SINK(w); })
/* vnsrl-based even/odd split of e64 -> via e32 narrow: 128 e32 -> 64 e32 */
BENCH(vnsrl_e64_e32_64, { vuint64m2_t v = __riscv_vle64_v_u64m2((uint64_t*)src, 32); vuint32m1_t w = __riscv_vnsrl_wx_u32m1(v, 0, 32); SINK(w); })

BENCH(unal_sd_x8, { long *s = (long*)(src + (i&7)*64); char *d = dst + (i&7)*160 + 3; long a0=s[0],a1=s[1],a2=s[2],a3=s[3],a4=s[4],a5=s[5],a6=s[6],a7=s[7]; \
	__builtin_memcpy(d,&a0,8); __builtin_memcpy(d+20,&a1,8); __builtin_memcpy(d+40,&a2,8); __builtin_memcpy(d+60,&a3,8); __builtin_memcpy(d+80,&a4,8); __builtin_memcpy(d+100,&a5,8); __builtin_memcpy(d+120,&a6,8); __builtin_memcpy(d+140,&a7,8); })
BENCH(vse_then_scalar_rows, { vint8m1_t v = __riscv_vle8_v_i8m1(src+(i&7)*128, 64); signed char buf[64] __attribute__((aligned(64))); __riscv_vse8_v_i8m1(buf, v, 64); \
	long *s = (long*)buf; char *d = dst + (i&7)*160 + 3; long a0=s[0],a1=s[1],a2=s[2],a3=s[3],a4=s[4],a5=s[5],a6=s[6],a7=s[7]; \
	__builtin_memcpy(d,&a0,8); __builtin_memcpy(d+20,&a1,8); __builtin_memcpy(d+40,&a2,8); __builtin_memcpy(d+60,&a3,8); __builtin_memcpy(d+80,&a4,8); __builtin_memcpy(d+100,&a5,8); __builtin_memcpy(d+120,&a6,8); __builtin_memcpy(d+140,&a7,8); })
BENCH(scalar_lut_x64, { volatile unsigned char *s = (unsigned char*)src + (i&7)*64; unsigned char *tb = (unsigned char*)src + 4096; volatile unsigned char *d = (unsigned char*)dst + (i&7)*64; for (int k = 0; k < 64; k++) d[k] = tb[s[k]]; })
BENCH(vlut_gather2_128, { vuint8m1_t x = __riscv_vle8_v_u8m1((uint8_t*)src, 128); vuint8m1_t t0 = __riscv_vle8_v_u8m1((uint8_t*)src+4096, 128); vuint8m1_t t1 = __riscv_vle8_v_u8m1((uint8_t*)src+4224, 128); \
	vuint8m1_t lo = __riscv_vrgather_vv_u8m1(t0, x, 128); vuint8m1_t hi = __riscv_vrgather_vv_u8m1(t1, __riscv_vsub_vx_u8m1(x, 128, 128), 128); __riscv_vse8_v_u8m1((uint8_t*)dst, __riscv_vor_vv_u8m1(lo, hi, 128), 128); })
BENCH(vmadot_2x4_step, { vint8m1_t a0 = __riscv_vle8_v_i8m1(src,128), a1 = __riscv_vle8_v_i8m1(src+128,128); vint8m1_t b0 = __riscv_vle8_v_i8m1(src+256,128), b1 = __riscv_vle8_v_i8m1(src+384,128), b2 = __riscv_vle8_v_i8m1(src+512,128), b3 = __riscv_vle8_v_i8m1(src+640,128); SINK(a0);SINK(a1);SINK(b0);SINK(b1);SINK(b2);SINK(b3); })

BENCH(unal_ld_asm_x8, { char *s = src + (i&7)*64 + 3; long a0,a1,a2,a3,a4,a5,a6,a7; \
	__asm__ volatile("ld %0,0(%8)\n ld %1,8(%8)\n ld %2,16(%8)\n ld %3,24(%8)\n ld %4,32(%8)\n ld %5,40(%8)\n ld %6,48(%8)\n ld %7,56(%8)" : "=&r"(a0),"=&r"(a1),"=&r"(a2),"=&r"(a3),"=&r"(a4),"=&r"(a5),"=&r"(a6),"=&r"(a7) : "r"(s) : "memory"); gsink = a0+a1+a2+a3+a4+a5+a6+a7; })
BENCH(al_ld_asm_x8, { char *s = src + (i&7)*64; long a0,a1,a2,a3,a4,a5,a6,a7; \
	__asm__ volatile("ld %0,0(%8)\n ld %1,8(%8)\n ld %2,16(%8)\n ld %3,24(%8)\n ld %4,32(%8)\n ld %5,40(%8)\n ld %6,48(%8)\n ld %7,56(%8)" : "=&r"(a0),"=&r"(a1),"=&r"(a2),"=&r"(a3),"=&r"(a4),"=&r"(a5),"=&r"(a6),"=&r"(a7) : "r"(s) : "memory"); gsink = a0+a1+a2+a3+a4+a5+a6+a7; })
BENCH(unal_sd_asm_x8, { char *d = dst + (i&7)*64 + 3; long a = i; \
	__asm__ volatile("sd %0,0(%1)\n sd %0,8(%1)\n sd %0,16(%1)\n sd %0,24(%1)\n sd %0,32(%1)\n sd %0,40(%1)\n sd %0,48(%1)\n sd %0,56(%1)" : : "r"(a), "r"(d) : "memory"); })
BENCH(al_sd_asm_x8, { char *d = dst + (i&7)*64; long a = i; \
	__asm__ volatile("sd %0,0(%1)\n sd %0,8(%1)\n sd %0,16(%1)\n sd %0,24(%1)\n sd %0,32(%1)\n sd %0,40(%1)\n sd %0,48(%1)\n sd %0,56(%1)" : : "r"(a), "r"(d) : "memory"); })
BENCH(alu_72_ops, { unsigned long a = i, b = i*3+1, m = 0x00FF00FF00FF00FFul; for (int k = 0; k < 12; k++) { unsigned long t = ((a >> 8) ^ b) & m; b ^= t; a ^= t << 8; __asm__ volatile("" : "+r"(a), "+r"(b)); } gsink = a + b; })

int main(int argc, char **argv)
{
	int cpu = argc > 1 ? atoi(argv[1]) : 8;
	if (mb_place_current_thread(cpu)) { fprintf(stderr, "placement failed\n"); return 2; }
	for (int i = 0; i < 16384; i++) src[i] = (signed char)(i * 7 + 3);
	printf("cpu=%d vlmax_e8m1=%zu\n", cpu, (size_t)__riscv_vsetvlmax_e8m1());
	P(vle8_128); P(vle8_128_unal); P(vse8_128); P(vle8_16);
	P(vsseg8e8_vl64); P(vsseg8e8_vl8); P(vlseg8e8_vl64); P(vlseg8e8_vl8); P(vlseg2e8_vl64);
	P(vsseg2e64_m4_vl16); P(vsseg2e64_m1_vl8); P(vlseg4e64_m2_vl16);
	P(vsse64_vl8); P(vlse64_vl8); P(vlse64_vl8_s16); P(vsse8_vl8);
	P(scalar_ld_sd_x8); P(scalar_lb_sb_x64);
	P(vrgather_e8m1_128); P(vrgather_e8m2_64); P(vluxei16_64);
	P(vwmul_e8); P(vfdiv_f32m2_64); P(vfmul_f32m2_64); P(vsmul_vnclip_m2_64); P(vcvt_rmm_m2_64);
	P(vslideup_e8m1); P(vnsrl_e64_e32_64);
	P(unal_sd_x8); P(vse_then_scalar_rows); P(scalar_lut_x64); P(vlut_gather2_128); P(vmadot_2x4_step);
	P(unal_ld_asm_x8); P(al_ld_asm_x8); P(unal_sd_asm_x8); P(al_sd_asm_x8); P(alu_72_ops);
	return 0;
}
