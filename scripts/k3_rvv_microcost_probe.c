/* K3 RVV micro-cost probe. Usage: k3_rvv_microcost_probe <cpu>
 * Build zvl256b for CPU 0, zvl1024b for CPU 8. VLEN-agnostic (vl from vsetvl).
 * Timer: rdtime @24 MHz; each measurement = median of REPS runs of an inner loop. */
#define _GNU_SOURCE
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_placement.h"

#define REPS 41
#define TICK_NS (1000.0 / 24.0)
#define SINK(v) __asm__ volatile("" ::"vr"(v) : "memory")
#define SINKA(v) __asm__ volatile("" : "+vr"(v)::"memory")

static inline uint64_t rdt(void) { uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); return t; }
static int cmpu(const void *a, const void *b) { uint64_t x = *(uint64_t *)a, y = *(uint64_t *)b; return x < y ? -1 : x > y; }
static double med_ns(uint64_t *t) { qsort(t, REPS, sizeof(*t), cmpu); return t[REPS / 2] * TICK_NS; }

static signed char src[8192] __attribute__((aligned(64)));
static signed char dst[8192] __attribute__((aligned(64)));
static volatile long gsink;

#define BENCH(NAME, ITERS, ...) \
	static double NAME(void) { uint64_t t[REPS]; \
		for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt(); __VA_ARGS__; uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; } \
		return med_ns(t) / (ITERS); }

/* (a) loads: 2048 elements per pass */
#define NEL 2048
BENCH(b_vle8_m1, 1, { size_t vl = __riscv_vsetvl_e8m1(NEL); for (int i = 0; i < NEL; i += vl) { vint8m1_t v = __riscv_vle8_v_i8m1(src + i, vl); SINK(v);} })
BENCH(b_vle8_m4, 1, { size_t vl = __riscv_vsetvl_e8m4(NEL); for (int i = 0; i < NEL; i += vl) { vint8m4_t v = __riscv_vle8_v_i8m4(src + i, vl); SINK(v);} })
BENCH(b_vlse8_m1, 1, { size_t vl = __riscv_vsetvl_e8m1(NEL); for (int i = 0; i < NEL; i += vl) { vint8m1_t v = __riscv_vlse8_v_i8m1(src + 2 * i, 2, vl); SINK(v);} })
BENCH(b_vlse8_m4, 1, { size_t vl = __riscv_vsetvl_e8m4(NEL); for (int i = 0; i < NEL; i += vl) { vint8m4_t v = __riscv_vlse8_v_i8m4(src + 2 * i, 2, vl); SINK(v);} })

/* (b) vredsum e32: chained (latency) and 4-independent (throughput), 256 ops */
#define NR 256
BENCH(b_red_m1_lat, NR, { size_t vl = __riscv_vsetvl_e32m1(1000); vint32m1_t v = __riscv_vle32_v_i32m1((int *)src, vl); vint32m1_t z = __riscv_vmv_s_x_i32m1(0, 1); for (int i = 0; i < NR; i++) { z = __riscv_vredsum_vs_i32m1_i32m1(v, z, vl); } SINK(z); })
BENCH(b_red_m4_lat, NR, { size_t vl = __riscv_vsetvl_e32m4(1000); vint32m4_t v = __riscv_vle32_v_i32m4((int *)src, vl); vint32m1_t z = __riscv_vmv_s_x_i32m1(0, 1); for (int i = 0; i < NR; i++) { z = __riscv_vredsum_vs_i32m4_i32m1(v, z, vl); } SINK(z); })
BENCH(b_red_m1_l16, NR, { size_t vl = __riscv_vsetvl_e32m1(16); vint32m1_t v = __riscv_vle32_v_i32m1((int *)src, vl); vint32m1_t z = __riscv_vmv_s_x_i32m1(0, 1); for (int i = 0; i < NR; i++) { z = __riscv_vredsum_vs_i32m1_i32m1(v, z, vl); } SINK(z); })
BENCH(b_red_m4_l16, NR, { size_t vl = __riscv_vsetvl_e32m4(16); vint32m4_t v = __riscv_vle32_v_i32m4((int *)src, vl); vint32m1_t z = __riscv_vmv_s_x_i32m1(0, 1); for (int i = 0; i < NR; i++) { z = __riscv_vredsum_vs_i32m4_i32m1(v, z, vl); } SINK(z); })
BENCH(b_red_m4_thr, NR, { size_t vl = __riscv_vsetvl_e32m4(1000); vint32m4_t v = __riscv_vle32_v_i32m4((int *)src, vl); vint32m1_t z0 = __riscv_vmv_s_x_i32m1(0, 1), z1 = z0, z2 = z0, z3 = z0; for (int i = 0; i < NR; i += 4) { z0 = __riscv_vredsum_vs_i32m4_i32m1(v, z0, vl); z1 = __riscv_vredsum_vs_i32m4_i32m1(v, z1, vl); z2 = __riscv_vredsum_vs_i32m4_i32m1(v, z2, vl); z3 = __riscv_vredsum_vs_i32m4_i32m1(v, z3, vl);} SINK(z0); SINK(z1); SINK(z2); SINK(z3); })
BENCH(b_red_m1_thr, NR, { size_t vl = __riscv_vsetvl_e32m1(1000); vint32m1_t v = __riscv_vle32_v_i32m1((int *)src, vl); vint32m1_t z0 = __riscv_vmv_s_x_i32m1(0, 1), z1 = z0, z2 = z0, z3 = z0; for (int i = 0; i < NR; i += 4) { z0 = __riscv_vredsum_vs_i32m1_i32m1(v, z0, vl); z1 = __riscv_vredsum_vs_i32m1_i32m1(v, z1, vl); z2 = __riscv_vredsum_vs_i32m1_i32m1(v, z2, vl); z3 = __riscv_vredsum_vs_i32m1_i32m1(v, z3, vl);} SINK(z0); SINK(z1); SINK(z2); SINK(z3); })

/* (c) 32-byte output scatter. dst stride 64 (NCHW plane stride stand-in). 64 groups per run */
#define NG 64
BENCH(b_scatter_scalar, NG, { size_t vl = __riscv_vsetvl_e8m1(32); vint8m1_t v = __riscv_vle8_v_i8m1(src, vl); \
	for (int g = 0; g < NG; g++) { signed char obuf[32] __attribute__((aligned(16))); __riscv_vse8_v_i8m1(obuf, v, vl); signed char *o = dst + g; for (int c = 0; c < 32; c++) o[c * 64 + 0] = obuf[c]; } })
BENCH(b_scatter_vsse, NG, { size_t vl = __riscv_vsetvl_e8m1(32); vint8m1_t v = __riscv_vle8_v_i8m1(src, vl); \
	for (int g = 0; g < NG; g++) { __riscv_vsse8_v_i8m1(dst + g, 64, v, vl); } })
BENCH(b_vse_reload, NG, { size_t vl = __riscv_vsetvl_e8m1(32); vint8m1_t v = __riscv_vle8_v_i8m1(src, vl); long s = 0; \
	for (int g = 0; g < NG; g++) { signed char obuf[32] __attribute__((aligned(16))); __riscv_vse8_v_i8m1(obuf, v, vl); for (int c = 0; c < 32; c++) s += obuf[c]; } gsink = s; })
BENCH(b_vse_only, NG, { size_t vl = __riscv_vsetvl_e8m1(32); vint8m1_t v = __riscv_vle8_v_i8m1(src, vl); \
	for (int g = 0; g < NG; g++) { __riscv_vse8_v_i8m1(dst + g * 32, v, vl); } })

/* (d) vsetvli churn, 512 vsetvli per run */
#define NV 256
BENCH(b_vset_const, 2 * NV, { for (int i = 0; i < NV; i++) __asm__ volatile("vsetvli t0, zero, e8, m1, ta, ma\n vsetvli t0, zero, e8, m1, ta, ma" ::: "t0"); })
BENCH(b_vset_alt, 2 * NV, { for (int i = 0; i < NV; i++) __asm__ volatile("vsetvli t0, zero, e8, m1, ta, ma\n vsetvli t0, zero, e32, m4, ta, ma" ::: "t0"); })
BENCH(b_vset_alt_ops, 2 * NV, { for (int i = 0; i < NV; i++) __asm__ volatile("vsetvli t0, zero, e8, m1, ta, ma\n vadd.vv v8, v8, v9\n vsetvli t0, zero, e32, m4, ta, ma\n vadd.vv v16, v16, v20" ::: "t0", "v8", "v16"); })
BENCH(b_vset_const_ops, 2 * NV, { for (int i = 0; i < NV; i++) __asm__ volatile("vsetvli t0, zero, e8, m1, ta, ma\n vadd.vv v8, v8, v9\n vsetvli t0, zero, e8, m1, ta, ma\n vadd.vv v16, v16, v20" ::: "t0", "v8", "v16"); })

/* (e) vwmacc throughput: 4 independent accumulators, 1024 instrs/run; MACs = vl per instr */
#define NW 1024
#define WM8(NAME, VLREQ) BENCH(NAME, NW, { size_t vl = __riscv_vsetvl_e8m1(VLREQ); vint8m1_t a = __riscv_vle8_v_i8m1(src, vl); \
	vint16m2_t c0 = __riscv_vmv_v_x_i16m2(0, vl), c1 = c0, c2 = c0, c3 = c0; \
	for (int i = 0; i < NW; i += 4) { c0 = __riscv_vwmacc_vx_i16m2(c0, 3, a, vl); c1 = __riscv_vwmacc_vx_i16m2(c1, 5, a, vl); c2 = __riscv_vwmacc_vx_i16m2(c2, 7, a, vl); c3 = __riscv_vwmacc_vx_i16m2(c3, 9, a, vl); SINKA(c0); SINKA(c1); SINKA(c2); SINKA(c3);} })
#define WM16(NAME, VLREQ) BENCH(NAME, NW, { size_t vl = __riscv_vsetvl_e16m1(VLREQ); vint16m1_t a = __riscv_vle16_v_i16m1((short *)src, vl); \
	vint32m2_t c0 = __riscv_vmv_v_x_i32m2(0, vl), c1 = c0, c2 = c0, c3 = c0; \
	for (int i = 0; i < NW; i += 4) { c0 = __riscv_vwmacc_vx_i32m2(c0, 3, a, vl); c1 = __riscv_vwmacc_vx_i32m2(c1, 5, a, vl); c2 = __riscv_vwmacc_vx_i32m2(c2, 7, a, vl); c3 = __riscv_vwmacc_vx_i32m2(c3, 9, a, vl); SINKA(c0); SINKA(c1); SINKA(c2); SINKA(c3);} })
WM8(b_wm8_max, 100000) WM8(b_wm8_32, 32) WM8(b_wm8_16, 16)
WM16(b_wm16_max, 100000) WM16(b_wm16_32, 32) WM16(b_wm16_16, 16)
/* chained single-accumulator (latency) */
BENCH(b_wm8_lat, NW, { size_t vl = __riscv_vsetvl_e8m1(100000); vint8m1_t a = __riscv_vle8_v_i8m1(src, vl); vint16m2_t c0 = __riscv_vmv_v_x_i16m2(0, vl); for (int i = 0; i < NW; i++) { c0 = __riscv_vwmacc_vx_i16m2(c0, 3, a, vl); SINKA(c0);} })

/* (f) scalar int8 MAC loop, 1024 MACs/run */
BENCH(b_scalar, 1024, { int s0 = 0, s1 = 0; for (int i = 0; i < 1024; i += 2) { s0 += src[i] * 3; s1 += src[i + 1] * 5; __asm__ volatile("" : "+r"(s0), "+r"(s1)); } gsink = s0 + s1; })

int main(int argc, char **argv)
{
	int cpu = argc > 1 ? atoi(argv[1]) : 0;
	if (mb_place_current_thread(cpu)) { fprintf(stderr, "placement failed\n"); return 2; }
	for (int i = 0; i < 8192; i++) src[i] = (signed char)(i * 7 + 3);
	unsigned long vlenb; __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
	size_t v8 = __riscv_vsetvlmax_e8m1();
	printf("cpu=%d vlenb=%lu vlmax_e8m1=%zu\n", cpu, vlenb, v8);
	double vm1 = (double)v8;
	double a1 = b_vle8_m1(), a4 = b_vle8_m4(), s1 = b_vlse8_m1(), s4 = b_vlse8_m4();
	printf("(a) 2048 elems/pass: vle8 e8m1 %.1f ns (%.2f elem/ns) | vle8 e8m4 %.1f ns (%.2f) | vlse8 s2 m1 %.1f ns (%.2f) | vlse8 s2 m4 %.1f ns (%.2f)\n",
	       a1, NEL / a1, a4, NEL / a4, s1, NEL / s1, s4, NEL / s4);
	printf("    per-instr ns: vle8m1 %.2f vle8m4 %.2f vlse8m1 %.2f vlse8m4 %.2f\n", a1 / (NEL / vm1), a4 / (NEL / (4 * vm1)), s1 / (NEL / vm1), s4 / (NEL / (4 * vm1)));
	printf("(b) vredsum e32 ns/op: m1 lat(vlmax) %.2f | m4 lat(vlmax) %.2f | m1 lat(vl16) %.2f | m4 lat(vl16) %.2f | m1 thr %.2f | m4 thr %.2f\n",
	       b_red_m1_lat(), b_red_m4_lat(), b_red_m1_l16(), b_red_m4_l16(), b_red_m1_thr(), b_red_m4_thr());
	printf("(c) per 32B group ns: vse8+scalar scatter(stride64) %.2f | vsse8 stride64 %.2f | vse8+scalar reload %.2f | vse8 only %.2f\n",
	       b_scatter_scalar(), b_scatter_vsse(), b_vse_reload(), b_vse_only());
	printf("(d) vsetvli ns/instr: const %.2f | alt e8m1<->e32m4 %.2f | alt+vadd %.2f | const+vadd %.2f\n",
	       b_vset_const(), b_vset_alt(), b_vset_alt_ops(), b_vset_const_ops());
	double w1 = b_wm8_max(), w2 = b_wm8_32(), w3 = b_wm8_16(), x1 = b_wm16_max(), x2 = b_wm16_32(), x3 = b_wm16_16();
	size_t v16 = __riscv_vsetvl_e16m1(100000), l32 = __riscv_vsetvl_e8m1(32), l16 = __riscv_vsetvl_e8m1(16), m32 = __riscv_vsetvl_e16m1(32), m16 = __riscv_vsetvl_e16m1(16);
	printf("(e) vwmacc ns/instr (MAC/ns): i8->i16 vlmax(vl=%zu) %.3f (%.1f) | vl=%zu %.3f (%.1f) | vl=%zu %.3f (%.1f) | chained vlmax %.3f\n",
	       v8, w1, v8 / w1, l32, w2, l32 / w2, l16, w3, l16 / w3, b_wm8_lat());
	printf("    i16->i32 vlmax(vl=%zu) %.3f (%.1f) | vl=%zu %.3f (%.1f) | vl=%zu %.3f (%.1f)\n", v16, x1, v16 / x1, m32, x2, m32 / x2, m16, x3, m16 / x3);
	printf("(f) scalar int8 MAC loop: %.3f ns/MAC (%.2f MAC/ns)\n", b_scalar(), 1.0 / b_scalar());
	return 0;
}
