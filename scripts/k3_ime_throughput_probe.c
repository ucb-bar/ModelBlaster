/* K3 IME vmadot throughput/latency probe. Usage: k3_ime_throughput_probe <cpu>
 * rdtime 24 MHz, median of REPS. 1024 vmadots per run (16 per loop iteration x 64). */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_placement.h"
#define REPS 41
#define TICK_NS (1000.0 / 24.0)
static inline uint64_t rdt(void) { uint64_t t; __asm__ volatile("rdtime %0" : "=r"(t)); return t; }
static int cmpu(const void *a, const void *b) { uint64_t x = *(uint64_t *)a, y = *(uint64_t *)b; return x < y ? -1 : x > y; }
static double med_ns(uint64_t *t) { qsort(t, REPS, sizeof(*t), cmpu); return t[REPS / 2] * TICK_NS; }
static int8_t buf[4096] __attribute__((aligned(64)));

#define VM(vd) ".insn r 0x2b, 3, 0x71, x" #vd ", x0, x4\n\t"
#define VM4(vd) ".insn r 0x2b, 3, 0x61, x" #vd ", x0, x4\n\t"
#define VMU(vd) ".insn r 0x2b, 0, 0x71, x" #vd ", x0, x4\n\t"
/* 16 vmadots per iteration spread over 1/2/4/8 chains (v8,v10,... pairs) */
#define C1 VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8) VM(8)
#define C2 VM(8) VM(10) VM(8) VM(10) VM(8) VM(10) VM(8) VM(10) VM(8) VM(10) VM(8) VM(10) VM(8) VM(10) VM(8) VM(10)
#define C4 VM(8) VM(10) VM(12) VM(14) VM(8) VM(10) VM(12) VM(14) VM(8) VM(10) VM(12) VM(14) VM(8) VM(10) VM(12) VM(14)
#define C8 VM(8) VM(10) VM(12) VM(14) VM(16) VM(18) VM(20) VM(22) VM(8) VM(10) VM(12) VM(14) VM(16) VM(18) VM(20) VM(22)
#define C8U VMU(8) VMU(10) VMU(12) VMU(14) VMU(16) VMU(18) VMU(20) VMU(22) VMU(8) VMU(10) VMU(12) VMU(14) VMU(16) VMU(18) VMU(20) VMU(22)
#define C8I4 VM4(8) VM4(10) VM4(12) VM4(14) VM4(16) VM4(18) VM4(20) VM4(22) VM4(8) VM4(10) VM4(12) VM4(14) VM4(16) VM4(18) VM4(20) VM4(22)
/* register-blocked microkernel: 2 A loads, 2 B loads, 4 vmadot (2x2 tiles), x4 per iteration = 16 vmadot, 8+8 loads */
#define LD(r, off) "vle8.v v" #r ", (%[p" "])\n\t"
#define MK ("addi t2, %[p], 0\n\t")

#define KERN(NAME, BODY) \
static double NAME(size_t vl) { uint64_t t[REPS]; \
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt(); \
		__asm__ volatile("vsetvli t0, %[vl], e8, m1, ta, ma\n\t vle8.v v0, (%[p])\n\t vle8.v v4, (%[p])\n\t" \
			"vsetvli t0, zero, e32, m1, ta, ma\n\t" \
			"vmv.v.i v8,0\n\t vmv.v.i v9,0\n\t vmv.v.i v10,0\n\t vmv.v.i v11,0\n\t vmv.v.i v12,0\n\t vmv.v.i v13,0\n\t vmv.v.i v14,0\n\t vmv.v.i v15,0\n\t" \
			"vmv.v.i v16,0\n\t vmv.v.i v17,0\n\t vmv.v.i v18,0\n\t vmv.v.i v19,0\n\t vmv.v.i v20,0\n\t vmv.v.i v21,0\n\t vmv.v.i v22,0\n\t vmv.v.i v23,0\n\t" \
			"vsetvli t0, %[vl], e8, m1, ta, ma\n\t li t1, 64\n\t 1:\n\t" BODY "addi t1, t1, -1\n\t bnez t1, 1b\n\t" \
			: : [vl] "r"(vl), [p] "r"(buf) : "t0", "t1", "t2", "memory", "v0", "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23"); \
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; } \
	return med_ns(t) / 1024.0; }
KERN(k_c1, C1) KERN(k_c2, C2) KERN(k_c4, C4) KERN(k_c8, C8) KERN(k_c8u, C8U) KERN(k_c8i4, C8I4)
/* streaming: per iter 4 vmadot (2x2 blocking) from 2 A + 2 B fresh loads (L1-resident), x4 => 16 vmadot, 16 vle8 */
#define MKB \
	"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v4, (%[p3])\n\t vle8.v v5, (%[p4])\n\t" \
	".insn r 0x2b, 3, 0x71, x8, x0, x4\n\t .insn r 0x2b, 3, 0x71, x10, x0, x5\n\t .insn r 0x2b, 3, 0x71, x12, x1, x4\n\t .insn r 0x2b, 3, 0x71, x14, x1, x5\n\t"
#define MKB4 MKB MKB MKB MKB
static double k_stream(size_t vl) { uint64_t t[REPS]; size_t vb = vl;
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt();
		__asm__ volatile("vsetvli t0, %[vl], e8, m1, ta, ma\n\t li t1, 64\n\t 1:\n\t" MKB4 "addi t1, t1, -1\n\t bnez t1, 1b\n\t"
			: : [vl] "r"(vl), [p] "r"(buf), [p2] "r"(buf + vb), [p3] "r"(buf + 2 * vb), [p4] "r"(buf + 3 * vb)
			: "t0", "t1", "memory", "v0", "v1", "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15");
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; }
	return med_ns(t) / 1024.0; }
/* loads only, same pattern, for subtracting */
static double k_loadonly(size_t vl) { uint64_t t[REPS]; size_t vb = vl;
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt();
		__asm__ volatile("vsetvli t0, %[vl], e8, m1, ta, ma\n\t li t1, 64\n\t 1:\n\t"
			"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v4, (%[p3])\n\t vle8.v v5, (%[p4])\n\t"
			"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v4, (%[p3])\n\t vle8.v v5, (%[p4])\n\t"
			"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v4, (%[p3])\n\t vle8.v v5, (%[p4])\n\t"
			"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v4, (%[p3])\n\t vle8.v v5, (%[p4])\n\t"
			"addi t1, t1, -1\n\t bnez t1, 1b\n\t"
			: : [vl] "r"(vl), [p] "r"(buf), [p2] "r"(buf + vb), [p3] "r"(buf + 2 * vb), [p4] "r"(buf + 3 * vb)
			: "t0", "t1", "memory", "v0", "v1", "v4", "v5");
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; }
	return med_ns(t) / 1024.0; }

/* 2x4 blocking: 2 A + 4 B loads, 8 vmadot (8 acc pairs v8..v22); x2 per iteration = 16 vmadot / 12 loads */
#define I(vd, a, b) ".insn r 0x2b, 3, 0x71, x" #vd ", x" #a ", x" #b "\n\t"
#define MK24 \
	"vle8.v v0, (%[p])\n\t vle8.v v1, (%[p2])\n\t vle8.v v2, (%[p3])\n\t vle8.v v3, (%[p4])\n\t vle8.v v5, (%[p5])\n\t vle8.v v6, (%[p6])\n\t" \
	I(8,0,2) I(10,0,3) I(12,0,5) I(14,0,6) I(16,1,2) I(18,1,3) I(20,1,5) I(22,1,6)
static double k_stream24(size_t vl) { uint64_t t[REPS]; size_t vb = vl;
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt();
		__asm__ volatile("vsetvli t0, %[vl], e8, m1, ta, ma\n\t li t1, 64\n\t 1:\n\t" MK24 MK24 "addi t1, t1, -1\n\t bnez t1, 1b\n\t"
			: : [vl] "r"(vl), [p] "r"(buf), [p2] "r"(buf + vb), [p3] "r"(buf + 2 * vb), [p4] "r"(buf + 3 * vb), [p5] "r"(buf + 4 * vb), [p6] "r"(buf + 5 * vb)
			: "t0", "t1", "memory", "v0", "v1", "v2", "v3", "v5", "v6", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; }
	return med_ns(t) / 1024.0; }

/* 2x4 blocking with whole-register group loads: vl2re8 (A: v0,v1) + vl4re8 (B: v4..v7) = 2 loads / 8 vmadot */
#define MK24W \
	"vl2re8.v v0, (%[p])\n\t vl4re8.v v4, (%[p3])\n\t" \
	I(8,0,4) I(10,0,5) I(12,0,6) I(14,0,7) I(16,1,4) I(18,1,5) I(20,1,6) I(22,1,7)
static double k_stream24w(size_t vl) { uint64_t t[REPS]; size_t vb = vl;
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt();
		__asm__ volatile("vsetvli t0, %[vl], e8, m1, ta, ma\n\t li t1, 64\n\t 1:\n\t" MK24W MK24W "addi t1, t1, -1\n\t bnez t1, 1b\n\t"
			: : [vl] "r"(vl), [p] "r"(buf), [p3] "r"(buf + 2 * vb)
			: "t0", "t1", "memory", "v0", "v1", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; }
	return med_ns(t) / 1024.0; }
static double k_ldw(size_t vl) { uint64_t t[REPS]; /* 16 x vl4re8 per iter => bytes/ns */
	for (int r = -3; r < REPS; r++) { uint64_t t0 = rdt();
		__asm__ volatile("li t1, 64\n\t 1:\n\t"
			"vl4re8.v v4, (%[p])\n\t vl4re8.v v8, (%[p])\n\t vl4re8.v v12, (%[p])\n\t vl4re8.v v16, (%[p])\n\t vl4re8.v v4, (%[p])\n\t vl4re8.v v8, (%[p])\n\t vl4re8.v v12, (%[p])\n\t vl4re8.v v16, (%[p])\n\t"
			"addi t1, t1, -1\n\t bnez t1, 1b\n\t" : : [p] "r"(buf) : "t1", "memory", "v4", "v8", "v12", "v16");
		uint64_t t1 = rdt(); if (r >= 0) t[r] = t1 - t0; }
	return med_ns(t) / 512.0; }

int main(int argc, char **argv)
{
	int cpu = argc > 1 ? atoi(argv[1]) : 0;
	if (mb_place_current_thread(cpu)) { fprintf(stderr, "placement failed\n"); return 2; }
	for (int i = 0; i < 4096; i++) buf[i] = (int8_t)(i * 7 + 3);
	unsigned long vb; __asm__ volatile("csrr %0, vlenb" : "=r"(vb));
	double macs = vb == 128 ? 8 * 16 * 8 : 4 * 8 * 4; /* int8 MACs per vmadot */
	printf("cpu=%d vlenb=%lu int8 MACs/vmadot=%.0f\n", cpu, vb, macs);
	double l1 = k_c1(vb), c2 = k_c2(vb), c4 = k_c4(vb), c8 = k_c8(vb), u8 = k_c8u(vb);
	printf("dependent chain (1 acc): %.3f ns/vmadot (latency)  [%.1f MAC/ns]\n", l1, macs / l1);
	printf("2 indep accs: %.3f ns [%.1f MAC/ns] | 4 accs: %.3f ns [%.1f] | 8 accs: %.3f ns [%.1f]\n", c2, macs / c2, c4, macs / c4, c8, macs / c8);
	printf("8 accs unsigned (f3=0): %.3f ns [%.1f MAC/ns]\n", u8, macs / u8);
	printf("8 accs, vl=32 (K3 ignores vl?): %.3f ns [%.1f MAC/ns]\n", k_c8(32), macs / k_c8(32));
	if (vb == 128) { double i4 = k_c8i4(vb); printf("int4 (f7=0x61, 8x32x8=2048 MAC) 8 accs: %.3f ns [%.1f MAC/ns]\n", i4, 2048.0 / i4); }
	double st = k_stream(vb), lo = k_loadonly(vb);
	printf("2x2-blocked microkernel, 4 loads per 4 vmadot (L1): %.3f ns/vmadot [%.1f MAC/ns]; load-only %.3f ns/vmadot-equiv\n", st, macs / st, lo);
	double s24 = k_stream24(vb);
	printf("2x4-blocked microkernel (6 loads per 8 vmadot, L1): %.3f ns/vmadot [%.1f MAC/ns]\n", s24, macs / s24);
	double s24w = k_stream24w(vb);
	printf("2x4-blocked, whole-reg loads vl2re8+vl4re8 (2 loads per 8 vmadot, L1): %.3f ns/vmadot [%.1f MAC/ns]\n", s24w, macs / s24w);
	double lw = k_ldw(vb);
	printf("vl4re8 load: %.3f ns each (%.1f B/ns)\n", lw, 4.0 * vb / lw);
	return 0;
}
