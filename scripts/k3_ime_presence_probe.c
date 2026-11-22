/* K3 IME presence probe. Usage: k3_ime_presence_probe <cpu>
 * Executes one vmadot (.insn r 0x2b, funct3, funct7) per variant under a SIGILL handler.
 * Variants: funct3 3=s 0=u 1=us 2=su at funct7 0x71 (int8); funct3 3 at funct7 0x61 (int4).
 * Also tries vl=32 and vl=VLMAX at e8m1, and LMUL=2 (expected trap). */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mb_placement.h"

static sigjmp_buf jb;
static void on_ill(int s) { (void)s; siglongjmp(jb, 1); }
static int8_t buf[2048] __attribute__((aligned(64)));
static int32_t out[64] __attribute__((aligned(64)));

#define DEFVAR(NAME, F3, F7, LMUL) \
static int NAME(size_t vlreq) { \
	if (sigsetjmp(jb, 1)) return 0; \
	__asm__ volatile( \
	  "vsetvli t0, %[n], e8, m1, ta, ma\n\t" \
	  "vle8.v v0, (%[p])\n\t vle8.v v4, (%[p])\n\t" \
	  "vmv.v.i v8, 0\n\t vmv.v.i v9, 0\n\t" \
	  "vsetvli t0, %[n], e8, " LMUL ", ta, ma\n\t" \
	  ".insn r 0x2b, " #F3 ", " #F7 ", x8, x0, x4\n\t" \
	  "vsetvli t0, zero, e32, m1, ta, ma\n\t" \
	  "vse32.v v8, (%[o])\n\t" \
	  : : [n] "r"(vlreq), [p] "r"(buf), [o] "r"(out) : "t0", "memory", "v0", "v4", "v8", "v9"); \
	return 1; }
DEFVAR(v_s8, 3, 0x71, "m1")
DEFVAR(v_u8, 0, 0x71, "m1")
DEFVAR(v_us8, 1, 0x71, "m1")
DEFVAR(v_su8, 2, 0x71, "m1")
DEFVAR(v_s4, 3, 0x61, "m1")
DEFVAR(v_s8_m2, 3, 0x71, "m2")

int main(int argc, char **argv)
{
	int cpu = argc > 1 ? atoi(argv[1]) : 0;
	struct sigaction sa = {0}; sa.sa_handler = on_ill; sigemptyset(&sa.sa_mask);
	sigaction(SIGILL, &sa, 0);
	if (mb_place_current_thread(cpu)) { fprintf(stderr, "placement failed\n"); return 2; }
	for (int i = 0; i < 2048; i++) buf[i] = (int8_t)(i % 7 - 3);
	unsigned long vlenb; __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
	unsigned long mvendorid, marchid, mimpid; /* may trap in U-mode; not read */
	(void)mvendorid; (void)marchid; (void)mimpid;
	size_t vmax = vlenb; /* e8m1 vlmax = vlenb */
	printf("cpu=%d vlenb=%lu\n", cpu, vlenb);
	size_t vls[3] = {32, vmax, 64};
	for (int i = 0; i < 3; i++) {
		size_t vl = vls[i];
		printf("cpu=%d vl=%zu: s8=%s u8=%s us8=%s su8=%s s4(0x61)=%s s8_lmul2=%s\n", cpu, vl,
		       v_s8(vl) ? "EXEC" : "SIGILL", v_u8(vl) ? "EXEC" : "SIGILL", v_us8(vl) ? "EXEC" : "SIGILL",
		       v_su8(vl) ? "EXEC" : "SIGILL", v_s4(vl) ? "EXEC" : "SIGILL", v_s8_m2(vl) ? "EXEC" : "SIGILL");
	}
	return 0;
}
