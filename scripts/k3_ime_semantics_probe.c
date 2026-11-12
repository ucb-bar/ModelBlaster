/* K3 IME vmadot semantics probe. Usage: k3_ime_semantics_probe <cpu>
 * Discovers tile shape / layouts / output order empirically (single-hot basis scan), then
 * verifies a reference C model on random data for all 4 signedness variants, accumulate,
 * vl dependence, vd/vd+1 order, odd vd, LMUL2. Operands: A in v0 (vs1), B in v4 (vs2), C in v8/v9. */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mb_placement.h"

static sigjmp_buf jb;
static void on_ill(int s) { (void)s; siglongjmp(jb, 1); }
static size_t VB; /* vlenb */

/* variant: 0 s(f3=3) 1 u(f3=0) 2 us(f3=1) 3 su(f3=2) 4 int4 s (f7=0x61) 5 lmul2 6 odd-vd(x9) 7 vd=x16 pair */
/* returns 1 if executed, 0 on SIGILL. cin/cout: 2*VB bytes (v8|v9 as e32m1 contiguous). */
#define BODY(INSN, LM) \
	"vsetvli t0, zero, e32, m1, ta, ma\n\t vle32.v v8, (%[ci])\n\t add t1, %[ci], %[vb]\n\t vle32.v v9, (t1)\n\t" \
	"vsetvli t0, zero, e8, m1, ta, ma\n\t vle8.v v0, (%[a])\n\t vle8.v v4, (%[b])\n\t" \
	"vsetvli t0, %[vl], e8, " LM ", ta, ma\n\t" INSN "\n\t" \
	"vsetvli t0, zero, e32, m1, ta, ma\n\t vse32.v v8, (%[co])\n\t add t1, %[co], %[vb]\n\t vse32.v v9, (t1)\n\t"
#define RUN(INSN, LM) __asm__ volatile(BODY(INSN, LM) : : [ci] "r"(cin), [a] "r"(A), [b] "r"(B), [co] "r"(cout), [vl] "r"(vl), [vb] "r"(VB) : "t0", "t1", "memory", "v0", "v4", "v8", "v9")
static int vm(int var, size_t vl, const int8_t *A, const int8_t *B, const int32_t *cin, int32_t *cout)
{
	if (sigsetjmp(jb, 1)) return 0;
	switch (var) {
	case 0: RUN(".insn r 0x2b, 3, 0x71, x8, x0, x4", "m1"); break;
	case 1: RUN(".insn r 0x2b, 0, 0x71, x8, x0, x4", "m1"); break;
	case 2: RUN(".insn r 0x2b, 1, 0x71, x8, x0, x4", "m1"); break;
	case 3: RUN(".insn r 0x2b, 2, 0x71, x8, x0, x4", "m1"); break;
	case 4: RUN(".insn r 0x2b, 3, 0x61, x8, x0, x4", "m1"); break;
	case 5: RUN(".insn r 0x2b, 3, 0x71, x8, x0, x4", "m2"); break;
	case 6: RUN(".insn r 0x2b, 3, 0x71, x9, x0, x4", "m1"); break;
	}
	return 1;
}

static int M, K, N;
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }

/* Reference model (the discovered layout):
 * A[i*K+k] (M x K, row-major), B[j*K+k] (N x K, i.e. B^T row-major = "column-major" B),
 * C[i*N+j] int32 row-major across the concatenated pair (vd | vd+1).  C += sum_k a*b. */
static void ref_vmadot(const int8_t *A, const int8_t *B, int32_t *C, int sa, int sb)
{
	for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
		int32_t s = 0;
		for (int k = 0; k < K; k++) {
			int32_t a = sa ? A[i * K + k] : (uint8_t)A[i * K + k];
			int32_t b = sb ? B[j * K + k] : (uint8_t)B[j * K + k];
			s += a * b;
		}
		C[i * N + j] += s;
	}
}
/* int4: each byte = 2 nibbles, low nibble = lower k; K4 = 2*K values per row; signed. */
static int nib(const int8_t *p, int idx, int sgn) { int b = (uint8_t)p[idx >> 1]; int v = (idx & 1) ? b >> 4 : b & 15; return sgn && v > 7 ? v - 16 : v; }
static void ref_vmadot4(const int8_t *A, const int8_t *B, int32_t *C, int M4, int K4, int N4)
{
	for (int i = 0; i < M4; i++) for (int j = 0; j < N4; j++) {
		int32_t s = 0;
		for (int k = 0; k < K4; k++) s += nib(A, i * K4 + k, 1) * nib(B, j * K4 + k, 1);
		C[i * N4 + j] += s;
	}
}

int main(int argc, char **argv)
{
	int cpu = argc > 1 ? atoi(argv[1]) : 0;
	struct sigaction sa = {0}; sa.sa_handler = on_ill; sigemptyset(&sa.sa_mask); sigaction(SIGILL, &sa, 0);
	if (mb_place_current_thread(cpu)) { fprintf(stderr, "placement failed\n"); return 2; }
	unsigned long vlenb; __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb)); VB = vlenb;
	int NE = 2 * VB / 4; /* int32 outputs over the pair */
	int8_t *A = aligned_alloc(64, VB), *B = aligned_alloc(64, VB);
	int32_t *cin = aligned_alloc(64, 2 * VB), *cout = aligned_alloc(64, 2 * VB), *cref = aligned_alloc(64, 2 * VB);
	printf("cpu=%d vlenb=%zu int32 outputs over vd,vd+1 = %d\n", cpu, VB, NE);

	/* ---- 1. single-hot scan: A[pa]=1, B[pb]=1, C0=0 */
	int16_t *map = malloc(VB * VB * sizeof(int16_t)); int nbad = 0;
	memset(cin, 0, 2 * VB);
	for (size_t pa = 0; pa < VB; pa++) for (size_t pb = 0; pb < VB; pb++) {
		memset(A, 0, VB); memset(B, 0, VB); A[pa] = 1; B[pb] = 1;
		if (!vm(0, VB, A, B, cin, cout)) { printf("SIGILL in scan\n"); return 1; }
		int hit = -1, cnt = 0;
		for (int e = 0; e < NE; e++) if (cout[e]) { hit = e; cnt++; if (cout[e] != 1) nbad++; }
		map[pa * VB + pb] = cnt == 1 ? hit : (cnt == 0 ? -1 : -2);
	}
	/* derive: pb hits for pa=0 */
	int n0 = 0, firstpb[64], stride = 0;
	for (size_t pb = 0; pb < VB; pb++) if (map[pb] >= 0) { if (n0 < 64) firstpb[n0] = pb; n0++; }
	int Nd = n0;
	int Kd = Nd > 1 ? firstpb[1] - firstpb[0] : 0; (void)stride;
	M = VB / Kd; K = Kd; N = Nd;
	printf("derived from scan: pa=0 couples to %d B-bytes (first at %d, stride %d) => N=%d K=%d M=%zu/K=%d\n", Nd, firstpb[0], Kd, N, K, VB, M);
	int mism = 0, bad2 = 0;
	for (size_t pa = 0; pa < VB; pa++) for (size_t pb = 0; pb < VB; pb++) {
		int i = pa / K, ka = pa % K, j = pb / K, kb = pb % K;
		int exp = (ka == kb) ? i * N + j : -1;
		int got = map[pa * VB + pb];
		if (got != exp) { if (mism < 8) printf("  MISMATCH pa=%zu pb=%zu exp=%d got=%d\n", pa, pb, exp, got); mism++; }
		if (got == -2) bad2++;
	}
	printf("TILE M=%d K=%d N=%d (int8); layout hypothesis A[i*K+k], B[j*K+k], C[i*N+j] (vd|vd+1 concatenated): scan mismatches=%d of %zu, multi-hit=%d, nonunit=%d\n", M, K, N, mism, VB * VB, bad2, nbad);

	/* ---- 2. random verification, 4 signedness variants, find (sa,sb) per funct3 */
	static const int f3[4] = {3, 0, 1, 2}; const char *nm[4] = {"s(f3=3)", "u(f3=0)", "us(f3=1)", "su(f3=2)"};
	for (int var = 0; var < 4; var++) {
		int ok[4] = {0}; int T = 20000;
		for (int t = 0; t < T; t++) {
			for (size_t i = 0; i < VB; i++) { A[i] = (int8_t)rnd(); B[i] = (int8_t)rnd(); }
			if (t % 4 == 0) for (size_t i = 0; i < VB; i++) { A[i] = (i & 1) ? -128 : 127; B[i] = (t & 8) ? -128 : 127; }
			for (int e = 0; e < NE; e++) cin[e] = (int32_t)rnd() >> 8;
			vm(var, VB, A, B, cin, cout);
			for (int c = 0; c < 4; c++) { memcpy(cref, cin, 2 * VB); ref_vmadot(A, B, cref, c & 1, c >> 1); if (!memcmp(cref, cout, 2 * VB)) ok[c]++; }
		}
		printf("variant %s: matches over %d trials (accum onto random C0): [A=u,B=u]=%d [A=s,B=u]=%d [A=u,B=s]=%d [A=s,B=s]=%d\n", nm[var], T, ok[0], ok[1], ok[2], ok[3]);
	}
	/* ---- 3. accumulate (twice) + signed values */
	{
		for (size_t i = 0; i < VB; i++) { A[i] = (int8_t)(i % 13 - 6); B[i] = (int8_t)(i % 11 - 5); }
		memset(cin, 0, 2 * VB); vm(0, VB, A, B, cin, cout); memcpy(cin, cout, 2 * VB); vm(0, VB, A, B, cin, cout);
		memset(cref, 0, 2 * VB); ref_vmadot(A, B, cref, 1, 1); for (int e = 0; e < NE; e++) cref[e] *= 2;
		printf("accumulate twice (neg values) == 2x ref: %s\n", memcmp(cref, cout, 2 * VB) ? "NO" : "YES");
	}
	/* ---- 4. vl dependence: random data, compare output at vl to full-vl reference */
	{
		size_t vls[] = {VB, 96, 64, 48, 32, 16, 8, 4, 1}; 
		for (unsigned vi = 0; vi < sizeof vls / sizeof *vls; vi++) {
			size_t vl = vls[vi]; if (vl > VB) continue;
			int eqfull = 0, eqzero = 0, T = 200; memset(cin, 0, 2 * VB);
			for (int t = 0; t < T; t++) {
				for (size_t i = 0; i < VB; i++) { A[i] = (int8_t)rnd(); B[i] = (int8_t)rnd(); }
				int r = vm(0, vl, A, B, cin, cout); if (!r) { eqfull = -1; break; }
				memset(cref, 0, 2 * VB); ref_vmadot(A, B, cref, 1, 1);
				if (!memcmp(cref, cout, 2 * VB)) eqfull++;
				int allz = 1; for (int e = 0; e < NE; e++) if (cout[e]) allz = 0; eqzero += allz;
			}
			printf("vl=%zu: %s (matches full-vl ref %d/%d, all-zero out %d)\n", vl, eqfull < 0 ? "SIGILL" : "exec", eqfull, T, eqzero);
		}
	}
	/* ---- 5. vl=32 with K1-style 4x4x8 model? (only meaningful when VB>32) */
	if (VB > 32) {
		int ok = 0, T = 200, ok2 = 0; memset(cin, 0, 2 * VB);
		for (int t = 0; t < T; t++) {
			for (size_t i = 0; i < VB; i++) { A[i] = (int8_t)rnd(); B[i] = (int8_t)rnd(); }
			vm(0, 32, A, B, cin, cout);
			int32_t r4[16] = {0};
			for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 8; k++) r4[i * 4 + j] += A[i * 8 + k] * B[j * 8 + k];
			if (!memcmp(r4, cout, 64)) ok++;
			/* maybe vl=32 -> 4x4x8 with results in first 16 elems */
			int lim = 1; for (int e = 16; e < NE; e++) if (cout[e]) lim = 0; ok2 += lim;
		}
		printf("vl=32 as K1-style 4x4x8 (C[0..15], rest zero): match=%d/%d, upper outputs zero=%d/%d\n", ok, T, ok2, T);
	}
	/* ---- 6. register pair / LMUL2 / odd vd */
	{
		for (size_t i = 0; i < VB; i++) { A[i] = 1; B[i] = 1; }
		memset(cin, 0, 2 * VB);
		printf("LMUL=2: %s\n", vm(5, VB, A, B, cin, cout) ? "EXEC" : "SIGILL");
		printf("odd vd (x9): %s\n", vm(6, VB, A, B, cin, cout) ? "EXEC" : "SIGILL");
		/* which half of the output lives in v8 vs v9: put C0 only in v8 half */
		memset(cin, 0, 2 * VB); for (int e = 0; e < NE / 2; e++) cin[e] = 1000; for (size_t i = 0; i < VB; i++) { A[i] = 0; B[i] = 0; }
		vm(0, VB, A, B, cin, cout);
		int pass = 1; for (int e = 0; e < NE; e++) if (cout[e] != cin[e]) pass = 0;
		printf("C0 passthrough with zero operands (C index e<%d in vd, rest in vd+1): %s\n", NE / 2, pass ? "YES" : "NO");
	}
	/* ---- 7. int4 (funct7 0x61): try 2*K nibbles/row hypothesis */
	{
		int M4 = M, N4 = N, K4 = 2 * K; int ok = 0, T = 5000, ex = 1;
		for (int t = 0; t < T; t++) {
			for (size_t i = 0; i < VB; i++) { A[i] = (int8_t)rnd(); B[i] = (int8_t)rnd(); }
			for (int e = 0; e < NE; e++) cin[e] = 0;
			if (!vm(4, VB, A, B, cin, cout)) { ex = 0; break; }
			memset(cref, 0, 2 * VB); ref_vmadot4(A, B, cref, M4, K4, N4);
			if (!memcmp(cref, cout, 2 * VB)) ok++;
		}
		printf("int4 f7=0x61: %s; matches %dx%dx%d nibble-model (low nibble first, signed): %d/%d\n", ex ? "exec" : "SIGILL", M4, K4, N4, ok, T);
		if (ex && ok == 0) { /* single-hot scan on nibbles to show structure */
			memset(cin, 0, 2 * VB); int shown = 0;
			for (size_t pa = 0; pa < 4 * 2 && shown < 6; pa++) for (int hi = 0; hi < 2; hi++) {
				memset(A, 0, VB); memset(B, 0x11, VB); A[pa] = hi ? 0x10 : 0x01;
				vm(4, VB, A, B, cin, cout); printf("  int4 A byte %zu %s nibble=1, B all nibbles=1: nonzero e:", pa, hi ? "hi" : "lo");
				for (int e = 0; e < NE; e++) if (cout[e]) printf(" [%d]=%d", e, cout[e]); printf("\n"); shown++;
			}
		}
	}
	return 0;
}
