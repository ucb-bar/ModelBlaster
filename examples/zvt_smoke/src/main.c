/* Zvt linear_s8 smoke test. Prints ZVT_SMOKE: PASS when the Zvt kernel is
 * bit-exact with the scalar reference on every shape. Cycle counts are
 * spike's (~1 instruction/cycle): instruction-count proxies, not timing. */
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/arch/riscv/zvt.h>
#include <stdint.h>
#include <string.h>

typedef void (*linear_fn)(const int8_t *, const int8_t *, const int32_t *, int8_t *,
                          int, int, int, int, int, int, int, int, int, int);
void ref_linear_s8(const int8_t *, const int8_t *, const int32_t *, int8_t *,
                   int, int, int, int, int, int, int, int, int, int);
void zvt_linear_s8(const int8_t *, const int8_t *, const int32_t *, int8_t *,
                   int, int, int, int, int, int, int, int, int, int);
void rvv_linear_s8(const int8_t *, const int8_t *, const int32_t *, int8_t *,
                   int, int, int, int, int, int, int, int, int, int);

static Z_RISCV_ZVT_SAVE_AREA_DEFINE(zvt_area);

#define MAXM 64
#define MAXK 512
#define MAXN 128
static int8_t in[MAXM * MAXK], w[MAXN * MAXK], out_ref[MAXM * MAXN], out_zvt[MAXM * MAXN],
	out_rvv[MAXM * MAXN];
static int32_t bias[MAXN];

static uint32_t rng = 12345;
static int8_t rnd8(void)
{
	rng = rng * 1664525u + 1013904223u;
	return (int8_t)(rng >> 24);
}

static inline uint64_t rdcycle(void)
{
	uint64_t c;
	__asm__ volatile("rdcycle %0" : "=r"(c));
	return c;
}

struct shape { int M, K, N, io, fo, oo; };

static const struct shape shapes[] = {
	/* LINEAR_S8.extra_shapes-style MLP shapes */
	{1, 16, 32, 0, 0, 0}, {1, 64, 64, 0, 0, 0}, {4, 128, 10, 0, 0, 0},
	/* tile / K edges */
	{32, 128, 32, 0, 0, 0}, {33, 37, 65, 0, 0, 0}, {5, 3, 7, 0, 0, 0},
	{1, 1, 1, 0, 0, 0}, {64, 512, 128, 0, 0, 0}, {17, 255, 33, 0, 0, 0},
	/* asymmetric offsets exercise the correction path */
	{9, 70, 40, 3, -2, 5}, {32, 64, 32, -7, 4, -3},
};

static uint64_t run(linear_fn f, const struct shape *s, int8_t *out, int mult, int shift)
{
	uint64_t t0 = rdcycle();
	f(in, w, bias, out, s->M, s->K, s->N, s->io, s->fo, s->oo, mult, shift, -128, 127);
	return rdcycle() - t0;
}

int main(void)
{
	int rc = arch_riscv_zvt_enable(zvt_area, sizeof(zvt_area));
	int fails = 0;

	printk("ZVT_SMOKE: arch_riscv_zvt_enable=%d\n", rc);
	if (rc != 0) {
		printk("ZVT_SMOKE: FAIL\n");
		return 0;
	}

	printk("%-14s %10s %10s %10s %8s %s\n", "M x K x N", "ref", "rvv", "zvt",
	       "ref/zvt", "result");
	for (size_t i = 0; i < ARRAY_SIZE(shapes); i++) {
		const struct shape *s = &shapes[i];
		int mult = 1073741824 + (int)(rng % 100000), shift = 6 + (int)(i % 4);

		for (int j = 0; j < s->M * s->K; j++) {
			in[j] = rnd8();
		}
		for (int j = 0; j < s->N * s->K; j++) {
			w[j] = rnd8();
		}
		for (int j = 0; j < s->N; j++) {
			bias[j] = (int32_t)(rnd8()) * 37;
		}
		memset(out_zvt, 0x55, sizeof(out_zvt));

		uint64_t c_ref = run(ref_linear_s8, s, out_ref, mult, shift);
		uint64_t c_rvv = run(rvv_linear_s8, s, out_rvv, mult, shift);
		uint64_t c_zvt = run(zvt_linear_s8, s, out_zvt, mult, shift);

		int bad = -1, bad_rvv = memcmp(out_ref, out_rvv, (size_t)s->M * s->N) != 0;

		for (int j = 0; j < s->M * s->N; j++) {
			if (out_zvt[j] != out_ref[j]) {
				bad = j;
				break;
			}
		}
		char name[24];

		snprintk(name, sizeof(name), "%dx%dx%d%s", s->M, s->K, s->N,
			 (s->io || s->fo || s->oo) ? "*" : "");
		printk("%-14s %10llu %10llu %10llu %7llu.%01llux %s%s\n", name,
		       (unsigned long long)c_ref, (unsigned long long)c_rvv,
		       (unsigned long long)c_zvt, (unsigned long long)(c_ref / c_zvt),
		       (unsigned long long)((c_ref * 10 / c_zvt) % 10),
		       bad < 0 ? "bit-exact" : "MISMATCH", bad_rvv ? " (rvv mismatch)" : "");
		if (bad >= 0) {
			printk("  first mismatch at %d: zvt %d ref %d\n", bad, out_zvt[bad],
			       out_ref[bad]);
			fails++;
		}
	}
	printk("(* = nonzero offsets)\nZVT_SMOKE: %s\n", fails ? "FAIL" : "PASS");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
