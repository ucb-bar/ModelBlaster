/* Zvt kernel smoke test: linear_s8, conv2d_s8 and fp32 linear tile
 * kernels vs the pipeline's reference implementations (bit-exact), with
 * curated RVV kernels as a baseline where one exists for the same layout.
 * Prints ZVT_SMOKE: PASS when every case is bit-exact.
 *
 * The kernels size their tiles from the hart at runtime, so the same ELF
 * runs on any Zvt geometry (TE, VLEN). Cycle counts are spike's
 * (~1 instruction/cycle): instruction-count proxies, not timing. */
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/arch/riscv/zvt.h>
#include <stdint.h>
#include <string.h>

#define LIN_ARGS const int8_t *, const int8_t *, const int32_t *, int8_t *, \
	int, int, int, int, int, int, int, int, int, int
void ref_linear_s8(LIN_ARGS);
void zvt_linear_s8(LIN_ARGS);
void rvv_linear_s8(LIN_ARGS);
#define CONV_ARGS const int8_t *, const int8_t *, const int32_t *, int8_t *, \
	int, int, int, int, int, int, int, int, int, int, int, \
	int, int, int, int, int, int, int
void ref_conv2d_s8(CONV_ARGS);
void zvt_conv2d_s8(CONV_ARGS);
#define F32_ARGS const float *, const float *, const float *, float *, int, int, int
void ref_linear_f32(F32_ARGS);
void zvt_linear_f32(F32_ARGS);
void rvv_linear_f32(F32_ARGS);

static Z_RISCV_ZVT_SAVE_AREA_DEFINE(zvt_area);

#define BUF (128 * 1024)
static int8_t a8[BUF], b8[BUF], o_ref[BUF], o_zvt[BUF], o_rvv[BUF];
static float af[BUF / 4], bf[BUF / 4], fo_ref[BUF / 4], fo_zvt[BUF / 4], fo_rvv[BUF / 4];
static int32_t bias[1024];
static float biasf[1024];
static int fails;

static uint32_t rng = 12345;
static uint32_t rnd(void)
{
	rng = rng * 1664525u + 1013904223u;
	return rng;
}
static int8_t rnd8(void)
{
	return (int8_t)(rnd() >> 24);
}
static float rndf(void)
{
	return ((int32_t)(rnd() >> 8) - (1 << 23)) / (float)(1 << 23);
}
static inline uint64_t rdcycle(void)
{
	uint64_t c;

	__asm__ volatile("rdcycle %0" : "=r"(c));
	return c;
}
#define TIME(var, call) do { uint64_t _t = rdcycle(); call; var = rdcycle() - _t; } while (0)

static void row(const char *name, uint64_t c_ref, uint64_t c_rvv, uint64_t c_zvt, int bad,
		const char *note)
{
	char rv[16] = "-";

	if (c_rvv) {
		snprintk(rv, sizeof(rv), "%llu", (unsigned long long)c_rvv);
	}
	printk("%-26s %10llu %10s %10llu %6llu.%01llux %s%s\n", name,
	       (unsigned long long)c_ref, rv, (unsigned long long)c_zvt,
	       (unsigned long long)(c_ref / c_zvt), (unsigned long long)((c_ref * 10 / c_zvt) % 10),
	       bad < 0 ? "bit-exact" : "MISMATCH", note);
	if (bad >= 0) {
		fails++;
	}
}

static int first_diff8(const int8_t *x, const int8_t *y, int n)
{
	for (int i = 0; i < n; i++) {
		if (x[i] != y[i]) {
			printk("  first mismatch at %d: zvt %d ref %d\n", i, y[i], x[i]);
			return i;
		}
	}
	return -1;
}

static int first_difff(const float *x, const float *y, int n)
{
	for (int i = 0; i < n; i++) {
		uint32_t ux, uy;

		memcpy(&ux, &x[i], 4);
		memcpy(&uy, &y[i], 4);
		if (ux != uy) {
			printk("  first mismatch at %d: zvt %08x ref %08x\n", i, uy, ux);
			return i;
		}
	}
	return -1;
}

struct lin { int M, K, N, io, fo, oo; };
static const struct lin lins[] = {
	{1, 16, 32, 0, 0, 0}, {4, 128, 10, 0, 0, 0}, {32, 128, 32, 0, 0, 0},
	{33, 37, 65, 0, 0, 0}, {5, 3, 7, 0, 0, 0}, {1, 1, 1, 0, 0, 0},
	{64, 512, 128, 0, 0, 0}, {130, 70, 140, 0, 0, 0}, {9, 70, 40, 3, -2, 5},
	{1, 2048, 1, 0, 0, 0}, {3, 50, 5, 2, 1, -1}, /* GEMV path (N < 8) */
};

struct conv { int N, IC, IH, IW, OC, KH, KW, SH, SW, PH, PW, io, fo, oo; };
static const struct conv convs[] = {
	{1, 3, 16, 16, 8, 3, 3, 1, 1, 1, 1, 0, 0, 0},    /* first layer, pad */
	{1, 16, 12, 12, 32, 3, 3, 2, 2, 1, 1, 0, 0, 0},  /* stride 2 */
	{1, 32, 8, 8, 64, 1, 1, 1, 1, 0, 0, 0, 0, 0},    /* pointwise */
	{1, 8, 10, 9, 150, 3, 3, 1, 1, 0, 0, 0, 0, 0},   /* OC > 4 tiles, odd W */
	{2, 3, 7, 7, 5, 5, 5, 1, 1, 2, 2, 0, 0, 0},      /* batch 2, K=75 tail */
	{1, 6, 9, 9, 12, 3, 3, 1, 1, 1, 1, 3, -2, 1},    /* offsets */
	{1, 32, 24, 24, 64, 3, 3, 1, 1, 1, 1, 0, 0, 0},  /* DroNet-sized */
	{1, 3, 112, 112, 32, 3, 3, 2, 2, 1, 1, 0, 0, 0}, /* DroNet layer 0 */
	{1, 64, 7, 7, 64, 3, 3, 1, 1, 1, 1, 0, 0, 0},    /* DroNet layer 5 */
};

struct flin { int M, K, N; bool bias; };
static const struct flin flins[] = {
	{1, 16, 32, true}, {4, 128, 10, true}, {33, 37, 65, false}, {64, 256, 64, true},
	{1, 1, 1, true},
};

int main(void)
{
	unsigned int te, cols;
	int rc = arch_riscv_zvt_enable(zvt_area, sizeof(zvt_area));

	if (rc != 0 || arch_riscv_zvt_geometry(&te, &cols) != 0) {
		printk("ZVT_SMOKE: enable=%d\nZVT_SMOKE: FAIL\n", rc);
		sys_reboot(SYS_REBOOT_COLD);
		return 0;
	}
	printk("ZVT_SMOKE: TE=%u cols=%u\n", te, cols);
	printk("%-26s %10s %10s %10s %8s\n", "case", "ref", "rvv", "zvt", "ref/zvt");

#ifdef ZVT_SMOKE_ONLY_CONV
	for (size_t i = 0; i < 0; i++) {
#else
	for (size_t i = 0; i < ARRAY_SIZE(lins); i++) {
#endif
		const struct lin *s = &lins[i];
		int mult = 1073741824 + (int)(rnd() % 100000), shift = 6 + (int)(i % 4);
		uint64_t cr, cv, cz;

		for (int j = 0; j < s->M * s->K; j++) a8[j] = rnd8();
		for (int j = 0; j < s->N * s->K; j++) b8[j] = rnd8();
		for (int j = 0; j < s->N; j++) bias[j] = (int32_t)rnd8() * 37;
		memset(o_zvt, 0x55, (size_t)s->M * s->N);
		TIME(cr, ref_linear_s8(a8, b8, bias, o_ref, s->M, s->K, s->N, s->io, s->fo, s->oo,
				       mult, shift, -128, 127));
		TIME(cv, rvv_linear_s8(a8, b8, bias, o_rvv, s->M, s->K, s->N, s->io, s->fo, s->oo,
				       mult, shift, -128, 127));
		TIME(cz, zvt_linear_s8(a8, b8, bias, o_zvt, s->M, s->K, s->N, s->io, s->fo, s->oo,
				       mult, shift, -128, 127));
		char name[32];

		snprintk(name, sizeof(name), "linear_s8 %dx%dx%d%s", s->M, s->K, s->N,
			 (s->io || s->fo || s->oo) ? "*" : "");
		row(name, cr, cv, cz, first_diff8(o_ref, o_zvt, s->M * s->N),
		    memcmp(o_ref, o_rvv, (size_t)s->M * s->N) ? " (rvv mismatch)" : "");
	}

	for (size_t i = 0; i < ARRAY_SIZE(convs); i++) {
#ifdef ZVT_SMOKE_ONLY_CONV
		if (i != ZVT_SMOKE_ONLY_CONV) {
			continue;
		}
#endif
		const struct conv *c = &convs[i];
		int OH = (c->IH + 2 * c->PH - c->KH) / c->SH + 1;
		int OW = (c->IW + 2 * c->PW - c->KW) / c->SW + 1;
		int nin = c->N * c->IC * c->IH * c->IW, nw = c->OC * c->IC * c->KH * c->KW;
		int nout = c->N * c->OC * OH * OW;
		int mult = 1073741824 + (int)(rnd() % 100000), shift = 7 + (int)(i % 3);
		uint64_t cr, cz;

		for (int j = 0; j < nin; j++) a8[j] = rnd8();
		for (int j = 0; j < nw; j++) b8[j] = rnd8();
		for (int j = 0; j < c->OC; j++) bias[j] = (int32_t)rnd8() * 53;
		memset(o_zvt, 0x55, (size_t)nout);
		TIME(cr, ref_conv2d_s8(a8, b8, bias, o_ref, c->N, c->IC, c->IH, c->IW, c->OC, c->KH,
				       c->KW, c->SH, c->SW, c->PH, c->PW, c->io, c->fo, c->oo, mult,
				       shift, -128, 127));
		TIME(cz, zvt_conv2d_s8(a8, b8, bias, o_zvt, c->N, c->IC, c->IH, c->IW, c->OC, c->KH,
				       c->KW, c->SH, c->SW, c->PH, c->PW, c->io, c->fo, c->oo, mult,
				       shift, -128, 127));
		char name[40];

		snprintk(name, sizeof(name), "conv %dx%dx%dx%d>%d k%d s%d p%d%s", c->N, c->IC, c->IH,
			 c->IW, c->OC, c->KH, c->SH, c->PH, (c->io || c->fo || c->oo) ? "*" : "");
		row(name, cr, 0, cz, first_diff8(o_ref, o_zvt, nout), "");
	}

#ifdef ZVT_SMOKE_ONLY_CONV
	for (size_t i = 0; i < 0; i++) {
#else
	for (size_t i = 0; i < ARRAY_SIZE(flins); i++) {
#endif
		const struct flin *s = &flins[i];
		uint64_t cr, cv, cz;

		for (int j = 0; j < s->M * s->K; j++) af[j] = rndf();
		for (int j = 0; j < s->N * s->K; j++) bf[j] = rndf();
		for (int j = 0; j < s->N; j++) biasf[j] = rndf();
		const float *bp = s->bias ? biasf : NULL;

		TIME(cr, ref_linear_f32(af, bf, bp, fo_ref, s->M, s->K, s->N));
		TIME(cv, rvv_linear_f32(af, bf, bp, fo_rvv, s->M, s->K, s->N));
		TIME(cz, zvt_linear_f32(af, bf, bp, fo_zvt, s->M, s->K, s->N));
		char name[32];

		snprintk(name, sizeof(name), "linear_f32 %dx%dx%d%s", s->M, s->K, s->N,
			 s->bias ? "" : " nobias");
		row(name, cr, cv, cz, first_difff(fo_ref, fo_zvt, s->M * s->N),
		    memcmp(fo_ref, fo_rvv, (size_t)s->M * s->N * 4) ? " (rvv differs: FMA)" : "");
	}

	printk("tile stats: nested_traps=%u lazy_traps=%u stores=%u loads=%u\n",
	       z_riscv_zvt_stats.nested, z_riscv_zvt_stats.lazy_traps, z_riscv_zvt_stats.stores,
	       z_riscv_zvt_stats.loads);
	printk("(* = nonzero offsets)\nZVT_SMOKE: %s\n", fails ? "FAIL" : "PASS");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
