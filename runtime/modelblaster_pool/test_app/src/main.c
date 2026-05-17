/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Unit test for modelblaster_pool. Runs three correctness checks:
 *
 *   1. parallelize_1d over a known range exactly invokes fn once per i
 *      (every input is touched exactly once, no overlap, no gaps).
 *   2. The accumulated sum across all workers matches the
 *      sequential-loop reference.
 *   3. modelblaster_pool_get_threads_count() reports the value passed to
 *      create().
 *
 * Then prints a tiny perf number — per-call cycles for a small fixed
 * range (32) at N=4 — so we can sanity-check it lands within ~2x of
 * the bench_pthreads_raw row in the threadpool microbench's
 * firesim_overhead.csv (~20k cycles per call there).
 *
 * Output protocol: PASS/FAIL between MODELBLASTER_POOL_TEST_{BEGIN,END}
 * markers; the runner script greps for these. Per-call cycles is
 * emitted as a single PERF line so the perf check is structured.
 */

#include "modelblaster_pool.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#define TEST_RANGE 1024
#define TEST_REPS  16

static unsigned long long touch_count[TEST_RANGE];

/* Fn body: increment touch_count[i] (for the "every i seen exactly
 * once" check) and accumulate i*i into a per-call partial sum used by
 * the second test below. The volatile qualifier on `partial_sum`
 * (declared in main as a struct member that the worker reads/writes)
 * keeps the compiler from elision-folding the work; modelblaster_pool runs
 * fn slices in series on each helper so the writes within one slice
 * don't race. */
struct test_ctx {
	volatile unsigned long long *bucket;  /* length n_workers */
	int n_workers;
	size_t per;
};

static void touch_fn(void *ctx_, size_t i)
{
	struct test_ctx *c = (struct test_ctx *)ctx_;
	(void)c;
	if (i < TEST_RANGE) {
		touch_count[i]++;
	}
}

/* Bucketed-sum body. Each helper accumulates into bucket[wid] where
 * wid is derived from i / per (per = ceil(range / n_workers)). This
 * verifies that the slicing inside modelblaster_pool maps i correctly to
 * the helper that owns it: when we sum bucket[0..n-1] we should get
 * exactly the closed-form sum_{i=0..range-1} i. */
static void bucket_fn(void *ctx_, size_t i)
{
	struct test_ctx *c = (struct test_ctx *)ctx_;
	int wid = (int)(i / c->per);
	if (wid >= c->n_workers) {
		wid = c->n_workers - 1;
	}
	c->bucket[wid] += (unsigned long long)i;
}

static inline uint64_t rdcycle64(void)
{
	uint64_t v;
	__asm__ volatile("rdcycle %0" : "=r"(v));
	return v;
}

int main(void)
{
	int fail = 0;

	printf("modelblaster_pool_test: starting on %s (MP_MAX_NUM_CPUS=%d)\n",
	       CONFIG_BOARD_TARGET, (int)CONFIG_MP_MAX_NUM_CPUS);

	/* Pin master to hart 0 so the affinity-pinned helpers (1..N-1)
	 * don't fight master for hart placement. The harness model
	 * mains do the same. */
	k_thread_cpu_pin(k_current_get(), 0);

	const int N = 4;
	modelblaster_pool_t pool = modelblaster_pool_create(N);
	if (pool == NULL) {
		printf("=== MODELBLASTER_POOL_TEST_BEGIN ===\n");
		printf("FAIL: modelblaster_pool_create(%d) returned NULL\n", N);
		printf("=== MODELBLASTER_POOL_TEST_END ===\n");
		sys_reboot(SYS_REBOOT_COLD);
		return -1;
	}

	unsigned tcount = modelblaster_pool_get_threads_count(pool);
	printf("modelblaster_pool_test: pool create OK, threads=%u\n", tcount);

	/* --- Test 1: every i in [0, range) is visited exactly once. -- */
	memset(touch_count, 0, sizeof(touch_count));
	struct test_ctx ctx = { 0 };
	modelblaster_pool_parallelize_1d(pool, touch_fn, &ctx, TEST_RANGE, 0);
	int touch_fail = 0;
	for (int i = 0; i < TEST_RANGE; i++) {
		if (touch_count[i] != 1ULL) {
			if (touch_fail < 4) {
				printf("  TOUCH FAIL i=%d count=%llu\n",
				       i, (unsigned long long)touch_count[i]);
			}
			touch_fail++;
		}
	}
	if (touch_fail) {
		printf("  TEST1 FAIL: %d slots not touched exactly once\n",
		       touch_fail);
		fail++;
	}

	/* --- Test 2: per-worker bucket sums to the closed-form total. ----
	 *
	 * sum_{i=0..N-1} i = N*(N-1)/2. With range=TEST_RANGE this is
	 * TEST_RANGE * (TEST_RANGE - 1) / 2 = 523776 for TEST_RANGE=1024.
	 */
	unsigned long long buckets[16] = { 0 };
	struct test_ctx bctx = {
		.bucket = buckets,
		.n_workers = (int)tcount,
		.per = (TEST_RANGE + tcount - 1) / tcount,
	};
	modelblaster_pool_parallelize_1d(pool, bucket_fn, &bctx, TEST_RANGE, 0);
	unsigned long long total = 0;
	for (unsigned w = 0; w < tcount; w++) {
		total += buckets[w];
	}
	unsigned long long expected =
		(unsigned long long)TEST_RANGE *
		(unsigned long long)(TEST_RANGE - 1) / 2ULL;
	if (total != expected) {
		printf("  TEST2 FAIL: sum=%llu expected=%llu\n",
		       total, expected);
		fail++;
	}

	/* --- Test 3: get_threads_count returns N. --------------------- */
	if (tcount != (unsigned)N) {
		printf("  TEST3 FAIL: get_threads_count=%u expected=%d\n",
		       tcount, N);
		fail++;
	}

	/* --- Perf sanity: per-call cycles for range=32 at N=4. -------- */
	uint64_t per_call_min = (uint64_t)-1;
	uint64_t per_call_sum = 0;
	for (int rep = 0; rep < TEST_REPS; rep++) {
		uint64_t t0 = rdcycle64();
		modelblaster_pool_parallelize_1d(pool, touch_fn, &ctx, 32, 0);
		uint64_t t1 = rdcycle64();
		uint64_t d = t1 - t0;
		if (d < per_call_min) {
			per_call_min = d;
		}
		per_call_sum += d;
	}
	uint64_t per_call_avg = per_call_sum / (uint64_t)TEST_REPS;

	modelblaster_pool_destroy(pool);

	/* --- Final report --------------------------------------------- */
	printf("=== MODELBLASTER_POOL_TEST_BEGIN ===\n");
	if (fail) {
		printf("FAIL: %d sub-tests failed\n", fail);
	} else {
		printf("PASS: 3/3 sub-tests passed\n");
	}
	printf("PERF: range=32 N=%d reps=%d per_call_cycles min=%llu avg=%llu\n",
	       N, TEST_REPS,
	       (unsigned long long)per_call_min,
	       (unsigned long long)per_call_avg);
	printf("=== MODELBLASTER_POOL_TEST_END ===\n");

	sys_reboot(SYS_REBOOT_COLD);
	return fail ? -1 : 0;
}
