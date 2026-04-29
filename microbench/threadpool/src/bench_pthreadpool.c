/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bench #1: pthreadpool dispatch overhead.
 *
 * Mirrors the way agents/harness_multi/ uses pthreadpool: create a pool
 * of N workers, then call pthreadpool_parallelize_1d in a tight loop
 * with a trivial body. Per-call cycles are read via rdcycle() at the
 * caller (the master thread), bracketing each parallelize_1d call —
 * this is the exact cost the model loop pays per dispatch.
 *
 * pthreadpool's portable backend (the one we vendor via executorch) is
 * pthread_cond_*-based by default; PTHREADPOOL_SYNC_PRIMITIVE='condvar'
 * in its CMakeLists.txt. The master thread inside parallelize already
 * spins (PTHREADPOOL_SPIN_WAIT_ITERATIONS = 1M) before falling back to
 * pthread_cond_wait, so on a fast SoC we'd see the spin path; on
 * FireSim's quad-rocket-saturn the wait can fall through to the
 * cond_wait path because the worker takes long enough to land on the
 * other hart.
 *
 * NOTE: there is no in-band way to time the master wake -> first worker
 * cycle here (pthreadpool's internals are private). The other two
 * harnesses (k_thread + raw POSIX) use explicit semaphores so they CAN
 * measure that breakdown — the comparison across the three is exactly
 * how we attribute cost to the wrapper layer.
 *
 * Print everything AFTER the timed loops; printing on hart 0 during
 * timing would block the master while the HTIF UART drains. (See the
 * xpurt walker incident referenced in the project memory.)
 */

#include <pthreadpool.h>

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zephyr/arch/cpu.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "bench_common.h"

/* Workload buffers — file-static so they're in BSS, not on the main
 * stack (Zephyr's main stack is 512 KB on the FireSim build, and we
 * don't want VLA pressure in this critical path). */
static float bench_in[BENCH_MAX_RANGE];
static volatile float bench_out[BENCH_MAX_RANGE];

/* pthreadpool's task signature: (context, index). For 1d
 * parallelize_1d the worker is invoked once per index in [0, range). We
 * coalesce contiguous index ranges into one body call by using
 * parallelize_1d_tile_1d instead — but pthreadpool's tile variant adds
 * its own quanta-management overhead. Stick with the plain 1d API: it's
 * the one models actually use, so its dispatch cost is what we want to
 * characterize. */
static void task_1d(void *ctx, size_t i)
{
	bench_workload_t *wl = (bench_workload_t *)ctx;
	wl->out[i] = wl->in[i] * 2.0f;
}

static int memalign_shim_used; /* keep the linker happy */

/* picolibc + pthreadpool wants memalign + posix_memalign in the linked
 * binary. The multi-model harness's generated multi_main.c provides
 * these for the same reason; we re-declare them locally. */
void *memalign(size_t alignment, size_t size)
{
	memalign_shim_used = 1;
	return aligned_alloc(alignment, size);
}

int posix_memalign(void **out, size_t alignment, size_t size)
{
	void *p = aligned_alloc(alignment, size);
	if (!p) {
		return 12; /* ENOMEM */
	}
	*out = p;
	return 0;
}

static void run_one(bench_row_t *row, pthreadpool_t pool,
		    int n_workers, int range)
{
	bench_workload_t wl = { .in = bench_in, .out = bench_out,
				.range = (size_t)range };

	/* Warmup so any first-call setup (lazy worker thread spawn,
	 * cache lines for the threadpool struct) doesn't pollute the
	 * timed samples. */
	for (int i = 0; i < BENCH_WARMUP_REPS; i++) {
		pthreadpool_parallelize_1d(pool, task_1d, &wl,
					   (size_t)range, 0);
	}

	static uint64_t samples[BENCH_TIMED_REPS];
	for (int i = 0; i < BENCH_TIMED_REPS; i++) {
		uint64_t t0 = bench_rdcycle();
		pthreadpool_parallelize_1d(pool, task_1d, &wl,
					   (size_t)range, 0);
		uint64_t t1 = bench_rdcycle();
		samples[i] = t1 - t0;
	}

	row->harness = "pthreadpool";
#ifdef BENCH_PTHREADPOOL_VARIANT_NAME
	row->variant = BENCH_PTHREADPOOL_VARIANT_NAME;
#else
	row->variant = "default";
#endif
	row->n_workers = n_workers;
	row->range = range;
	row->reps = BENCH_TIMED_REPS;
	row->per_call_min = bench_min(samples, BENCH_TIMED_REPS);
	row->per_call_max = bench_max(samples, BENCH_TIMED_REPS);
	row->per_call_med = bench_median_inplace(samples, BENCH_TIMED_REPS);
	/* pthreadpool internals are opaque to us — leave the breakdown
	 * fields zero. The k_thread + raw-pthread harnesses fill these in. */
	row->wake_to_first_worker = 0;
	row->wake_to_all_finished = 0;
	row->finish_to_observed = 0;
}

/* Single-shot k_sem ping/pong on the master hart, to baseline the
 * primitive-itself cost (hart-local, no IPI). One round = give + take.
 * Used to interpret the cross-hart numbers from the other harnesses. */
static struct k_sem ping_sem;

static void bench_local_sem(uint64_t *out_med, uint64_t *out_min,
			    uint64_t *out_max)
{
	k_sem_init(&ping_sem, 0, 1);
	for (int i = 0; i < BENCH_WARMUP_REPS; i++) {
		k_sem_give(&ping_sem);
		k_sem_take(&ping_sem, K_FOREVER);
	}
	static uint64_t s[BENCH_TIMED_REPS];
	for (int i = 0; i < BENCH_TIMED_REPS; i++) {
		uint64_t t0 = bench_rdcycle();
		k_sem_give(&ping_sem);
		k_sem_take(&ping_sem, K_FOREVER);
		uint64_t t1 = bench_rdcycle();
		s[i] = t1 - t0;
	}
	*out_min = bench_min(s, BENCH_TIMED_REPS);
	*out_max = bench_max(s, BENCH_TIMED_REPS);
	*out_med = bench_median_inplace(s, BENCH_TIMED_REPS);
}

int main(void)
{
	printf("bench_pthreadpool: starting on %s "
	       "(MP_MAX_NUM_CPUS=%d)\n",
	       CONFIG_BOARD_TARGET, (int)CONFIG_MP_MAX_NUM_CPUS);
	/* Pin master to hart 0 so cross-rdcycle pairs always land on
	 * the same per-hart cycle counter on spike. (FireSim's harts
	 * share a coherent counter so this is a no-op there.) */
	k_thread_cpu_pin(k_current_get(), 0);

	/* Touch the input buffer so it's resident, not zero-page-COW. */
	for (int i = 0; i < BENCH_MAX_RANGE; i++) {
		bench_in[i] = (float)i;
	}

	/* Local-sem baseline first — independent of pool state. */
	uint64_t sem_med = 0, sem_min = 0, sem_max = 0;
	bench_local_sem(&sem_med, &sem_min, &sem_max);

	/* Sweep N x range. We tear down + recreate the pool per N
	 * because pthreadpool_create takes a fixed worker count. */
	bench_row_t rows[BENCH_NWORKERS_COUNT * BENCH_RANGE_COUNT];
	int rcount = 0;
	for (int ni = 0; ni < BENCH_NWORKERS_COUNT; ni++) {
		int n = BENCH_NWORKERS[ni];
		pthreadpool_t pool = pthreadpool_create((size_t)n);
		if (!pool) {
			printf("FATAL: pthreadpool_create(%d) returned NULL\n",
			       n);
			sys_reboot(SYS_REBOOT_COLD);
			return -1;
		}
		for (int ri = 0; ri < BENCH_RANGE_COUNT; ri++) {
			run_one(&rows[rcount++], pool, n, BENCH_RANGES[ri]);
		}
		pthreadpool_destroy(pool);
	}

	/* Print the local-sem baseline as a synthetic row first so it
	 * lands inside the same parser block. n_workers=0 marks it. */
	bench_row_t base = {
		.harness = "k_sem_pingpong",
		.variant = "local",
		.n_workers = 0,
		.range = 0,
		.reps = BENCH_TIMED_REPS,
		.per_call_min = sem_min,
		.per_call_med = sem_med,
		.per_call_max = sem_max,
	};
	bench_emit_csv(&base, 1);
	bench_emit_csv(rows, rcount);

	printf("bench_pthreadpool: done\n");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
