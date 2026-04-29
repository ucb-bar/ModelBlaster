/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bench #3: raw POSIX pthreads (no pthreadpool wrapper) over Zephyr's
 * POSIX layer.
 *
 * Spawns N pthreads, each pinned to a hart via
 * pthread_attr_setaffinity_np (the vendored Phase-A patch — see
 * agents/harness_multi/zephyr_patches/posix-affinity.patch). Each
 * worker spins on a per-worker semaphore, processes its slice, signals
 * a per-worker done semaphore. The master gives all start sems then
 * takes all done sems.
 *
 * Functionally identical wake/wait pattern to bench_zephyr_threads but
 * crossing pthread boundaries — `pthread_create` paths through
 * Zephyr's POSIX shim onto k_thread under the hood. Comparing this
 * harness's per-call cycles to bench_zephyr_threads.c isolates the
 * POSIX-layer overhead; comparing it to bench_pthreadpool isolates
 * pthreadpool's own wrapper overhead (queue management, command-state
 * machine, FPU-save fences) on top of POSIX.
 *
 * We use Zephyr k_sem (not POSIX sem_t) for the rendezvous so the
 * primitive cost matches bench_zephyr_threads. The variable being
 * isolated here is *thread spawn / scheduling layer*, not the sem.
 */

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

#define BENCH_MAX_WORKERS 4

static struct k_sem start_sem[BENCH_MAX_WORKERS];
static struct k_sem done_sem[BENCH_MAX_WORKERS];

struct bench_iter_state {
	const float *in;
	volatile float *out;
	size_t range;
	int n_workers;
	int active;
	volatile uint64_t first_worker;
	volatile uint64_t last_done;
};
static volatile struct bench_iter_state iter;

static void *worker_fn(void *arg)
{
	int wid = (int)(uintptr_t)arg;
	for (;;) {
		k_sem_take(&start_sem[wid], K_FOREVER);
		/* mtime, NOT rdcycle — see bench_zephyr_threads.c for
		 * the cross-hart counter caveat. */
		uint64_t t_wake = bench_rdmtime();
		if (iter.active == 0) {
			return NULL;
		}
		uint64_t cur = iter.first_worker;
		if (cur == 0 || t_wake < cur) {
			iter.first_worker = t_wake;
		}
		size_t r = iter.range;
		size_t per = r / (size_t)iter.n_workers;
		size_t rem = r % (size_t)iter.n_workers;
		size_t start = (size_t)wid * per +
				(size_t)(wid < (int)rem ? wid : (int)rem);
		size_t end = start + per + (wid < (int)rem ? 1 : 0);
		bench_workload_t wl = { .in = iter.in, .out = iter.out,
					.range = r };
		if (wid < iter.n_workers) {
			bench_apply(&wl, start, end);
		}
		uint64_t t_done = bench_rdmtime();
		uint64_t prev = iter.last_done;
		if (t_done > prev) {
			iter.last_done = t_done;
		}
		k_sem_give(&done_sem[wid]);
	}
}

static float bench_in[BENCH_MAX_RANGE];
static volatile float bench_out[BENCH_MAX_RANGE];

static void run_one(bench_row_t *row, int n_workers, int range)
{
	iter.in = bench_in;
	iter.out = bench_out;
	iter.n_workers = n_workers;
	iter.active = 1;

	for (int i = 0; i < BENCH_WARMUP_REPS; i++) {
		iter.range = (size_t)range;
		iter.first_worker = 0;
		iter.last_done = 0;
		for (int w = 0; w < n_workers; w++) {
			k_sem_give(&start_sem[w]);
		}
		for (int w = 0; w < n_workers; w++) {
			k_sem_take(&done_sem[w], K_FOREVER);
		}
	}

	static uint64_t per_call[BENCH_TIMED_REPS];
	uint64_t sum_w2f = 0, sum_w2a = 0, sum_f2o = 0;
	for (int i = 0; i < BENCH_TIMED_REPS; i++) {
		iter.range = (size_t)range;
		iter.first_worker = 0;
		iter.last_done = 0;

		/* per-call: rdcycle on master (per-hart, but master is
		 * pinned to hart 0 so values are self-consistent).
		 * Breakdown: mtime (global) for the cross-hart deltas. */
		uint64_t t0 = bench_rdcycle();
		uint64_t m0 = bench_rdmtime();
		for (int w = 0; w < n_workers; w++) {
			k_sem_give(&start_sem[w]);
		}
		for (int w = 0; w < n_workers; w++) {
			k_sem_take(&done_sem[w], K_FOREVER);
		}
		uint64_t m1 = bench_rdmtime();
		uint64_t t1 = bench_rdcycle();
		per_call[i] = t1 - t0;

		uint64_t fw = iter.first_worker;
		uint64_t ld = iter.last_done;
		if (fw && fw > m0) {
			sum_w2f += (fw - m0);
		}
		if (ld && ld > m0) {
			sum_w2a += (ld - m0);
		}
		if (ld && m1 > ld) {
			sum_f2o += (m1 - ld);
		}
	}

	row->harness = "pthreads_raw";
	row->variant = "k_sem";
	row->n_workers = n_workers;
	row->range = range;
	row->reps = BENCH_TIMED_REPS;
	row->per_call_min = bench_min(per_call, BENCH_TIMED_REPS);
	row->per_call_max = bench_max(per_call, BENCH_TIMED_REPS);
	row->per_call_med = bench_median_inplace(per_call, BENCH_TIMED_REPS);
	row->wake_to_first_worker = sum_w2f / BENCH_TIMED_REPS;
	row->wake_to_all_finished = sum_w2a / BENCH_TIMED_REPS;
	row->finish_to_observed = sum_f2o / BENCH_TIMED_REPS;
}

int main(void)
{
	printf("bench_pthreads_raw: starting on %s "
	       "(MP_MAX_NUM_CPUS=%d)\n",
	       CONFIG_BOARD_TARGET, (int)CONFIG_MP_MAX_NUM_CPUS);
	/* Pin master to hart 0 — see bench_zephyr_threads.c for why. */
	k_thread_cpu_pin(k_current_get(), 0);
	for (int i = 0; i < BENCH_MAX_RANGE; i++) {
		bench_in[i] = (float)i;
	}

	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		k_sem_init(&start_sem[w], 0, 1);
		k_sem_init(&done_sem[w], 0, 1);
	}

	pthread_t tids[BENCH_MAX_WORKERS];
	pthread_attr_t attrs[BENCH_MAX_WORKERS];

	/* Pre-spawn workers, pinned. Same as the affinity_smoketest. */
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		int rc = pthread_attr_init(&attrs[w]);
		if (rc != 0) {
			printf("FATAL: pthread_attr_init[%d] = %d\n", w, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}
		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(w, &cs);
		rc = pthread_attr_setaffinity_np(&attrs[w], sizeof(cs), &cs);
		if (rc != 0) {
			printf("FATAL: pthread_attr_setaffinity_np[%d] = %d\n",
			       w, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}
		rc = pthread_create(&tids[w], &attrs[w], worker_fn,
				    (void *)(uintptr_t)w);
		if (rc != 0) {
			printf("FATAL: pthread_create[%d] = %d\n", w, rc);
			sys_reboot(SYS_REBOOT_COLD);
		}
	}

	/* Sweep. */
	bench_row_t rows[BENCH_NWORKERS_COUNT * BENCH_RANGE_COUNT];
	int rcount = 0;
	for (int ni = 0; ni < BENCH_NWORKERS_COUNT; ni++) {
		int n = BENCH_NWORKERS[ni];
		for (int ri = 0; ri < BENCH_RANGE_COUNT; ri++) {
			run_one(&rows[rcount++], n, BENCH_RANGES[ri]);
		}
	}

	/* Shut workers down. */
	iter.active = 0;
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		k_sem_give(&start_sem[w]);
	}
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		pthread_join(tids[w], NULL);
		pthread_attr_destroy(&attrs[w]);
	}

	bench_emit_csv(rows, rcount);

	printf("bench_pthreads_raw: done\n");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
