/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * Bench #2: native Zephyr k_thread + k_sem dispatch overhead.
 *
 * Pattern follows samples/tiled_matmul_mt_pool: pre-spawn N k_threads,
 * each pinned via k_thread_cpu_pin to a distinct hart, each spinning
 * in a `for(;;) k_sem_take(start_sem); ... k_sem_give(done_sem);` loop.
 * The master dispatches by giving every start_sem then taking every
 * done_sem in order.
 *
 * No POSIX, no pthreadpool: this is the absolute baseline for
 * cross-hart work dispatch using only Zephyr kernel objects. The cost
 * here is essentially:
 *   - master k_sem_give × N   (each potentially triggers a wake on the
 *                              target hart's run-queue, plus an IPI if
 *                              the target hart was idle)
 *   - workers run their body
 *   - master k_sem_take × N   (each may sleep until the worker calls
 *                              k_sem_give on done_sem, then IPI back)
 *
 * Crucially we instrument:
 *   wake_master_t   = rdcycle() right before the first start_sem give
 *   first_worker_t  = the earliest cycle any worker observed wakeup
 *                     (stamped by the worker via rdcycle right after
 *                     it returns from k_sem_take). One shared atomic
 *                     "min" register, written if smaller.
 *   last_done_t     = the largest cycle stamped by any worker right
 *                     after it called k_sem_give(done_sem)
 *   wake_observed_t = rdcycle() right after the master finishes its
 *                     last k_sem_take(done_sem)
 *
 * From those:
 *   wake_to_first_worker = first_worker_t - wake_master_t
 *   wake_to_all_finished = last_done_t   - wake_master_t
 *   finish_to_observed   = wake_observed_t - last_done_t
 */

#include <pthread.h>            /* only for the affinity attr in raw-pthread bench;
				 * unused here but the build needs the include
				 * present in case POSIX_API is on. We still
				 * exclude POSIX threads in this binary's prj. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <zephyr/arch/cpu.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/reboot.h>

#include "bench_common.h"

#define BENCH_MAX_WORKERS 4
#define BENCH_WORKER_STACK_SIZE 16384

K_THREAD_STACK_ARRAY_DEFINE(bench_worker_stacks,
			    BENCH_MAX_WORKERS,
			    BENCH_WORKER_STACK_SIZE);
static struct k_thread bench_worker_data[BENCH_MAX_WORKERS];
static k_tid_t bench_worker_tids[BENCH_MAX_WORKERS];

static struct k_sem start_sem[BENCH_MAX_WORKERS];
static struct k_sem done_sem[BENCH_MAX_WORKERS];

/* Per-iteration shared state. Written by master, read+stamped by
 * workers, written by workers, read by master after the take loop. */
struct bench_iter_state {
	const float *in;
	volatile float *out;
	size_t range;
	int n_workers;
	int active;	/* set to 1 when shutting workers down */

	/* Cycle stamps (all rdcycle()). volatile so the master sees the
	 * worker's stores after the take loop, without us needing
	 * explicit barriers — we're on the same coherent memory and the
	 * sem give/take pair already orders these reads. */
	volatile uint64_t first_worker;
	volatile uint64_t last_done;
};
static volatile struct bench_iter_state iter;

static void worker_fn(void *arg1, void *arg2, void *arg3)
{
	int wid = (int)(uintptr_t)arg1;
	(void)arg2; (void)arg3;
	for (;;) {
		k_sem_take(&start_sem[wid], K_FOREVER);
		/* mtime, NOT rdcycle: rdcycle is per-hart and isn't
		 * comparable to the master's sample read on hart 0.
		 * mtime is a single global counter (CLINT mtime memory-
		 * mapped) so timestamps from different harts compose. */
		uint64_t t_wake = bench_rdmtime();
		if (iter.active == 0) {
			/* Shutdown signal — exit the worker. We won't
			 * give done_sem on this final iteration; main
			 * uses k_thread_join. */
			return;
		}
		/* Stamp first-worker observation. We use a CAS-free
		 * "if smaller" update because we only care about the
		 * earliest wake, and racy updates can only ever
		 * over-estimate (write a slightly later value); the
		 * 0-initial sentinel is special-cased. */
		uint64_t cur = iter.first_worker;
		if (cur == 0 || t_wake < cur) {
			iter.first_worker = t_wake;
		}

		/* Slice [start, end) for this worker. */
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
		/* Update last_done with max — same racy-but-monotonic
		 * pattern as first_worker. */
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

	/* Warmup. */
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

		/* per-call cycles: rdcycle on master only. Local-hart
		 * counter, gives high-resolution dispatch cost. */
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

		/* Breakdown uses mtime (global), so master m0 / m1 and
		 * worker first_worker / last_done are directly comparable.
		 * mtime ticks at 1 MHz on FireSim quad-rocket-saturn so
		 * each unit = 1000 target cycles at 1 GHz. */
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

	row->harness = "k_thread";
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
	printf("bench_zephyr_threads: starting on %s "
	       "(MP_MAX_NUM_CPUS=%d)\n",
	       CONFIG_BOARD_TARGET, (int)CONFIG_MP_MAX_NUM_CPUS);
	/* Pin main to hart 0. Without this, when the master blocks on
	 * k_sem_take the scheduler is free to migrate it to another
	 * hart, and our t0/t1 rdcycle pair lands on different per-hart
	 * counters — producing nonsense per-call deltas (negative when
	 * cast back from uint64). On real hardware the cycle counters
	 * are coherent so this isn't an issue, but keeping the pin is
	 * a no-cost robustness win. */
	k_thread_cpu_pin(k_current_get(), 0);
	for (int i = 0; i < BENCH_MAX_RANGE; i++) {
		bench_in[i] = (float)i;
	}

	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		k_sem_init(&start_sem[w], 0, 1);
		k_sem_init(&done_sem[w], 0, 1);
	}

	/* Spawn all 4 workers up front, pinned to harts 0..3 (master
	 * runs on hart 0 too, but Zephyr SMP scheduler is fine
	 * coexisting; the master only ever spends time inside k_sem_*).
	 * On FireSim quad-rocket-saturn, harts are 0..3 and the prj
	 * overlay sets MP_MAX_NUM_CPUS=4 (see backends/firesim_chipyard.conf). */
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		bench_worker_tids[w] = k_thread_create(
			&bench_worker_data[w], bench_worker_stacks[w],
			BENCH_WORKER_STACK_SIZE, worker_fn,
			(void *)(uintptr_t)w, NULL, NULL,
			K_PRIO_PREEMPT(1), 0, K_FOREVER);
		k_thread_cpu_pin(bench_worker_tids[w], w);
		k_thread_start(bench_worker_tids[w]);
	}

	/* Sweep. Each row uses the first n_workers of the pre-spawned
	 * thread set; the others stay parked on their start_sem. */
	bench_row_t rows[BENCH_NWORKERS_COUNT * BENCH_RANGE_COUNT];
	int rcount = 0;
	for (int ni = 0; ni < BENCH_NWORKERS_COUNT; ni++) {
		int n = BENCH_NWORKERS[ni];
		for (int ri = 0; ri < BENCH_RANGE_COUNT; ri++) {
			run_one(&rows[rcount++], n, BENCH_RANGES[ri]);
		}
	}

	/* Tell every worker to exit, then join. */
	iter.active = 0;
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		k_sem_give(&start_sem[w]);
	}
	for (int w = 0; w < BENCH_MAX_WORKERS; w++) {
		k_thread_join(bench_worker_tids[w], K_FOREVER);
	}

	bench_emit_csv(rows, rcount);

	printf("bench_zephyr_threads: done\n");
	sys_reboot(SYS_REBOOT_COLD);
	return 0;
}
