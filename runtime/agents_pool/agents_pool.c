/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * agents_pool implementation — see agents_pool.h for design rationale
 * and the microbench under agents/microbench/threadpool/ for the
 * measurements that motivated this layout.
 *
 * Lifecycle of one parallelize_1d call:
 *
 *     master                          helpers (1..N-1, pre-pinned)
 *     ------                          ----------------------------
 *     publish (fn, ctx, range)
 *     k_sem_give(start[1..N-1])  ---> k_sem_take(start[wid])
 *     run slice 0 itself              run slice wid
 *     k_sem_take(done[1..N-1])  <---  k_sem_give(done[wid])
 *
 * Workers loop forever waiting on `start`; on shutdown, master sets
 * `quit=1` then gives every start sem so each worker wakes, sees the
 * flag, and returns from its top-level fn (then pthread_join collects
 * them in destroy).
 */

#include "agents_pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <zephyr/kernel.h>

/* Hard cap so we can size the embedded sem arrays without dynamic
 * malloc per worker. The bench under agents/microbench/threadpool/
 * uses 4 (CONFIG_MP_MAX_NUM_CPUS on the FireSim quad-rocket-saturn
 * build); pick a generous bound — 16 covers any plausible Chipyard
 * hwconfig we'd build for. Pools that ask for more than this fail
 * create() returning NULL. */
#define AGENTS_POOL_MAX_WORKERS 16

struct agents_pool_state {
	int n_workers;          /* total thread count (master + helpers) */
	int quit;               /* set on destroy; workers re-read after wake */

	/* Job descriptor — published by master under no lock; helpers only
	 * read it after their start sem fires (sem-give is the
	 * happens-before edge), so plain volatile is sufficient — no
	 * additional barriers required. */
	void (*fn)(void *, size_t);
	void *ctx;
	size_t range;
	int n_active_slices;    /* may be < n_workers if range < n_workers */

	/* Per-helper state. Index 0 is unused (= master); index w in
	 * [1..n_workers-1] is the w'th helper. Sized for the max so we
	 * can place the structs in a single allocation. */
	pthread_t      tids[AGENTS_POOL_MAX_WORKERS];
	pthread_attr_t attrs[AGENTS_POOL_MAX_WORKERS];
	struct k_sem   start[AGENTS_POOL_MAX_WORKERS];
	struct k_sem   done[AGENTS_POOL_MAX_WORKERS];
};

/* Worker arg packs (pool_state*, my worker index) without forcing the
 * helper to read its index out of the state struct (avoids a small
 * per-iteration indirection). */
struct agents_pool_worker_arg {
	struct agents_pool_state *pool;
	int wid;
};

static void *agents_pool_worker_fn(void *arg_)
{
	struct agents_pool_worker_arg *wa = (struct agents_pool_worker_arg *)arg_;
	struct agents_pool_state *p = wa->pool;
	int wid = wa->wid;

	for (;;) {
		k_sem_take(&p->start[wid], K_FOREVER);
		if (p->quit) {
			free(wa);
			return NULL;
		}

		/* Slice [start, end) of [0, range). When range < n_workers
		 * (e.g. parallel_linear with N == n_workers - 1) the
		 * master only posts start sems for slices 0..range-1; this
		 * helper might still get woken via shutdown, which the
		 * quit-check above already handled. The active-slice gate
		 * is also defended on the helper side: if our wid >=
		 * n_active_slices, do nothing. */
		if (wid < p->n_active_slices) {
			size_t per = p->range / (size_t)p->n_active_slices;
			size_t rem = p->range % (size_t)p->n_active_slices;
			size_t s = (size_t)wid * per +
				   (size_t)(wid < (int)rem ? wid : (int)rem);
			size_t e = s + per + (wid < (int)rem ? 1 : 0);
			for (size_t i = s; i < e; i++) {
				p->fn(p->ctx, i);
			}
		}

		k_sem_give(&p->done[wid]);
	}
}

agents_pool_t agents_pool_create(int n_workers)
{
	if (n_workers <= 0) {
		/* Match pthreadpool_create(0) behavior: caller treats NULL
		 * as "no pool, run sequentially". */
		return NULL;
	}
	if (n_workers > AGENTS_POOL_MAX_WORKERS) {
		return NULL;
	}

	struct agents_pool_state *p = calloc(1, sizeof(*p));
	if (p == NULL) {
		return NULL;
	}
	p->n_workers = n_workers;
	p->quit = 0;

	/* Init every sem up to n_workers-1 (helpers). Master doesn't use
	 * its slot. */
	for (int w = 1; w < n_workers; w++) {
		k_sem_init(&p->start[w], 0, 1);
		k_sem_init(&p->done[w], 0, 1);
	}

	/* Spawn helpers, each pinned to hart `w`. Master is responsible
	 * for being on hart 0 (callers that care should
	 * k_thread_cpu_pin(k_current_get(), 0) before creating the pool —
	 * we don't do it here because we don't want to force a global
	 * affinity change on whoever instantiated us). */
	int spawned = 0;
	for (int w = 1; w < n_workers; w++) {
		struct agents_pool_worker_arg *wa = malloc(sizeof(*wa));
		if (wa == NULL) {
			goto fail;
		}
		wa->pool = p;
		wa->wid = w;

		if (pthread_attr_init(&p->attrs[w]) != 0) {
			free(wa);
			goto fail;
		}
#ifdef CONFIG_POSIX_THREADS_AFFINITY
		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(w, &cs);
		if (pthread_attr_setaffinity_np(&p->attrs[w], sizeof(cs), &cs) != 0) {
			pthread_attr_destroy(&p->attrs[w]);
			free(wa);
			goto fail;
		}
#endif
		if (pthread_create(&p->tids[w], &p->attrs[w],
				   agents_pool_worker_fn, wa) != 0) {
			pthread_attr_destroy(&p->attrs[w]);
			free(wa);
			goto fail;
		}
		spawned++;
	}
	return p;

fail:
	/* Tear down any helpers we already spawned: signal quit, post
	 * their start sems so they wake and return, join. */
	if (spawned > 0) {
		p->quit = 1;
		for (int w = 1; w <= spawned; w++) {
			k_sem_give(&p->start[w]);
		}
		for (int w = 1; w <= spawned; w++) {
			pthread_join(p->tids[w], NULL);
			pthread_attr_destroy(&p->attrs[w]);
		}
	}
	free(p);
	return NULL;
}

void agents_pool_destroy(agents_pool_t pool)
{
	if (pool == NULL) {
		return;
	}
	pool->quit = 1;
	for (int w = 1; w < pool->n_workers; w++) {
		k_sem_give(&pool->start[w]);
	}
	for (int w = 1; w < pool->n_workers; w++) {
		pthread_join(pool->tids[w], NULL);
		pthread_attr_destroy(&pool->attrs[w]);
	}
	free(pool);
}

unsigned agents_pool_get_threads_count(agents_pool_t pool)
{
	if (pool == NULL) {
		return 0;
	}
	return (unsigned)pool->n_workers;
}

void agents_pool_parallelize_1d(agents_pool_t pool,
				void (*fn)(void *, size_t),
				void *ctx,
				size_t range,
				unsigned flags)
{
	(void)flags;
	if (range == 0) {
		return;
	}
	if (pool == NULL || pool->n_workers <= 1) {
		/* No fanout — just run the loop ourselves. Matches
		 * pthreadpool's behavior when N==1 (or NULL). */
		for (size_t i = 0; i < range; i++) {
			fn(ctx, i);
		}
		return;
	}

	/* Number of slices = min(range, n_workers). When range <
	 * n_workers, only the first `range` workers are active; the
	 * remaining helpers are NOT signaled (their start sem stays
	 * down). This matches our generated parallel_<op> wrappers,
	 * which only ever invoke parallelize_1d with range ==
	 * pool_threads_count anyway, but it's worth being correct in
	 * the off chance a future kernel passes a smaller range. */
	int n_slices = (range < (size_t)pool->n_workers)
		       ? (int)range
		       : pool->n_workers;

	/* Publish the job. Plain stores: the sem-give below is the
	 * happens-before edge and forces these to be visible to helpers
	 * before they wake. */
	pool->fn = fn;
	pool->ctx = ctx;
	pool->range = range;
	pool->n_active_slices = n_slices;

	/* Wake helpers 1..n_slices-1. */
	for (int w = 1; w < n_slices; w++) {
		k_sem_give(&pool->start[w]);
	}

	/* Master runs slice 0 itself. */
	{
		size_t per = range / (size_t)n_slices;
		size_t rem = range % (size_t)n_slices;
		size_t s = 0;
		size_t e = per + (0 < (int)rem ? 1 : 0);
		for (size_t i = s; i < e; i++) {
			fn(ctx, i);
		}
	}

	/* Gather the helpers. */
	for (int w = 1; w < n_slices; w++) {
		k_sem_take(&pool->done[w], K_FOREVER);
	}
}
