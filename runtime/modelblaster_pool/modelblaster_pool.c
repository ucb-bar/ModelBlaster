/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * modelblaster_pool implementation — see modelblaster_pool.h for design rationale
 * and the microbench under modelblaster/microbench/threadpool/ for the
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
 * (On Linux each helper first places itself -- see mb_placement.h -- and
 * reports to the creator through a per-helper `ready` sem before looping.)
 *
 * Workers loop forever waiting on `start`; on shutdown, master sets
 * `quit=1` then gives every start sem so each worker wakes, sees the
 * flag, and returns from its top-level fn (then pthread_join collects
 * them in destroy).
 */

#include "modelblaster_pool.h"

#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MB_PLATFORM_POSIX
#include "mb_posix.h"   /* hosted Linux: k_sem -> sem_t, see runtime/mb_posix.h */
#elif defined(MODELBLASTER_PLATFORM_LINUX)
#include "mb_posix_compat.h"  /* Agustin's K1 harness build */
#else
#include <zephyr/kernel.h>
#endif
#include "mb_placement.h"  /* self-placement of helpers on hosted Linux (K3 A100) */

/* Hard cap so we can size the embedded sem arrays without dynamic
 * malloc per worker. The bench under modelblaster/microbench/threadpool/
 * uses 4 (CONFIG_MP_MAX_NUM_CPUS on the FireSim quad-rocket-saturn
 * build); pick a generous bound — 16 covers any plausible Chipyard
 * hwconfig we'd build for. Pools that ask for more than this fail
 * create() returning NULL. */
/* MODELBLASTER_POOL_MAX_WORKERS (16) is defined in modelblaster_pool.h. */

struct modelblaster_pool_state {
	int n_workers;          /* total thread count (master + helpers) */
	int quit;               /* set on destroy; workers re-read after wake */
	int is_gang;            /* 1: no threads; helpers are gang_serve() callers */

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
	pthread_t      tids[MODELBLASTER_POOL_MAX_WORKERS];
	pthread_attr_t attrs[MODELBLASTER_POOL_MAX_WORKERS];
	struct k_sem   start[MODELBLASTER_POOL_MAX_WORKERS];
	struct k_sem   done[MODELBLASTER_POOL_MAX_WORKERS];
#if MB_PLACEMENT_LINUX
	/* Per-helper placement handshake (Linux only). A helper places ITSELF
	 * (it needs its own TID to register on a K3 A100 CPU), so the creator
	 * must wait for the verdict: `ready[w]` is given once the helper has
	 * finished mb_place_current_thread, `place_rc[w]` holds its result. */
	struct k_sem   ready[MODELBLASTER_POOL_MAX_WORKERS];
	int            place_rc[MODELBLASTER_POOL_MAX_WORKERS];
#endif
};

/* Worker arg packs (pool_state*, my worker index) without forcing the
 * helper to read its index out of the state struct (avoids a small
 * per-iteration indirection). */
struct modelblaster_pool_worker_arg {
	struct modelblaster_pool_state *pool;
	int wid;
	int hart;   /* CPU this helper must place itself on (Linux path) */
};

static void *modelblaster_pool_worker_fn(void *arg_)
{
	struct modelblaster_pool_worker_arg *wa = (struct modelblaster_pool_worker_arg *)arg_;
	struct modelblaster_pool_state *p = wa->pool;
	int wid = wa->wid;

#if MB_PLACEMENT_LINUX
	/* Place first, then tell the creator how it went. Not via
	 * pthread_attr_setaffinity_np: on K3 an A100 thread must write its own
	 * TID to /proc/set_ai_thread BEFORE it may pin, and there is no TID until
	 * the thread runs. A failed helper exits without entering the loop; the
	 * creator joins it. Fail loud: a helper left on the wrong core would make
	 * every multi-core number a fiction. */
	int prc = mb_place_current_thread(wa->hart);
	p->place_rc[wid] = prc;
	k_sem_give(&p->ready[wid]);
	if (prc != 0) {
		free(wa);
		return NULL;
	}
#endif

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

/* ---- static gangs: see modelblaster_pool.h ---------------------------------
 * Shared state is one cache-line-aligned block per gang: the job descriptor
 * (written by the master only), one event word (sequence number + waiter
 * flag: an eventcount; a futex on Linux, spun on under Zephyr -- see
 * xs_os.h), and one done sequence per helper (each written by its helper
 * only). Portable since 2026-11-26 (xs_os.h). */
#include <stdatomic.h>
#include "xs_os.h"

#ifndef MB_GANG_SPIN_ITERS
#define MB_GANG_SPIN_ITERS 20000      /* about 100 us before a waiter sleeps */
#endif

struct mb_gang {
	_Alignas(64) _Atomic unsigned long long open_tok;
	_Atomic unsigned long long closed_tok;
	_Atomic unsigned long long job_tok;
	_Atomic uint32_t job_seq;
	_Alignas(64) _Atomic uint32_t evt;       /* futex word */
	_Atomic uint32_t waiters;
	struct { _Alignas(64) _Atomic uint32_t done_seq; uint32_t served; }
		h[MODELBLASTER_POOL_MAX_WORKERS];
};

static inline struct mb_gang *mb_gang_of(struct modelblaster_pool_state *p)
{
	return (struct mb_gang *)(void *)(p + 1);
}

static inline void mb_gang_signal(struct mb_gang *g)
{
	atomic_fetch_add(&g->evt, 1);
	if (atomic_load(&g->waiters) != 0)
		xs_os_evt_wake(&g->evt, 1);
}

/* Wait until cond(g) holds: spin `spin` iterations, then sleep on the event
 * word. The waiter count is raised BEFORE the last check and the signaller
 * bumps the word BEFORE reading the count, so a change is never missed. */
#define MB_GANG_WAIT(g, cond, spin)                                            \
	do {                                                                   \
		long sp_ = (spin);                                             \
		while (!(cond)) {                                              \
			if (sp_-- > 0) continue;                               \
			uint32_t s_ = atomic_load(&(g)->evt);                  \
			atomic_fetch_add(&(g)->waiters, 1);                    \
			if (!(cond))                                           \
				(void)xs_os_evt_wait(&(g)->evt, s_, 0);        \
			atomic_fetch_sub(&(g)->waiters, 1);                    \
		}                                                              \
	} while (0)

modelblaster_pool_t modelblaster_pool_create_gang(int n_workers)
{
	if (n_workers <= 1 || n_workers > MODELBLASTER_POOL_MAX_WORKERS)
		return NULL;
	size_t sz = sizeof(struct modelblaster_pool_state) + sizeof(struct mb_gang) + 64;
	struct modelblaster_pool_state *p = NULL;
	p = aligned_alloc(64, (sz + 63) & ~(size_t)63);
	if (p == NULL)
		return NULL;
	memset(p, 0, sz);
	p->n_workers = n_workers;
	p->is_gang = 1;
	return p;
}

void modelblaster_pool_gang_open(modelblaster_pool_t p, unsigned long long token)
{
	struct mb_gang *g = mb_gang_of(p);
	atomic_store(&g->open_tok, token);
	mb_gang_signal(g);
}

void modelblaster_pool_gang_close(modelblaster_pool_t p, unsigned long long token)
{
	struct mb_gang *g = mb_gang_of(p);
	atomic_store(&g->closed_tok, token);
	mb_gang_signal(g);
}

static void mb_gang_run_slice(struct modelblaster_pool_state *p, int wid)
{
	if (wid >= p->n_active_slices)
		return;
	size_t per = p->range / (size_t)p->n_active_slices;
	size_t rem = p->range % (size_t)p->n_active_slices;
	size_t s = (size_t)wid * per + (size_t)(wid < (int)rem ? wid : (int)rem);
	size_t e = s + per + (wid < (int)rem ? 1 : 0);
	for (size_t i = s; i < e; i++)
		p->fn(p->ctx, i);
}

void modelblaster_pool_gang_serve(modelblaster_pool_t p, int wid,
				  unsigned long long token, int spin_first)
{
	struct mb_gang *g = mb_gang_of(p);
	/* the master has not reached this entry yet: wait for it */
	MB_GANG_WAIT(g, atomic_load(&g->open_tok) >= token ||
			atomic_load(&g->closed_tok) >= token,
		     spin_first ? MB_GANG_SPIN_ITERS : 0);
	for (;;) {
		uint32_t j = atomic_load(&g->job_seq);
		if (j != g->h[wid].served && atomic_load(&g->job_tok) == token) {
			mb_gang_run_slice(p, wid);
			g->h[wid].served = j;
			atomic_store(&g->h[wid].done_seq, j);
			mb_gang_signal(g);
			continue;
		}
		if (atomic_load(&g->closed_tok) >= token)
			return;
		MB_GANG_WAIT(g, (atomic_load(&g->job_seq) != g->h[wid].served &&
				 atomic_load(&g->job_tok) == token) ||
				atomic_load(&g->closed_tok) >= token,
			     MB_GANG_SPIN_ITERS);
	}
}

static void mb_gang_parallelize(struct modelblaster_pool_state *p,
				void (*fn)(void *, size_t), void *ctx, size_t range)
{
	struct mb_gang *g = mb_gang_of(p);
	int n_slices = (range < (size_t)p->n_workers) ? (int)range : p->n_workers;
	if (n_slices <= 1) {
		for (size_t i = 0; i < range; i++)
			fn(ctx, i);
		return;
	}
	p->fn = fn;
	p->ctx = ctx;
	p->range = range;
	p->n_active_slices = n_slices;
	atomic_store(&g->job_tok, atomic_load(&g->open_tok));
	uint32_t j = atomic_fetch_add(&g->job_seq, 1) + 1;
	mb_gang_signal(g);
	mb_gang_run_slice(p, 0);
	/* Every helper acknowledges every job (also one it has no slice of),
	 * so the next job never overwrites a descriptor still being read. */
	for (int w = 1; w < p->n_workers; w++)
		MB_GANG_WAIT(g, atomic_load(&g->h[w].done_seq) == j, MB_GANG_SPIN_ITERS);
}

modelblaster_pool_t modelblaster_pool_create(int n_workers)
{
	if (n_workers <= 0 || n_workers > MODELBLASTER_POOL_MAX_WORKERS) {
		return NULL;
	}
	int harts[MODELBLASTER_POOL_MAX_WORKERS];
	for (int w = 0; w < n_workers; w++) {
		harts[w] = w;
	}
	return modelblaster_pool_create_on_harts(n_workers, harts);
}

modelblaster_pool_t modelblaster_pool_create_on_harts(int n_workers,
						      const int *harts)
{
	if (n_workers <= 0) {
		/* Match pthreadpool_create(0) behavior: caller treats NULL
		 * as "no pool, run sequentially". */
		return NULL;
	}
	if (n_workers > MODELBLASTER_POOL_MAX_WORKERS) {
		return NULL;
	}

	struct modelblaster_pool_state *p = calloc(1, sizeof(*p));
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
#if MB_PLACEMENT_LINUX
		k_sem_init(&p->ready[w], 0, 1);
#endif
	}

	/* Spawn helpers, each pinned to hart `w`. Master is responsible
	 * for being on hart 0 (callers that care should
	 * k_thread_cpu_pin(k_current_get(), 0) before creating the pool —
	 * we don't do it here because we don't want to force a global
	 * affinity change on whoever instantiated us). */
	int spawned = 0;
	for (int w = 1; w < n_workers; w++) {
		struct modelblaster_pool_worker_arg *wa = malloc(sizeof(*wa));
		if (wa == NULL) {
			goto fail;
		}
		wa->pool = p;
		wa->wid = w;
		wa->hart = harts[w];

		if (pthread_attr_init(&p->attrs[w]) != 0) {
			free(wa);
			goto fail;
		}
#if defined(CONFIG_POSIX_THREADS_AFFINITY) && !MB_PLACEMENT_LINUX
		cpu_set_t cs;
		CPU_ZERO(&cs);
		CPU_SET(harts[w], &cs);
		if (pthread_attr_setaffinity_np(&p->attrs[w], sizeof(cs), &cs) != 0) {
			pthread_attr_destroy(&p->attrs[w]);
			free(wa);
			goto fail;
		}
#endif
		if (pthread_create(&p->tids[w], &p->attrs[w],
				   modelblaster_pool_worker_fn, wa) != 0) {
			pthread_attr_destroy(&p->attrs[w]);
			free(wa);
			goto fail;
		}
		spawned++;
#if MB_PLACEMENT_LINUX
		/* Wait for this helper's placement verdict. On failure the
		 * helper has already exited its function (and freed `wa`);
		 * the fail path below joins it like any other. */
		k_sem_take(&p->ready[w], K_FOREVER);
		if (p->place_rc[w] != 0) {
			fprintf(stderr, "modelblaster_pool: helper %d could not be "
					"placed on cpu %d; aborting pool create\n",
				w, harts[w]);
			goto fail;
		}
#endif
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

void modelblaster_pool_destroy(modelblaster_pool_t pool)
{
	if (pool == NULL) {
		return;
	}
	if (pool->is_gang) {
		free(pool);
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

unsigned modelblaster_pool_get_threads_count(modelblaster_pool_t pool)
{
	if (pool == NULL) {
		return 0;
	}
	return (unsigned)pool->n_workers;
}

void modelblaster_pool_parallelize_1d(modelblaster_pool_t pool,
				void (*fn)(void *, size_t),
				void *ctx,
				size_t range,
				unsigned flags)
{
	(void)flags;
	if (range == 0) {
		return;
	}
	if (pool != NULL && pool->is_gang) {
		mb_gang_parallelize(pool, fn, ctx, range);
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
