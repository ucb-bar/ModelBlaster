/*
 * Copyright (c) 2026 Dima Nikiforov <vnikiforov@berkeley.edu>
 * SPDX-License-Identifier: Apache-2.0
 *
 * modelblaster_pool — minimal raw-pthread parallel-for pool for the modelblaster
 * runtime. Drop-in replacement for the subset of pthreadpool that the
 * generated `parallel_<op>` wrappers actually use:
 *
 *     pthreadpool_create(N)             -> modelblaster_pool_create(N)
 *     pthreadpool_destroy(p)            -> modelblaster_pool_destroy(p)
 *     pthreadpool_get_threads_count(p)  -> modelblaster_pool_get_threads_count(p)
 *     pthreadpool_parallelize_1d(p, fn, ctx, range, flags)
 *                                       -> modelblaster_pool_parallelize_1d(...)
 *
 * The microbench in modelblaster/microbench/threadpool/ established that
 * pthreadpool's wrapper costs ~13 M target cycles per dispatch on the
 * FireSim quad-rocket-saturn build, while a raw pthreads + k_sem
 * rendezvous (the pattern this lib promotes from
 * bench_pthreads_raw.c) costs ~20k. That's a ~650x reduction in
 * dispatch overhead and is the entire motivation for this lib.
 *
 * Implementation notes:
 *
 *  - Workers are POSIX pthreads spawned at create time and pinned via
 *    pthread_attr_setaffinity_np (vendored Phase A patch — see
 *    modelblaster/harness_multi/zephyr_patches/posix-affinity.patch). We use
 *    pthreads, not k_thread, so the API stays portable to non-Zephyr
 *    builds; the bench measured the POSIX layer cost as essentially
 *    zero on top of k_thread.
 *
 *  - We use Zephyr `k_sem` (not POSIX sem_t) for the rendezvous because
 *    the bench harness measured it as the cheapest cross-hart wakeup
 *    primitive available on the platform. The pool object is therefore
 *    Zephyr-only at runtime; any non-Zephyr port would swap k_sem for
 *    a POSIX equivalent and rely on the bench's "POSIX layer is free"
 *    finding.
 *
 *  - Master participates in its own slice (slice index 0). The master
 *    posts start sems for workers 1..N-1, runs slice 0, then takes done
 *    sems from workers 1..N-1. This matches pthreadpool's behavior
 *    where the calling thread is counted as one of the N workers.
 *
 *  - n_workers == 0 returns NULL. Callers (the codegen-emitted
 *    `parallel_<op>` wrappers) already check for NULL pool and run
 *    sequentially in that case — preserving the existing behavior with
 *    pthreadpool_create(0).
 *
 *  - n_workers == 1 returns a non-NULL pool whose
 *    modelblaster_pool_get_threads_count() reports 1; in that case
 *    parallelize_1d runs the function on the master with no fanout.
 */

#ifndef MODELBLASTER_RUNTIME_MODELBLASTER_POOL_H
#define MODELBLASTER_RUNTIME_MODELBLASTER_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct modelblaster_pool_state;
typedef struct modelblaster_pool_state *modelblaster_pool_t;

/* Create a pool with `n_workers` total threads (caller participates as
 * worker 0; n_workers-1 helper threads are spawned and pinned via
 * pthread_attr_setaffinity_np to harts 1..n_workers-1).
 *
 * Returns NULL on n_workers==0 (callers treat NULL as "run sequentially")
 * or on hard failure (out of memory, pthread_create failed). On any
 * partial failure during create we tear down anything we already
 * allocated and return NULL — never leak workers. */
modelblaster_pool_t modelblaster_pool_create(int n_workers);

/* Tear down: post a "die" signal to every worker, join them, free the
 * state. Safe on a NULL pointer (no-op). */
void modelblaster_pool_destroy(modelblaster_pool_t pool);

/* Total thread count (caller + helpers). pthreadpool_get_threads_count
 * returns the same convention. */
unsigned modelblaster_pool_get_threads_count(modelblaster_pool_t pool);

/* Parallel-for across [0, range): invoke fn(ctx, i) for every i. The
 * range is partitioned across `min(range, T)` slices where T is the
 * pool's thread count; helpers run their slices on their pinned harts,
 * the master runs slice 0 itself, then all done sems are gathered.
 *
 * `flags` is reserved for future use (e.g. matching pthreadpool's
 * NO_YIELD bit) — currently ignored.
 *
 * If pool == NULL, runs the loop sequentially on the calling thread. */
void modelblaster_pool_parallelize_1d(modelblaster_pool_t pool,
                                void (*fn)(void *ctx, size_t i),
                                void *ctx,
                                size_t range,
                                unsigned flags);

#ifdef __cplusplus
}
#endif

#endif /* MODELBLASTER_RUNTIME_MODELBLASTER_POOL_H */
